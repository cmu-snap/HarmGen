# Compare two competing CCAs across Mahak's predicted harm surface, pick the network conditions where they differ the most (or the least)
#
# Usage:
#   python3 plot_mahak_compare.py <beta_cca> <alpha_old> <alpha_new> <time_limit> \
#                                 <flow_type> <mode> [top_n] [short_flow_harm_metric]
#
#   mode = max-improvement  -> largest harm(alpha_old) - harm(alpha_new)
#                              i.e. where alpha_new improves most  (paper Fig 16)
#   mode = min-difference   -> smallest |harm(alpha_old) - harm(alpha_new)|
#                              i.e. where alpha_new changes nothing (paper Fig 17)
#
#   long-flow:   python3 plot_mahak_compare.py cubic bbr1 bbr 240.0 long-flow  max-improvement 1
#   short-flow:  python3 plot_mahak_compare.py cubic bbr1 bbr 240.0 short-flow min-difference  1 recovery_time

import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

from tc_single_run import (
    ALPHA_START_TIMES_MAP,
    ExperimentDataError,
    calculate_queue_size,
    find_convergence_time,
    load_throughput_data,
    plot_long_flow_results_in_pdf,
    plot_short_flow_results_in_pdf,
    run_tc_long_flow_experiment,
    run_tc_short_flow_experiment,
)

if len(sys.argv) < 7:
    sys.exit("usage: plot_mahak_compare.py <beta_cca> <alpha_old> <alpha_new> "
             "<time_limit> <flow_type> <mode> [top_n] [short_flow_harm_metric] [min_harm]\n"
             "  mode: max-improvement | min-difference\n"
             "  min_harm: min-difference only -- ignore conditions where "
             "<alpha_old> harm is below this (default 0.1)")

beta_cca = sys.argv[1]
alpha_old = sys.argv[2]
alpha_new = sys.argv[3]
time_limit = float(sys.argv[4])
flow_type = sys.argv[5]
mode = sys.argv[6]
top_n = int(sys.argv[7]) if len(sys.argv) > 7 else 1
short_flow_harm_metric = sys.argv[8] if len(sys.argv) > 8 else "recovery_time"
min_harm = float(sys.argv[9]) if len(sys.argv) > 9 else 0.1

if flow_type not in ("long-flow", "short-flow"):
    sys.exit(f"flow_type must be long-flow or short-flow, got {flow_type}")
if mode not in ("max-improvement", "min-difference"):
    sys.exit(f"mode must be max-improvement or min-difference, got {mode}")

results_root = "./mahak_results"
old_csv = f"{results_root}/{beta_cca}-{alpha_old}-{time_limit}-{flow_type}/final_predictions.csv"
new_csv = f"{results_root}/{beta_cca}-{alpha_new}-{time_limit}-{flow_type}/final_predictions.csv"
for path in (old_csv, new_csv):
    if not os.path.exists(path):
        sys.exit(f"No predictions at {path}\nRun mahak.py for that pair first.")

df_old = pd.read_csv(old_csv)
df_new = pd.read_csv(new_csv)

keys = ["bw", "rtt", "bdp_ratio", "num_beta_flows", "num_alpha_flows"]
if flow_type == "short-flow" and "alpha_start" in df_old.columns:
    keys.append("alpha_start")
keys = [k for k in keys if k in df_old.columns and k in df_new.columns]

merged = df_old.merge(df_new, on=keys, suffixes=("_old", "_new"))
if merged.empty:
    sys.exit("The two prediction grids do not overlap -- were they run with the "
             "same -bw/-rtt/-bdp_ratio steps?")

merged["improvement"] = merged["harm_old"] - merged["harm_new"]
merged["abs_difference"] = merged["improvement"].abs()

if mode == "max-improvement":
    ranked = merged.sort_values("improvement", ascending=False)
else:
    eligible = merged[merged["harm_old"] >= min_harm]
    if eligible.empty:
        sys.exit(f"No condition has {alpha_old} harm >= {min_harm} "
                 f"(max predicted is {merged['harm_old'].max():.4f}). "
                 f"Lower min_harm.")
    print(f"min-difference: keeping {len(eligible)}/{len(merged)} conditions "
          f"where {alpha_old} harm >= {min_harm}")
    ranked = eligible.sort_values("abs_difference", ascending=True)

out_dir = f"{results_root}/compare-{alpha_old}-vs-{alpha_new}-{time_limit}-{flow_type}-{mode}"
os.makedirs(out_dir, exist_ok=True)
summary_path = f"{out_dir}/selected_conditions.csv"

print(f"Merged {len(merged)} predicted conditions from")
print(f"  {old_csv}")
print(f"  {new_csv}")
print(f"mode={mode}, looking for the top {top_n} usable condition(s) into {out_dir}\n")


def traces_needed(num_beta, num_alpha):
    """Trace files the *_in_pdf plotters read for one experiment."""
    names = ["beta_solo_beta_thr_0.txt"]
    if flow_type == "long-flow":
        names += [f"target_cc_beta_thr_{i}.txt" for i in range(num_beta)]
        names += [f"target_cc_alpha_thr_{i}.txt" for i in range(num_alpha)]
    else:
        for prefix in ("compete_a", "compete_b"):
            names += [f"{prefix}_beta_thr_{i}.txt" for i in range(num_beta)]
            names += [f"{prefix}_alpha_thr_{i}.txt" for i in range(num_alpha)]
    return names


def find_complete_exp_dir(alpha_cca, condition, num_beta, num_alpha):
    """Reuse Mahak's own experiment dir for this setting if it has every trace."""
    base = f"{results_root}/{beta_cca}-{alpha_cca}-{time_limit}-{flow_type}/experiments"
    for candidate in sorted(glob.glob(f"{base}/{condition}*")):
        if all(os.path.exists(f"{candidate}/{n}") for n in traces_needed(num_beta, num_alpha)):
            return candidate
    return None


RUN_PREFIXES = ["beta_solo"] + (["target_cc"] if flow_type == "long-flow"
                                else ["compete_a", "compete_b"])


def check_traces(exp_dir, num_beta):
    """Reject a run whose beta long flow recorded no throughput -- its harm is a
    missing measurement rather than competition."""
    for prefix in RUN_PREFIXES:
        total = 0.0
        for i in range(num_beta):
            try:
                _, rates = load_throughput_data(exp_dir, prefix, "beta", i)
            except (OSError, ValueError):
                return f"{prefix} beta trace {i} is missing or unreadable"
            if len(rates):
                total += float(np.sum(rates))
        if total <= 0:
            return f"{prefix} beta flow recorded no throughput"
    return None


rank = 0
skipped = []
selected_rows = []

for _, row in ranked.iterrows():
    if rank >= top_n:
        break

    bw = int(row["bw"])
    rtt = int(row["rtt"])
    bdp_ratio = float(row["bdp_ratio"])
    num_beta = int(row["num_beta_flows"])
    num_alpha = int(row["num_alpha_flows"])
    alpha_start = int(row["alpha_start"]) if "alpha_start" in row else 1
    queue = calculate_queue_size(bw, rtt, bdp_ratio)
    setting = f"{bw}bw-{rtt}rtt-{bdp_ratio}bdp ({queue}q), {num_beta} {beta_cca} vs {num_alpha} alpha"

    print(f"=== candidate: {setting} ===")
    print(f"    predicted harm  {alpha_old}={row['harm_old']:.4f}  "
          f"{alpha_new}={row['harm_new']:.4f}  "
          f"improvement={row['improvement']:.4f}", flush=True)

    prepared = []
    problem = None

    for alpha_cca in (alpha_old, alpha_new):
        # Same experiment directory naming as genetic-algorithm.py / mahak.py
        condition = (f"{bw}bw-{rtt}rtt-{queue}q-"
                     f"{beta_cca}-{num_beta}-{alpha_cca}-{num_alpha}")

        exp_dir = find_complete_exp_dir(alpha_cca, condition, num_beta, num_alpha)
        if exp_dir:
            print(f"  [{alpha_cca}] reusing traces in {exp_dir}", flush=True)
        else:
            exp_dir = f"{out_dir}/rank{rank + 1}-{condition}"
            os.makedirs(exp_dir, exist_ok=True)
            print(f"  [{alpha_cca}] running into {exp_dir}", flush=True)
            try:
                if flow_type == "long-flow":
                    alpha_starts = [0, 0, 0, 0, 0][:num_alpha]
                    run_tc_long_flow_experiment(
                        exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_starts,
                        beta_cca, alpha_cca, dualpi2=False)
                else:
                    run_tc_short_flow_experiment(
                        exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_start,
                        beta_cca, alpha_cca, short_flow_harm_metric, dualpi2=False)
            except ExperimentDataError as exc:
                problem = f"[{alpha_cca}] {exc}"
                break

        trace_problem = check_traces(exp_dir, num_beta)
        if trace_problem:
            problem = f"[{alpha_cca}] {trace_problem}"
            break

        conv_time = converged = None
        if flow_type == "long-flow":
            flows = {}
            for i in range(num_beta):
                t, r = load_throughput_data(exp_dir, "target_cc", "beta", i)
                flows[f"beta_{i}"] = {"times": t, "rates": list(r)}
            for i in range(num_alpha):
                t, r = load_throughput_data(exp_dir, "target_cc", "alpha", i)
                flows[f"alpha_{i}"] = {"times": t, "rates": list(r)}
            conv_time, converged = find_convergence_time(flows)

        prepared.append((alpha_cca, condition, exp_dir, conv_time, converged))

    if problem:
        print(f"  SKIP: {problem}\n", flush=True)
        skipped.append((setting, problem))
        continue

    rank += 1
    selected_rows.append(row)

    for alpha_cca, condition, exp_dir, conv_time, converged in prepared:
        if flow_type == "long-flow":
            plot_long_flow_results_in_pdf(
                exp_dir, bw, rtt, queue, num_beta, num_alpha,
                beta_cca, alpha_cca, conv_time, converged, dualpi2=False)

            src = f"{exp_dir}/long_flow_result.pdf"
            dst = f"{out_dir}/top{rank}_{condition}.pdf"
            if os.path.exists(src):
                shutil.move(src, dst)
                print(f"  wrote {dst}", flush=True)
            else:
                print(f"  WARNING: {src} was not produced", flush=True)
        else:
            alpha_starts = ALPHA_START_TIMES_MAP[alpha_start][:num_alpha]
            for panel in ("alpha", "beta"):
                plot_short_flow_results_in_pdf(
                    exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_starts,
                    beta_cca, alpha_cca, panel, short_flow_harm_metric,
                    dualpi2=False)

                src = f"{exp_dir}/short_flow_{panel}_new.pdf"
                dst = f"{out_dir}/top{rank}_{condition}_{panel}.pdf"
                if os.path.exists(src):
                    shutil.move(src, dst)
                    print(f"  wrote {dst}", flush=True)
                else:
                    print(f"  WARNING: {src} was not produced", flush=True)
    print(flush=True)

if selected_rows:
    pd.DataFrame(selected_rows).to_csv(summary_path, index=False)
    print(f"selection written to {summary_path}\n")

if skipped:
    print(f"Skipped {len(skipped)} higher-ranked condition(s) with unusable data:")
    for setting, problem in skipped:
        print(f"  {setting} -- {problem}")
    print()

if rank < top_n:
    print(f"WARNING: only {rank} usable condition(s) found, {top_n} were requested.")

print(f"Done. PDFs are in {out_dir}")
