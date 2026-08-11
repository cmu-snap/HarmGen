# Pick the highest-harm settings found by a HarmGen (genetic-algorithm.py) run.
#
# Usage: python3 plot_top_harm.py <beta_cca> <alpha_cca> <time_limit> <flow_type> [metric] [top_n]
#   long-flow:   python3 plot_top_harm.py cubic prague 30.0 long-flow
#   short-flow:  python3 plot_top_harm.py prague cubic 30.0 short-flow download_bytes

import glob
import os
import pickle
import shutil
import sys

import numpy as np

from tc_single_run import (
    ALPHA_START_TIMES_MAP,
    ExperimentDataError,
    find_convergence_time,
    load_throughput_data,
    plot_long_flow_results_in_pdf,
    plot_short_flow_results_in_pdf,
    run_tc_long_flow_experiment,
    run_tc_short_flow_experiment,
)

if len(sys.argv) < 5:
    sys.exit("usage: plot_top_harm.py <beta_cca> <alpha_cca> <time_limit> "
             "<flow_type> [short_flow_harm_metric] [top_n]")

beta_cca = sys.argv[1]
alpha_cca = sys.argv[2]
time_limit = float(sys.argv[3])
flow_type = sys.argv[4]
short_flow_harm_metric = sys.argv[5] if len(sys.argv) > 5 else "download_bytes"
top_n = int(sys.argv[6]) if len(sys.argv) > 6 else 3

if flow_type not in ("long-flow", "short-flow"):
    sys.exit(f"flow_type must be long-flow or short-flow, got {flow_type}")

result_dir = f"./results/{beta_cca}-{alpha_cca}-{time_limit}-{flow_type}"

RUN_PREFIXES = ["beta_solo"] + (["target_cc"] if flow_type == "long-flow"
                                else ["compete_a", "compete_b"])


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


def find_complete_exp_dir(condition, num_beta, num_alpha):
    """Return the GA's own experiment dir for this setting if it has every trace."""
    for candidate in sorted(glob.glob(f"{result_dir}/raygen-{beta_cca}-{alpha_cca}-*/{condition}")):
        if all(os.path.exists(f"{candidate}/{n}") for n in traces_needed(num_beta, num_alpha)):
            return candidate
    return None


def check_traces(exp_dir, num_beta):
    """Reject a run whose beta long flow recorded no throughput.

    That makes its absolute harm exactly 1.0 and the reported harm large, but
    it only means the flow never started or was never captured.
    """
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


# Same pickle lookup rule as extract_harm_*.py
budget = os.environ.get("HARMGEN_BUDGET")
label = os.environ.get("HARMGEN_LABEL")
if budget and label:
    pickle_path = (f"{result_dir}/genetic_algorithm_{beta_cca}_{alpha_cca}"
                   f"_harm_dict_{budget}_budget_{label}.pckl")
else:
    pattern = (f"{result_dir}/genetic_algorithm_{beta_cca}_{alpha_cca}"
               f"_harm_dict_*_budget_*.pckl")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        sys.exit(f"No harm-dict pickle found matching {pattern}")
    pickle_path = matches[-1]

if not os.path.exists(pickle_path):
    sys.exit(f"No harm-dict pickle at {pickle_path}")

with open(pickle_path, "rb") as f:
    harm_dict = pickle.load(f)

candidates = [
    (chrom, harm) for chrom, harm in harm_dict.items()
    if isinstance(chrom, tuple) and len(chrom) == 6 and harm is not None
]
if not candidates:
    sys.exit(f"No usable results in {pickle_path}")

candidates.sort(key=lambda item: item[1], reverse=True)

use_dualpi2 = (beta_cca == "prague" or alpha_cca == "prague")
out_dir = f"{result_dir}/top_harm"
os.makedirs(out_dir, exist_ok=True)

print(f"Loaded {len(candidates)} results from {pickle_path}")
print(f"Looking for the top {top_n} usable settings, writing into {out_dir}\n")

rank = 0
skipped = []

for chrom, harm in candidates:
    if rank >= top_n:
        break

    bw, rtt, queue, num_beta, num_alpha, alpha_start = chrom
    condition = (f"{bw}bw-{rtt}rtt-{queue}q-"
                 f"{beta_cca}-{num_beta}-{alpha_cca}-{num_alpha}")

    print(f"--- candidate {condition}  (recorded harm {harm:.4f}) ---", flush=True)

    exp_dir = find_complete_exp_dir(condition, num_beta, num_alpha)
    if exp_dir:
        problem = check_traces(exp_dir, num_beta)
        if problem:
            print(f"SKIP: {problem}\n", flush=True)
            skipped.append((condition, harm, problem))
            continue
        print(f"reusing existing traces in {exp_dir}", flush=True)
        rerun = False
    else:
        exp_dir = f"{out_dir}/rank{rank + 1}-{condition}"
        os.makedirs(exp_dir, exist_ok=True)
        print(f"traces incomplete, re-running into {exp_dir}", flush=True)
        rerun = True

    try:
        if flow_type == "long-flow":
            if rerun:
                alpha_starts = [0, 0, 0, 0, 0][:num_alpha]
                _, conv_time, converged = run_tc_long_flow_experiment(
                    exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_starts,
                    beta_cca, alpha_cca, dualpi2=use_dualpi2)
                problem = check_traces(exp_dir, num_beta)
                if problem:
                    raise ExperimentDataError(problem)
            else:
                flows = {}
                for i in range(num_beta):
                    t, r = load_throughput_data(exp_dir, "target_cc", "beta", i)
                    flows[f"beta_{i}"] = {"times": t, "rates": list(r)}
                for i in range(num_alpha):
                    t, r = load_throughput_data(exp_dir, "target_cc", "alpha", i)
                    flows[f"alpha_{i}"] = {"times": t, "rates": list(r)}
                conv_time, converged = find_convergence_time(flows)
        else:
            if rerun:
                run_tc_short_flow_experiment(
                    exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_start,
                    beta_cca, alpha_cca, short_flow_harm_metric, dualpi2=use_dualpi2)
                problem = check_traces(exp_dir, num_beta)
                if problem:
                    raise ExperimentDataError(problem)
    except ExperimentDataError as exc:
        print(f"SKIP: {exc}\n", flush=True)
        skipped.append((condition, harm, str(exc)))
        continue

    rank += 1

    if flow_type == "long-flow":
        plot_long_flow_results_in_pdf(
            exp_dir, bw, rtt, queue, num_beta, num_alpha,
            beta_cca, alpha_cca, conv_time, converged, dualpi2=use_dualpi2)

        src = f"{exp_dir}/long_flow_result.pdf"
        dst = f"{out_dir}/top{rank}_{condition}.pdf"
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"wrote {dst}\n", flush=True)
        else:
            print(f"WARNING: {src} was not produced\n", flush=True)
    else:
        alpha_starts = ALPHA_START_TIMES_MAP[alpha_start][:num_alpha]
        for panel in ("alpha", "beta"):
            plot_short_flow_results_in_pdf(
                exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_starts,
                beta_cca, alpha_cca, panel, short_flow_harm_metric,
                dualpi2=use_dualpi2)

            src = f"{exp_dir}/short_flow_{panel}_new.pdf"
            dst = f"{out_dir}/top{rank}_{condition}_{panel}.pdf"
            if os.path.exists(src):
                shutil.move(src, dst)
                print(f"wrote {dst}", flush=True)
            else:
                print(f"WARNING: {src} was not produced", flush=True)
        print(flush=True)

if skipped:
    print(f"Skipped {len(skipped)} higher-ranked candidate(s) with unusable data:")
    for condition, harm, problem in skipped:
        print(f"  {condition}  harm={harm:.4f}  -- {problem}")
    print()

if rank < top_n:
    print(f"WARNING: only {rank} usable setting(s) found, {top_n} were requested.")

print(f"Done. PDFs are in {out_dir}")
