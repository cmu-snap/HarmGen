"""Measure and plot relative harm across the five CCA pairs in Figure 12."""

import argparse
import json
import random
from functools import partial
from pathlib import Path

from section6_compare import (
    DEFAULT_SEED,
    Evaluations,
    enter_run,
    experiment_lock,
    rank_results,
    run_cli,
)
from section6_search import random_search, sweep_grid


# Each pair is (beta, alpha). The artifact kernel exposes BBRv3 as bbr.
PAIRS = {
    "bbr1-vs-bbr3": ("bbr1", "bbr"),
    "cubic-vs-bbr1": ("cubic", "bbr1"),
    "cubic-vs-bbr3": ("cubic", "bbr"),
    "reno-vs-westwood": ("reno", "westwood"),
    "reno-vs-cubic": ("reno", "cubic"),
}
CCA_NAMES = {"bbr1": "BBRv1", "bbr": "BBRv3", "cubic": "Cubic", "reno": "Reno", "westwood": "Westwood"}


def analyze(output):
    from section6_plots import plot_random_pairs

    budgets = json.loads((output / "run.json").read_text())["budgets"]
    if budgets.keys() != PAIRS.keys():
        raise ValueError(f"{output}: expected a Figure 12 run")
    ranked = rank_results(output, budgets)
    distributions = {
        f"{CCA_NAMES[beta]} vs {CCA_NAMES[alpha]}": [
            float(row["relative_harm"]) for row in ranked[name]
        ]
        for name, (beta, alpha) in PAIRS.items()
    }
    plot_random_pairs(output, distributions)
    print(f"Wrote the Figure 12 comparison to {output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="measure 4 scenarios per pair with 6-second runs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, help="output directory (also used to resume)")
    parser.add_argument("--analyze-only", action="store_true", help="regenerate the plot from a completed run")
    args = parser.parse_args()
    run_name = "section6-figure12-smoke" if args.smoke else "section6-figure12"
    output = args.output or Path("results", run_name)
    budget = 4 if args.smoke else 300
    duration = 6 if args.smoke else 180
    config = {
        "pairs": {name: list(pair) for name, pair in PAIRS.items()},
        "seed": args.seed,
        "flow_duration_s": duration,
        "budgets": dict.fromkeys(PAIRS, budget),
    }

    with experiment_lock():
        if args.analyze_only:
            analyze(output)
            return

        from section6_tc import check_testbed, measure

        check_testbed({cca for pair in PAIRS.values() for cca in pair})
        enter_run(output, config)
        grid = sweep_grid()
        for name, (beta, alpha) in PAIRS.items():
            measure_pair = partial(measure, duration=duration, beta_cca=beta, alpha_cca=alpha)
            evaluate = Evaluations(name, budget, measure_pair)
            # Reuse the same ordered sample so the curves compare the CCA
            # pairs over the same network and workload settings.
            random_search(evaluate, grid, random.Random(args.seed))
            evaluate.check_complete()
            print(f"{name}: {budget} completed measurements", flush=True)
        analyze(Path("."))


if __name__ == "__main__":
    run_cli(main)
