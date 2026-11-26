"""Run the search comparison for Figures 13 and 14."""

import argparse
import csv
import fcntl
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from functools import partial
from itertools import combinations
from pathlib import Path
from statistics import median

from section6_search import Scenario, harmgen, hill_climbing, parameter_sweep, random_search, sweep_grid


DEFAULT_SEED = 701
METHODS = {
    "harmgen": harmgen,
    "random-search": random_search,
    "hill-climbing": hill_climbing,
    "parameter-sweep": parameter_sweep,
}
COLUMNS = (
    "evaluation", *Scenario._fields, "relative_harm", "solo_mbps",
    "competing_mbps", "solo_cutoff_s", "competing_cutoff_s", "trace_directory",
)


class Evaluations:
    """Count each distinct measurement and replay checkpoints on resume."""

    def __init__(self, method, budget, measure):
        self.method = method
        self.budget = budget
        self.measure = measure
        self.path = Path(f"{method}.csv")
        self.rows = []
        self.seen = {}
        self.saved = iter(read_csv(self.path) if self.path.exists() else [])
        self.trace_dir = Path("experiments", method)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def remaining(self):
        return self.budget - len(self.rows)

    def __contains__(self, scenario):
        return scenario in self.seen

    def __call__(self, scenario):
        if scenario in self.seen:
            return self.seen[scenario]
        if not self.remaining:
            raise RuntimeError(f"{self.method} exceeded its budget")
        index = len(self.rows) + 1
        saved = next(self.saved, None)
        if saved is None:
            directory = tempfile.mkdtemp(prefix=f"{index:04d}-", dir=self.trace_dir)
            print(f"{self.method} {index}/{self.budget} {scenario}", flush=True)
            result = self.measure(scenario, directory)
            if not math.isfinite(result["relative_harm"]):
                raise ValueError(f"{directory}: relative harm must be finite")
            row = {
                "evaluation": index,
                **scenario._asdict(),
                **result,
                "trace_directory": directory,
            }
            write_csv(self.path, COLUMNS, [*self.rows, row])
        else:
            saved_scenario = Scenario(*(int(saved[field]) for field in Scenario._fields))
            if saved_scenario != scenario or int(saved["evaluation"]) != index:
                raise ValueError(f"{self.path}: checkpoint differs at evaluation {index}")
            if not math.isfinite(float(saved["relative_harm"])):
                raise ValueError(f"{self.path}: non-finite harm at evaluation {index}")
            row = saved
        self.rows.append(row)
        self.seen[scenario] = float(row["relative_harm"])
        return self.seen[scenario]

    def check_complete(self):
        if self.remaining or next(self.saved, None) is not None:
            raise RuntimeError(f"{self.method}: expected {self.budget} measurements")


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@contextmanager
def experiment_lock():
    with Path("/tmp/harmgen-section6.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another Section 6 command is running in this VM") from None
        yield


def enter_run(output, config):
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    metadata = Path("run.json")
    if metadata.exists():
        if json.loads(metadata.read_text()) != config:
            raise ValueError("Checkpoint settings differ. Specify a new --output directory.")
    else:
        temporary = metadata.with_suffix(".tmp")
        temporary.write_text(json.dumps(config, indent=2) + "\n")
        temporary.replace(metadata)


def rank_results(output, budgets):
    ranked = {}
    for method, budget in budgets.items():
        path = output / f"{method}.csv"
        rows = read_csv(path)
        keys = {tuple(row[field] for field in Scenario._fields) for row in rows}
        if len(rows) != budget or len(keys) != budget:
            raise ValueError(f"{path}: expected {budget} distinct measured scenarios")
        if any(not math.isfinite(float(row["relative_harm"])) for row in rows):
            raise ValueError(f"{path}: expected finite relative harm in every row")
        ranked[method] = sorted(rows, key=lambda row: float(row["relative_harm"]), reverse=True)
    return ranked


def scenario_vectors(rows):
    for row in rows:
        scenario = tuple(int(row[field]) for field in Scenario._fields)
        yield tuple(math.log2(value) for value in scenario[:3]) + scenario[3:]


def analyze(output, *, skip_sweep=False):
    from section6_plots import plot

    budgets = json.loads((output / "run.json").read_text())["budgets"]
    if skip_sweep:
        del budgets["parameter-sweep"]
    ranked = rank_results(output, budgets)
    report = output / "without-sweep" if skip_sweep else output
    report.mkdir(exist_ok=True)
    top_n = min(100, min(budgets.values()))
    diversity_n = min(20, top_n)
    top_harms = {
        method: [float(row["relative_harm"]) for row in rows[:top_n]]
        for method, rows in ranked.items()
    }

    # Normalize the methods against the same measured feature ranges.
    features = zip(*(vector for rows in ranked.values() for vector in scenario_vectors(rows)))
    ranges = [(min(feature), max(feature)) for feature in features]
    distances = {}
    for method, rows in ranked.items():
        vectors = [
            [(value - low) / (high - low) if high > low else 0
             for value, (low, high) in zip(vector, ranges)]
            for vector in scenario_vectors(rows[:diversity_n])
        ]
        distances[method] = [math.dist(a, b) for a, b in combinations(vectors, 2)]

    summary = []
    for method, harms in top_harms.items():
        paired = list(zip(top_harms["harmgen"], harms))
        differences = [100 * (ga - baseline) / abs(baseline)
                       for ga, baseline in paired if baseline]
        compare = method != "harmgen"
        summary.append({
            "method": method,
            "evaluations": budgets[method],
            "compared_ranks": top_n,
            "highest_relative_harm": harms[0],
            "median_top_relative_harm": median(harms),
            "harmgen_higher_ranks": sum(ga > baseline for ga, baseline in paired) if compare else "",
            "median_harmgen_minus_method_pct": median(differences) if compare and differences else "",
            "diversity_scenarios": diversity_n,
            "median_pairwise_distance": median(distances[method]),
        })
    write_csv(report / "comparison.csv", summary[0], summary)
    write_csv(
        report / "ranked_harm.csv", ("method", "rank", *COLUMNS),
        ({"method": method, "rank": rank, **row}
         for method, rows in ranked.items()
         for rank, row in enumerate(rows[:top_n], start=1)),
    )
    plot(report, top_harms, distances, budgets)
    print(f"Wrote the Figures 13 and 14 comparisons to {report.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="measure 4 scenarios per method with 6-second runs")
    parser.add_argument("--skip-sweep", action="store_true", help="compare HarmGen, random search, and hill climbing")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, help="output directory (also used to resume)")
    parser.add_argument("--method", choices=METHODS, help="run only this method")
    parser.add_argument("--analyze-only", action="store_true", help="regenerate figures from a completed run")
    args = parser.parse_args()
    if args.method and args.analyze_only:
        parser.error("--method and --analyze-only are mutually exclusive")
    if args.skip_sweep and args.method == "parameter-sweep":
        parser.error("--skip-sweep and --method parameter-sweep are mutually exclusive")

    profile = "smoke" if args.smoke else "full"
    run_name = "section6-figures13-14-smoke" if args.smoke else "section6-figures13-14"
    output = args.output or Path("results", run_name)
    grid = sweep_grid()
    search_budget = 4 if args.smoke else 300
    population = 4 if args.smoke else 60
    duration = 6 if args.smoke else 180
    budgets = {name: search_budget for name in METHODS}
    budgets["parameter-sweep"] = search_budget if args.smoke else len(grid)
    config = {
        "alpha": "westwood",
        "beta": "reno",
        "profile": profile,
        "seed": args.seed,
        "flow_duration_s": duration,
        "population_size": population,
        "budgets": budgets,
    }
    with experiment_lock():
        if args.analyze_only:
            analyze(output, skip_sweep=args.skip_sweep)
            return

        from section6_tc import check_testbed, measure

        check_testbed()
        enter_run(output, config)

        methods = (args.method,) if args.method else METHODS
        for name in methods:
            if args.skip_sweep and name == "parameter-sweep":
                continue
            evaluate = Evaluations(name, budgets[name], partial(measure, duration=duration))
            search = partial(harmgen, population_size=population) if name == "harmgen" else METHODS[name]
            search(evaluate, grid, random.Random(args.seed))
            evaluate.check_complete()
            print(f"{name}: {budgets[name]} completed measurements", flush=True)

        if not args.method:
            analyze(Path("."), skip_sweep=args.skip_sweep)


def run_cli(command):
    try:
        command()
    except KeyboardInterrupt:
        sys.exit("Interrupted. Rerun the same command to resume from the last saved measurement.")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        sys.exit(f"Section 6: {error}")


if __name__ == "__main__":
    run_cli(main)
