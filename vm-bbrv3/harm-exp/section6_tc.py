"""Measure Section 6 relative harm on the artifact's tc testbed."""

import math
import subprocess
from pathlib import Path
from statistics import mean

from tc_single_run import (
    CONVERGENCE_BACKUP_TIME,
    LONG_FLOW_EXP_FINISH_TIME,
    ExperimentDataError,
    compute_average_throughput_after_convergence,
    find_convergence_time,
    run_tc_experiment,
)


def check_testbed(required_ccas=("reno", "westwood")):
    subprocess.run(["sudo", "-n", "true"], check=True)
    allowed = subprocess.check_output(
        ["sysctl", "-n", "net.ipv4.tcp_allowed_congestion_control"], text=True
    ).split()
    missing = set(required_ccas).difference(allowed)
    if missing:
        names = ", ".join(sorted(missing))
        raise RuntimeError(f"Enable {names} in the VM as described in Reproducing Section 6")


def measurement_start(flows, duration):
    if duration != LONG_FLOW_EXP_FINISH_TIME:
        return 2  # Short wiring check with a fixed warm-up.
    cutoff, converged = find_convergence_time(flows)
    return cutoff if converged else CONVERGENCE_BACKUP_TIME


def flow_throughputs(flows, cutoff):
    throughputs = []
    for name, flow in flows.items():
        times, rates = flow["times"], flow["rates"]
        if not any(times >= cutoff):
            raise ExperimentDataError(f"{name} has no throughput samples after {cutoff}s")
        throughput = compute_average_throughput_after_convergence(times, rates, cutoff)
        if not math.isfinite(throughput) or throughput <= 0:
            raise ExperimentDataError(f"{name} recorded invalid throughput after {cutoff}s")
        throughputs.append(float(throughput))
    return throughputs


def measure(scenario, exp_dir, duration, *, beta_cca="reno", alpha_cca="westwood"):
    bw, rtt, queue, alpha, beta = scenario
    # The existing tc runner adds five seconds to the iperf3 duration.
    tc_duration = duration - 5
    try:
        solo, _ = run_tc_experiment(
            exp_dir, bw, rtt, queue, beta, 0, [], beta_cca, alpha_cca,
            tc_duration, None, "beta_solo", dualpi2=False,
        )
        competing, alpha_flows = run_tc_experiment(
            exp_dir, bw, rtt, queue, beta, alpha, [0] * alpha, beta_cca, alpha_cca,
            tc_duration, None, "target_cc", dualpi2=False,
        )
    finally:
        for capture in Path(exp_dir).glob("*.pcap"):
            capture.unlink()

    solo_cutoff = measurement_start(solo, duration)
    compete_cutoff = measurement_start({**competing, **alpha_flows}, duration)
    solo_mbps = mean(flow_throughputs(solo, solo_cutoff))
    compete_mbps = mean(flow_throughputs(competing, compete_cutoff))
    flow_throughputs(alpha_flows, compete_cutoff)
    harm = beta / (alpha + beta) - compete_mbps / solo_mbps
    return {
        "relative_harm": harm,
        "solo_mbps": solo_mbps,
        "competing_mbps": compete_mbps,
        "solo_cutoff_s": solo_cutoff,
        "competing_cutoff_s": compete_cutoff,
    }
