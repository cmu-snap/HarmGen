#!/usr/bin/env bash
# Paper Section 7.2 -- L4S: TCP Prague vs Cubic.
#
# RUN THIS ON vm-dualpi2-prague ONLY (needs the 5.15.72 DualPI2 + Prague kernel).
# Run from ~/harm-exp inside the VM.
#
# PREREQUISITE (once per boot, see README):
#   sudo modprobe sch_dualpi2 tcp_prague
#   sudo sysctl -w net.ipv4.tcp_ecn=3
#   sudo sysctl -w net.ipv4.tcp_congestion_control=prague
set -u

BUDGET=600
LABEL=section7-2
TESTBED=3
SEED=701
MUTATION=mutate_v2
MUTATION_PROB=0.5
POP_SIZE=60
CROSSOVER=0.5
TIME_LIMIT=30.0

export HARMGEN_BUDGET=${BUDGET}
export HARMGEN_LABEL=${LABEL}

GA_ARGS="${BUDGET} ${LABEL} ${TESTBED} ${SEED} ${MUTATION} ${MUTATION_PROB} ${POP_SIZE} ${CROSSOVER} ${TIME_LIMIT}"

echo "=== [1/3] HarmGen long-flow: harm Prague does to Cubic (beta=cubic, alpha=prague) ==="
python3 genetic-algorithm.py cubic prague ${GA_ARGS}
python3 extract_harm_long_flows.py HarmGen cubic prague ${TIME_LIMIT}
python3 draw_distribution_long_flows.py HarmGen cubic prague 30 ${TIME_LIMIT}
python3 plot_top_harm.py cubic prague ${TIME_LIMIT} long-flow

echo "=== [2/3] HarmGen long-flow: harm Cubic does to Prague (beta=prague, alpha=cubic) ==="
python3 genetic-algorithm.py prague cubic ${GA_ARGS}
python3 extract_harm_long_flows.py HarmGen prague cubic ${TIME_LIMIT}
python3 draw_distribution_long_flows.py HarmGen prague cubic 30 ${TIME_LIMIT}
python3 plot_top_harm.py prague cubic ${TIME_LIMIT} long-flow

echo "=== [3/3] HarmGen short-flow: Cubic short flows vs a long Prague flow (Fig 18) ==="
python3 genetic-algorithm.py prague cubic ${GA_ARGS} download_bytes
python3 extract_harm_short_flows.py HarmGen prague cubic ${TIME_LIMIT} download_bytes
python3 draw_distribution_short_flows.py HarmGen prague cubic 30 ${TIME_LIMIT} download_bytes
python3 plot_top_harm.py prague cubic ${TIME_LIMIT} short-flow download_bytes

echo "=== Done.  Results in ./results/ ==="
