#!/usr/bin/env bash
# HarmGen, short-flow workload: Cubic short flows against a long Prague flow (Section 7.2, Fig 18).
# RUN ON vm-dualpi2-prague (needs DualPI2 + Prague).
# Prerequisites once per boot -- see README:
#   sudo modprobe sch_dualpi2 tcp_prague
#   sudo sysctl -w net.ipv4.tcp_ecn=3
#   sudo sysctl -w net.ipv4.tcp_congestion_control=prague
set -u

BUDGET=600
LABEL=prague-cubic-short-flow
TIME_LIMIT=30.0
METRIC=download_bytes        # or: recovery_time

export HARMGEN_BUDGET=${BUDGET}
export HARMGEN_LABEL=${LABEL}

python3 genetic-algorithm.py prague cubic ${BUDGET} ${LABEL} 1 701 mutate_v2 0.5 60 0.5 ${TIME_LIMIT} ${METRIC}

python3 extract_harm_short_flows.py HarmGen prague cubic ${TIME_LIMIT} ${METRIC}
python3 draw_distribution_short_flows.py HarmGen prague cubic 30 ${TIME_LIMIT} ${METRIC}
