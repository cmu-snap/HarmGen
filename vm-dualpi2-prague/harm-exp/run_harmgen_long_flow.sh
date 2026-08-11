#!/usr/bin/env bash
# HarmGen, long-flow workload: harm that Prague does to Cubic (Section 7.2).
# RUN ON vm-dualpi2-prague (needs DualPI2 + Prague).
# Prerequisites once per boot -- see README:
#   sudo modprobe sch_dualpi2 tcp_prague
#   sudo sysctl -w net.ipv4.tcp_ecn=3
#   sudo sysctl -w net.ipv4.tcp_congestion_control=prague
set -u

BUDGET=600
LABEL=cubic-prague-long-flow
TIME_LIMIT=30.0

export HARMGEN_BUDGET=${BUDGET}
export HARMGEN_LABEL=${LABEL}

python3 genetic-algorithm.py cubic prague ${BUDGET} ${LABEL} 1 701 mutate_v2 0.5 60 0.5 ${TIME_LIMIT}

python3 extract_harm_long_flows.py HarmGen cubic prague ${TIME_LIMIT}
python3 draw_distribution_long_flows.py HarmGen cubic prague 30 ${TIME_LIMIT}
