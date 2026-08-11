#!/usr/bin/env bash
# HarmGen, long-flow workload: harm that BBRv3 does to Cubic (Section 7.1).
# RUN ON vm-bbrv3 (needs the BBRv3 kernel; "bbr" is v3, "bbr1" is v1).
set -u

BUDGET=600
LABEL=cubic-bbr-long-flow
TIME_LIMIT=30.0

export HARMGEN_BUDGET=${BUDGET}
export HARMGEN_LABEL=${LABEL}

python3 genetic-algorithm.py cubic bbr ${BUDGET} ${LABEL} 1 701 mutate_v2 0.5 60 0.5 ${TIME_LIMIT}

python3 extract_harm_long_flows.py HarmGen cubic bbr ${TIME_LIMIT}
python3 draw_distribution_long_flows.py HarmGen cubic bbr 30 ${TIME_LIMIT}
