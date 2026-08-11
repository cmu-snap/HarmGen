#!/usr/bin/env bash
# HarmGen, short-flow workload: BBRv3 short flows against a long Cubic flow.
# RUN ON vm-bbrv3 (needs the BBRv3 kernel; "bbr" is v3, "bbr1" is v1).
set -u

BUDGET=600
LABEL=cubic-bbr-short-flow
TIME_LIMIT=30.0
METRIC=download_bytes        # or: recovery_time

export HARMGEN_BUDGET=${BUDGET}
export HARMGEN_LABEL=${LABEL}

python3 genetic-algorithm.py cubic bbr ${BUDGET} ${LABEL} 1 701 mutate_v2 0.5 60 0.5 ${TIME_LIMIT} ${METRIC}

python3 extract_harm_short_flows.py HarmGen cubic bbr ${TIME_LIMIT} ${METRIC}
python3 draw_distribution_short_flows.py HarmGen cubic bbr 30 ${TIME_LIMIT} ${METRIC}
