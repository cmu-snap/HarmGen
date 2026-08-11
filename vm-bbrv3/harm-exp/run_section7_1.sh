#!/usr/bin/env bash
# Paper Section 7.1 -- Cubic vs BBRv3 (and vs BBRv1 as the release-to-release baseline).
#
# RUN THIS ON vm-bbrv3 ONLY (needs the 6.13.7+BBRv3 kernel: bbr == v3, bbr1 == v1).
# Run from ~/harm-exp inside the VM.
set -u

BUDGET=300
TIME_LIMIT=240.0
TOP_N=3
MIN_HARM=0.1
LONG_SPACE="-bdp_ratio 0.25,5,0.5 -rtt 10,100,20 -bw 25,200,35 -alpha_flows 1,5,1 -beta_flows 1,5,1"
SHORT_SPACE="-bdp_ratio 0.25,5,0.25 -rtt 10,100,10 -bw 25,200,25 -alpha_flows 1,5,1 -beta_flows 1,1,1"

echo "=== [1/4] Mahak long-flow: Cubic vs BBRv1 (Fig 15a baseline) ==="
python3 mahak.py -cc cubic -compete_cc bbr1 -budget ${BUDGET} -time_limit ${TIME_LIMIT} \
    -experiment_type long-flow ${LONG_SPACE} -convergence True

echo "=== [2/4] Mahak long-flow: Cubic vs BBRv3 (Fig 15b) ==="
python3 mahak.py -cc cubic -compete_cc bbr -budget ${BUDGET} -time_limit ${TIME_LIMIT} \
    -experiment_type long-flow ${LONG_SPACE} -convergence True

echo "=== [3/4] Mahak short-flow: Cubic vs BBRv1, recovery_time (Fig 17a) ==="
python3 mahak.py -cc cubic -compete_cc bbr1 -budget ${BUDGET} -time_limit ${TIME_LIMIT} \
    -experiment_type short-flow -short_flow_harm_metric recovery_time \
    ${SHORT_SPACE} -alpha_start 1,1,1

echo "=== [4/4] Mahak short-flow: Cubic vs BBRv3, recovery_time (Fig 17b) ==="
python3 mahak.py -cc cubic -compete_cc bbr -budget ${BUDGET} -time_limit ${TIME_LIMIT} \
    -experiment_type short-flow -short_flow_harm_metric recovery_time \
    ${SHORT_SPACE} -alpha_start 1,1,1

echo "=== Plotting Fig 15 heatmaps ==="
python3 draw_heatmap.py cubic bbr1 ${TIME_LIMIT} long-flow
python3 draw_heatmap.py cubic bbr  ${TIME_LIMIT} long-flow

echo "=== Plotting Mahak sampling heatmaps (Fig 8) ==="
python3 draw_training.py cubic bbr1 ${TIME_LIMIT} long-flow
python3 draw_training.py cubic bbr  ${TIME_LIMIT} long-flow
python3 draw_training.py cubic bbr1 ${TIME_LIMIT} short-flow
python3 draw_training.py cubic bbr  ${TIME_LIMIT} short-flow

echo "=== BBRv1 vs BBRv3 comparison PDFs from Mahak's predictions ==="
python3 plot_mahak_compare.py cubic bbr1 bbr ${TIME_LIMIT} long-flow  max-improvement ${TOP_N}
python3 plot_mahak_compare.py cubic bbr1 bbr ${TIME_LIMIT} short-flow min-difference  ${TOP_N} recovery_time ${MIN_HARM}

echo "=== Done.  Results in ./mahak_results/ ==="
