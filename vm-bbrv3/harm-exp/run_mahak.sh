#!/usr/bin/env bash
# Mahak, long-flow workload: Cubic vs BBRv1 (Section 7.1 baseline).
# RUN ON vm-bbrv3.  Swap "-compete_cc bbr1" for "-compete_cc bbr" to get BBRv3.
set -u

python3 mahak.py -cc cubic -compete_cc bbr1 -budget 300 -time_limit 240.0 \
    -experiment_type long-flow \
    -bdp_ratio 0.25,5,0.5 -rtt 10,100,20 -bw 25,200,35 \
    -alpha_flows 1,5,1 -beta_flows 1,5,1 -convergence True

python3 draw_heatmap.py cubic bbr1 240.0 long-flow
