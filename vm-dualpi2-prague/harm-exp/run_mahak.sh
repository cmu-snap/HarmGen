#!/usr/bin/env bash
# Mahak, long-flow workload: Cubic vs Prague over DualPI2.
# RUN ON vm-dualpi2-prague.  Swap the two CCAs to measure harm in the other direction.
# Prerequisites once per boot -- see README:
#   sudo modprobe sch_dualpi2 tcp_prague
#   sudo sysctl -w net.ipv4.tcp_ecn=3
#   sudo sysctl -w net.ipv4.tcp_congestion_control=prague
set -u

python3 mahak.py -cc cubic -compete_cc prague -budget 300 -time_limit 240.0 \
    -experiment_type long-flow \
    -bdp_ratio 0.25,5,0.5 -rtt 10,100,20 -bw 25,200,35 \
    -alpha_flows 1,5,1 -beta_flows 1,5,1 -convergence True

python3 draw_heatmap.py cubic prague 240.0 long-flow
