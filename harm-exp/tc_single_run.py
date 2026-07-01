import numpy as np
import matplotlib.pylab as plt
import seaborn
import os
import sys
import subprocess
import logging
import psutil
import time as ti
from scapy.all import *
import pandas as pd
import ruptures as rpt
from ruptures.exceptions import BadSegmentationParameters

LONG_FLOW_EXP_FINISH_TIME = 180
SHORT_FLOW_EXP_FINISH_TIME = 120
SHORT_FLOW_DURATION = 10
SHORT_FLOW_BETA_ALONE_TIME = 60
PACKET_SIZE_BYTES = 1514
CONVERGENCE_WINDOW_SIZE = 60
CONVERGENCE_RPT_MIN_SIZE = 5
CONVERGENCE_ABS_THRESH = 0.2
CONVERGENCE_PCT_THRESH = 0.2
CONVERGENCE_MAX_TIME = 150
CONVERGENCE_BACKUP_TIME = 60

ALPHA_START_TIMES_MAP = {
    0: [0, 0, 0, 0, 0],
    1: [25, 25, 25, 25, 25],
    2: [15, 20, 25, 30, 35],
    3: [2, 14, 26, 38, 50]
}

BDP_MULTIPLES = [.25, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

def calculate_queue_size(bw_mbps, rtt_ms, bdp_mult):
    try:
        size = max(4, int(bdp_mult * bw_mbps * 1e6 * rtt_ms / 1e3 / 8 / PACKET_SIZE_BYTES))
    except:
        size = 4
    return size


def get_queue_pkts_to_bdp_multiple(bw, rtt, queue_size, bytes_per_packet=1514):
    bdp = get_bdp(bw, rtt, bytes_per_packet=bytes_per_packet)
    result = queue_size / bdp
    return get_closest_bdp_multiple(result)


def get_queue_pkts_to_bdp(bw, rtt, queue_size, bytes_per_packet=1514):
    bdp = get_bdp(bw, rtt, bytes_per_packet=bytes_per_packet)
    result = queue_size / bdp
    return result


def get_bdp(bw, rtt, bytes_per_packet=1514):
    bps = bw * 1e6 / 8
    seconds = rtt / 1e3
    return (bps * seconds) / bytes_per_packet


def get_closest_bdp_multiple(value):
    return min(BDP_MULTIPLES, key=lambda x: abs(x - value))

def generate_trace_file(bw, duration):
    os.makedirs("traces", exist_ok=True)
    trace_path = f'traces/{bw}.trace'
    num_packets = int(float(bw) * 1e6 / 12000 * duration)
    timestamps = np.linspace(0, duration * 1000, num=num_packets, endpoint=False)
    with open(trace_path, 'w') as f:
        for ts in timestamps:
            f.write(f'{int(ts)}\n')
    return trace_path


def find_available_port(start_port=45000):
    port = start_port
    while True:
        try:
            result = subprocess.check_output(['netstat', '-at'], stderr=subprocess.STDOUT, text=True)
            grep_result = subprocess.run(
                ['grep', str(port)],
                input=result,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if grep_result.returncode != 0 or not grep_result.stdout:
                return port
            port += 10
        except subprocess.CalledProcessError as e:
            return port


def setup_network_namespace(bw, rtt, queue, dualpi2=True):
    cleanup_cmd = """
        sudo tc qdisc del dev veth0 root 2>/dev/null || true
        sudo ip netns del client_ns 2>/dev/null || true
        sudo ip netns del netem_ns 2>/dev/null || true
        sudo ip link del veth0 2>/dev/null || true
        sudo ip link del veth2 2>/dev/null || true
        sleep 0.5
    """

    if dualpi2:
        queue_cmd = f"sudo ip netns exec netem_ns tc qdisc add dev veth2 parent 1:1 handle 20: dualpi2"
    else:
        queue_cmd = f"sudo ip netns exec netem_ns tc qdisc add dev veth2 parent 1:1 handle 20: pfifo limit {queue}"
    
    setup_cmd = f"""
        sudo ip netns add netem_ns
        sudo ip netns add client_ns
        
        sudo ip link add veth0 type veth peer name veth1
        sudo ip link set veth1 netns netem_ns
        sudo ip link set veth0 up
        sudo ip netns exec netem_ns ip link set veth1 up
        sudo ip netns exec netem_ns ip link set lo up
        
        sudo ip netns exec netem_ns ip link add veth2 type veth peer name veth3
        sudo ip netns exec netem_ns ip link set veth3 netns client_ns
        sudo ip netns exec netem_ns ip link set veth2 up
        sudo ip netns exec client_ns ip link set veth3 up
        sudo ip netns exec client_ns ip link set lo up
        
        sudo ip addr add 10.0.0.1/24 dev veth0
        sudo ip netns exec netem_ns ip addr add 10.0.0.2/24 dev veth1
        sudo ip netns exec netem_ns ip addr add 10.0.1.1/24 dev veth2
        sudo ip netns exec client_ns ip addr add 10.0.1.2/24 dev veth3
        
        sudo ip route add 10.0.1.0/24 via 10.0.0.2
        sudo ip netns exec netem_ns sysctl -w net.ipv4.ip_forward=1 > /dev/null
        sudo ip netns exec netem_ns ip route add default via 10.0.0.1
        sudo ip netns exec client_ns ip route add default via 10.0.1.1
        
        sudo ip netns exec netem_ns tc qdisc add dev veth1 root handle 1: netem delay {rtt}ms limit {queue * 2}
        
        sudo ip netns exec netem_ns tc qdisc add dev veth2 root handle 1: htb default 1
        sudo ip netns exec netem_ns tc class add dev veth2 parent 1: classid 1:1 htb rate {bw}mbit ceil {bw}mbit
        {queue_cmd}
    """
    
    os.system(cleanup_cmd)
    os.system(setup_cmd)


def cleanup_network_namespace():
    cmd = """
        sudo ip netns del client_ns 2>/dev/null || true
        sudo ip netns del netem_ns 2>/dev/null || true
        sudo ip link del veth0 2>/dev/null || true
    """
    os.system(cmd)


def start_iperf_servers(num_servers, base_port, exp_dir):
    server_ports = []
    for i in range(num_servers):
        port = base_port + i
        log_file = f"{exp_dir}/iperf_srv_{port}.log"
        os.system(f"iperf3 -s -p {port} > {log_file} 2>&1 &")
        server_ports.append(port)
        ti.sleep(0.1)
    ti.sleep(0.5)
    return server_ports


def build_client_commands(num_beta, num_alpha, base_port, alpha_starts, cca_beta, cca_alpha, duration, alpha_dur=None):
    cmds = ""
    for i in range(num_beta):
        port = base_port + i
        cmds += f"iperf3 -c 10.0.0.1 -p {port} -t {duration} -i 0.5 -C {cca_beta} -R & "
    
    for i in range(num_alpha):
        port = base_port + num_beta + i
        start = alpha_starts[i]
        cmds += f"(sleep {start}; iperf3 -c 10.0.0.1 -p {port} -t {alpha_dur if alpha_dur is not None else duration} -i 0.5 -C {cca_alpha} -R) & "
    
    return cmds


def extract_throughput_from_pcap(pcap_file, port, interval=1.0):
    output = subprocess.check_output(
        f"sudo tcpstat -o '%b\n' -r {pcap_file} {interval} -f 'src port {port}'",
        shell=True,
        text=True
    )
    rates_bps = [float(line) for line in output.strip().split('\n')]
    rates_mbps = [rate / 1e6 for rate in rates_bps]
    times = np.arange(0, len(rates_mbps)) * interval
    return times, rates_mbps


def find_breakpoints(series_stats, rpt_min_size):
    algo = rpt.Dynp(model="l2", min_size=rpt_min_size)
    algo.fit(series_stats)
    result = algo.predict(n_bkps=1)
    return result[0]


def recursive_breakpoints_with_mean_validation(time_series, window_size, abs_thresh, pct_thresh, rpt_min_size):
    rolling_stdevs = pd.Series(time_series).rolling(window=window_size).std().dropna()
    
    if len(rolling_stdevs) < 2 * rpt_min_size:
        return float('inf')

    std_devs = rolling_stdevs.copy()
    abs_diff = float('inf')
    pct_diff = float('inf')
    break_point = 0
    overall_breakpoint = 0
    num_splits = 0
    
    while (abs_diff > abs_thresh and pct_diff > pct_thresh) or num_splits < 1:
        std_devs = std_devs[break_point:]
        overall_breakpoint += break_point
        
        if len(std_devs) < 2 * rpt_min_size:
            if num_splits >= 1:
                break
            else:
                return float('inf')

        try:
            break_point = find_breakpoints(std_devs.values, rpt_min_size)
        except BadSegmentationParameters:
            if num_splits >= 1:
                break
            else:
                return float('inf')
        
        left_mean = std_devs[:break_point].mean()
        right_mean = std_devs[break_point:].mean()
        num_splits += 1
        
        abs_diff = abs(left_mean - right_mean)
        pct_diff = abs_diff / left_mean if left_mean > 1e-6 else 0.0
    
    if overall_breakpoint > CONVERGENCE_MAX_TIME:
        return float('inf')
    
    return overall_breakpoint


def find_convergence_time(throughput_dict):
    convergence_times = []
    
    for flow_id, data in throughput_dict.items():
        times = data['times']
        rates = data['rates']
        
        if len(rates) < CONVERGENCE_WINDOW_SIZE:
            convergence_times.append(LONG_FLOW_EXP_FINISH_TIME)
            continue
        
        conv_time = recursive_breakpoints_with_mean_validation(
            rates, 
            CONVERGENCE_WINDOW_SIZE,
            CONVERGENCE_ABS_THRESH,
            CONVERGENCE_PCT_THRESH,
            CONVERGENCE_RPT_MIN_SIZE
        )
        
        if conv_time == float('inf'):
            convergence_times.append(LONG_FLOW_EXP_FINISH_TIME)
        else:
            convergence_times.append(conv_time)
    
    if not convergence_times or all(t == LONG_FLOW_EXP_FINISH_TIME for t in convergence_times):
        return LONG_FLOW_EXP_FINISH_TIME, False
    
    max_conv_time = max(convergence_times)
    if max_conv_time == float('inf') or max_conv_time >= LONG_FLOW_EXP_FINISH_TIME:
        return LONG_FLOW_EXP_FINISH_TIME, False
    
    return max_conv_time, True


def compute_average_throughput_after_convergence(times, rates, conv_time):
    mask = times >= conv_time
    if not any(mask):
        return np.mean(rates) if rates else 0
    rates_after_conv = np.array(rates)[mask]
    return np.mean(rates_after_conv) if len(rates_after_conv) > 0 else 0


def save_throughput_data(exp_dir, flow_type, flow_idx, times, rates):
    filename = f"{exp_dir}/{flow_type}_thr_{flow_idx}.txt"
    with open(filename, "w") as f:
        for t, r in zip(times, rates):
            f.write(f"{t} {r}\n")


def run_tc_experiment(exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_starts, cca_beta, cca_alpha, duration, alpha_dur, pcap_name, dualpi2=True):
    generate_trace_file(bw, duration + 30)
    base_port = find_available_port()
    total_flows = num_beta + num_alpha

    setup_network_namespace(bw, rtt, queue, dualpi2=dualpi2)
    
    server_ports = start_iperf_servers(total_flows, base_port, exp_dir)
    
    client_cmds = build_client_commands(
        num_beta, num_alpha, base_port, alpha_starts, cca_beta, cca_alpha, duration + 5, alpha_dur
    )
    
    all_ports = " or ".join(str(p) for p in server_ports)
    pcap_file = f"{exp_dir}/{pcap_name}.pcap"
    
    run_cmd = f"""
        sudo ip netns exec client_ns script -q -c "sudo tcpdump port {all_ports} -i veth3 -w {pcap_file}" &
        sudo ip netns exec client_ns bash -c '
            sleep 0.5
            {client_cmds}
            sleep {duration}
        ' &
    """
    
    os.system(run_cmd)
    ti.sleep(duration + 10)
    os.system("sudo killall -9 iperf3 tcpdump")
    ti.sleep(0.2)
    
    cleanup_network_namespace()
    
    beta_flows = {}
    for i in range(num_beta):
        port = base_port + i
        times, rates = extract_throughput_from_pcap(pcap_file, port, interval=1.0)
        save_throughput_data(exp_dir, "beta", i, times, rates)
        beta_flows[f'beta_{i}'] = {'port': port, 'rates': rates, 'times': times}
    
    alpha_flows = {}
    for i in range(num_alpha):
        port = base_port + num_beta + i
        times, rates = extract_throughput_from_pcap(pcap_file, port, interval=1.0)
        times = times + alpha_starts[i]
        save_throughput_data(exp_dir, "alpha", i, times, rates)
        alpha_flows[f'alpha_{i}'] = {'port': port, 'rates': rates, 'times': times}
    
    return beta_flows, alpha_flows

def generate_color_palette(num_colors, colormap):
    if num_colors == 0:
        return []
    colors = plt.get_cmap(colormap)(np.linspace(0.3, 0.9, num_colors))
    return colors


def plot_tc_long_flow_results(exp_dir, bw, rtt, queue, baseline_data, beta_flows, alpha_flows, cca_beta, cca_alpha, conv_time, converged, dualpi2=True):
    bdp_multiple_ = get_queue_pkts_to_bdp_multiple(bw, rtt, queue)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    if not converged:
        conv_time = CONVERGENCE_BACKUP_TIME
    
    cap_times = np.arange(0, LONG_FLOW_EXP_FINISH_TIME + 1, 1)
    cap_rates = [bw] * len(cap_times)
    ax.plot(cap_times, cap_rates, label='Link Capacity', color='dodgerblue', alpha=0.7)
    ax.fill_between(cap_times, cap_rates, 0, color='dodgerblue', alpha=0.2)
    
    ax.plot(baseline_data['times'], baseline_data['rates'], 
            label=f'{cca_beta} Solo', color='blue', alpha=0.7, linestyle='-')
    
    if len(beta_flows) > 0:
        beta_colors = generate_color_palette(len(beta_flows), 'YlOrBr')
        for i, (flow_id, flow) in enumerate(beta_flows.items()):
            color = beta_colors[i]
            color = np.array([color[0], color[1] * 0.9, color[2] * 0.7, color[3]])
            ax.plot(flow['times'], flow['rates'], 
                   label=f'{cca_beta} Beta Flow {i+1}', color=color, alpha=0.7, linestyle='-')
    
    if len(alpha_flows) > 0:
        alpha_colors = generate_color_palette(len(alpha_flows), 'PuRd')
        for i, (flow_id, flow) in enumerate(alpha_flows.items()):
            color = alpha_colors[i]
            color = np.array([min(color[0] * 1.2, 1), color[1] * 0.8, min(color[2] * 1.1, 1), color[3]])
            ax.plot(flow['times'], flow['rates'], 
                   label=f'{cca_alpha} Alpha Flow {i+1}', color=color, alpha=0.7, linestyle='--')
    
    if converged:
        ax.axvline(x=conv_time, color='red', linestyle='--', linewidth=2, label=f'Convergence at {conv_time:.0f}s')
    
    ax.set_ylim(0, bw * 1.1)
    ax.set_xlim(0, LONG_FLOW_EXP_FINISH_TIME)
    ax.set_title('Throughput: Exp1 (Beta Solo) vs Exp2 (Competition)', fontsize=14)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Throughput (Mbps)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right', fontsize=8)
    
    baseline_avg = compute_average_throughput_after_convergence(
        baseline_data['times'], baseline_data['rates'], conv_time
    )
    
    beta_avgs = []
    for flow in beta_flows.values():
        avg = compute_average_throughput_after_convergence(flow['times'], flow['rates'], conv_time)
        beta_avgs.append(avg)
    
    beta_compete_avg = sum(beta_avgs) if beta_avgs else 0
    
    num_total = len(beta_flows) + len(alpha_flows)
    harm = (len(beta_flows) / num_total - beta_compete_avg / baseline_avg) if baseline_avg else 0
    
    conv_status = f"Converged at {conv_time:.0f}s" if converged else f"Not converged, backup to {conv_time:.0f}s"
    harm_text = f"Harm: {harm:.3f} Alone Avg: {baseline_avg:.2f} Mbps Compete Avg: {beta_compete_avg:.2f} Mbps\n{conv_status}"
    
    plt.title(f'Harm: {cca_beta} vs {cca_alpha} with {"DualPI2" if dualpi2 else "pfifo"} (BW={bw}Mbps, RTT={rtt}ms, Queue={queue}, BDP={bdp_multiple_})\n{harm_text}', 
             fontsize=16)
    
    plot_path = f"{exp_dir}/thr_latency.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    return harm, conv_time, converged


def run_tc_long_flow_experiment(exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_starts, cca_beta, cca_alpha, dualpi2=True, convergence=True):
    os.makedirs(exp_dir, exist_ok=True)
    
    solo_res, _ = run_tc_experiment(exp_dir, bw, rtt, queue, 1, 0, [], cca_beta, cca_alpha, LONG_FLOW_EXP_FINISH_TIME, None, "beta_solo", dualpi2=dualpi2)
    baseline_data = solo_res['beta_0']
    
    beta_flows, alpha_flows = run_tc_experiment(
        exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_starts, cca_beta, cca_alpha, LONG_FLOW_EXP_FINISH_TIME, None, "target_cc", dualpi2=dualpi2
    )

    if convergence:
        all_flows = {**beta_flows, **alpha_flows}
        conv_time, converged = find_convergence_time(all_flows)
    else:
        conv_time = 0
        converged = True

    harm, conv_time_used, conv_status = plot_tc_long_flow_results(
        exp_dir, bw, rtt, queue, baseline_data, beta_flows, alpha_flows, cca_beta, cca_alpha, conv_time, converged, dualpi2=dualpi2
    )

    os.system(f"rm -rf {exp_dir}/*.pcap")
    
    return harm, conv_time_used, conv_status


def plot_tc_short_flow_results(exp_dir, bw, rtt, queue, baseline_data, comp_a_beta, comp_a_alpha, comp_b_beta, comp_b_alpha, cca_beta, cca_alpha, alpha_starts, short_flow_harm_metric, dualpi2=True):
    bdp_multiple_ = get_queue_pkts_to_bdp_multiple(bw, rtt, queue)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    cap_times = np.arange(0, SHORT_FLOW_EXP_FINISH_TIME + 1, 1)
    cap_rates = [bw] * len(cap_times)

    ax1.plot(cap_times, cap_rates, label='Link Capacity', color='dodgerblue', alpha=0.7)
    ax1.fill_between(cap_times, cap_rates, 0, color='dodgerblue', alpha=0.2)

    ax1.plot(baseline_data['times'], baseline_data['rates'], 
             label=f'{cca_beta} Solo', color='blue', alpha=0.7, linestyle='-')

    if len(comp_a_beta) > 0:
        beta_color = "orange"
        for i, (flow_id, flow) in enumerate(comp_a_beta.items()):
            ax1.plot(flow['times'], flow['rates'], 
                    label=f'{cca_beta} Beta Flow {i+1}', color=beta_color, alpha=0.7, linestyle='-')
    if len(comp_a_alpha) > 0:
        alpha_colors = generate_color_palette(len(comp_a_alpha), 'PuRd')
        for i, (flow_id, flow) in enumerate(comp_a_alpha.items()):
            color = alpha_colors[i]
            color = np.array([min(color[0] * 1.2, 1), color[1] * 0.8, min(color[2] * 1.1, 1), color[3]])
            ax1.plot(flow['times'], flow['rates'], 
                    label=f'{cca_alpha} Alpha Flow {i+1}', color=color, alpha=0.7, linestyle='--')
    ax1.set_ylim(0, bw * 1.1)
    ax1.set_xlim(0, SHORT_FLOW_EXP_FINISH_TIME)
    ax1.set_title('Competition with Alpha', fontsize=16)
    ax1.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel('Throughput (Mbps)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(cap_times, cap_rates, label='Link Capacity', color='dodgerblue', alpha=0.7)
    ax2.fill_between(cap_times, cap_rates, 0, color='dodgerblue', alpha=0.2)

    ax2.plot(baseline_data['times'], baseline_data['rates'], 
             label=f'{cca_beta} Solo', color='blue', alpha=0.7, linestyle='-')
    
    if len(comp_b_beta) > 0:
        beta_color = "green"
        for i, (flow_id, flow) in enumerate(comp_b_beta.items()):
            ax2.plot(flow['times'], flow['rates'], 
                    label=f'{cca_beta} Beta Flow {i+1}', color=beta_color, alpha=0.7, linestyle='-')
    if len(comp_b_alpha) > 0:
        alpha_colors = plt.cm.Greens(np.linspace(0.6, 0.9, len(comp_b_alpha)))
        for i, (flow_id, flow) in enumerate(comp_b_alpha.items()):
            color = alpha_colors[i]
            color = np.array([alpha_colors[i][0] * 0.7, min(alpha_colors[i][1] * 1.3, 1), alpha_colors[i][2] * 0.7, alpha_colors[i][3]])
            ax2.plot(flow['times'], flow['rates'], 
                    label=f'{cca_beta} Alpha Flow {i+1}', color=color, alpha=0.7, linestyle='--')
    ax2.set_ylim(0, bw * 1.1)
    ax2.set_xlim(0, SHORT_FLOW_EXP_FINISH_TIME)
    ax2.set_title('Competition with Beta', fontsize=16)
    ax2.set_xlabel('Time (s)', fontsize=14)
    ax2.set_ylabel('Throughput (Mbps)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)

    ts = SHORT_FLOW_BETA_ALONE_TIME + min(alpha_starts) if alpha_starts else SHORT_FLOW_BETA_ALONE_TIME
    
    max_alpha_end_time = 0
    if len(comp_a_alpha) > 0:
        for flow in comp_a_alpha.values():
            max_alpha_end_time = max(max_alpha_end_time, flow['times'][-1])

    if short_flow_harm_metric == "download_bytes":
        te = ts + SHORT_FLOW_DURATION * 2

        baseline_mask = (baseline_data['times'] >= ts) & (baseline_data['times'] <= te)
        baseline_bytes = sum(np.array(baseline_data['rates'])[baseline_mask]) / 8

        comp_a_beta_bytes = 0
        for flow in comp_a_beta.values():
            flow_mask = (flow['times'] >= ts) & (flow['times'] <= te)
            comp_a_beta_bytes += sum(np.array(flow['rates'])[flow_mask]) / 8
        
        comp_b_beta_bytes = 0
        for flow in comp_b_beta.values():
            flow_mask = (flow['times'] >= ts) & (flow['times'] <= te)
            comp_b_beta_bytes += sum(np.array(flow['rates'])[flow_mask]) / 8
        
        absolute_harm_comp_alpha = (baseline_bytes - comp_a_beta_bytes) / baseline_bytes if baseline_bytes else 0
        absolute_harm_comp_beta = (baseline_bytes - comp_b_beta_bytes) / baseline_bytes if baseline_bytes else 0
        harm = absolute_harm_comp_alpha - absolute_harm_comp_beta
        
        ax1.axvline(x=ts, color='purple', linestyle='-', linewidth=2, label='ts (Harm Calculation Start)')
        ax1.axvline(x=te, color='brown', linestyle='-', linewidth=2, label='te (Harm Calculation End)')
        ts_idx = int(ts) if int(ts) < len(comp_a_beta['beta_0']['times']) else len(comp_a_beta['beta_0']['times']) - 1
        te_idx = int(te+1) if int(te+1) < len(comp_a_beta['beta_0']['times']) else len(comp_a_beta['beta_0']['times'])
        ax1.fill_between(comp_a_beta['beta_0']['times'][ts_idx:te_idx], comp_a_beta['beta_0']['rates'][ts_idx:te_idx], 0, color='yellow', alpha=0.3)
        ax1.legend(loc='lower right', fontsize=10)

        ax2.axvline(x=ts, color='purple', linestyle='-', linewidth=2, label='ts (Harm Calculation Start)')
        ax2.axvline(x=te, color='brown', linestyle='-', linewidth=2, label='te (Harm Calculation End)')
        ts_idx = int(ts) if int(ts) < len(comp_b_beta['beta_0']['times']) else len(comp_b_beta['beta_0']['times']) - 1
        te_idx = int(te+1) if int(te+1) < len(comp_b_beta['beta_0']['times']) else len(comp_b_beta['beta_0']['times'])
        ax2.fill_between(comp_b_beta['beta_0']['times'][ts_idx:te_idx], comp_b_beta['beta_0']['rates'][ts_idx:te_idx], 0, color='yellow', alpha=0.3)
        ax2.legend(loc='lower right', fontsize=10)

        harm_text = f"Harm (Download Bytes): {harm:.3f}\n" \
                    f"Absolute Harm Compete Alpha: {absolute_harm_comp_alpha:.3f}\n" \
                    f"Absolute Harm Compete Beta: {absolute_harm_comp_beta:.3f}\n" \
                    f"Compete Alpha Bytes: {comp_a_beta_bytes:.2f} MB\n" \
                    f"Compete Beta Bytes: {comp_b_beta_bytes:.2f} MB\n" \
                    f"Baseline Bytes: {baseline_bytes:.2f} MB"
    
    elif short_flow_harm_metric == "recovery_time":
        beta_vs_alpha_recovery_time = SHORT_FLOW_BETA_ALONE_TIME
        if len(comp_a_beta) > 0 and len(baseline_data['times']) > 0:
            comp_a_flow = list(comp_a_beta.values())[0]
            for time, rate in zip(comp_a_flow['times'], comp_a_flow['rates']):
                if time < max_alpha_end_time:
                    continue
                
                baseline_idx = np.searchsorted(baseline_data['times'], time)
                if baseline_idx >= len(baseline_data['rates']):
                    break
                    
                baseline_rate = baseline_data['rates'][baseline_idx]
                if rate >= 0.95 * baseline_rate:
                    beta_vs_alpha_recovery_time = time - max_alpha_end_time
                    break

        beta_vs_beta_recovery_time = SHORT_FLOW_BETA_ALONE_TIME
        if len(comp_b_beta) > 0 and len(baseline_data['times']) > 0:
            comp_b_flow = list(comp_b_beta.values())[0]
            for time, rate in zip(comp_b_flow['times'], comp_b_flow['rates']):
                if time < max_alpha_end_time:
                    continue
                
                baseline_idx = np.searchsorted(baseline_data['times'], time)
                if baseline_idx >= len(baseline_data['rates']):
                    break
                    
                baseline_rate = baseline_data['rates'][baseline_idx]
                if rate >= 0.95 * baseline_rate:
                    beta_vs_beta_recovery_time = time - max_alpha_end_time
                    break
        
        harm = (beta_vs_alpha_recovery_time - beta_vs_beta_recovery_time) / (SHORT_FLOW_DURATION * 2)

        ax1.axvline(x=ts, color='purple', linestyle='-', linewidth=2, label='ts (Short Flow Start)')
        ax1.axvline(x=max_alpha_end_time, color='brown', linestyle='-', linewidth=2, label='Max Alpha End Time')
        ax1.axvline(x=max_alpha_end_time + beta_vs_alpha_recovery_time, color='green', linestyle='--', linewidth=2, label='Beta vs Alpha Recovery Time')
        ax1.legend(loc='lower right', fontsize=10)

        ax2.axvline(x=ts, color='purple', linestyle='-', linewidth=2, label='ts (Short Flow Start)')
        ax2.axvline(x=max_alpha_end_time, color='brown', linestyle='-', linewidth=2, label='Max Alpha End Time')
        ax2.axvline(x=max_alpha_end_time + beta_vs_beta_recovery_time, color='green', linestyle='--', linewidth=2, label='Beta vs Beta Recovery Time')
        ax2.legend(loc='lower right', fontsize=10)

        harm_text = f"Harm (Recovery Time): {harm:.3f}\n" \
                    f"Beta vs Alpha Recovery Time: {beta_vs_alpha_recovery_time:.2f} s\n" \
                    f"Beta vs Beta Recovery Time: {beta_vs_beta_recovery_time:.2f} s\n" \
                    f"ts (Short Flow Start): {ts:.2f} s\n" \
                    f"Max Alpha End Time: {max_alpha_end_time:.2f} s"
    
    else:
        harm = 0
        harm_text = "Harm calculation method not recognized."

    plt.suptitle(f'Short Flow Experiment: {cca_beta} vs {cca_alpha} with {"DualPI2" if dualpi2 else "pfifo"} (BW={bw}Mbps, RTT={rtt}ms, Queue={queue}, BDP={bdp_multiple_})', fontsize=18)
    plt.figtext(0.6, 0.01, harm_text, ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plot_path = f"{exp_dir}/short_flow_thr.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    return harm


def run_tc_short_flow_experiment(exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_start_type, cca_beta, cca_alpha, short_flow_harm_metric, dualpi2=True):
    os.makedirs(exp_dir, exist_ok=True)
    original_alpha_starts = ALPHA_START_TIMES_MAP[alpha_start_type][:num_alpha]
    adjusted_alpha_starts = [t + SHORT_FLOW_BETA_ALONE_TIME for t in original_alpha_starts]

    solo_res, _ = run_tc_experiment(exp_dir, bw, rtt, queue, 1, 0, [], cca_beta, cca_alpha, SHORT_FLOW_EXP_FINISH_TIME, None, "beta_solo", dualpi2=dualpi2)
    baseline_data = solo_res['beta_0']

    comp_a_beta, comp_a_alpha = run_tc_experiment(
        exp_dir, bw, rtt, queue, num_beta, num_alpha, adjusted_alpha_starts, cca_beta, cca_alpha, SHORT_FLOW_EXP_FINISH_TIME, SHORT_FLOW_DURATION, "compete_a", dualpi2=dualpi2
    )

    comp_b_beta, comp_b_alpha = run_tc_experiment(
        exp_dir, bw, rtt, queue, num_beta, num_alpha, adjusted_alpha_starts, cca_beta, cca_beta, SHORT_FLOW_EXP_FINISH_TIME, SHORT_FLOW_DURATION, "compete_b", dualpi2=dualpi2
    )

    harm = plot_tc_short_flow_results(exp_dir, bw, rtt, queue, baseline_data,
                                   comp_a_beta, comp_a_alpha,
                                   comp_b_beta, comp_b_alpha,
                                   cca_beta, cca_alpha,
                                   original_alpha_starts, short_flow_harm_metric, dualpi2=dualpi2)

    os.system(f"rm -rf {exp_dir}/*.pcap")

    return harm


MORE_COLORS = seaborn.color_palette("colorblind", 10)
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use(['seaborn-v0_8-paper','seaborn-v0_8-colorblind'])
plt.rc('font',**{'size':15, 'family':'sans-serif'})
plt.rc('axes', **{'titlesize':15,
                'titleweight':'bold',
                'labelsize':15,
                'grid':False,
                'axisbelow':True,
                'spines.right':False,
                'spines.top':False,
                'grid.axis':'x'})
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('legend', fontsize=15)
plt.rc('figure', titlesize=15, figsize=(6, 3))
plt.rc('lines', linewidth=2)
plt.rc('hatch', color='white')
plt.rc('savefig', transparent=True, bbox='tight', dpi=100)
plt.rc('legend', loc='upper center', fancybox=False, frameon=False, handlelength=1.5, columnspacing=1.5)


def load_throughput_data(exp_dir, prefix, flow_type, flow_idx):
    filename = f"{exp_dir}/{prefix}_{flow_type}_thr_{flow_idx}.txt"
    times = []
    rates = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                times.append(float(parts[0]))
                rates.append(float(parts[1]))
    return np.array(times), np.array(rates)


def plot_long_flow_results_in_pdf(exp_dir, bw, rtt, queue, num_beta, num_alpha, cca_beta, cca_alpha, conv_time, converged, dualpi2=True):
    bdp_multiple_ = get_queue_pkts_to_bdp_multiple(bw, rtt, queue)

    if cca_alpha == "bbr":
        cca_alpha = "bbr3"

    if not converged:
        conv_time = CONVERGENCE_BACKUP_TIME
    
    fig, ax = plt.subplots(figsize=(6, 3))

    # conv_time = 0
    
    for i in range(num_beta):
        times, rates = load_throughput_data(exp_dir, "target_cc", "beta", i)
        mask = times >= conv_time
        times_filtered = times[mask]
        rates_filtered = rates[mask]
        ax.plot(times_filtered, rates_filtered, 
               label=f'{cca_beta}-{i}', color=MORE_COLORS[i % len(MORE_COLORS)], alpha=0.8, linewidth=2)
    
    for i in range(num_alpha):
        times, rates = load_throughput_data(exp_dir, "target_cc", "alpha", i)
        mask = times >= conv_time
        times_filtered = times[mask]
        rates_filtered = rates[mask]
        ax.plot(times_filtered, rates_filtered, 
               label=f'{cca_alpha}-{i}', color=MORE_COLORS[(num_beta + i) % len(MORE_COLORS)], 
               alpha=0.8, linewidth=2, linestyle='-')
    
    ax.set_ylim(0, bw * 1.1)
    ax.set_xlim(conv_time, LONG_FLOW_EXP_FINISH_TIME)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Throughput (Mbps)')
    
    plot_path = f"{exp_dir}/long_flow_result.pdf"

    # add a dashed line here to indicate convergence time
    ax.axvline(x=conv_time, color='gray', linestyle='--', linewidth=1, label='Converged')

    avg_beta_throughputs = 0
    
    for i in range(num_beta):
        times, rates = load_throughput_data(exp_dir, "target_cc", "beta", i)
        mask = times >= conv_time
        times_filtered = times[mask]
        rates_filtered = rates[mask]
        avg_beta_throughputs += np.mean(rates_filtered) if len(rates_filtered) > 0 else 0
        print(avg_beta_throughputs)

    solo_time, solo_rates = load_throughput_data(exp_dir, "beta_solo", "beta", 0)
    solo_mask = solo_time >= conv_time
    solo_times_filtered = solo_time[solo_mask]
    solo_rates_filtered = solo_rates[solo_mask]
    solo_avg = np.mean(solo_rates_filtered) if len(solo_rates_filtered) > 0 else 0
    harm = (solo_avg - avg_beta_throughputs) / solo_avg - num_alpha / (num_alpha + num_beta)
    print(f"Average beta throughput after convergence: {avg_beta_throughputs:.2f} Mbps")
    print(f"Calculated harm: {harm:.4f}")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, fancybox=True, columnspacing=0.7)

    plt.savefig(plot_path, transparent=True, bbox_inches='tight', dpi=1200)
    plt.close(fig)


def plot_short_flow_results_in_pdf(exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_starts,
                                cca_beta, cca_alpha, flow_type, short_flow_harm_metric, dualpi2=True):
    bdp_multiple_ = get_queue_pkts_to_bdp_multiple(bw, rtt, queue)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    
    start_time = 60
    
    # baseline_times, baseline_rates = load_throughput_data(exp_dir, "beta_solo", "beta", 0)
    # mask = baseline_times >= start_time
    # times_filtered = baseline_times[mask]
    # rates_filtered = baseline_rates[mask]
    # ax.plot(times_filtered, rates_filtered, label=f'{cca_beta}-solo', color='gray', alpha=0.7, linewidth=2)
    
    prefix = "compete_a" if flow_type == 'alpha' else "compete_b"
    
    beta_color_offset = 0
    for i in range(num_beta):
        times, rates = load_throughput_data(exp_dir, prefix, "beta", i)
        mask = times >= start_time
        times_filtered = times[mask]
        rates_filtered = rates[mask]
        ax.plot(times_filtered, rates_filtered, 
               label=f'{cca_beta}-long', color=MORE_COLORS[beta_color_offset % len(MORE_COLORS)], 
               alpha=0.8, linewidth=2)
        beta_color_offset += 1
    
    alpha_label = cca_alpha if flow_type == 'alpha' else cca_beta
    alpha_color_offset = 3 if flow_type == 'alpha' else 5
    for i in range(num_alpha):
        times, rates = load_throughput_data(exp_dir, prefix, "alpha", i)
        mask = times >= start_time
        times_filtered = times[mask]
        rates_filtered = rates[mask]
        ax.plot(times_filtered, rates_filtered, 
               label=f'{alpha_label}-{i}', 
               color=MORE_COLORS[(alpha_color_offset + i) % len(MORE_COLORS)], 
               alpha=0.8, linewidth=2)
    
    if short_flow_harm_metric == "download_bytes":
        ts = SHORT_FLOW_BETA_ALONE_TIME + min(alpha_starts) if alpha_starts else SHORT_FLOW_BETA_ALONE_TIME
        te = ts + SHORT_FLOW_DURATION * 2
        print(f"Shading between {ts}s and {te}s for short flow duration.")
        
        ax.axvline(x=ts, color='purple', linestyle='-', linewidth=2, label='Harm Calculation Start')
        ax.axvline(x=te, color='brown', linestyle='-', linewidth=2, label='Harm Calculation End')
        
        beta_times, beta_rates = load_throughput_data(exp_dir, prefix, "beta", 0)
        mask = (beta_times >= ts) & (beta_times <= te)
        times_fill = beta_times[mask]
        rates_fill = beta_rates[mask]
        ax.fill_between(times_fill, rates_fill, 0, color='yellow', alpha=0.4)

        prefix = "beta_solo"
        solo_time, solo_rates = load_throughput_data(exp_dir, prefix, "beta", 0)
        solo_mask = (solo_time >= ts) & (solo_time <= te)
        solo_times_fill = solo_time[solo_mask]
        solo_rates_fill = solo_rates[solo_mask]
        baseline_bytes = sum(np.array(solo_rates_fill)) / 8

        prefix = "compete_a"
        comp_a_time, comp_a_rates = load_throughput_data(exp_dir, prefix, "beta", 0)
        comp_a_mask = (comp_a_time >= ts) & (comp_a_time <= te)
        comp_a_times_fill = comp_a_time[comp_a_mask]
        comp_a_rates_fill = comp_a_rates[comp_a_mask]
        comp_a_beta_bytes = sum(np.array(comp_a_rates_fill)) / 8

        prefix = "compete_b"
        comp_b_time, comp_b_rates = load_throughput_data(exp_dir, prefix, "beta", 0)
        comp_b_mask = (comp_b_time >= ts) & (comp_b_time <= te)
        comp_b_times_fill = comp_b_time[comp_b_mask]
        comp_b_rates_fill = comp_b_rates[comp_b_mask]
        comp_b_beta_bytes = sum(np.array(comp_b_rates_fill)) / 8
        
        absolute_harm_comp_alpha = (baseline_bytes - comp_a_beta_bytes) / baseline_bytes if baseline_bytes else 0
        absolute_harm_comp_beta = (baseline_bytes - comp_b_beta_bytes) / baseline_bytes if baseline_bytes else 0
        harm = absolute_harm_comp_alpha - absolute_harm_comp_beta

        print(f"Baseline beta bytes: {baseline_bytes:.2f} MB")
        print(f"Compete A beta bytes: {comp_a_beta_bytes:.2f} MB")
        print(f"Compete B beta bytes: {comp_b_beta_bytes:.2f} MB")
        print(f"Absolute Harm comp a: {absolute_harm_comp_alpha:.4f}")
        print(f"Absolute Harm comp b: {absolute_harm_comp_beta:.4f}")
        print(f"Relative Harm: {harm:.4f}")

    else:
        alpha_end_times = []
        for i in range(num_alpha):
            alpha_times, _ = load_throughput_data(exp_dir, prefix, "alpha", i)
            if len(alpha_times) > 0:
                alpha_end_times.append(alpha_times[-1])
        
        ts = SHORT_FLOW_BETA_ALONE_TIME + min(alpha_starts) if alpha_starts else SHORT_FLOW_BETA_ALONE_TIME
        print(f"Beta harm calculation starts at {ts}s.")

        baseline_times, baseline_rates = load_throughput_data(exp_dir, "beta_solo", "beta", 0)
        baseline_mask = baseline_times >= ts
        baseline_rates_after_ts = np.array(baseline_rates)[baseline_mask]
        baseline_avg = np.mean(baseline_rates_after_ts) if len(baseline_rates_after_ts) > 0 else 0
        recovery_threshold = baseline_avg * 0.95

        print(f"Beta recovery threshold set at {recovery_threshold:.2f} Mbps.")

        beta_times, beta_rates = load_throughput_data(exp_dir, prefix, "beta", 0)
        beta_mask = beta_times >= ts
        beta_times = beta_times[beta_mask]
        beta_rates = np.array(beta_rates)[beta_mask]
        recovery_times = beta_times[beta_rates >= recovery_threshold]
        te = recovery_times[0] if len(recovery_times) > 0 and recovery_times[0] >= ts else SHORT_FLOW_EXP_FINISH_TIME
        
        print(f"Beta recovery time: {te:.2f}s.")
        
        ax.axvline(x=ts, color='purple', linestyle='-', linewidth=2, label='Harm Calculation Start')
        ax.axvline(x=te, color='brown', linestyle='-', linewidth=2, label='Harm Calculation End')
    
    ax.set_ylim(0, bw * 1.1)
    ax.set_xlim(start_time, SHORT_FLOW_EXP_FINISH_TIME)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Throughput (Mbps)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.42), ncol=3, fancybox=True, columnspacing=0.7)
    
    plot_path = f"{exp_dir}/short_flow_{flow_type}_new.pdf"
    plt.savefig(plot_path, transparent=True, bbox_inches='tight', dpi=1200)
    plt.close(fig)


if __name__ == "__main__":
    exp_dir = "./results/cubic-prague-0.1-long-flow/tc"
    bw = 100
    rtt = 10
    queue = calculate_queue_size(bw, rtt, 1)
    num_beta = 1
    num_alpha = 1
    cca_beta = "cubic"
    cca_alpha = "prague"
    alpha_starts = ALPHA_START_TIMES_MAP[0]
    harm = run_tc_long_flow_experiment(
        exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_starts, cca_beta, cca_alpha
    )
    print(f"Harm: {harm}")
