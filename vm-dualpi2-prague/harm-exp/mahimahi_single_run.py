import numpy as np
import matplotlib.pylab as plt
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
from tc_single_run import ExperimentDataError


LONG_FLOW_EXP_FINISH_TIME = 180
SHORT_FLOW_EXP_FINISH_TIME = 60
SHORT_FLOW_DURATION = 10
PACKET_SIZE_BYTES = 1514
CONVERGENCE_WINDOW_SIZE = 60
CONVERGENCE_RPT_MIN_SIZE = 5
CONVERGENCE_ABS_THRESH = 0.2
CONVERGENCE_PCT_THRESH = 0.2
CONVERGENCE_MAX_TIME = 50

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
    return f'{bw}.trace'


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
        except subprocess.CalledProcessError:
            return port


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
        cmds += f"(iperf3 -c $MAHIMAHI_BASE -p {port} -t {duration} -i 0.5 -C {cca_beta} -R) & "
    
    for i in range(num_alpha):
        port = base_port + num_beta + i
        start = alpha_starts[i]
        flow_duration = alpha_dur if alpha_dur is not None else duration
        cmds += f"(sleep {start}; iperf3 -c $MAHIMAHI_BASE -p {port} -t {flow_duration} -i 0.5 -C {cca_alpha} -R) & "
    
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


def save_throughput_data(exp_dir, prefix, flow_type, flow_idx, times, rates):
    filename = f"{exp_dir}/{prefix}_{flow_type}_thr_{flow_idx}.txt"
    with open(filename, "w") as f:
        for t, r in zip(times, rates):
            f.write(f"{t} {r}\n")


def run_mahimahi_experiment(exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_starts, cca_beta, cca_alpha, duration, alpha_dur, pcap_name):
    mm_delay = rtt // 2
    trace_dir = generate_trace_file(bw, duration + 30)
    
    base_port = find_available_port()
    total_flows = num_beta + num_alpha
    server_ports = start_iperf_servers(total_flows, base_port, exp_dir)
    
    client_cmds = build_client_commands(
        num_beta, num_alpha, base_port, alpha_starts, cca_beta, cca_alpha, duration + 5, alpha_dur
    )
    
    all_ports = " or ".join(str(p) for p in server_ports)
    pcap_file = f"{exp_dir}/{pcap_name}.pcap"
    
    cmd = (
        f"mm-delay {mm_delay} "
        f"mm-link traces/{trace_dir} traces/{trace_dir} "
        f"--uplink-queue=droptail --uplink-queue-args=\"packets={queue}\" "
        f"--downlink-queue=droptail --downlink-queue-args=\"packets={queue}\" "
        f"-- sh -c '"
        f"script -q -c \"sudo tcpdump port {all_ports} -i ingress -w {pcap_file}\" & "
        f"{client_cmds}"
        f"sleep {duration}' &"
    )
    
    os.system(cmd)
    ti.sleep(duration + 10)
    os.system("sudo killall -9 iperf3 tcpdump")
    ti.sleep(0.2)
    
    beta_flows = {}
    for i in range(num_beta):
        port = base_port + i
        times, rates = extract_throughput_from_pcap(pcap_file, port, interval=0.5)
        save_throughput_data(exp_dir, pcap_name, "beta", i, times, rates)
        beta_flows[f'beta_{i}'] = {'port': port, 'rates': rates, 'times': times}
    
    alpha_flows = {}
    for i in range(num_alpha):
        port = base_port + num_beta + i
        times, rates = extract_throughput_from_pcap(pcap_file, port, interval=0.5)
        times = times + alpha_starts[i]
        save_throughput_data(exp_dir, pcap_name, "alpha", i, times, rates)
        alpha_flows[f'alpha_{i}'] = {'port': port, 'rates': rates, 'times': times}
    
    return beta_flows, alpha_flows


def generate_color_palette(num_colors, colormap):
    if num_colors == 0:
        return []
    colors = plt.get_cmap(colormap)(np.linspace(0.3, 0.9, num_colors))
    return colors


def plot_mahimahi_long_flow_results(exp_dir, bw, rtt, queue, baseline_data, beta_flows, alpha_flows, cca_beta, cca_alpha, conv_time, converged):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
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
    
    conv_status = f"Converged at {conv_time:.0f}s" if converged else f"Not converged (using {conv_time:.0f}s)"
    harm_text = f"Harm: {harm:.3f} Alone Avg: {baseline_avg:.2f} Mbps Compete Avg: {beta_compete_avg:.2f} Mbps\n{conv_status}"
    
    plt.title(f'Long Flow Harm: {cca_beta} vs {cca_alpha} (BW={bw}Mbps, RTT={rtt}ms, Queue={queue})\n{harm_text}', 
             fontsize=16)
    
    plot_path = f"{exp_dir}/thr_latency.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    if baseline_avg <= 0:
        raise ExperimentDataError(
            f"{exp_dir}: beta solo run recorded no throughput after {conv_time:.0f}s")
    if beta_compete_avg <= 0:
        raise ExperimentDataError(
            f"{exp_dir}: beta flows recorded no throughput while competing "
            f"(solo was {baseline_avg:.2f} Mbps)")

    return harm, conv_time, converged


def run_mahimahi_long_flow_experiment(exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_starts, cca_beta, cca_alpha):
    os.makedirs(exp_dir, exist_ok=True)
    
    solo_beta, _ = run_mahimahi_experiment(
        exp_dir, bw, rtt, queue, 1, 0, [], cca_beta, cca_alpha, 
        LONG_FLOW_EXP_FINISH_TIME, None, "beta_solo"
    )
    baseline_data = solo_beta['beta_0']
    
    beta_flows, alpha_flows = run_mahimahi_experiment(
        exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_starts, cca_beta, cca_alpha,
        LONG_FLOW_EXP_FINISH_TIME, None, "target_cc"
    )
    
    all_flows = {**beta_flows, **alpha_flows}
    conv_time, converged = find_convergence_time(all_flows)
    
    harm, conv_time_used, conv_status = plot_mahimahi_long_flow_results(
        exp_dir, bw, rtt, queue, baseline_data, beta_flows, alpha_flows, cca_beta, cca_alpha, conv_time, converged
    )
    
    os.system(f"rm -rf {exp_dir}/*.pcap")
    
    return harm, conv_time_used, conv_status


def plot_mahimahi_short_flow_results(exp_dir, bw, rtt, queue, baseline_data, comp_a_beta, comp_a_alpha, comp_b_beta, comp_b_alpha, cca_beta, cca_alpha, alpha_starts, short_flow_harm_metric):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    cap_times = np.arange(0, SHORT_FLOW_EXP_FINISH_TIME + 1, 1)
    cap_rates = [bw] * len(cap_times)
    
    ax1.plot(cap_times, cap_rates, label='Link Capacity', color='dodgerblue', alpha=0.7)
    ax1.fill_between(cap_times, cap_rates, 0, color='dodgerblue', alpha=0.2)
    
    ax1.plot(baseline_data['times'], baseline_data['rates'], 
            label=f'{cca_beta} Solo', color='blue', alpha=0.7, linestyle='-')
    
    for i, (flow_id, flow) in enumerate(comp_a_beta.items()):
        ax1.plot(flow['times'], flow['rates'], 
               label=f'{cca_beta} Beta Flow', color='orange', alpha=0.7, linestyle='-')
    
    if len(comp_a_alpha) > 0:
        alpha_colors = generate_color_palette(len(comp_a_alpha), 'PuRd')
        for i, (flow_id, flow) in enumerate(comp_a_alpha.items()):
            color = alpha_colors[i]
            color = np.array([min(color[0] * 1.2, 1), color[1] * 0.8, min(color[2] * 1.1, 1), color[3]])
            ax1.plot(flow['times'], flow['rates'], 
                   label=f'{cca_alpha} Alpha Flow {i+1}', color=color, alpha=0.7, linestyle='--')
    
    ax1.set_ylim(0, bw * 1.1)
    ax1.set_xlim(0, SHORT_FLOW_EXP_FINISH_TIME)
    ax1.set_title('Experiment 1 (Beta Solo) vs Experiment 2 (Competition)', fontsize=16)
    ax1.set_xlabel('Time (seconds)', fontsize=14)
    ax1.set_ylabel('Throughput (Mbps)', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.plot(cap_times, cap_rates, label='Link Capacity', color='dodgerblue', alpha=0.7)
    ax2.fill_between(cap_times, cap_rates, 0, color='dodgerblue', alpha=0.2)
    
    ax2.plot(baseline_data['times'], baseline_data['rates'], 
            label=f'{cca_beta} Solo', color='blue', alpha=0.7, linestyle='-')
    
    for i, (flow_id, flow) in enumerate(comp_b_beta.items()):
        ax2.plot(flow['times'], flow['rates'], 
               label=f'{cca_beta} Beta Flow' if i == 0 else "", color='green', alpha=0.7, linestyle='-')
    
    if len(comp_b_alpha) > 0:
        alpha_colors = plt.cm.Greens(np.linspace(0.6, 0.9, len(comp_b_alpha)))
        for i, (flow_id, flow) in enumerate(comp_b_alpha.items()):
            color = alpha_colors[i]
            color = np.array([color[0] * 0.7, min(color[1] * 1.3, 1), color[2] * 0.7, color[3]])
            ax2.plot(flow['times'], flow['rates'], 
                   label=f'{cca_beta} Alpha Flow {i+1}', color=color, alpha=0.7, linestyle='--')
    
    ax2.set_ylim(0, bw * 1.1)
    ax2.set_xlim(0, SHORT_FLOW_EXP_FINISH_TIME)
    ax2.set_title('Experiment 1 (Beta Solo) vs Experiment 3 (Beta Solo)', fontsize=16)
    ax2.set_xlabel('Time (seconds)', fontsize=14)
    ax2.set_ylabel('Throughput (Mbps)', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='lower right')
    
    ts = min(alpha_starts) if alpha_starts else 0
    max_alpha_end_time = 0
    for flow in comp_a_alpha.values():
        if len(flow['times']) > 0:
            max_alpha_end_time = max(max_alpha_end_time, flow['times'][-1])
    
    baseline_times = baseline_data['times']
    baseline_rates = baseline_data['rates']
    
    comp_a_beta_data = list(comp_a_beta.values())[0] if comp_a_beta else None
    comp_b_beta_data = list(comp_b_beta.values())[0] if comp_b_beta else None
    
    time_interval = 0.5
    
    invalid_reason = None

    if short_flow_harm_metric == 'download_bytes':
        te = ts + 20
        
        d1 = 0
        for time, rate in zip(baseline_times, baseline_rates):
            if time >= te:
                break
            start_of_interval = time - time_interval
            end_of_interval = time
            start_of_integration = max(start_of_interval, ts)
            end_of_integration = min(end_of_interval, max_alpha_end_time)
            duration_in_range = max(0, end_of_integration - start_of_integration)
            d1 += rate * duration_in_range
        
        beta_vs_alpha_d2 = 0
        if comp_a_beta_data:
            for time, rate in zip(comp_a_beta_data['times'], comp_a_beta_data['rates']):
                if time >= te:
                    break
                start_of_interval = time - time_interval
                end_of_interval = time
                start_of_integration = max(start_of_interval, ts)
                end_of_integration = min(end_of_interval, max_alpha_end_time)
                duration_in_range = max(0, end_of_integration - start_of_integration)
                beta_vs_alpha_d2 += rate * duration_in_range
        
        beta_vs_beta_d2 = 0
        if comp_b_beta_data:
            for time, rate in zip(comp_b_beta_data['times'], comp_b_beta_data['rates']):
                if time >= te:
                    break
                start_of_interval = time - time_interval
                end_of_interval = time
                start_of_integration = max(start_of_interval, ts)
                end_of_integration = min(end_of_interval, max_alpha_end_time)
                duration_in_range = max(0, end_of_integration - start_of_integration)
                beta_vs_beta_d2 += rate * duration_in_range
        
        absolute_harm_vs_alpha = (d1 - beta_vs_alpha_d2) / d1 if d1 > 0 else 0
        absolute_harm_vs_beta = (d1 - beta_vs_beta_d2) / d1 if d1 > 0 else 0
        harm = absolute_harm_vs_alpha - absolute_harm_vs_beta

        if d1 <= 0:
            invalid_reason = "beta solo downloaded nothing in the harm window"
        elif beta_vs_alpha_d2 <= 0:
            invalid_reason = f"beta long flow downloaded nothing while competing with {cca_alpha}"
        elif beta_vs_beta_d2 <= 0:
            invalid_reason = f"beta long flow downloaded nothing while competing with {cca_beta}"
        
        ax1.axvline(x=ts, color='purple', linestyle='-', linewidth=2, label='ts (Harm Calculation Start)')
        ax1.axvline(x=te, color='brown', linestyle='-', linewidth=2, label='te (Harm Calculation End)')
        if comp_a_beta_data:
            start_idx = int(ts * 2)
            end_idx = int(te * 2 + 1)
            ax1.fill_between(comp_a_beta_data['times'][start_idx:end_idx], 
                           comp_a_beta_data['rates'][start_idx:end_idx], 0, color='yellow', alpha=0.3)
        ax1.legend(loc='lower right')
        
        ax2.axvline(x=ts, color='purple', linestyle='-', linewidth=2, label='ts (Harm Calculation Start)')
        ax2.axvline(x=te, color='brown', linestyle='-', linewidth=2, label='te (Harm Calculation End)')
        if comp_b_beta_data:
            start_idx = int(ts * 2)
            end_idx = int(te * 2 + 1)
            ax2.fill_between(comp_b_beta_data['times'][start_idx:end_idx], 
                           comp_b_beta_data['rates'][start_idx:end_idx], 0, color='yellow', alpha=0.3)
        ax2.legend(loc='lower right')
        
        harm_text = f"Harm: {harm:.3f}\n" \
                f"Absolute Harm vs Alpha: {absolute_harm_vs_alpha:.3f}\n" \
                f"Absolute Harm vs Beta: {absolute_harm_vs_beta:.3f}\n" \
                f"D1 (Beta Solo Download Bytes): {d1:.2f}\n" \
                f"D2 (Beta vs Alpha Download Bytes): {beta_vs_alpha_d2:.2f}\n" \
                f"D2 (Beta vs Beta Download Bytes): {beta_vs_beta_d2:.2f}\n" \
                f"ts (Harm Calculation Start): {ts:.2f} s\n" \
                f"te (Harm Calculation End): {te:.2f} s\n" \
                f"Max Alpha End Time: {max_alpha_end_time:.2f} s"
    
    else:
        def _beta_recorded(d):
            return bool(d) and len(np.asarray(d['rates'])) > 0 and np.sum(d['rates']) > 0

        if not _beta_recorded(baseline_data):
            invalid_reason = "beta solo recorded no throughput"
        elif not _beta_recorded(comp_a_beta_data):
            invalid_reason = f"beta long flow recorded no throughput while competing with {cca_alpha}"
        elif not _beta_recorded(comp_b_beta_data):
            invalid_reason = f"beta long flow recorded no throughput while competing with {cca_beta}"

        beta_vs_alpha_recovery_time = 60
        if comp_a_beta_data and baseline_data:
            for time, rate, beta_solo_rate in zip(comp_a_beta_data['times'], comp_a_beta_data['rates'], baseline_rates):
                if time < max_alpha_end_time:
                    continue
                if rate >= 0.95 * beta_solo_rate:
                    beta_vs_alpha_recovery_time = time - max_alpha_end_time
                    break
        
        beta_vs_beta_recovery_time = 60
        if comp_b_beta_data and baseline_data:
            for time, rate, beta_solo_rate in zip(comp_b_beta_data['times'], comp_b_beta_data['rates'], baseline_rates):
                if time < max_alpha_end_time:
                    continue
                if rate >= 0.99 * beta_solo_rate:
                    beta_vs_beta_recovery_time = time - max_alpha_end_time
                    break
        
        harm = (beta_vs_alpha_recovery_time - beta_vs_beta_recovery_time) / 10.0
        
        ax1.axvline(x=ts, color='purple', linestyle='-', linewidth=2, label='ts (Short Flow Start)')
        ax1.axvline(x=max_alpha_end_time, color='brown', linestyle='-', linewidth=2, label='Max Alpha End Time')
        ax1.axvline(x=max_alpha_end_time + beta_vs_alpha_recovery_time, color='green', linestyle='--', linewidth=2, label='Beta vs Alpha Recovery Time')
        ax1.legend(loc='lower right')
        
        ax2.axvline(x=ts, color='purple', linestyle='-', linewidth=2, label='ts (Short Flow Start)')
        ax2.axvline(x=max_alpha_end_time, color='brown', linestyle='-', linewidth=2, label='Max Alpha End Time')
        ax2.axvline(x=max_alpha_end_time + beta_vs_beta_recovery_time, color='green', linestyle='--', linewidth=2, label='Beta vs Beta Recovery Time')
        ax2.legend(loc='lower right')
        
        harm_text = f"Harm: {harm:.3f}\n" \
                f"Beta vs Alpha Recovery Time: {beta_vs_alpha_recovery_time:.2f} s\n" \
                f"Beta vs Beta Recovery Time: {beta_vs_beta_recovery_time:.2f} s\n" \
                f"ts (Short Flow Start): {ts:.2f} s\n" \
                f"Max Alpha End Time: {max_alpha_end_time:.2f} s"
    
    plt.figtext(0.6, 0.01, harm_text, ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    start_times_str = ",".join(map(str, alpha_starts))
    plt.suptitle(f'Harm Analysis: {cca_beta} vs {cca_alpha} (BW={bw}Mbps, Delay={rtt//2}ms, Queue={queue}\n Alpha Starts={start_times_str})', fontsize=18)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    plot_path = f"{exp_dir}/thr.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    os.system(f"rm -rf {exp_dir}/*.pcap")

    if invalid_reason:
        raise ExperimentDataError(f"{exp_dir}: {invalid_reason}")

    return harm


def run_mahimahi_short_flow_experiment(exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_start_type, cca_beta, cca_alpha, short_flow_harm_metric):
    os.makedirs(exp_dir, exist_ok=True)
    
    alpha_starts = ALPHA_START_TIMES_MAP[alpha_start_type]
    alpha_starts = alpha_starts[:num_alpha]
    start_times_str = ",".join(map(str, alpha_starts))
    
    solo_beta, _ = run_mahimahi_experiment(
        exp_dir, bw, rtt, queue, 1, 0, [], cca_beta, cca_alpha,
        SHORT_FLOW_EXP_FINISH_TIME, None, "beta_solo"
    )
    baseline_data = solo_beta['beta_0']
    
    comp_a_beta, comp_a_alpha = run_mahimahi_experiment(
        exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_starts, cca_beta, cca_alpha,
        SHORT_FLOW_EXP_FINISH_TIME, SHORT_FLOW_DURATION, "target_cc"
    )
    
    comp_b_beta, comp_b_alpha = run_mahimahi_experiment(
        exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_starts, cca_beta, cca_beta,
        SHORT_FLOW_EXP_FINISH_TIME, SHORT_FLOW_DURATION, "solo"
    )
    
    harm = plot_mahimahi_short_flow_results(
        exp_dir, bw, rtt, queue, baseline_data,
        comp_a_beta, comp_a_alpha,
        comp_b_beta, comp_b_alpha,
        cca_beta, cca_alpha,
        alpha_starts, short_flow_harm_metric
    )
    
    return harm


if __name__ == "__main__":
    exp_dir = "./results/cubic-bbr-short-flow/long_single_run"
    bw = 25
    rtt = 10
    queue = calculate_queue_size(bw, rtt, 2.0)
    num_beta = 1
    num_alpha = 2
    alpha_starts = [0, 0]
    cca_beta = "cubic"
    cca_alpha = "cubic"

    harm = run_mahimahi_long_flow_experiment(
        exp_dir, bw, rtt, queue, num_beta, num_alpha, alpha_starts, cca_beta, cca_alpha
    )
    print(f"Long Flow Harm: {harm}")