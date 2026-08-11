import math
import random
import numpy as np
from functools import total_ordering
import statistics
import os
import pickle
import sys
import subprocess
import logging
import psutil
import time as ti
from scapy.all import *
from tc_single_run import ExperimentDataError, run_tc_long_flow_experiment, run_tc_short_flow_experiment
from mahimahi_single_run import run_mahimahi_long_flow_experiment, run_mahimahi_short_flow_experiment

alpha_cca = sys.argv[2]  # prague
beta_cca = sys.argv[1]   # cubic
pop_size = int(sys.argv[9])
prob_crossover = float(sys.argv[10])
num_generations = 15
experiment_budget = int(sys.argv[3])
label = sys.argv[4]
testbed = int(sys.argv[5])
seed = int(sys.argv[6])
mutation_version = sys.argv[7]
mutation_prob = float(sys.argv[8])
time_limit_hours = float(sys.argv[11])
time_limit_seconds = time_limit_hours * 3600

MIN_BW = 25
MAX_BW = 200
MIN_RTT = 10
MAX_RTT = 100

experiment_type = "short-flow" if len(sys.argv) > 12 else "long-flow"
short_flow_harm_metric = sys.argv[12] if len(sys.argv) > 12 else "download_bytes"

bdp_dict = {}

alpha_flow_start_times_long_flow = [0, 0, 0, 0, 0]

dir_path = f"./results/{beta_cca}-{alpha_cca}-{time_limit_hours}-{experiment_type}"
os.makedirs(dir_path, exist_ok=True)

logging.basicConfig(filename=f"{dir_path}/ga_{beta_cca}_{alpha_cca}_{label}.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
bdp_multiples = [.25, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
# bdp_multiples = [2]
bdp_multiples_to_int = {bdp_multiples[i]: i for i in range(len(bdp_multiples))}

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
random.seed(seed)

def ceil_pow_two(val):
    return 2 ** math.ceil(math.log2(val))

def get_bdp_to_queue_pkts(bw_Mbps, rtt_ms, que_mult, packet_size_B=1514):
    try:
        size = max(4, int(que_mult * bw_Mbps * 1e6 * rtt_ms / 1e3 / 8 / packet_size_B))
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
    return min(bdp_multiples, key=lambda x: abs(x - value))

def get_initial_pop(initial_pop_size, min_bw=MIN_BW, max_bw=MAX_BW, min_rtt=MIN_RTT, max_rtt=MAX_RTT, min_queue=32, max_queue=8192, min_flows=1, max_flows=5, min_start=10, max_start=50):
    initial_pop = []
    min_bw_log = math.log(min_bw,2)
    max_bw_log = math.log(max_bw,2)
    min_rtt_log = math.log(min_rtt,2)
    max_rtt_log = math.log(max_rtt,2)
    
    for x in range(initial_pop_size):
        bw = round(2 ** random.uniform(min_bw_log, max_bw_log))
        rtt = round(2 ** random.uniform(min_rtt_log, max_rtt_log))
        bdp_multiple = random.choice(bdp_multiples)
        num_alpha_flows = random.randint(min_flows, max_flows)
        num_beta_flows = random.randint(min_flows, max_flows) if experiment_type == "long-flow" else 1
        alpha_start = 1
        
        if x <= 10:
            bw = min_bw
        elif x <= 20:
            bw = max_bw
        elif x <= 30:
            rtt = min_rtt
        elif x <= 40:
            rtt = max_rtt
        elif x <= 50:
            bdp_multiple = bdp_multiples[0]
        elif x <= 60:
            bdp_multiple = bdp_multiples[-1]
        
        queue = get_bdp_to_queue_pkts(bw, rtt, bdp_multiple)
        chrom = (bw, rtt, queue, num_beta_flows, num_alpha_flows, alpha_start)
        bdp_dict[chrom] = bdp_multiple
        initial_pop.append(chrom)
    return initial_pop

def selection(population, fitness_per_exp):
    population_to_fitness = {chrom: fitness_per_exp[chrom] for chrom in population
                             if chrom in fitness_per_exp and fitness_per_exp[chrom] is not None}
    top_half = sorted(population_to_fitness.items(), key=lambda item: item[1], reverse=True)[:(len(population) // 2)]
    return [chrom for (chrom,fitness) in top_half]

def crossover(chrom1, chrom2, prob_crossover=prob_crossover):
    assert len(chrom1) == len(chrom2)
    child1 = list(chrom1)
    child2 = list(chrom2)
    
    if random.random() > prob_crossover:
        return [tuple(child1), tuple(child2)]
    
    fixed_len = 6
    bdp_multiple1 = get_queue_pkts_to_bdp_multiple(child1[0], child1[1], child1[2])
    bdp_multiple2 = get_queue_pkts_to_bdp_multiple(child2[0], child2[1], child2[2])
    
    for gene_idx in range(fixed_len):
        coin_flip = random.randrange(2)
        if coin_flip == 1:
            if gene_idx == 2:
                bdp_multiple1, bdp_multiple2 = blend_queue_size(child1, child2)
            elif gene_idx == 3:
                new_val1, new_val2 = blend_values(child1[4], child2[4])
                new_val1 = max(1, min(5, round(new_val1))) if experiment_type == "long-flow" else 1
                new_val2 = max(1, min(5, round(new_val2))) if experiment_type == "long-flow" else 1
                child1[3] = new_val1
                child2[3] = new_val2
            elif gene_idx == 4:
                new_val1, new_val2 = blend_values(child1[4], child2[4])
                new_val1 = max(1, min(5, round(new_val1)))
                new_val2 = max(1, min(5, round(new_val2)))
                child1[4] = new_val1
                child2[4] = new_val2
            elif gene_idx == 0 or gene_idx == 1:
                val1, val2 = get_blended_vals(gene_idx, child1[gene_idx], child2[gene_idx])
                child1[gene_idx] = val1
                child2[gene_idx] = val2
            else:
                child1[gene_idx] = chrom1[gene_idx]
                child2[gene_idx] = chrom2[gene_idx]
    
    child1[2] = get_bdp_to_queue_pkts(child1[0], child1[1], bdp_multiple1)
    child2[2] = get_bdp_to_queue_pkts(child2[0], child2[1], bdp_multiple2)
    
    child1 = tuple(child1)
    child2 = tuple(child2)
    
    bdp_dict[child1] = bdp_multiple1
    bdp_dict[child2] = bdp_multiple2
    
    return [child1, child2]

def blend_queue_size(a_chrom, b_chrom):
    a_bdp = get_queue_pkts_to_bdp_multiple(a_chrom[0], a_chrom[1], a_chrom[2])
    b_bdp = get_queue_pkts_to_bdp_multiple(b_chrom[0], b_chrom[1], b_chrom[2])
    a_int = bdp_multiples_to_int[a_bdp]
    b_int = bdp_multiples_to_int[b_bdp]
    a_int_new, b_int_new = blend_values(a_int, b_int)
    return bdp_multiples[a_int_new], bdp_multiples[b_int_new]

def get_blended_vals(i, a, b):
    if i == 0 or i == 1:
        return blend_values_log(a,b)
    if i == 3 or i == 4:
        return blend_values(a,b)

def blend_values(a, b):
    beta = random.uniform(0, 1)
    new_a = round(a - (beta * (a - b)))
    new_b = round(b + (beta * (a - b)))
    return new_a, new_b

def blend_values_log(a, b):
    beta = random.uniform(0, 1)
    new_a = round(2 ** (math.log(a,2) - (beta * (math.log(a,2) - math.log(b,2)))))
    new_b = round(2 ** (math.log(b,2) + (beta * (math.log(a,2) - math.log(b,2)))))
    return new_a, new_b

def mutate_v2(chromosome, min_bw=MIN_BW, max_bw=MAX_BW, min_rtt=MIN_RTT, max_rtt=MAX_RTT, min_queue=32, max_queue=8192, min_flows=1, max_flows=5, min_start=10, max_start=50, prob_mutate=mutation_prob):
    min_bw_log = math.log(min_bw,2)
    max_bw_log = math.log(max_bw,2)
    min_rtt_log = math.log(min_rtt,2)
    max_rtt_log = math.log(max_rtt,2)
    chromosome = list(chromosome)
    fixed_len = 6
    
    bdp_multiple = get_queue_pkts_to_bdp_multiple(chromosome[0], chromosome[1], chromosome[2])
    
    for idx in range(fixed_len):
        coin_flip = random.random()
        if coin_flip < prob_mutate:
            if idx == 0:
                chromosome[idx] = round(2 ** random.uniform(min_bw_log, max_bw_log))
            elif idx == 1:
                chromosome[idx] = round(2 ** random.uniform(min_rtt_log, max_rtt_log))
            elif idx == 2:
                bdp_multiple = random.choice(bdp_multiples)
            elif idx == 3:
                chromosome[idx] = random.randint(min_flows, max_flows) if experiment_type == "long-flow" else 1
            elif idx == 4:
                chromosome[idx] = random.randint(min_flows, max_flows)
            else:
                chromosome[idx] = chromosome[idx]
    
    chromosome[2] = get_bdp_to_queue_pkts(chromosome[0], chromosome[1], bdp_multiple)
    chromosome = tuple(chromosome)
    bdp_dict[chromosome] = bdp_multiple
    return chromosome

def mutate_v3(chromosome, min_bw=MIN_BW, max_bw=MAX_BW, min_rtt=MIN_RTT, max_rtt=MAX_RTT, min_queue=32, max_queue=8192, min_flows=1, max_flows=5, min_start=10, max_start=50, prob_mutate=0.2):
    min_bw_log = math.log(min_bw,2)
    max_bw_log = math.log(max_bw,2)
    min_rtt_log = math.log(min_rtt,2)
    max_rtt_log = math.log(max_rtt,2)
    chromosome = list(chromosome)
    
    bdp_multiple = get_queue_pkts_to_bdp_multiple(chromosome[0], chromosome[1], chromosome[2])
    
    if random.random() < prob_mutate:
        gene_type = random.choice(['bw', 'rtt', 'queue', 'beta_flows', 'alpha_flows', 'start_time'])
        
        if gene_type == 'bw':
            chromosome[0] = round(2 ** random.uniform(min_bw_log, max_bw_log))
        elif gene_type == 'rtt':
            chromosome[1] = round(2 ** random.uniform(min_rtt_log, max_rtt_log))
        elif gene_type == 'queue':
            bdp_multiple = random.choice(bdp_multiples)
        elif gene_type == 'beta_flows':
            chromosome[3] = random.randint(min_flows, max_flows) if experiment_type == "long-flow" else 1
        elif gene_type == 'alpha_flows':
            new_alpha_flows = random.randint(min_flows, max_flows)
            chromosome[4] = new_alpha_flows
        else:
            chromosome[5] = random.randint(1, 3)
    
    chromosome[2] = get_bdp_to_queue_pkts(chromosome[0], chromosome[1], bdp_multiple)
    chromosome = tuple(chromosome)
    bdp_dict[chromosome] = bdp_multiple
    return chromosome

def mutate_v1(chromosome, min_bw=MIN_BW, max_bw=MAX_BW, min_rtt=MIN_RTT, max_rtt=MAX_RTT, min_queue=32, max_queue=8192, min_flows=1, max_flows=5, min_start=10, max_start=50, prob_mutate=0.3):
    min_bw_log = math.log(min_bw,2)
    max_bw_log = math.log(max_bw,2)
    min_rtt_log = math.log(min_rtt,2)
    max_rtt_log = math.log(max_rtt,2)
    chromosome = list(chromosome)
    
    bdp_multiple = get_queue_pkts_to_bdp_multiple(chromosome[0], chromosome[1], chromosome[2])
    
    if random.random() > prob_mutate:
        return tuple(chromosome)
    
    gene_idx = random.randrange(len(chromosome))
    
    if gene_idx == 0:
        chromosome[0] = round(2 ** random.uniform(min_bw_log, max_bw_log))
    elif gene_idx == 1:
        chromosome[1] = round(2 ** random.uniform(min_rtt_log, max_rtt_log))
    elif gene_idx == 2:
        bdp_multiple = random.choice(bdp_multiples)
    elif gene_idx == 3:
        chromosome[3] = random.randint(min_flows, max_flows) if experiment_type == "long-flow" else 1
    elif gene_idx == 4:
        new_alpha_flows = random.randint(min_flows, max_flows)
        chromosome[4] = new_alpha_flows
    else:
        chromosome[5] = random.randint(1, 3)
    
    chromosome[2] = get_bdp_to_queue_pkts(chromosome[0], chromosome[1], bdp_multiple)
    chromosome = tuple(chromosome)
    bdp_dict[chromosome] = bdp_multiple
    return chromosome

def mutate(chromosome, min_bw=MIN_BW, max_bw=MAX_BW, min_rtt=MIN_RTT, max_rtt=MAX_RTT, min_queue=32, max_queue=8192, min_flows=1, max_flows=5, min_start=10, max_start=50, prob_mutate=0.3):
    if mutation_version == "mutate_v1":
        return mutate_v1(chromosome)
    elif mutation_version == "mutate_v2":
        return mutate_v2(chromosome)
    elif mutation_version == "mutate_v3":
        return mutate_v3(chromosome)
    else:
        raise Exception("Mutation version not accepted")

def trace_generator(bw, last_time):
    os.makedirs("traces", exist_ok=True)
    save_name = 'traces/' + str(bw) + '.trace'
    num_packets = int(float(bw) * 1e6 / 12000 * last_time)
    ts_list = np.linspace(0, last_time*1000, num=num_packets, endpoint=False)
    with open(save_name, 'w') as trace:
        for ts in ts_list:
            trace.write('%d\n' % ts)
    return str(bw) + '.trace'

def open_port_finder(port):
    while True:
        try:
            result = subprocess.check_output(['netstat', '-at'], stderr=subprocess.STDOUT, text=True)
            grep_result = subprocess.run(['grep', str(port)], input=result, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if grep_result.returncode != 0 or not grep_result.stdout:
                break
            port += 10
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
    return port

def rerun_experiments_until_complete(all_harm_dict, experiments_to_run, beta_cca, alpha_cca):
    for experiment in experiments_to_run:
        if experiment not in all_harm_dict:
            try:
                bw = experiment[0]
                rtt_ms = experiment[1]
                queue = experiment[2]
                num_beta_flows = experiment[3]
                num_alpha_flows = experiment[4]
                exp_dir = f"{dir_path}/raygen-{beta_cca}-{alpha_cca}-{label}/{bw}bw-{rtt_ms}rtt-{queue}q-{beta_cca}-{num_beta_flows}-{alpha_cca}-{num_alpha_flows}"
                os.makedirs(exp_dir, exist_ok=True)

                if experiment_type == "short-flow":
                    start_type = experiment[5]
                    if beta_cca == "prague" or alpha_cca == "prague":
                        harm = run_tc_short_flow_experiment(exp_dir, bw, rtt_ms, queue, num_beta_flows, num_alpha_flows, start_type, beta_cca, alpha_cca, short_flow_harm_metric)
                    else:
                        harm = run_tc_short_flow_experiment(exp_dir, bw, rtt_ms, queue, num_beta_flows, num_alpha_flows, start_type, beta_cca, alpha_cca, short_flow_harm_metric, dualpi2=False)
                    logger.info(f"Experiment {experiment} completed with harm {harm:.4f}")
                    all_harm_dict[experiment] = harm
                else:
                    alpha_start_times = alpha_flow_start_times_long_flow
                    alpha_start_times = alpha_start_times[:num_alpha_flows]
                    if beta_cca == "prague" or alpha_cca == "prague":
                        harm, conv_time, converged = run_tc_long_flow_experiment(exp_dir, bw, rtt_ms, queue, num_beta_flows, num_alpha_flows, alpha_start_times, beta_cca, alpha_cca)
                    else:
                        harm, conv_time, converged = run_tc_long_flow_experiment(exp_dir, bw, rtt_ms, queue, num_beta_flows, num_alpha_flows, alpha_start_times, beta_cca, alpha_cca, dualpi2=False)
                    if converged:
                        logger.info(f"Experiment {experiment} converged in {conv_time:.2f}s with harm {harm:.4f}")
                    else:
                        logger.warning(f"Experiment {experiment} did not converge, fall back to 60s")
                    all_harm_dict[experiment] = harm

            except Exception as e:
                logger.error(f"Experiment {experiment} failed: {str(e)}")
                all_harm_dict[experiment] = None
            ti.sleep(0.2)

def get_pop_stats(population, gen, all_harm_dict):
    pop_with_harm = []
    for chrom in population:
        if chrom in all_harm_dict:
            pop_with_harm.append((chrom, all_harm_dict[chrom]))
        else:
            pop_with_harm.append((chrom, -1))
    
    if not pop_with_harm:
        return {'generation': gen, 'avg': 0, 'best': 0, 'size_of_set': 0, 'median': 0}
    
    harms = [harm for _, harm in pop_with_harm if harm is not None and harm != -1]
    
    if not harms:
        return {'generation': gen, 'avg': 0, 'best': 0, 'size_of_set': 0, 'median': 0}
    
    mean = sum(harms) / len(harms)
    best = max(harms)
    median = statistics.median(harms)
    size_of_set = len(set(pop_with_harm))
    
    logger.info(f"Gen {gen} - Avg: {mean:.4f}, Best: {best:.4f}, Median: {median:.4f}, Size: {size_of_set}")
    return {'generation': gen, 'avg': mean, 'best': best, 'size_of_set': size_of_set, 'median': median}

def save_pickle(alpha_cca, beta_cca, populations, all_harm_dict):
    populations_file = open(f'{dir_path}/genetic_algorithm_{beta_cca}_{alpha_cca}_populations_{experiment_budget}_budget_{label}.pckl', 'wb')
    all_harm_dict_file = open(f'{dir_path}/genetic_algorithm_{beta_cca}_{alpha_cca}_harm_dict_{experiment_budget}_budget_{label}.pckl', 'wb')
    pickle.dump(populations, populations_file)
    pickle.dump(all_harm_dict, all_harm_dict_file)
    populations_file.close()
    all_harm_dict_file.close()

all_harm_dict = {}
generation_results = []
populations = []
gen = 0
identical_child = 0
new_pop = get_initial_pop(pop_size)
logger.info("Starting tc genetic algorithm with DualPI2")
process = psutil.Process(os.getpid())
initial_mem = process.memory_info().rss / (1024 * 1024)
logger.info(f"Initial memory: {initial_mem:.2f} MB")
total_start_time = ti.time()

while (ti.time() - total_start_time) < time_limit_seconds:
    logger.info(f"Generation: {gen}")
    start_time = ti.time()
    if len(all_harm_dict) >= experiment_budget:
        logger.info("Budget reached")
        break
    
    experiments_to_run = [exp for exp in new_pop if exp not in all_harm_dict.keys()]
    logger.info(f"New experiments: {len(experiments_to_run)}")
    
    if len(all_harm_dict) + len(experiments_to_run) >= experiment_budget:
        logger.info("Budget will be exceeded, breaking")
        break
    
    rerun_experiments_until_complete(all_harm_dict, experiments_to_run, beta_cca, alpha_cca)
    generation_results.append(get_pop_stats(new_pop, gen, all_harm_dict))
    populations.append(new_pop)
    
    mating_pool = selection(new_pop, all_harm_dict)
    random.shuffle(mating_pool)
    offspring = []
    
    for i in range(0, len(mating_pool)-1, 2):
        chrom1, chrom2 = mating_pool[i], mating_pool[i+1]
        children = crossover(chrom1, chrom2)
        child1 = mutate(children[0])
        child2 = mutate(children[1])
        
        if child1 == chrom1 or child1 == chrom2:
            identical_child += 1
        if child2 == chrom1 or child2 == chrom2:
            identical_child += 1
        
        offspring.extend([child1, child2])
    
    new_pop = offspring
    new_pop.extend([chromosome for chromosome in mating_pool])
    
    save_pickle(alpha_cca, beta_cca, populations, all_harm_dict)
    
    elapsed = ti.time() - start_time
    mem_usage = process.memory_info().rss / (1024 * 1024)
    remaining = time_limit_seconds - (ti.time() - total_start_time)
    logger.info(f"Gen {gen} done - Time: {elapsed:.2f}s, Remaining: {remaining/3600:.2f}h, Mem: {mem_usage:.2f}MB")
    gen += 1

logger.info(f"Identical children: {identical_child}")
final_mem = process.memory_info().rss / (1024 * 1024)
total_elapsed = ti.time() - total_start_time
logger.info(f"Total time: {total_elapsed/3600:.2f}h, Final mem: {final_mem:.2f}MB, Increase: {(final_mem - initial_mem):.2f}MB")
logger.info("Genetic algorithm completed")
