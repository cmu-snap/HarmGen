import argparse
import numpy as np
from modAL.models import ActiveLearner
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
import os
import pandas as pd
import itertools
import sys
from modAL.uncertainty import entropy_sampling
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import time
import shutil
import logging
import psutil
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tc_single_run import run_tc_long_flow_experiment, run_tc_short_flow_experiment
from mahimahi_single_run import run_mahimahi_long_flow_experiment, run_mahimahi_short_flow_experiment
from tc_single_run import calculate_queue_size

ALPHA_FLOW_START_TIMES_LONG_FLOW = [0, 0, 0, 0, 0]

parser = argparse.ArgumentParser(description="Mahak's Parser")

parser.add_argument('-cc', type=str, help='Target Congestion Control Scheme')
parser.add_argument('-compete_cc', type=str, help='CCA that competing flows uses')
parser.add_argument('-budget', type=int, help='Desired Computation Budget', default=300)
parser.add_argument('-time_limit', type=float, help='Desired Time Limit in Hours', default=12.0)
parser.add_argument('-experiment_type', type=str, help='Experiment type: long-flow or short-flow', default='long-flow')
parser.add_argument('-short_flow_harm_metric', type=str, help='Short flow harm metric: download_bytes or recovery_time', default='download_bytes')

parser.add_argument('-bdp_ratio', type=str, help='Min, Max and Search Step for BDP Ratio. Ex: 0.25,5,0.25', default='0.25,5,0.25')
parser.add_argument('-rtt', type=str, help='Min, Max and Search Step for RTT Space. Ex: 10,320,10', default='10,320,10')
parser.add_argument('-bw', type=str, help='Min, Max and Search Step for BW Space. Ex: 25,400,25', default='25,400,25')
parser.add_argument('-alpha_flows', type=str, help='Min, Max and Search Step for Number of Alpha flows. Ex: 1,5,1', default='1,5,1')
parser.add_argument('-beta_flows', type=str, help='Min, Max and Search Step for Number of Beta flows. Ex: 1,5,1', default='1,5,1')
parser.add_argument('-alpha_start', type=str, help='Min, Max and Search Step for Alpha start type. Ex: 1,3,1', default='1,1,1')

parser.add_argument('-convergence', type=bool, help='Whether to use convergence algorithm in long-flow experiments', default=True)

args = parser.parse_args()
beta_cca = args.cc
alpha_cca = args.compete_cc
time_limit_hours = args.time_limit 
budget = args.budget
experiment_type = args.experiment_type
short_flow_harm_metric = args.short_flow_harm_metric

convergence = args.convergence

bdp_ratio_list = args.bdp_ratio.split(',')
min_bdp_ratio = float(bdp_ratio_list[0])
max_bdp_ratio = float(bdp_ratio_list[1])
bdp_ratio_step = float(bdp_ratio_list[2])

rtt_list = args.rtt.split(',')
min_rtt = int(rtt_list[0])
max_rtt = int(rtt_list[1])
rtt_step = int(rtt_list[2])

bw_list = args.bw.split(',')
min_bw = int(bw_list[0])
max_bw = int(bw_list[1])
bw_step = int(bw_list[2])

alpha_flows_list = args.alpha_flows.split(',')
min_alpha_flows = int(alpha_flows_list[0])
max_alpha_flows = int(alpha_flows_list[1])
alpha_flows_step = int(alpha_flows_list[2])

beta_flows_list = args.beta_flows.split(',')
min_beta_flows = int(beta_flows_list[0])
max_beta_flows = int(beta_flows_list[1])
beta_flows_step = int(beta_flows_list[2])

alpha_start_list = args.alpha_start.split(',')
min_alpha_start = int(alpha_start_list[0])
max_alpha_start = int(alpha_start_list[1])
alpha_start_step = int(alpha_start_list[2])

dir_path = f"./mahak_results/{beta_cca}-{alpha_cca}-{time_limit_hours}-{experiment_type}"
os.makedirs(dir_path, exist_ok=True)

log_filename = f'{dir_path}/mahak_{beta_cca}_{alpha_cca}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('MahakMonitor')

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(console_handler)

bdp_ratio_space = list(np.arange(min_bdp_ratio, max_bdp_ratio + 0.1, bdp_ratio_step))
rtt_space = list(np.arange(min_rtt, max_rtt + 1, rtt_step))
bw_space = list(np.arange(min_bw, max_bw + 1, bw_step))
alpha_flows_space = list(np.arange(min_alpha_flows, max_alpha_flows + 1, alpha_flows_step))
beta_flows_space = list(np.arange(min_beta_flows, max_beta_flows + 1, beta_flows_step))

if experiment_type == "short-flow":
    alpha_start_space = list(np.arange(min_alpha_start, max_alpha_start + alpha_start_step, alpha_start_step))
    beta_flows_space = list(np.arange(1, 1 + 1, 1))  # Fix beta flows to 1 for short-flow experiments
    F = [bw_space, rtt_space, bdp_ratio_space, beta_flows_space, alpha_flows_space, alpha_start_space]
    column_names = ["bw", "rtt", "bdp_ratio", "num_beta_flows", "num_alpha_flows", "alpha_start"]
else:
    F = [bw_space, rtt_space, bdp_ratio_space, beta_flows_space, alpha_flows_space]
    column_names = ["bw", "rtt", "bdp_ratio", "num_beta_flows", "num_alpha_flows"]

logger.info(f"Starting Mahak for {beta_cca} vs {alpha_cca} with experiment type {experiment_type}")
logger.info(f"Budget: {budget}, Time limit: {time_limit_hours} hours")

total_points = np.prod([len(space) for space in F])
logger.info(f"Total search space size: {total_points:,}")
logger.info(f"Search space dimensions: BW {len(bw_space)}, RTT {len(rtt_space)}, "
            f"BDP Ratio {len(bdp_ratio_space)}, Beta Flows {len(beta_flows_space)}, "
            f"Alpha Flows {len(alpha_flows_space)}")

process = psutil.Process(os.getpid())
initial_mem = process.memory_info().rss / (1024 * 1024)
logger.info(f"Initial memory usage: {initial_mem:.2f} MB")

data = []
for element in itertools.product(*F):
    data.append(element)

TrainData = np.array(data)
TrainData = pd.DataFrame(TrainData, columns=column_names)

logger.info(f"TrainData shape: {TrainData.shape}")
logger.info(f"TrainData memory: {TrainData.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

scaler = MinMaxScaler()
T_data = scaler.fit_transform(TrainData)
pca = PCA()
D_data = pca.fit_transform(T_data)

n_initial = 1
initial_idx = np.random.choice(range(len(D_data)), size=n_initial)

def oracle_query_harm(sample_row, beta_cca, alpha_cca, experiment_type, short_flow_harm_metric):
    bw = int(sample_row[0])
    rtt = int(sample_row[1])
    bdp_multiple = float(sample_row[2])
    num_beta_flows = int(sample_row[3])
    num_alpha_flows = int(sample_row[4])
    
    queue = calculate_queue_size(bw, rtt, bdp_multiple)
    
    if experiment_type == "short-flow":
        alpha_start_type = int(sample_row[5])
        exp_dir = f"{dir_path}/experiments/{bw}bw-{rtt}rtt-{queue}q-{beta_cca}-{num_beta_flows}-{alpha_cca}-{num_alpha_flows}-{alpha_start_type}start"
        os.makedirs(exp_dir, exist_ok=True)
        
        # if beta_cca == "prague" or alpha_cca == "prague":
        harm = run_tc_short_flow_experiment(exp_dir, bw, rtt, queue, num_beta_flows, num_alpha_flows, alpha_start_type, beta_cca, alpha_cca, short_flow_harm_metric, dualpi2=False)
        # else:
            # harm = run_mahimahi_short_flow_experiment(exp_dir, bw, rtt, queue, num_beta_flows, num_alpha_flows, alpha_start_type, beta_cca, alpha_cca, short_flow_harm_metric)
        
        logger.info(f"Short flow experiment completed: BW={bw}, RTT={rtt}, Queue={queue}, Harm={harm:.4f}")
        return np.array([harm])
    else:
        alpha_start_times = ALPHA_FLOW_START_TIMES_LONG_FLOW[:num_alpha_flows]
        exp_dir = f"{dir_path}/experiments/{bw}bw-{rtt}rtt-{queue}q-{beta_cca}-{num_beta_flows}-{alpha_cca}-{num_alpha_flows}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # if beta_cca == "prague" or alpha_cca == "prague":
        harm, conv_time, converged = run_tc_long_flow_experiment(exp_dir, bw, rtt, queue, num_beta_flows, num_alpha_flows, alpha_start_times, beta_cca, alpha_cca, dualpi2=False, convergence=convergence)
        # else:
            # harm, conv_time, converged = run_mahimahi_long_flow_experiment(exp_dir, bw, rtt, queue, num_beta_flows, num_alpha_flows, alpha_start_times, beta_cca, alpha_cca)
        
        if converged:
            logger.info(f"Long flow experiment converged at {conv_time:.2f}s: BW={bw}, RTT={rtt}, Queue={queue}, Harm={harm:.4f}")
            return np.array([harm])
        else:
            logger.warning(f"Long flow experiment did not converge: BW={bw}, RTT={rtt}, Queue={queue}, Harm={harm:.4f}")
            return np.array([harm])

X_training = D_data[initial_idx,:]
y_training = oracle_query_harm(np.array(TrainData.iloc[initial_idx])[0], beta_cca, alpha_cca, experiment_type, short_flow_harm_metric)

selected_samples = []

initial_sample = TrainData.iloc[initial_idx].copy()
initial_sample = initial_sample.reset_index(drop=True)
initial_sample['harm'] = y_training[0]
initial_sample['iteration'] = 0
selected_samples.append(initial_sample)

time.sleep(1)

kernel = RBF(length_scale=0.1, length_scale_bounds=(1e-60, 1e5))

def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    return np.argmax(std)

def GP_regression_result(regressor, X):
    mean = regressor.predict(X)
    return mean

regressor = ActiveLearner(
    estimator=GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True, random_state=100),
    query_strategy=GP_regression_std,
    X_training=X_training, 
    y_training=y_training
)

logger.info(f"ActiveLearner initialized, memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")

start_time_total = time.time()
total_time_used = 0.0
iteration_count = 0

prediction_percentages = [0.01, 0.1, 0.5, 1, 2, 3, 4]
predictions_saved = set()

while total_time_used < float(time_limit_hours) * 3600 and iteration_count < budget:
    iteration_start = time.time()
    iteration_count += 1

    query_idx, query_instance = regressor.query(D_data)
    new_data_original = np.array(TrainData.iloc[query_idx])
    new_data_transformed = D_data[query_idx,:].reshape(1, -1)
    new_label = oracle_query_harm(new_data_original, beta_cca, alpha_cca, experiment_type, short_flow_harm_metric)
    
    regressor.teach(new_data_transformed, new_label)

    current_sample = TrainData.iloc[[query_idx]].copy()
    current_sample['harm'] = new_label
    current_sample['iteration'] = iteration_count
    selected_samples.append(current_sample)

    iteration_end = time.time()
    iteration_duration = iteration_end - iteration_start
    total_time_used = time.time() - start_time_total
    mem_usage = process.memory_info().rss / (1024 * 1024)
    
    logger.info(f"Iteration {iteration_count}/{budget} completed")
    logger.info(f"  Selected point: {new_data_original}")
    logger.info(f"  Harm value: {new_label[0]:.4f}")
    logger.info(f"  Time elapsed: {iteration_duration:.2f} seconds")
    logger.info(f"  Total time used: {total_time_used/3600:.2f} hours (Limit: {time_limit_hours} hours)")
    logger.info(f"  Memory usage: {mem_usage:.2f} MB")
    
    current_percentage = (iteration_count / total_points) * 100
    for pct in prediction_percentages:
        if current_percentage >= pct and pct not in predictions_saved:
            logger.info(f"Reached {pct}% of search space, saving predictions...")
            
            final_result = GP_regression_result(regressor, D_data)
            prediction_df = TrainData.copy()
            prediction_df['harm'] = final_result
            
            prediction_filename = f'{dir_path}/predictions_at_{pct}pct.csv'
            prediction_df.to_csv(prediction_filename, index=False)
            
            predictions_saved.add(pct)
            logger.info(f"Predictions saved to {prediction_filename}")
    
    logger.info("-" * 50)

final_result = GP_regression_result(regressor, D_data)

TrainData['harm'] = final_result
final_prediction_file = f'{dir_path}/final_predictions.csv'
TrainData.to_csv(final_prediction_file, index=False)

selected_samples_df = pd.concat(selected_samples, ignore_index=True)
selection_filename = f'{dir_path}/selected_samples.csv'
selected_samples_df.to_csv(selection_filename, index=False)

final_mem = process.memory_info().rss / (1024 * 1024)
logger.info(f"Total iterations completed: {iteration_count}")
logger.info(f"Total time used: {total_time_used/3600:.2f} hours (Limit: {time_limit_hours} hours)")
logger.info(f"Final memory usage: {final_mem:.2f} MB")
logger.info(f"Total memory increase: {(final_mem - initial_mem):.2f} MB")
logger.info(f"Final predictions saved to {final_prediction_file}")
logger.info(f"Selected samples saved to {selection_filename}")
logger.info("Mahak completed successfully")