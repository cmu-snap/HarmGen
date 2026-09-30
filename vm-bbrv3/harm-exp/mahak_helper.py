import numpy as np
import os
import time
import sys
import subprocess
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import warnings
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
import csv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tc_single_run import ExperimentDataError, run_tc_long_flow_experiment, run_tc_short_flow_experiment
from mahimahi_single_run import run_mahimahi_long_flow_experiment, run_mahimahi_short_flow_experiment
from tc_single_run import calculate_queue_size

ALPHA_FLOW_START_TIMES_LONG_FLOW = [0, 0, 0, 0, 0]

logger = logging.getLogger('MahakMonitor')

def initialize(exp_conf, experiment_type):
    bw = int(exp_conf[0])
    rtt = int(exp_conf[1])
    bdp_ratio = float(exp_conf[2])
    num_beta_flows = int(exp_conf[3])
    num_alpha_flows = int(exp_conf[4])
    
    if experiment_type == "short-flow":
        alpha_start_type = int(exp_conf[5])
        return bw, rtt, bdp_ratio, num_beta_flows, num_alpha_flows, alpha_start_type
    else:
        return bw, rtt, bdp_ratio, num_beta_flows, num_alpha_flows, None

def append_to_csv(file_path, header, row_data):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row_data)

def calculate_distances(points, reference_point):
    return np.linalg.norm(points - reference_point, axis=1)

# Invalid samples keep their slot as NaN so that labels stay aligned with S;
# drop_invalid removes them before any model is fitted.
def label_generator(harm, metric):
    if metric == 'harm':
        return np.nan if harm is None else harm
    else:
        raise NotImplementedError("Metric not implemented yet")

def drop_invalid(S, labels):
    S = np.asarray(S)
    labels = np.asarray(labels, dtype=float)
    valid = ~np.isnan(labels)
    if not valid.any():
        raise ValueError("No valid samples to fit: every experiment produced unusable data")
    return S[valid], labels[valid]

# Same contract as mahak.py: (harm, None) for a valid measurement, or (None, reason)
# when the experiment produced unusable data twice. A failure is never harm 0.0.
def oracle_query_harm(sample_config, beta_cca, alpha_cca, experiment_type, short_flow_harm_metric, dir_path):
    bw = int(sample_config[0])
    rtt = int(sample_config[1])
    bdp_multiple = float(sample_config[2])
    num_beta_flows = int(sample_config[3])
    num_alpha_flows = int(sample_config[4])

    queue = calculate_queue_size(bw, rtt, bdp_multiple)
    use_dualpi2 = (beta_cca == "prague" or alpha_cca == "prague")

    if experiment_type == "short-flow":
        alpha_start_type = int(sample_config[5])
        exp_dir = f"{dir_path}/experiments/{bw}bw-{rtt}rtt-{queue}q-{beta_cca}-{num_beta_flows}-{alpha_cca}-{num_alpha_flows}-{alpha_start_type}start"
        os.makedirs(exp_dir, exist_ok=True)

        def run():
            return run_tc_short_flow_experiment(exp_dir, bw, rtt, queue, num_beta_flows, num_alpha_flows, alpha_start_type, beta_cca, alpha_cca, short_flow_harm_metric, dualpi2=use_dualpi2)
    else:
        alpha_start_times = ALPHA_FLOW_START_TIMES_LONG_FLOW[:num_alpha_flows]
        exp_dir = f"{dir_path}/experiments/{bw}bw-{rtt}rtt-{queue}q-{beta_cca}-{num_beta_flows}-{alpha_cca}-{num_alpha_flows}"
        os.makedirs(exp_dir, exist_ok=True)

        # A run that did not converge still reports harm over the fallback window;
        # mahak.py and genetic-algorithm.py keep that value too.
        def run():
            harm, conv_time, converged = run_tc_long_flow_experiment(exp_dir, bw, rtt, queue, num_beta_flows, num_alpha_flows, alpha_start_times, beta_cca, alpha_cca, dualpi2=use_dualpi2)
            return harm

    try:
        return run(), None
    except ExperimentDataError as exc:
        logger.warning(f"Experiment produced unusable data ({exc}); retrying once")
    try:
        return run(), None
    except ExperimentDataError as exc2:
        logger.error(f"Experiment still unusable ({exc2}); marking sample invalid")
        return None, str(exc2)

def run_experiment(Z, max_index, beta_cca, alpha_cca, experiment_type, short_flow_harm_metric, dir_path, metric, labels):
    experiment_config = Z[max_index]
    
    if experiment_type == "short-flow":
        bw, rtt, bdp_ratio, num_beta_flows, num_alpha_flows, alpha_start_type = initialize(experiment_config, experiment_type)
        header_columns = ['bw', 'rtt', 'bdp_ratio', 'num_beta_flows', 'num_alpha_flows', 'alpha_start_type', 'harm', 'status', 'error']
        row_data = [bw, rtt, bdp_ratio, num_beta_flows, num_alpha_flows, alpha_start_type]
    else:
        bw, rtt, bdp_ratio, num_beta_flows, num_alpha_flows, _ = initialize(experiment_config, experiment_type)
        header_columns = ['bw', 'rtt', 'bdp_ratio', 'num_beta_flows', 'num_alpha_flows', 'harm', 'status', 'error']
        row_data = [bw, rtt, bdp_ratio, num_beta_flows, num_alpha_flows]

    harm, error = oracle_query_harm(experiment_config, beta_cca, alpha_cca, experiment_type, short_flow_harm_metric, dir_path)

    label = label_generator(harm, metric)
    if harm is None:
        row_data += [np.nan, 'invalid', error]
    else:
        row_data += [harm, 'valid', '']
    
    selected_samples_file = f'{dir_path}/selected_samples.csv'
    append_to_csv(selected_samples_file, header_columns, row_data)
    
    updated_labels = np.append(labels, label)
    
    return updated_labels

def update_configuration_arrays(Z, normalized_Z, max_index, S, normalized_S):
    S.append(Z[max_index])
    normalized_S.append(normalized_Z[max_index])
    
    Z = np.delete(Z, max_index, axis=0)
    normalized_Z = np.delete(normalized_Z, max_index, axis=0)
    
    return Z, normalized_Z, S, normalized_S

def train_and_save_models(budget_lst, gsx_budget, normalized_S, labels, beta_cca, alpha_cca, save_path, seed, metric, alg):
    for indi_budget in budget_lst:
        final_budget = indi_budget + gsx_budget
        
        if isinstance(normalized_S, list):
            tmp_S = np.vstack(normalized_S[:final_budget])
        else:
            tmp_S = np.array(normalized_S[:final_budget])
        
        temp_labels = np.array(labels[:final_budget])
        tmp_S, temp_labels = drop_invalid(tmp_S, temp_labels)

        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=seed)
        rf_regressor.fit(tmp_S, temp_labels)
        
        model_filename = f'{save_path}/{beta_cca}_{alpha_cca}_{alg}_{metric}_{indi_budget}_{seed}.pkl'
        dump(rf_regressor, model_filename)

def train_and_save_unif_models(budget_lst, gsx_budget, normalized_S, labels, beta_cca, alpha_cca, save_path, seed, metric, alg):
    if isinstance(normalized_S, list):
        normalized_S_array = np.vstack(normalized_S)
    else:
        normalized_S_array = np.array(normalized_S)
    
    labels_array = np.array(labels)
    
    for indi_budget in budget_lst:
        final_budget = indi_budget + gsx_budget
        
        step_size = max(1, len(normalized_S_array) // final_budget)
        samples = np.arange(0, len(normalized_S_array), step_size)
        samples = samples[:final_budget]
        
        tmp_S = normalized_S_array[samples]
        temp_labels = labels_array[samples]
        tmp_S, temp_labels = drop_invalid(tmp_S, temp_labels)

        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=seed)
        rf_regressor.fit(tmp_S, temp_labels)
        
        model_filename = f'{save_path}/{beta_cca}_{alpha_cca}_{alg}_{metric}_{indi_budget}_{seed}.pkl'
        dump(rf_regressor, model_filename)

def generate_final_predictions(normalized_S, labels, original_df, beta_cca, alpha_cca, dir_path, seed, metric, alg):
    if isinstance(normalized_S, list):
        normalized_S_array = np.vstack(normalized_S)
    else:
        normalized_S_array = np.array(normalized_S)
    
    labels_array = np.array(labels)
    normalized_S_array, labels_array = drop_invalid(normalized_S_array, labels_array)

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=seed)
    rf_regressor.fit(normalized_S_array, labels_array)
    
    scaler = MinMaxScaler(feature_range=(1, 2))
    full_space_normalized = scaler.fit_transform(original_df.values)
    
    predictions = rf_regressor.predict(full_space_normalized)
    
    prediction_df = original_df.copy()
    prediction_df['harm'] = predictions
    
    final_prediction_file = f'{dir_path}/final_predictions.csv'
    prediction_df.to_csv(final_prediction_file, index=False)
    
    return final_prediction_file