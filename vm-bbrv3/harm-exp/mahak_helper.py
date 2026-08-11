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
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
import csv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tc_single_run import run_tc_long_flow_experiment, run_tc_short_flow_experiment
from mahimahi_single_run import run_mahimahi_long_flow_experiment, run_mahimahi_short_flow_experiment
from tc_single_run import calculate_queue_size

ALPHA_FLOW_START_TIMES_LONG_FLOW = [0, 0, 0, 0, 0]

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

def label_generator(harm, metric):
    if metric == 'harm':
        return harm
    else:
        raise NotImplementedError("Metric not implemented yet")

def oracle_query_harm(sample_config, beta_cca, alpha_cca, experiment_type, short_flow_harm_metric, dir_path):
    bw = int(sample_config[0])
    rtt = int(sample_config[1])
    bdp_multiple = float(sample_config[2])
    num_beta_flows = int(sample_config[3])
    num_alpha_flows = int(sample_config[4])
    
    queue = calculate_queue_size(bw, rtt, bdp_multiple)
    
    if experiment_type == "short-flow":
        alpha_start_type = int(sample_config[5])
        exp_dir = f"{dir_path}/experiments/{bw}bw-{rtt}rtt-{queue}q-{beta_cca}-{num_beta_flows}-{alpha_cca}-{num_alpha_flows}-{alpha_start_type}start"
        os.makedirs(exp_dir, exist_ok=True)
        
        harm = run_tc_short_flow_experiment(exp_dir, bw, rtt, queue, num_beta_flows, num_alpha_flows, alpha_start_type, beta_cca, alpha_cca, short_flow_harm_metric, dualpi2=False)
        
        return harm
    else:
        alpha_start_times = ALPHA_FLOW_START_TIMES_LONG_FLOW[:num_alpha_flows]
        exp_dir = f"{dir_path}/experiments/{bw}bw-{rtt}rtt-{queue}q-{beta_cca}-{num_beta_flows}-{alpha_cca}-{num_alpha_flows}"
        os.makedirs(exp_dir, exist_ok=True)
        
        harm, conv_time, converged = run_tc_long_flow_experiment(exp_dir, bw, rtt, queue, num_beta_flows, num_alpha_flows, alpha_start_times, beta_cca, alpha_cca, dualpi2=False)
        
        if converged:
            return harm
        else:
            return 0.0

def run_experiment(Z, max_index, beta_cca, alpha_cca, experiment_type, short_flow_harm_metric, dir_path, metric, labels):
    experiment_config = Z[max_index]
    
    if experiment_type == "short-flow":
        bw, rtt, bdp_ratio, num_beta_flows, num_alpha_flows, alpha_start_type = initialize(experiment_config, experiment_type)
        header_columns = ['bw', 'rtt', 'bdp_ratio', 'num_beta_flows', 'num_alpha_flows', 'alpha_start_type', 'harm']
        row_data = [bw, rtt, bdp_ratio, num_beta_flows, num_alpha_flows, alpha_start_type]
    else:
        bw, rtt, bdp_ratio, num_beta_flows, num_alpha_flows, _ = initialize(experiment_config, experiment_type)
        header_columns = ['bw', 'rtt', 'bdp_ratio', 'num_beta_flows', 'num_alpha_flows', 'harm']
        row_data = [bw, rtt, bdp_ratio, num_beta_flows, num_alpha_flows]
    
    harm = oracle_query_harm(experiment_config, beta_cca, alpha_cca, experiment_type, short_flow_harm_metric, dir_path)
    
    label = label_generator(harm, metric)
    row_data.append(harm)
    
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