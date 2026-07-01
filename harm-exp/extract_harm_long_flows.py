import pickle
import csv
import numpy as np
import sys

tool = sys.argv[1]
target_cc = sys.argv[2]
compete_cc = sys.argv[3]
times = float(sys.argv[4])

alpha_flow_start_times_aggregate = (25, 25, 25, 25, 25)
alpha_flow_start_times_overlap = (15, 20, 25, 30, 35)
alpha_flow_start_times_split = (2, 14, 26, 38, 50)

file_path = f"./results/{target_cc}-{compete_cc}-{times}-long-flow/genetic_algorithm_{target_cc}_{compete_cc}_harm_dict_600_budget_Haochen-text.pckl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

output_csv = f"./results/{target_cc}-{compete_cc}-{times}-long-flow/{tool}_{target_cc}_{compete_cc}.csv"
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['bdp_ratio', 'queue', 'mRTT', 'BW', 'alpha_flows', 'beta_flows', 'harm']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    
    for key, harm_value in data.items():
        if isinstance(key, tuple) and len(key) == 6:
            bw, mRTT, queue, beta_flows, alpha_flows = key[:5]
            
            rtt = mRTT / 1000 
            bdp = bw * rtt
            bdp_packets = int((bdp * 1e6) / (1500 * 8)) 
            bdp_ratio = queue / bdp_packets

            writer.writerow({
                'bdp_ratio': float(bdp_ratio),
                'queue': int(queue),
                'mRTT': int(mRTT),
                'BW': int(bw),
                'alpha_flows': int(alpha_flows),
                'beta_flows': int(beta_flows),
                'harm': float(harm_value)
            })
        else:
            print(f"skip: {key}")

print(f"CSV file successfully generated: {output_csv}")