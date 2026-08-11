import pickle
import csv
import glob
import os
import numpy as np
import sys

tool = sys.argv[1]
target_cc = sys.argv[2]
compete_cc = sys.argv[3]
times = float(sys.argv[4])

alpha_flow_start_times_aggregate = (25, 25, 25, 25, 25)
alpha_flow_start_times_overlap = (15, 20, 25, 30, 35)
alpha_flow_start_times_split = (2, 14, 26, 38, 50)

result_dir = f"./results/{target_cc}-{compete_cc}-{times}-long-flow"
budget = os.environ.get("HARMGEN_BUDGET")
label = os.environ.get("HARMGEN_LABEL")

if budget and label:
    file_path = f"{result_dir}/genetic_algorithm_{target_cc}_{compete_cc}_harm_dict_{budget}_budget_{label}.pckl"
else:
    pattern = f"{result_dir}/genetic_algorithm_{target_cc}_{compete_cc}_harm_dict_*_budget_*.pckl"
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        sys.exit(f"No harm-dict pickle found matching {pattern}\n"
                 f"Set HARMGEN_BUDGET and HARMGEN_LABEL, or run the GA first.")
    if len(matches) > 1:
        print(f"warning: {len(matches)} pickles matched, using newest: {matches[-1]}")
    file_path = matches[-1]

with open(file_path, "rb") as f:
    data = pickle.load(f)

output_csv = f"./results/{target_cc}-{compete_cc}-{times}-long-flow/{tool}_{target_cc}_{compete_cc}.csv"
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['bdp_ratio', 'queue', 'mRTT', 'BW', 'alpha_flows', 'beta_flows', 'harm']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    
    skipped_none = 0

    for key, harm_value in data.items():
        if harm_value is None:
            skipped_none += 1
        elif isinstance(key, tuple) and len(key) == 6:
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

if skipped_none:
    print(f"skipped {skipped_none} experiment(s) with unusable data (harm was None)")

print(f"CSV file successfully generated: {output_csv}")