import pickle
import csv
import sys

tool = sys.argv[1]
target_cc = sys.argv[2]
compete_cc = sys.argv[3]
times = float(sys.argv[4])
short_flow_harm_metric = sys.argv[5]

alpha_flow_start_times_aggregate = (25, 25, 25, 25, 25)
alpha_flow_start_times_overlap = (15, 20, 25, 30, 35)
alpha_flow_start_times_split = (2, 14, 26, 38, 50)

file_path = f"./results/{target_cc}-{compete_cc}-{times}-short-flow/genetic_algorithm_{target_cc}_{compete_cc}_harm_dict_600_budget_Haochen-text.pckl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

output_csv = f"./results/{target_cc}-{compete_cc}-{times}-short-flow/{tool}_{target_cc}_{compete_cc}.csv"
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['bdp_ratio', 'queue', 'mRTT', 'BW', 'alpha_flows', 'beta_flows', 'start_time_1', 'start_time_2', 'start_time_3', 'start_time_4', 'start_time_5', 'start_type', 'harm']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    
    for key, harm_value in data.items():
        if isinstance(key, tuple) and len(key) == 6:
            bw, mRTT, queue, beta_flows, alpha_flows, alpha_start_type = key[:6]
            
            rtt = mRTT / 1000 
            bdp = bw * rtt
            bdp_packets = int((bdp * 1e6) / (1500 * 8)) 
            bdp_ratio = queue / bdp_packets

            if alpha_start_type == 1:
                start_time_1, start_time_2, start_time_3, start_time_4, start_time_5 = alpha_flow_start_times_aggregate
            elif alpha_start_type == 2:
                start_time_1, start_time_2, start_time_3, start_time_4, start_time_5 = alpha_flow_start_times_overlap
            elif alpha_start_type == 3:
                start_time_1, start_time_2, start_time_3, start_time_4, start_time_5 = alpha_flow_start_times_split

            writer.writerow({
                'bdp_ratio': float(bdp_ratio),
                'queue': int(queue),
                'mRTT': int(mRTT),
                'BW': int(bw),
                'alpha_flows': int(alpha_flows),
                'beta_flows': int(beta_flows),
                'start_time_1': int(start_time_1),
                'start_time_2': int(start_time_2),
                'start_time_3': int(start_time_3),
                'start_time_4': int(start_time_4),
                'start_time_5': int(start_time_5),
                'start_type': int(alpha_start_type),
                'harm': float(harm_value)
            })
        else:
            print(f"skip: {key}")

print(f"CSV file successfully generated: {output_csv}")