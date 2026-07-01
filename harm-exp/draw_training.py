import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_and_prep_data(filepath):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    df = df[
        (df['bw'].between(25, 200)) &
        (df['rtt'].between(10, 100)) &
        (df['bdp_ratio'].between(0.25, 5)) &
        (df['num_alpha_flows'].between(1, 5)) &
        (df['num_beta_flows'].between(1, 5))
    ]
    int_cols = ['bw', 'rtt', 'num_alpha_flows', 'num_beta_flows']
    df[int_cols] = df[int_cols].astype(int)
    return df

def get_plot_params(flow_type):
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['axes.labelsize'] = 35
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['pdf.fonttype'] = 42
    
    labels = {
        'bw': 'Bandwidth (Mbps)',
        'rtt': 'RTT (ms)',
        'bdp_ratio': 'Queue Size (x BDP)',
        'num_alpha_flows': 'Number of Alpha Flows',
        'num_beta_flows': 'Number of Beta Flows'
    }
    y_vars = ['rtt', 'bdp_ratio', 'num_alpha_flows', 'num_beta_flows']
    
    if flow_type == 'short-flow':
        param_ranges = {
            'bw': list(range(25, 201, 25)),
            'rtt': list(range(10, 101, 10)),
            'bdp_ratio': [round(x, 2) for x in np.arange(0.25, 5.01, 0.25)],
            'num_alpha_flows': list(range(1, 6)),
            'num_beta_flows': [1]
        }
    else:
        param_ranges = {
            'bw': list(range(25, 201, 35)),
            'rtt': list(range(10, 101, 20)),
            'bdp_ratio': [round(x, 2) for x in np.arange(0.25, 5.01, 0.5)],
            'num_alpha_flows': list(range(1, 6)),
            'num_beta_flows': list(range(1, 6))
        }
    return labels, y_vars, param_ranges

def plot_individual_sampling(df, labels, y_vars, param_ranges, base_filename, draw_arrows=True):
    df_sorted = df.sort_values('iteration').reset_index(drop=True)

    for y_var in y_vars:
        plt.figure(figsize=(16, 12))
        x_vals = param_ranges['bw']
        y_vals = param_ranges[y_var]
        
        counts = df_sorted.groupby(['bw', y_var]).size().reset_index(name='count')
        
        unique_blocks = []
        block_to_ordinal = {}
        for _, row in df_sorted.iterrows():
            block = (row['bw'], row[y_var])
            if block not in block_to_ordinal:
                unique_blocks.append(block)
                block_to_ordinal[block] = len(unique_blocks)

        grid_count = pd.DataFrame(0, index=y_vals, columns=x_vals)
        grid_ordinal = pd.DataFrame(-1, index=y_vals, columns=x_vals)
        
        for _, row in counts.iterrows():
            if row[y_var] in grid_count.index and row['bw'] in grid_count.columns:
                grid_count.loc[row[y_var], row['bw']] = int(row['count'])
        
        for (bw, yv), ord_idx in block_to_ordinal.items():
            if yv in grid_ordinal.index and bw in grid_ordinal.columns:
                grid_ordinal.loc[yv, bw] = ord_idx
        
        grid_count = grid_count.sort_index(ascending=False)
        grid_ordinal = grid_ordinal.sort_index(ascending=False)
        
        annot_matrix = []
        for r in range(len(grid_count)):
            row_annots = []
            for c in range(len(grid_count.columns)):
                cnt = grid_count.iloc[r, c]
                ord_val = grid_ordinal.iloc[r, c]
                if cnt > 0:
                    row_annots.append(f"#{ord_val}\n({cnt})")
                else:
                    row_annots.append("")
            annot_matrix.append(row_annots)

        ax = sns.heatmap(
            grid_count,
            cmap="Blues",
            annot=np.array(annot_matrix),
            fmt="",
            # vmin=0,
            # vmax=30,
            cbar=True,
            cbar_kws={'label': 'Sampled Times'},
            linewidths=1,
            linecolor='black',
            annot_kws={'size': 30, 'weight': 'bold'}
        )
        
        if draw_arrows:
            y_map = {val: idx for idx, val in enumerate(grid_count.index)}
            x_map = {val: idx for idx, val in enumerate(grid_count.columns)}
            
            prev_pos = None
            for _, row in df_sorted.head(50).iterrows():
                if row['bw'] in x_map and row[y_var] in y_map:
                    curr_pos = (x_map[row['bw']] + 0.5, y_map[row[y_var]] + 0.5)
                    if prev_pos is not None:
                        plt.gca().annotate(
                            "",
                            xy=curr_pos,
                            xytext=prev_pos,
                            arrowprops=dict(
                                arrowstyle="->",
                                color="lightgray",
                                linestyle="--",
                                lw=2,
                                alpha=0.6,
                                mutation_scale=20
                            )
                        )
                    prev_pos = curr_pos

        plt.xlabel(labels['bw'], labelpad=15)
        plt.ylabel(labels[y_var], labelpad=15)
        
        cbar = ax.collections[0].colorbar
        cbar.set_label('Sampled Times', size=30, labelpad=15)
        cbar.ax.tick_params(labelsize=25)

        plt.tight_layout()
        output_path = f"{base_filename}_{y_var}.pdf"
        print(f"Saving plot to {output_path}...")
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    machine = "HarmGen_VM1"
    tool = "mahak"
    beta_cca = "cubic"
    alpha_cca = "bbr1"
    time = 240.0
    flow_type = "long-flow"
    
    draw_arrows_flag = False

    csv_file_path = f"./{machine}/harm-exp/mahak_results_with_conv/{beta_cca}-{alpha_cca}-{time}-{flow_type}/selected_samples.csv"
    
    data = load_and_prep_data(csv_file_path)
    if not data.empty:
        print(f"Loaded data with {len(data)} rows.")
        plot_labels, target_vars, ranges = get_plot_params(flow_type)
        base_name = f'sampling_{tool}_{beta_cca}_{alpha_cca}_{time}_{flow_type}_with_conv'
        plot_individual_sampling(data, plot_labels, target_vars, ranges, base_name, draw_arrows=draw_arrows_flag)