import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

machine = "HarmGen_VM1"
tool = "mahak"
beta_cca = "cubic"
alpha_cca = "bbr1"
time = 240.0
flow_type = "long-flow"
suffix = "_jfi"

csv_file_path = f"./{machine}/harm-exp/mahak_results{suffix}/{beta_cca}-{alpha_cca}-{time}-{flow_type}/final_predictions.csv"

df = pd.read_csv(csv_file_path)

plt.rcParams.update({'font.size': 30})
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30

df = df[
    (df['bw'].between(25, 200)) &
    (df['rtt'].between(10, 100)) &
    (df['bdp_ratio'].between(0.25, 5)) &
    (df['num_alpha_flows'].between(1, 5)) &
    (df['num_beta_flows'].between(1, 5))
]

int_cols = ['bw', 'rtt', 'num_alpha_flows', 'num_beta_flows']
df[int_cols] = df[int_cols].astype(int)

y_vars = ['rtt', 'bdp_ratio', 'num_alpha_flows', 'num_beta_flows']
labels = {
    'bw': 'Bandwidth (Mbps)',
    'rtt': 'RTT (ms)',
    'bdp_ratio': 'Queue Size (x BDP)',
    'num_alpha_flows': 'Number of Alpha Flows',
    'num_beta_flows': 'Number of Beta Flows'
}

# fig, axes = plt.subplots(2, 2, figsize=(36, 28))
# axes = axes.flatten()

# for i, y_var in enumerate(y_vars):
#     pivot_data = df.pivot_table(index=y_var, columns='bw', values='harm', aggfunc='max')
#     print(pivot_data)
#     pivot_data = pivot_data.sort_index(ascending=False)
    
#     sns.heatmap(
#         pivot_data, 
#         ax=axes[i], 
#         center=0,
#         vmin=0, 
#         vmax=1,
#         cmap='RdBu_r', 
#         annot=False, 
#         cbar=True,
#         cbar_kws={'label': 'Harm'}
#     )
    
#     axes[i].set_xlabel(labels['bw'], labelpad=20)
#     axes[i].set_ylabel(labels[y_var], labelpad=20)
    
#     cbar = axes[i].collections[0].colorbar
#     cbar.ax.tick_params(labelsize=40)
#     cbar.set_label('Harm', size=45, labelpad=20)

# plt.tight_layout()
# plt.savefig(f'heatmaps_{tool}_{beta_cca}_{alpha_cca}_{time}_{flow_type}.png')

# save individual heatmaps in pdf
for y_var in y_vars:
    plt.figure(figsize=(12, 10))
    pivot_data = df.pivot_table(index=y_var, columns='bw', values='harm', aggfunc='max')
    pivot_data = pivot_data.sort_index(ascending=False)
    
    annot_matrix = pivot_data.round(2).astype(str).values

    sns.heatmap(
        pivot_data,
        annot=annot_matrix, 
        fmt='', 
        center=0,
        vmin=0, 
        vmax=1,
        cmap='RdBu_r', 
        cbar=True,
        cbar_kws={'label': 'Harm'},
        annot_kws={'size': 25, 'weight': 'bold'}
    )
    
    plt.xlabel(labels['bw'], labelpad=20, size=35)
    plt.ylabel(labels[y_var], labelpad=20, size=35)
    
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label('Harm', size=35, labelpad=20)

    plt.tight_layout()
    plt.savefig(f'{tool}-{y_var}-heatmap-{beta_cca}-{alpha_cca}-{flow_type}{suffix}.pdf')
    plt.close()