import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

tool = sys.argv[1]
target_cc = sys.argv[2]
compete_cc = sys.argv[3]
top_num = int(sys.argv[4])
times = float(sys.argv[5])

df = pd.read_csv(f'results/{target_cc}-{compete_cc}-{times}-long-flow/{tool}_{target_cc}_{compete_cc}.csv')
top = df.sort_values(by='harm', ascending=False).head(top_num).copy()

# change the order of top
top = top.sort_values(by='harm', ascending=True)

plt.figure(figsize=(12, 8))
ax = plt.gca()

min_bubble = 50  
max_bubble = 500  
bdp_min = top['bdp_ratio'].min()
bdp_max = top['bdp_ratio'].max()
bubble_scale = (max_bubble - min_bubble) / (bdp_max - bdp_min)
top['bubble_size'] = min_bubble + (top['bdp_ratio'] - bdp_min) * bubble_scale

harm_min = -0.5
harm_max = 1

scatter = ax.scatter(
    x=top['BW'],
    y=top['mRTT'],
    s=top['bubble_size'],
    c=top['harm'],
    cmap='viridis_r',
    vmin=harm_min,
    vmax=harm_max,
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)

cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
cbar.set_label('Harm Value', fontsize=16)

ax.set_xlabel('Bandwidth (Mbps)', fontsize=16)
ax.set_ylabel('RTT (ms)', fontsize=16)
ax.set_xlim(25, 401)
ax.set_ylim(10, 321)
ax.tick_params(axis='both', which='major', labelsize=14)

ax.set_title(f'Top {top_num} Harmful Network Environments in {target_cc} vs {compete_cc}', fontsize=20)

ax.grid(True, linestyle='--', alpha=0.3)

max_harm = top.iloc[-1]
ax.annotate(f"Highest Harm: {max_harm['harm']:.3f}\nBDP Ratio: {max_harm['bdp_ratio']:.3f}", 
            (max_harm['BW'], max_harm['mRTT']),
            xytext=(-120, 50),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->", color='red'),
            fontsize=12)

ax.text(0.80, 0.75, 
        f"Color: Harm Value\nSize: BDP Ratio\n{target_cc} vs {compete_cc}", 
        transform=ax.transAxes, fontsize=12,
        verticalalignment='top')

def add_bubble_legend(ax):
    bdp_values = [bdp_min, (bdp_min + bdp_max)/2, bdp_max]
    sizes = [min_bubble + (v - bdp_min) * bubble_scale for v in bdp_values]
    labels = [f"{v:.2f}" for v in bdp_values]
    
    for size, label in zip(sizes, labels):
        ax.scatter([], [], s=size, c='gray', alpha=0.7, 
                   edgecolors='black', linewidth=0.5, label=label)
    
    legend = ax.legend(title='BDP Ratio', loc='upper right', 
                       frameon=True, framealpha=0.8,
                       labelspacing=1.2, handletextpad=1.5,
                       fontsize=12, title_fontsize=12)
    return legend

add_bubble_legend(ax)

plt.savefig(f'results/{target_cc}-{compete_cc}-{times}-long-flow/top_{top_num}_harm_{tool}_{target_cc}_{compete_cc}.png', dpi=300, bbox_inches='tight')