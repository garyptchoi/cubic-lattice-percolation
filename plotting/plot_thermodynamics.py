#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr
from scipy.ndimage import gaussian_filter1d
import warnings

warnings.filterwarnings('ignore')

DATA_DIRS = ["./simulations/AWS_Results", "./AWS_Results", "./data", "."]
OVERLEAF_DIR = "./plots"
os.makedirs(OVERLEAF_DIR, exist_ok=True)

all_files = []
for directory in DATA_DIRS:
    if os.path.exists(directory):
        all_files.extend(glob.glob(os.path.join(directory, "*.csv")))

if not all_files:
    print("[WARN] No thermodynamic files parsed. Script exited.")
    exit(0)

essential_cols = ["run_idx", "model_type", "L", "k", "max_susceptibility", "p_max_susceptibility", "max_jump_frac", "p_max_jump", "p_rigid"]
df_list = []
for file in all_files:
    try:
        header = pd.read_csv(file, nrows=0)
        cols_to_use = [col for col in essential_cols if col in header.columns]
        df = pd.read_csv(file, usecols=cols_to_use, on_bad_lines='skip')
        if 'model_type' not in df.columns: df['model_type'] = 'NN'
        if 'p_rigid' not in df.columns: df['p_rigid'] = np.float32(-1.0)
        df_list.append(df)
    except: pass

master_df = pd.concat(df_list, ignore_index=True)
master_df = master_df[master_df['L'] <= 192]
master_df['N'] = (master_df['L'].astype(np.int32) + 1)**3

def save_figure(fig, filename_base):
    fig.savefig(os.path.join(OVERLEAF_DIR, f"{filename_base}.pdf"), format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OVERLEAF_DIR, f"{filename_base}.png"), format='png', bbox_inches='tight', dpi=300)

def save_legend(handles, labels, filename_base):
    if not handles: return
    fig_leg = plt.figure(figsize=(6, 4))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis('off')
    legend = ax_leg.legend(handles, labels, loc='center', frameon=True, edgecolor='lightgray', facecolor='white')
    fig_leg.canvas.draw()
    bbox_padded = matplotlib.transforms.Bbox([[0,0], [1,1]]) # generic fallback
    try:
        bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
        bbox_padded = matplotlib.transforms.Bbox([[bbox.x0 - 0.05, bbox.y0 - 0.05], [bbox.x1 + 0.05, bbox.y1 + 0.05]])
    except: pass
    fig_leg.savefig(os.path.join(OVERLEAF_DIR, f"{filename_base}_legend.pdf"), format='pdf', bbox_inches=bbox_padded)
    fig_leg.savefig(os.path.join(OVERLEAF_DIR, f"{filename_base}_legend.png"), format='png', bbox_inches=bbox_padded)
    plt.close(fig_leg)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

colors = {1: '#1f77b4', 2: '#ff7f0e', 8: '#2ca02c', 32: '#d62728'}
target_ks = [1, 2, 8, 32]

for model in master_df['model_type'].unique():
    model_df = master_df[(master_df['model_type'] == model) & (master_df['k'].isin(target_ks))]
    if len(model_df) == 0: continue
    
    complete_L_values = model_df.groupby('L')['k'].nunique()
    complete_L_values = complete_L_values[complete_L_values == len(target_ks)].index.tolist()
    plot_L = max(complete_L_values) if complete_L_values else model_df['L'].max()
    
    p_crossings_by_k = {k: [] for k in target_ks}
    p_max_jump_by_k = {k: [] for k in target_ks}
    
    for file in all_files:
        try:
            fname = os.path.basename(file)
            if f"_{model}_" in fname and f"_L{plot_L}_" in fname:
                df_sub = pd.read_csv(file, usecols=['k', 'p_crossings', 'p_max_jump'], on_bad_lines='skip')
                for k in target_ks:
                    df_k_sub = df_sub[df_sub['k'] == k].dropna(subset=['p_crossings'])
                    for _, row in df_k_sub.iterrows():
                        p_crossings_by_k[k].append(str(row['p_crossings']))
                        p_max_jump_by_k[k].append(float(row['p_max_jump']))
        except: pass

    # Plot A: exact Phase Transition Curves
    fig_a, ax_a = plt.subplots(figsize=(6, 4.5))
    percentile_y = np.linspace(0.01, 1.0, 100)
    for k in target_ks:
        valid_p_arrays = []
        for p_str in p_crossings_by_k[k]:
            try:
                p_arr = np.array([float(x) for x in p_str.split(';')])
                if len(p_arr) == 100:
                    p_arr[p_arr < 0] = 1.0  
                    valid_p_arrays.append(np.maximum.accumulate(p_arr))
            except: continue
        if valid_p_arrays:
            ax_a.plot(np.mean(valid_p_arrays, axis=0), percentile_y, color=colors[k], linewidth=1.2, label=f'k={k}')
            
    ax_a.set_title(f"Exact Phase Transition Curve ($L={plot_L}$, {model})", fontsize=12, fontweight='bold')
    ax_a.set_xlabel(r"Edge Density ($p$)", fontsize=11)
    ax_a.set_ylabel(r"Giant Component Fraction ($S_{\max} / N$)", fontsize=11)
    save_figure(fig_a, f"{model}_a_PhaseCurve")
    save_legend(*ax_a.get_legend_handles_labels(), f"{model}_a_PhaseCurve")
    plt.close(fig_a)

    # Plot B: Bimodality
    fig_b, ax_b = plt.subplots(figsize=(6, 4.5))
    for k in target_ks:
        valid_p_arrays = []
        p_max_jumps = p_max_jump_by_k[k]
        for i, p_str in enumerate(p_crossings_by_k[k]):
            try:
                p_arr = np.array([float(x) for x in p_str.split(';')])
                if len(p_arr) == 100:
                    p_arr[p_arr < 0] = 1.0
                    valid_p_arrays.append((np.maximum.accumulate(p_arr), p_max_jumps[i]))
            except: continue
        if valid_p_arrays:
            pc_ensemble = np.median([item[1] for item in valid_p_arrays])
            s_vals = [np.searchsorted(p_arr, pc_ensemble) / 100.0 for p_arr, _ in valid_p_arrays]
            counts, bin_edges = np.histogram(s_vals, bins=np.linspace(-0.005, 1.005, 102), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            smoothed_counts = gaussian_filter1d(counts, sigma=1.5)
            ax_b.plot(bin_centers, smoothed_counts, color=colors[k], linewidth=1.2, label=f'k={k}')
            ax_b.fill_between(bin_centers, 0, smoothed_counts, color=colors[k], alpha=0.3)
            
    ax_b.set_title(f"Order Parameter Distribution at $p_c$ ($L={plot_L}$, {model})", fontsize=12, fontweight='bold')
    ax_b.set_xlabel(r"Giant Component Fraction ($S_{\max} / N$)", fontsize=11)
    ax_b.set_ylabel("Probability Density", fontsize=11)
    ax_b.set_xlim(-0.05, 1.05)
    save_figure(fig_b, f"{model}_b_Bimodal")
    save_legend(*ax_b.get_legend_handles_labels(), f"{model}_b_Bimodal")
    plt.close(fig_b)

    # Plot C: Jump Scaling
    fig_c, ax_c = plt.subplots(figsize=(6, 4.5))
    for k in target_ks:
        k_data = model_df[(model_df['k'] == k) & (model_df['max_jump_frac'] > 0)]
        agg_data = k_data.groupby('L')['max_jump_frac'].mean().reset_index().dropna()
        if not agg_data.empty:
            ax_c.plot(agg_data['L'], agg_data['max_jump_frac'], marker='o', markersize=6, linewidth=1.2, color=colors[k], label=f'k={k}')
    ax_c.set_title(f"Maximum Single-Step Jump vs System Size ({model})", fontsize=12, fontweight='bold')
    ax_c.set_xlabel(r"System Size ($L$)", fontsize=11)
    ax_c.set_ylabel(r"$\Delta S_{\max} / N$", fontsize=11)
    ax_c.set_ylim(0, 1.0)
    save_figure(fig_c, f"{model}_c_MaxJump")
    save_legend(*ax_c.get_legend_handles_labels(), f"{model}_c_MaxJump")
    plt.close(fig_c)

print(f"\n SUCCESS! Outputs compiled to: {OVERLEAF_DIR}")
plt.close('all')
