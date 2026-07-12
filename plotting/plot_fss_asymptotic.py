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
from scipy.stats import linregress
import warnings

warnings.filterwarnings('ignore')

DATA_DIRS = ["./simulations/AWS_Results", "./AWS_Results", "./data", "."]
OVERLEAF_DIR = "./plots"
os.makedirs(OVERLEAF_DIR, exist_ok=True)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    plt.style.use('ggplot')

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 12})
colors = {1: '#1f77b4', 2: '#ff7f0e', 8: '#2ca02c', 32: '#d62728'}
target_ks = [1, 2, 8, 32]

all_files = []
for directory in DATA_DIRS:
    if os.path.exists(directory):
        all_files.extend(glob.glob(os.path.join(directory, "*.csv")))

if not all_files:
    print("[WARN] No CSV files found. Scaling skipped.")
    exit(0)

df_list = []
essential_cols = ["model_type", "L", "k", "max_susceptibility"]
for file in all_files:
    try:
        header = pd.read_csv(file, nrows=0)
        cols_to_use = [col for col in essential_cols if col in header.columns]
        df = pd.read_csv(file, usecols=cols_to_use, on_bad_lines='skip')
        if 'model_type' not in df.columns:
            df['model_type'] = 'NN'
        df_list.append(df)
    except:
        pass

master_df = pd.concat(df_list, ignore_index=True)
master_df = master_df[master_df['L'] <= 192]
master_df['N'] = (master_df['L'].astype(np.int32) + 1)**3

for model in master_df['model_type'].unique():
    model_df = master_df[(master_df['model_type'] == model) & (master_df['k'].isin(target_ks))]
    if len(model_df) == 0: 
        continue
        
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    fig.subplots_adjust(top=0.90, bottom=0.15, left=0.15, right=0.95)
    
    for k in target_ks:
        k_data = model_df[(model_df['k'] == k) & (model_df['max_susceptibility'] > 0)]
        agg_data = k_data.groupby(['L', 'N'])['max_susceptibility'].mean().reset_index().dropna()
        if len(agg_data) >= 2:
            log_N = np.log(agg_data['N'])
            log_chi = np.log(agg_data['max_susceptibility'])
            
            # Asymptotic selection (L >= 24)
            fit_subset = agg_data[agg_data['L'] >= 24]
            if len(fit_subset) >= 2:
                slope, intercept, r_val, _, _ = linregress(np.log(fit_subset['N']), np.log(fit_subset['max_susceptibility']))
            else:
                slope, intercept, r_val, _, _ = linregress(log_N, log_chi)
            
            ax.scatter(log_N, log_chi, color=colors[k], s=45, zorder=5, edgecolor='k')
            ax.plot(log_N, intercept + slope * log_N, color=colors[k], linestyle='--', linewidth=1.2,
                     label=r'$k={}$ (Fit $\gamma={:.3f}$, $R^2={:.3f}$)'.format(k, slope, r_val**2))
                     
    ax.set_title(f"Finite-Size Scaling of Peak Susceptibility ({model})", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(r"$\ln(N)$", fontsize=11)
    ax.set_ylabel(r"$\ln(\chi'_{\max})$", fontsize=11)
    ax.legend(loc='upper left', frameon=True, edgecolor='lightgray', facecolor='white', fontsize=9.5)
    
    fig.savefig(os.path.join(OVERLEAF_DIR, f"{model}_d_FSS_Asymptotic.pdf"), format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OVERLEAF_DIR, f"{model}_d_FSS_Asymptotic.png"), format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)