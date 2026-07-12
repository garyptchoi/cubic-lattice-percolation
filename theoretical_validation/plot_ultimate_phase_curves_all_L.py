#!/usr/bin/env python3
import os
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize, LogNorm
from matplotlib.lines import Line2D
import tarfile
import re
import warnings

warnings.filterwarnings('ignore')

CACHE_FILE = "./data/optimized_campaign_cache_fast.pkl"
TAR_FILE = "./data/V1_Rigidity_20k.tar.gz"
OVERLEAF_DIR = "./plots"
os.makedirs(OVERLEAF_DIR, exist_ok=True)

SPECTRUM_CMAP = plt.get_cmap('YlOrBr')
main_ks = [1, 2, 4, 8, 16, 32]  
si_ks = [1, 2, 8, 32]           
y_vals = np.linspace(0.01, 1.0, 100)

if not os.path.exists(CACHE_FILE):
    print(f"[!] Cannot find connectivity cache: {CACHE_FILE}")
    exit(0)

with open(CACHE_FILE, 'rb') as f:
    conn_db = pickle.load(f)

rigidity_db = {'Intra': {k: {L: [] for L in range(1, 11)} for k in main_ks}}
file_pattern_tar = re.compile(r'results_L(\d+)_k(\d+)\.csv')
if os.path.exists(TAR_FILE):
    with tarfile.open(TAR_FILE, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith('.csv'): continue
            match = file_pattern_tar.search(member.name)
            if not match: continue
            L, k = int(match.group(1)), int(match.group(2))
            if L > 10 or k not in main_ks: continue
            
            f = tar.extractfile(member)
            df = pd.read_csv(f, usecols=['p_rigid'], on_bad_lines='skip')
            rigidity_db['Intra'][k][L].extend(pd.to_numeric(df['p_rigid'], errors='coerce').dropna().values.tolist())

# Scan raw backup paths as well
for r_dir in ["./simulations/AWS_Results", "./AWS_Results", "./data", "."]:
    if not os.path.exists(r_dir): continue
    for file in glob.glob(os.path.join(r_dir, "*.csv")):
        match = re.search(r'results_?L(\d+)_k(\d+)\.csv', os.path.basename(file))
        if match:
            L, k = int(match.group(1)), int(match.group(2))
            if L <= 10 and k in main_ks:
                try:
                    df = pd.read_csv(file, usecols=['p_rigid'], on_bad_lines='skip')
                    valid = pd.to_numeric(df['p_rigid'], errors='coerce').dropna().values
                    if len(rigidity_db['Intra'][k][L]) < len(valid):
                        rigidity_db['Intra'][k][L] = valid.tolist()
                except: pass

large_conn_db = {'NN': {k: {} for k in si_ks}, 'Intra': {k: {} for k in si_ks}}
possible_locations = ["./simulations/AWS_Results", "./AWS_Results", "./data", "."]
found_dir = next((loc for loc in possible_locations if os.path.exists(loc) and glob.glob(os.path.join(loc, "results_*_L24_*.csv"))), None)

if found_dir:
    all_files = glob.glob(os.path.join(found_dir, "*.csv"))
    for file in all_files:
        try:
            header = pd.read_csv(file, nrows=0)
            cols_to_use = [col for col in ['model_type', 'k', 'L', 'p_crossings'] if col in header.columns]
            df = pd.read_csv(file, usecols=cols_to_use, on_bad_lines='skip')
            if 'model_type' not in df.columns: df['model_type'] = 'NN'
            for (model, k, L), group in df.groupby(['model_type', 'k', 'L']):
                if k not in si_ks or L < 24: continue
                valid_p_arrays = []
                for p_str in group['p_crossings'].dropna():
                    p_arr = np.fromstring(str(p_str), sep=';')
                    if len(p_arr) == 100:
                        p_arr[p_arr < 0] = 1.0  
                        valid_p_arrays.append(np.maximum.accumulate(p_arr))
                if valid_p_arrays:
                    large_conn_db[model][k][L] = np.mean(valid_p_arrays, axis=0)
        except: pass

# Figure 1: Consistent L <= 10 Gap
norm_small = Normalize(vmin=1, vmax=10)
fig_main, axs_main = plt.subplots(2, 3, figsize=(20, 11), sharey=True, sharex=True)
for idx, k in enumerate(main_ks):
    ax = axs_main.ravel()[idx]
    for L in range(1, 11):
        if k not in conn_db["Intra"] or L not in conn_db["Intra"][k]: continue
        color = SPECTRUM_CMAP(0.3 + 0.7 * norm_small(L))
        ax.plot(conn_db["Intra"][k][L]['mean_crossings'], y_vals, color=color, linewidth=1.2)
        if len(rigidity_db['Intra'][k][L]) > 10:
            p_rig_arr = np.array(rigidity_db['Intra'][k][L])
            p_rig_arr.sort()
            cdf = np.arange(1, len(p_rig_arr) + 1) / len(p_rig_arr)
            ax.plot(p_rig_arr, cdf, color=color, linewidth=1.2, linestyle='--')
            
    ax.set_title(f'({chr(97 + idx)}) Intra ($k = {k}$)', loc='left', fontsize=20, fontweight='bold', pad=15)
    ax.set_xlim(0.0, 0.75)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6])
    if idx // 3 == 1: ax.set_xlabel('Edge Density ($p$)', fontsize=22, labelpad=10)
    if idx % 3 == 0: ax.set_ylabel(r'$S_{\max}/N$', fontsize=22, labelpad=10)

plt.subplots_adjust(top=0.82, bottom=0.10, left=0.08, right=0.85, hspace=0.35, wspace=0.25)
custom_lines_main = [Line2D([0], [0], color='k', linestyle='-', lw=1.5), Line2D([0], [0], color='k', linestyle='--', lw=1.5)]
fig_main.legend(custom_lines_main, ['Connectivity', 'Relaxed Rigidity'], loc='upper left', bbox_to_anchor=(0.80, 0.94), fontsize=16, frameon=True)
fig_main.colorbar(plt.cm.ScalarMappable(cmap=SPECTRUM_CMAP, norm=norm_small), ax=axs_main, pad=0.02, aspect=30).set_label('System Size ($L$)', fontsize=20, labelpad=10)
fig_main.savefig(os.path.join(OVERLEAF_DIR, "Fig_Rigidity_Connectivity_Gap_6Panel_L10.pdf"), format='pdf', dpi=300, bbox_inches='tight')

# Figure 2: Macroscopic Connectivity Spectrum (L <= 192)
log_norm = LogNorm(vmin=1, vmax=192)
fig_si, axs_si = plt.subplots(2, 4, figsize=(24, 10), sharey=True, sharex=True)
for i, model in enumerate(["NN", "Intra"]):
    for j, k in enumerate(si_ks):
        ax = axs_si[i, j]
        available_L = set(conn_db.get(model, {}).get(k, {}).keys()) | set(large_conn_db.get(model, {}).get(k, {}).keys())
        for L in sorted(list(available_L)):
            color = SPECTRUM_CMAP(log_norm(L))
            if L <= 10 and L in conn_db[model].get(k, {}):
                ax.plot(conn_db[model][k][L]['mean_crossings'], y_vals, color=color, linewidth=1.2)
            elif L in large_conn_db[model].get(k, {}):
                ax.plot(large_conn_db[model][k][L], y_vals, color=color, linewidth=1.2)
        
        ax.set_title(f'({chr(97 + i*4 + j)}) {model} ($k = {k}$)', loc='left', fontsize=20, fontweight='bold', pad=15)
        ax.set_xlim(0.0, 0.75)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6])
        if i == 1: ax.set_xlabel('Edge Density ($p$)', fontsize=22, labelpad=10)
        if j == 0: ax.set_ylabel(r'$S_{\max}/N$', fontsize=22, labelpad=10)

plt.subplots_adjust(top=0.85, bottom=0.10, left=0.06, right=0.85, hspace=0.35, wspace=0.25)
cbar = fig_si.colorbar(plt.cm.ScalarMappable(cmap=SPECTRUM_CMAP, norm=log_norm), ax=axs_si, pad=0.02, aspect=30)
cbar.set_label('System Size ($L$)', fontsize=20, labelpad=10)
cbar.set_ticks([1, 10, 100])
cbar.ax.set_yticklabels(['$1$', '$10$', '$10^2$'])
fig_si.savefig(os.path.join(OVERLEAF_DIR, "Fig_PhaseCurves_All_L_1_to_192.pdf"), format='pdf', dpi=300, bbox_inches='tight')
print("-> Completed ultimate phase plotting outputs.")