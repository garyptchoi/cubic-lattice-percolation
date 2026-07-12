#!/usr/bin/env python3
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
import tarfile
import re

TAR_FILE = "./data/V1_Rigidity_20k.tar.gz"
CACHE_FILE = "./data/optimized_campaign_cache_fast.pkl"
PLOT_DIR = "./plots"
os.makedirs(PLOT_DIR, exist_ok=True)

if not os.path.exists(CACHE_FILE) or not os.path.exists(TAR_FILE):
    print(f"[!] Error: Make sure both CACHE_FILE and TAR_FILE exist in ./data/")
    exit(0)

with open(CACHE_FILE, 'rb') as f:
    conn_db = pickle.load(f)

rigidity_db = {'Intra': {k: {L:[] for L in range(1, 11)} for k in [1, 2, 8, 32]}}
file_pattern = re.compile(r'results_L(\d+)_k(\d+)\.csv')

with tarfile.open(TAR_FILE, "r:gz") as tar:
    for member in tar.getmembers():
        if not member.name.endswith('.csv'): continue
        match = file_pattern.search(member.name)
        if not match: continue
        L, k = int(match.group(1)), int(match.group(2))
        if L > 10 or k not in [1, 2, 8, 32]: continue
        
        f = tar.extractfile(member)
        df = pd.read_csv(f, usecols=['p_rigid'], on_bad_lines='skip')
        p_rig = pd.to_numeric(df['p_rigid'], errors='coerce').dropna()
        rigidity_db['Intra'][k][L].extend(p_rig[p_rig > 0].values.tolist())

sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)
SPECTRUM_CMAP = plt.get_cmap('YlOrBr')
y_vals = np.linspace(0.01, 1.0, 100)
target_ks = [1, 2, 8, 32]

fig, axs = plt.subplots(2, 4, figsize=(24, 10), sharey=True, sharex=True)
norm = Normalize(vmin=1, vmax=10)

for i, model in enumerate(["NN", "Intra"]):
    for j, k in enumerate(target_ks):
        ax = axs[i, j]
        for L in range(1, 11):
            if k not in conn_db[model] or L not in conn_db[model][k]: continue
            color = SPECTRUM_CMAP(0.3 + 0.7 * norm(L))
            
            mean_cross = conn_db[model][k][L]['mean_crossings']
            ax.plot(mean_cross, y_vals, color=color, linewidth=1.8, linestyle='-')
            
            if model == "Intra" and len(rigidity_db['Intra'][k][L]) > 10:
                p_rig_arr = np.array(rigidity_db['Intra'][k][L])
                p_rig_arr.sort()
                cdf = np.arange(1, len(p_rig_arr) + 1) / len(p_rig_arr)
                ax.plot(p_rig_arr, cdf, color=color, linewidth=2.5, linestyle='--')
        
        letter = chr(97 + i*4 + j)
        ax.set_title(f'{model} (k = {k})', fontsize=20)
        ax.text(-0.05, 1.05, f'({letter})', transform=ax.transAxes, fontsize=20, fontweight='bold')
        ax.set_xlim(0.05, 1.0)
        ax.set_ylim(-0.02, 1.05)
        if i == 1: ax.set_xlabel('p', fontsize=22)
        if j == 0: ax.set_ylabel(r'$S_{\max}/N$', fontsize=22)

from matplotlib.lines import Line2D
custom_lines = [Line2D([0],[0], color='k', linestyle='-', lw=2), Line2D([0],[0], color='k', linestyle='--', lw=2.5)]
axs[0, 0].legend(custom_lines, ['Connectivity', 'Relaxed Rigidity'], loc='lower right', fontsize=16)

sm = plt.cm.ScalarMappable(cmap=SPECTRUM_CMAP, norm=norm)
fig.colorbar(sm, ax=axs, pad=0.02, aspect=30).set_label('System Size (L)', fontsize=20)
fig.savefig(os.path.join(PLOT_DIR, "Fig_Rigidity_Connectivity_Gap_8Panel_Ultimate_20k.png"), dpi=300, bbox_inches='tight')