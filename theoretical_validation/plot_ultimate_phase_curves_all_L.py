import os
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import tarfile
import re
import warnings

# Suppress warnings for clean console output
warnings.filterwarnings('ignore')

# ==============================================================================
# EXACT FILE PATHS
# ==============================================================================
CACHE_FILE = r"C:\Users\lautr\Downloads\optimized_campaign_cache_fast.pkl"
TAR_FILE = r"C:\Users\lautr\Downloads\V1_Rigidity_20k.tar.gz"
OVERLEAF_DIR = r"C:\Users\lautr\Downloads\Overleaf_Final_Figures"
os.makedirs(OVERLEAF_DIR, exist_ok=True)

target_ks =[1, 2, 8, 32]
y_vals = np.linspace(0.01, 1.0, 100)

print("==================================================")
print(" 1. LOADING L=1..10 CONNECTIVITY (CACHE)")
print("==================================================")
if not os.path.exists(CACHE_FILE):
    print(f"[!] Cannot find {CACHE_FILE}")
    exit()

with open(CACHE_FILE, 'rb') as f:
    small_conn_db = pickle.load(f)

print("==================================================")
print(" 2. LOADING L=1..10 RIGIDITY (TARBALL)")
print("==================================================")
rigidity_db = {'Intra': {k: {L:[] for L in range(1, 11)} for k in target_ks}}
file_pattern = re.compile(r'results_L(\d+)_k(\d+)\.csv')

if os.path.exists(TAR_FILE):
    with tarfile.open(TAR_FILE, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith('.csv'): continue
            match = file_pattern.search(member.name)
            if not match: continue
            
            L, k = int(match.group(1)), int(match.group(2))
            if L > 10 or k not in target_ks: continue
            
            f = tar.extractfile(member)
            df = pd.read_csv(f, usecols=['p_rigid'], on_bad_lines='skip')
            valid = pd.to_numeric(df['p_rigid'], errors='coerce').dropna()
            valid = valid[valid > 0].values
            rigidity_db['Intra'][k][L].extend(valid.tolist())
else:
    print(f"[!] Warning: {TAR_FILE} not found. Rigidity dashed lines will be missing.")

print("==================================================")
print(" 3. LOADING L=24..192 CONNECTIVITY (AUTO-SEARCH)")
print("==================================================")
large_conn_db = {'NN': {k: {} for k in target_ks}, 'Intra': {k: {} for k in target_ks}}

# --- SMART SEARCH LOGIC ---
possible_locations =[
    r"C:\Users\lautr\Downloads\AWS_Rigidity_Campaign\AWS_Results",
    r"C:\Users\lautr\Downloads\AWS_Results",
    os.path.join(os.getcwd(), "AWS_Results"),
    r"C:\Users\lautr\Downloads",
    os.getcwd()
]

found_dir = None
for loc in possible_locations:
    if not os.path.exists(loc): continue
    if glob.glob(os.path.join(loc, "results_*_L24_*.csv")):
        found_dir = loc
        print(f"[SUCCESS] Auto-detected Big L data in: {found_dir}")
        break

if not found_dir:
    print("\n[!] CRITICAL ERROR: Could not find the Big L data!")
    exit()

all_files = glob.glob(os.path.join(found_dir, "*.csv"))

for file in all_files:
    try:
        df = pd.read_csv(file, on_bad_lines='skip')
        if 'p_crossings' not in df.columns: continue
        if 'model_type' not in df.columns: df['model_type'] = 'NN'
        
        for (model, k, L), group in df.groupby(['model_type', 'k', 'L']):
            if k not in target_ks or L < 24: continue
            
            valid_p_arrays =[]
            for p_str in group['p_crossings'].dropna():
                try:
                    p_arr = np.array([float(x) for x in str(p_str).split(';')])
                    if len(p_arr) == 100:
                        p_arr[p_arr < 0] = 1.0  
                        p_arr = np.maximum.accumulate(p_arr) 
                        valid_p_arrays.append(p_arr)
                except: continue
                
            if valid_p_arrays:
                large_conn_db[model][k][L] = np.mean(valid_p_arrays, axis=0)
    except Exception as e:
        pass

# --- PRINT DIAGNOSTIC SUMMARY ---
print("\n==================================================")
print(" DATA LOAD SUMMARY (Check here for missing sizes)")
print("==================================================")
for model in["NN", "Intra"]:
    for k in target_ks:
        avail = list(small_conn_db.get(model, {}).get(k, {}).keys()) + \
                list(large_conn_db.get(model, {}).get(k, {}).keys())
        if avail:
            print(f" {model} (k={k:2d}): Extracted sizes from L=1 up to L={max(avail)}")

print("\n==================================================")
print(" 4. PLOTTING ULTIMATE PHASE CURVES (ALL L)")
print("==================================================")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)

# 1. Applying base colormap exactly as in your previous script
SPECTRUM_CMAP = plt.get_cmap('YlOrBr')

# 2. Applying the true Logarithmic Scale mapping (L=1 to L=192)
log_norm = LogNorm(vmin=1, vmax=192)

fig, axs = plt.subplots(2, 4, figsize=(24, 10), sharey=True, sharex=True)

for i, model in enumerate(["NN", "Intra"]):
    for j, k in enumerate(target_ks):
        ax = axs[i, j]
        
        available_L = set()
        if model in small_conn_db and k in small_conn_db[model]:
            available_L.update(small_conn_db[model][k].keys())
        if model in large_conn_db and k in large_conn_db[model]:
            available_L.update(large_conn_db[model][k].keys())
        
        for L in sorted(list(available_L)):
            
            
            # Use the full colormap starting from 0.0 (very pale yellow)
            color = SPECTRUM_CMAP(log_norm(L))
            
            # 1. Connectivity Curve (Solid)
            # Removed alpha=0.9 and synced linewidth=1.8 so it visually mirrors old plots
            if L <= 10 and L in small_conn_db[model].get(k, {}):
                p_curve = small_conn_db[model][k][L]['mean_crossings']
                ax.plot(p_curve, y_vals, color=color, linewidth=1.8, linestyle='-')
            elif L in large_conn_db[model].get(k, {}):
                p_curve = large_conn_db[model][k][L]
                ax.plot(p_curve, y_vals, color=color, linewidth=1.8, linestyle='-')
            
            # 2. Rigidity Curve (Dashed) - Only Intra, Only L<=10
            if model == "Intra" and L <= 10 and len(rigidity_db['Intra'][k][L]) > 10:
                p_rig_arr = np.array(rigidity_db['Intra'][k][L])
                p_rig_arr.sort()
                cdf = np.arange(1, len(p_rig_arr) + 1) / len(p_rig_arr)
                # Synced linewidth=2.5
                ax.plot(p_rig_arr, cdf, color=color, linewidth=2.5, linestyle='--')
        
        letter = chr(97 + i*4 + j)
        ax.set_title(f'{model} (k = {k})', fontsize=20)
        ax.text(-0.05, 1.05, f'({letter})', transform=ax.transAxes, fontsize=20, fontweight='bold')
        
        # UNTRUNCATED X-AXIS: Starts exactly at 0.0
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.02, 1.02)
        
        if i == 1: ax.set_xlabel('Edge Density ($p$)', fontsize=22)
        if j == 0: ax.set_ylabel(r'$S_{\max}/N$', fontsize=22)

# Applying the full Log Colorbar to match the lines
sm = plt.cm.ScalarMappable(cmap=SPECTRUM_CMAP, norm=log_norm)
cbar = fig.colorbar(sm, ax=axs, pad=0.02, aspect=30)
cbar.set_label('System Size ($L$)', fontsize=20)

cbar.set_ticks([1, 10, 100])
cbar.ax.set_yticklabels(['$1$', '$10$', '$10^2$'])

out_path = os.path.join(OVERLEAF_DIR, "Fig_PhaseCurves_All_L_1_to_192.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"-> SUCCESS! Saved Ultimate Figure to {out_path}")

# Explicitly output/display the figure
plt.show()