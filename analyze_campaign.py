#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D Cubic Lattice Percolation Analysis Script
Version: 4.4.2 (2025-09-03) - Final with All NumPy 2.0 Compatibility Fixes

- Performs the full, unabridged analysis pipeline.
- Generates comprehensive plots.
- Performs exhaustive statistical analysis.
- [FIX in v4.4.2] Added a second compatibility shim to handle the removal of `np.float`.
  The 'unidip' library's use of both np.msort and np.float caused crashes with
  NumPy >= 2.0. Both are now patched. This should be the final fix.
- Retains robust, deterministic caching and memory-safe processing from v4.3.

Reference:
Trenton Lau and Gary P. T. Choi, 
Explosive connectivity and mechanical rigidity in cubic lattice structures,
arXiv preprint arXiv:2511.01537, 2025.

Copyright (c) 2025, Trenton Lau and Gary P. T. Choi
https://github.com/garyptchoi/cubic-lattice-percolation
"""

import os
import json
import re
import datetime as dt
import pickle
from collections import defaultdict
from pathlib import Path
import multiprocessing as mp
import shutil
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats

# [FIX in v4.4.2] Compatibility shim for numpy>=2.0
# The 'unidip' library uses np.msort and np.float, which were removed in numpy 2.0.
# These lines manually restore them by pointing to their modern equivalents.
if not hasattr(np, 'msort'):
    np.msort = lambda a: np.sort(a, axis=0)
if not hasattr(np, 'float'):
    np.float = float

# Attempt to import the dip test library
try:
    import unidip.dip as ud
    HAS_UNIDIP = True
except ImportError:
    HAS_UNIDIP = False

# --- CONFIGURATION ---
ROOT_DATA_DIR = Path("./FULL_CAMPAIGN_L1-10")
CACHE_DIR = Path("./cache")
OUTPUT_DIR = Path("./plots")
NUM_WORKERS = mp.cpu_count() - 1 or 1
BOOTSTRAP_SAMPLES = 1000

# --- Global Plot Settings ---
sns.set_context("talk"); sns.set_style("whitegrid")
plt.rcParams.update({'figure.figsize': (10, 7), 'lines.linewidth': 2.5, 'axes.labelsize': 16,'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})
PLOT_VERSION = dt.datetime.now().strftime("%Y%m%d_%H%M")

# ==============================================================================
# PART 1 & 2: Data Handling and Aggregation (Unchanged and Stable)
# ==============================================================================
def get_cache_path(L, k, model, rule):
    model_safe = model.replace(' ', '_').replace('(', '').replace(')', '')
    return CACHE_DIR / f"agg_L{L}_k{k}_{model_safe}_{rule}.pkl"

def parse_filename(filename):
    pattern = re.compile(r"L(\d+)_k(\d+)_(NNShell1|IntraS1-S3)_(StdK1|MinProd)_rig(True|False)_run(\d+).json"); match = pattern.match(filename)
    if not match: return None
    L, k, model_safe, rule, rig, run = match.groups(); model = "NN (Shell 1)" if model_safe == "NNShell1" else "Intra (S1-S3)"
    return {"L": int(L), "k": int(k), "model": model, "rule": rule, "rigidity_on": rig == "True", "run": int(run)}

def build_file_index(root_dir):
    index_cache_file = CACHE_DIR / "file_index.pkl"
    if index_cache_file.exists():
        try:
            if index_cache_file.stat().st_size > 0:
                with open(index_cache_file, "rb") as f: return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            print("WARNING: Index cache corrupt. Rebuilding."); index_cache_file.unlink(missing_ok=True)
    print("Building file index from scratch...")
    file_index = defaultdict(list); machine_dirs = [d.path for d in os.scandir(root_dir) if d.is_dir() and d.name.startswith("machine_")]
    for machine_dir in tqdm(machine_dirs, desc="Scanning folders"):
        try:
            for file in os.scandir(machine_dir):
                if file.name.endswith(".json"):
                    params = parse_filename(file.name)
                    if params: file_index[(params['L'], params['k'], params['model'], params['rule'], params['rigidity_on'])].append(Path(file.path))
        except FileNotFoundError: print(f"Warning: Dir not found: {machine_dir}.")
    temp_file = index_cache_file.with_suffix('.tmp');
    with open(temp_file, "wb") as f: pickle.dump(dict(file_index), f)
    shutil.move(temp_file, index_cache_file); print(f"Index built with {len(file_index)} param groups.")
    return dict(file_index)

def get_aggregated_curves(args_tuple):
    file_index, L, k, model, rule = args_tuple
    cache_file = get_cache_path(L, k, model, rule)
    if cache_file.exists() and cache_file.stat().st_size > 0: return True
    rig_key = (L, k, model, rule, True); no_rig_key = (L, k, model, rule, False)
    conn_files = file_index.get(no_rig_key, file_index.get(rig_key, []));
    if not conn_files: return False
    try:
        with open(conn_files[0]) as f: p_values = np.array(json.load(f)['data']['p_hist'])
        num_p_steps = len(p_values)
    except: return False
    N = (L + 1)**3
    sum_conn = np.zeros(num_p_steps); sum_sq_conn = np.zeros(num_p_steps); n_conn = 0; conn_files_list = []
    for f in conn_files:
        try:
            with open(f) as fp: c = np.array(json.load(fp)['data']['smax_v_hist'])
            if len(c)==num_p_steps: sum_conn+=c; sum_sq_conn+=c**2; n_conn+=1; conn_files_list.append(f)
        except: continue
    if n_conn==0: return False
    mean_conn = (sum_conn/n_conn)/N; chi_conn = N * ((sum_sq_conn/n_conn)/N**2 - mean_conn**2)
    result = {'L':L,'k':k,'model':model,'rule':rule,'N':N,'p':p_values,'mean_conn_curve':mean_conn,'chi_conn_curve':chi_conn,'pc_conn':p_values[np.argmax(chi_conn)],'chi_max_conn':np.max(chi_conn),'conn_files':conn_files_list}
    rig_files = file_index.get(rig_key, [])
    if rig_files:
        sum_rig=np.zeros(num_p_steps); sum_sq_rig=np.zeros(num_p_steps); n_rig=0; rig_files_list=[]
        for f in rig_files:
            try:
                with open(f) as fp: c = np.array(json.load(fp)['data']['smax_rigid_pg_hist'])
                if len(c)==num_p_steps: sum_rig+=c; sum_sq_rig+=c**2; n_rig+=1; rig_files_list.append(f)
            except: continue
        if n_rig>0:
            mean_rig = (sum_rig/n_rig)/N; chi_rig = N * ((sum_sq_rig/n_rig)/N**2 - mean_rig**2)
            is_trans = np.any(mean_rig > 0.1); pc_rig_idx = np.argmax(chi_rig)
            result.update({'mean_rig_curve':mean_rig, 'chi_rig_curve':chi_rig, 'pc_rig':p_values[pc_rig_idx] if is_trans else np.nan, 'rig_files':rig_files_list})
    with open(cache_file.with_suffix('.tmp'), 'wb') as f: pickle.dump(result, f)
    shutil.move(cache_file.with_suffix('.tmp'), cache_file); return True

def load_cached_data(L, k, model, rule):
    cache_file = get_cache_path(L, int(k), model, rule)
    if cache_file.exists():
        try: return pickle.load(open(cache_file, 'rb'))
        except: return None
    return None

def get_available_data_from_cache():
    all_data = []
    for f in glob.glob(str(CACHE_DIR / "agg_*.pkl")):
        try: all_data.append(pickle.load(open(f, 'rb')))
        except: continue
    return pd.DataFrame([d for d in all_data if d]) if all_data else pd.DataFrame()

# ==============================================================================
# PART 3: Comprehensive Plotting (Stable)
# ==============================================================================
def generate_publication_plots(df):
    """Generates exhaustive, publication-quality plots for all data."""
    print("\n--- STAGE 3: Generating comprehensive publication plots. ---")
    if df.empty: print("No data available to plot."); return
    grouped = df.groupby(['model', 'k', 'rule'])
    for (model, k, rule), group in tqdm(grouped, desc="Generating plots"):
        group = group.sort_values('L'); model_safe = model.replace(' ','').replace('(','').replace(')','')
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        for _, row in group.iterrows(): ax1.plot(row['p'], row['mean_conn_curve'], 'o-', label=f"L={row['L']}", markersize=4)
        ax1.set_title(f"Order Parameter: {model}, k={k}"); ax1.set_xlabel("p"); ax1.set_ylabel("$S_{max}/N$"); ax1.legend()
        fig1.savefig(OUTPUT_DIR / f"{model_safe}_k{k}_{rule}_OrderParam_{PLOT_VERSION}.png", dpi=150, bbox_inches='tight'); plt.close(fig1)
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        for _, row in group.iterrows(): ax2.plot(row['p'], row['chi_conn_curve'], 'o-', label=f"L={row['L']}", markersize=4)
        ax2.set_title(f"Susceptibility: {model}, k={k}"); ax2.set_xlabel("p"); ax2.set_ylabel("$\\chi$"); ax2.legend()
        fig2.savefig(OUTPUT_DIR / f"{model_safe}_k{k}_{rule}_Susceptibility_{PLOT_VERSION}.png", dpi=150, bbox_inches='tight'); plt.close(fig2)
        if 'pc_rig' in group.columns and group['pc_rig'].notna().any():
            fig3, ax3 = plt.subplots(figsize=(12, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(group)))
            for i, (_, row) in enumerate(group.iterrows()):
                if 'mean_rig_curve' in row and isinstance(row['mean_rig_curve'], np.ndarray):
                     ax3.plot(row['p'], row['mean_conn_curve'], 'o-', color=colors[i], label=f"Conn, L={row['L']}", markersize=4)
                     ax3.plot(row['p'], row['mean_rig_curve'], 's--', color=colors[i], markersize=4)
            ax3.set_title(f"Rigidity vs Connectivity: {model}, k={k}"); ax3.set_xlabel("p"); ax3.set_ylabel("$S_{max}/N$"); ax3.legend()
            fig3.savefig(OUTPUT_DIR / f"{model_safe}_k{k}_{rule}_RigidityGap_{PLOT_VERSION}.png", dpi=150, bbox_inches='tight'); plt.close(fig3)
        if len(group) >= 3:
            log_N = np.log(group['N']); log_chi = np.log(group['chi_max_conn']); res = stats.linregress(log_N, log_chi)
            fig4, ax4 = plt.subplots(figsize=(10, 7))
            ax4.plot(log_N, log_chi, 'o', label='Data')
            ax4.plot(log_N, res.intercept + res.slope * log_N, 'r-', label=f'Fit: $\\gamma$={res.slope:.3f}, $R^2$={res.rvalue**2:.4f}')
            ax4.set_title(f"Susceptibility Scaling: {model}, k={k}"); ax4.set_xlabel("ln(N)"); ax4.set_ylabel("ln($\\chi_{max}$)"); ax4.legend()
            fig4.savefig(OUTPUT_DIR / f"{model_safe}_k{k}_{rule}_ChiScaling_{PLOT_VERSION}.png", dpi=150, bbox_inches='tight'); plt.close(fig4)
    print("--- STAGE 3 Complete: Plots saved to 'plots' directory. ---")

# ==============================================================================
# PART 4: Comprehensive Statistical Analysis (Stable)
# ==============================================================================
def bootstrap_critical_points(data, n_boot=1000):
    cache_path = get_cache_path(data['L'], data['k'], data['model'], data['rule'])
    boot_cache_file = CACHE_DIR / (cache_path.stem + "_bootstrap.pkl")
    if boot_cache_file.exists():
        with open(boot_cache_file, 'rb') as f: return pickle.load(f)

    p_vals, N = data['p'], data['N']
    pc_conn_samples, pc_rig_samples = [], []
    for _ in tqdm(range(n_boot), desc=f"  Bootstrapping L={data['L']},k={data['k']}", leave=False, ncols=80):
        conn_files, rig_files = data.get('conn_files', []), data.get('rig_files', [])
        if conn_files:
            indices = np.random.choice(len(conn_files), size=len(conn_files), replace=True)
            boot_curves = [np.array(json.load(open(conn_files[i]))['data']['smax_v_hist'])/N for i in indices]
            mean_c=np.mean(boot_curves,axis=0); chi_c=N*(np.mean(np.square(boot_curves),axis=0)-mean_c**2); pc_conn_samples.append(p_vals[np.argmax(chi_c)])
        if rig_files:
            indices = np.random.choice(len(rig_files), size=len(rig_files), replace=True)
            boot_curves = [np.array(json.load(open(rig_files[i]))['data']['smax_rigid_pg_hist'])/N for i in indices]
            mean_r=np.mean(boot_curves,axis=0); chi_r=N*(np.mean(np.square(boot_curves),axis=0)-mean_r**2);
            if np.any(mean_r>0.1): pc_rig_samples.append(p_vals[np.argmax(chi_r)])
    result = {'pc_conn': pc_conn_samples, 'pc_rig': pc_rig_samples}
    with open(boot_cache_file, 'wb') as f: pickle.dump(result, f)
    return result

def run_comprehensive_statistical_analysis(df):
    """Generates an exhaustive statistical report for all valid data subsets."""
    print("\n\n===================================================================")
    print("STAGE 4: COMPREHENSIVE STATISTICAL ANALYSIS")
    print("===================================================================")
    if df.empty: print("No data found for analysis."); return
    
    print("\n--- [1] Per-Group Analysis (Scaling, Bimodality, Gaps) ---\n")
    grouped = df.groupby(['model', 'k', 'rule'])
    for (model, k, rule), group in grouped:
        print(f"\n>>> Analyzing Group: Model={model}, k={k}, Rule={rule}")
        if len(group) >= 3:
            subset = group.sort_values('N'); log_N, log_chi = np.log(subset['N']), np.log(subset['chi_max_conn']); res = stats.linregress(log_N, log_chi)
            print(f"  [Scaling Test]       gamma={res.slope:.3f}, R^2={res.rvalue**2:.4f} (from {len(subset)} sizes)")
        else: print(f"  [Scaling Test]       SKIPPED (requires 3+ system sizes, found {len(group)})")
        
        largest_L_row = group.loc[group['L'].idxmax()]
        data = load_cached_data(largest_L_row['L'], largest_L_row['k'], largest_L_row['model'], largest_L_row['rule'])
        if HAS_UNIDIP and data and 'conn_files' in data and len(data['conn_files']) > 10:
            pc_idx = np.argmax(data['chi_conn_curve'])
            sizes_at_pc = [np.array(json.load(open(f))['data']['smax_v_hist'])[pc_idx]/data['N'] for f in data.get('conn_files',[])]
            dip, pval, _ = ud.diptst(np.array([s for s in sizes_at_pc if not np.isnan(s)]))
            verdict = "REJECT H0 (Bimodal)" if pval < 0.05 else "FAIL to reject H0 (Unimodal)"
            print(f"  [Bimodality Test L={data['L']}] p-value={pval:.4f}. {verdict}")
        else: print(f"  [Bimodality Test L={largest_L_row['L']}] SKIPPED (unidip missing or <10 runs)")

        rig_group = group.dropna(subset=['pc_rig'])
        if not rig_group.empty:
            largest_L_rig_row = rig_group.loc[rig_group['L'].idxmax()]
            data = load_cached_data(largest_L_rig_row['L'], largest_L_rig_row['k'], largest_L_rig_row['model'], largest_L_rig_row['rule'])
            if data and 'rig_files' in data and len(data['rig_files']) > 5:
                boot_data = bootstrap_critical_points(data, n_boot=BOOTSTRAP_SAMPLES)
                pc_c, pc_r = boot_data['pc_conn'], boot_data['pc_rig']
                if len(pc_c) > 10 and len(pc_r) > 10:
                    min_len = min(len(pc_c), len(pc_r)); t, pval = stats.ttest_rel(pc_r[:min_len], pc_c[:min_len], alternative='greater')
                    gap = np.mean(pc_r) - np.mean(pc_c); verdict = "REJECT H0 (Gap is significant)" if pval < 0.01 else "FAIL to reject H0"
                    print(f"  [Rigidity Gap Test L={data['L']}] Gap={gap:.4f}, p-value={pval:e}. {verdict}")
                else: print(f"  [Rigidity Gap Test L={data['L']}] SKIPPED (bootstrap failed)")
        else: print(f"  [Rigidity Gap Test]    SKIPPED (no rigidity data for this group)")

    print("\n\n--- [2] Trend Analysis (Varying k) ---\n")
    if not df.empty:
        max_L = df['L'].max(); trend_df = df[df['L'] == max_L]
        for model, group in trend_df.groupby('model'):
            group = group.sort_values('k');
            if len(group['k'].unique()) < 3: continue
            print(f">>> Analyzing Trends for Model={model}, L={max_L}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", stats.ConstantInputWarning)
                rho, pval = stats.spearmanr(group['k'], group['pc_conn'])
                print(f"  [pc_conn vs k]   Spearman's rho={rho:.4f}, p-value={pval:e}")
                gap_group = group.dropna(subset=['pc_rig', 'pc_conn'])
                if len(gap_group) >= 3:
                    gap = gap_group['pc_rig'] - gap_group['pc_conn']; rho, pval = stats.spearmanr(gap_group['k'], gap)
                    print(f"  [Gap vs k]         Spearman's rho={rho:.4f}, p-value={pval:e}")
                else: print(f"  [Gap vs k]         SKIPPED (requires 3+ k values with rigidity)")
    else: print("SKIPPED - No data available.")

# ==============================================================================
# PART 5: Main Controller
# ==============================================================================
def main():
    print(f"--- 3D Percolation Campaign Analysis (v4.4.2 - Final) ---")
    CACHE_DIR.mkdir(exist_ok=True); OUTPUT_DIR.mkdir(exist_ok=True)
    if not ROOT_DATA_DIR.exists(): print(f"FATAL: Data dir '{ROOT_DATA_DIR}' not found."); return
        
    file_index = build_file_index(ROOT_DATA_DIR)
    print(f"\n--- STAGE 2: Aggregating data using up to {NUM_WORKERS} cores. ---")
    tasks = sorted(list({(L,k,m,r) for L,k,m,r,_ in file_index.keys()}))
    with mp.Pool(processes=NUM_WORKERS) as pool:
        task_args = ((file_index, *t) for t in tasks)
        list(tqdm(pool.imap_unordered(get_aggregated_curves, task_args), total=len(tasks), desc="Processing param sets"))
    print("--- STAGE 2 Complete: Data processed & cached. ---")
    
    available_data_df = get_available_data_from_cache()

    generate_publication_plots(available_data_df)
    run_comprehensive_statistical_analysis(available_data_df)

    print("\n\n--- FULL ANALYSIS COMPLETE ---")
    print(f"Plots saved to '{OUTPUT_DIR}'. Statistical report printed above.")

if __name__ == "__main__":
    mp.freeze_support()
    main()
