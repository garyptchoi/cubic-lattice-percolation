#!/usr/bin/env python3
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless-safe rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, linregress, gaussian_kde

# Path setup
ROOT_DIR = "."
THERMO_DIR = os.path.join(ROOT_DIR, "simulations", "AWS_Results")
RIGIDITY_DIR = os.path.join(ROOT_DIR, "simulations", "AWS_Rigidity_Only_20k")
PLOT_DIR = os.path.join(ROOT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

print("[SCAN] Scanning for raw simulation output files on disk...")
thermo_files = glob.glob(os.path.join(THERMO_DIR, "results_*.csv")) if os.path.exists(THERMO_DIR) else []
rigidity_files = glob.glob(os.path.join(RIGIDITY_DIR, "results_*.csv")) if os.path.exists(RIGIDITY_DIR) else []

if not thermo_files and not rigidity_files:
    print("\n[!] Error: No simulation data found on disk.")
    print("    Please run the simulation scripts first to generate raw CSV data.")
    exit(0)

L_detected = []
k_detected = []
for f in thermo_files + rigidity_files:
    m = re.search(r"L(\d+)_k(\d+)\.csv", os.path.basename(f))
    if m:
        L_detected.append(int(m.group(1)))
        k_detected.append(int(m.group(2)))

L_list = sorted(list(set(L_detected)))
k_list = sorted(list(set(k_detected)))

# Live Calculations
data_jumps = {k: [] for k in [1, 2, 8, 32] if k in k_list}
data_fss = {k: {'L': [], 'N': [], 'chi_max': []} for k in [1, 2, 8, 32] if k in k_list}
data_dense_gaps = []
target_ks = [k for k in [1, 2, 8, 32] if k in k_list]
if not target_ks:
    target_ks = k_list[:4]

print("\n[CALC] Calculating physical quantities from raw CSVs...")
for k in target_ks:
    for L in L_list:
        for model in ["Intra", "NN"]:
            file_path = os.path.join(THERMO_DIR, f"results_{model}_L{L}_k{k}.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, usecols=['max_susceptibility', 'max_jump_frac'], on_bad_lines='skip')
                    valid_chi = pd.to_numeric(df['max_susceptibility'], errors='coerce').dropna()
                    valid_jump = pd.to_numeric(df['max_jump_frac'], errors='coerce').dropna()
                    if len(valid_chi) > 0:
                        N = (L + 1) ** 3
                        data_fss[k]['L'].append(L)
                        data_fss[k]['N'].append(N)
                        data_fss[k]['chi_max'].append(valid_chi.mean())
                    if len(valid_jump) > 0 and (L == max(L_list) or L == 10):
                        data_jumps[k] = valid_jump.values
                except Exception as e:
                    print(f"    [Warning] Failed to parse {file_path}: {e}")

for L in L_list:
    for k in k_list:
        rig_file = os.path.join(RIGIDITY_DIR, f"results_L{L}_k{k}.csv")
        conn_file = os.path.join(THERMO_DIR, f"results_Intra_L{L}_k{k}.csv")
        if os.path.exists(rig_file) and os.path.exists(conn_file):
            try:
                df_rig = pd.read_csv(rig_file, usecols=['p_rigid'], on_bad_lines='skip')
                df_conn = pd.read_csv(conn_file, usecols=['p_max_susceptibility'], on_bad_lines='skip')
                valid_p_rig = pd.to_numeric(df_rig['p_rigid'], errors='coerce').dropna().values
                valid_p_conn = pd.to_numeric(df_conn['p_max_susceptibility'], errors='coerce').dropna().values
                if len(valid_p_rig) > 0 and len(valid_p_conn) > 0:
                    mean_p_rig, mean_p_conn = np.mean(valid_p_rig), np.mean(valid_p_conn)
                    data_dense_gaps.append({'L': L, 'k': k, 'p_conn': mean_p_conn, 'p_rigid': mean_p_rig, 'gap': mean_p_rig - mean_p_conn})
            except Exception as e:
                print(f"    [Warning] Failed to calculate gap for L={L}, k={k}: {e}")

df_dense = pd.DataFrame(data_dense_gaps)

# FSS Fit Calculations
fss_results = {}
for k in target_ks:
    volumes = np.array(data_fss[k]['N'])
    peaks = np.array(data_fss[k]['chi_max'])
    if len(volumes) >= 2:
        slope, intercept, r_val, _, _ = linregress(np.log(volumes), np.log(peaks))
        fss_results[k] = (slope, r_val**2)

# Grid Plot Rendering
fig = plt.figure(figsize=(15, 11))
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.3)
colors = {1: '#1f77b4', 2: '#ff7f0e', 8: '#2ca02c', 32: '#d62728'}
markers = {1: 'o', 2: 's', 8: '^', 32: 'D'}

# (a) Rigidity Transition CDF
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("(a) Rigidity Transition CDF", fontweight='semibold')
ax1.set_xlabel("Edge Density ($p$)")
ax1.set_ylabel("Cumulative Distribution ($P(p)$)")
for k in target_ks:
    available_rig_files = glob.glob(os.path.join(RIGIDITY_DIR, f"results_L*_k{k}.csv"))
    if available_rig_files:
        Ls = [int(re.search(r"L(\d+)_", os.path.basename(f)).group(1)) for f in available_rig_files]
        max_L = max(Ls)
        filepath = os.path.join(RIGIDITY_DIR, f"results_L{max_L}_k{k}.csv")
        try:
            df = pd.read_csv(filepath, usecols=['p_rigid'], on_bad_lines='skip')
            valid_rig = pd.to_numeric(df['p_rigid'], errors='coerce').dropna()
            valid_rig = valid_rig[valid_rig > 0].values
            if len(valid_rig) > 0:
                valid_rig.sort()
                cdf = np.arange(1, len(valid_rig) + 1) / len(valid_rig)
                ax1.plot(valid_rig, cdf, label=f"k={k} (L={max_L})", color=colors.get(k, '#333'), linewidth=2.0)
        except Exception:
            pass
ax1.legend(loc="upper left", frameon=True, edgecolor='#cccccc')

# (b) OP Jump Size Probability Density
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("(b) OP Jump Size Probability Density", fontweight='semibold')
ax2.set_xlabel("Single-Step OP Jump ($\\Delta S_{\\max}/N$)")
ax2.set_ylabel("Probability Density Function (PDF)")
for k in target_ks:
    jumps = data_jumps.get(k, [])
    if len(jumps) > 0:
        ax2.hist(jumps[jumps > 0], bins=45, density=True, histtype='step', linewidth=2.0, color=colors.get(k, '#333'), label=f"k={k}")
ax2.set_yscale('log')
ax2.legend(loc="upper right", frameon=True, edgecolor='#cccccc')

# (c) Susceptibility Finite-Size Scaling
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_title("(c) Susceptibility Finite-Size Scaling", fontweight='semibold')
ax3.set_xlabel("Total System Volume ($N$)")
ax3.set_ylabel("Peak Susceptibility ($\\chi'_{L,\\max}$)")
for k in target_ks:
    v = np.array(data_fss[k]['N'])
    peaks = np.array(data_fss[k]['chi_max'])
    if len(v) >= 2:
        gamma, r2 = fss_results[k]
        ax3.loglog(v, peaks, marker=markers.get(k, 'o'), color=colors.get(k, '#333'), linestyle='None', markersize=7)
        fit_line = np.exp(gamma * np.log(v) + (np.log(peaks[0]) - gamma * np.log(v[0])))
        ax3.loglog(v, fit_line, color=colors.get(k, '#333'), linestyle='--', linewidth=1.5, label=f"k={k}: $\\gamma={gamma:.3f}$")
ax3.legend(loc="lower right", frameon=True, edgecolor='#cccccc', fontsize=9)

# (d) Rigidity-Connectivity Gap vs L
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_title("(d) Rigidity-Connectivity Gap vs $L$", fontweight='semibold')
ax4.set_xlabel("Linear System Size ($L$)")
ax4.set_ylabel("Gap Size ($\\Delta p_c = p_c^{\\mathrm{rigid}} - p_c^{\\mathrm{conn}}$)")
if not df_dense.empty:
    for k in target_ks:
        sub_k = df_dense[df_dense['k'] == k].sort_values(by='L')
        if not sub_k.empty:
            ax4.plot(sub_k['L'], sub_k['gap'], marker=markers.get(k, 'o'), linestyle='--', linewidth=1.8, color=colors.get(k, '#333'), label=f"k={k}")
ax4.legend(loc="upper left", frameon=True, edgecolor='#cccccc', fontsize=9.5)

# (e) Optimal Efficiency Crossover Curve
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_title("(e) Optimal Rigidification Efficiency", fontweight='semibold')
ax5.set_xlabel("Choice Parameter ($k$)")
ax5.set_ylabel("Rigidity-Connectivity Gap ($\\Delta p_c$)")
if not df_dense.empty:
    dense_colors = {24: '#9467bd', 32: '#008080', 48: '#e377c2', 10: '#7f7f7f'}
    markers_dense = {24: 'o', 32: 's', 48: '^', 10: 'D'}
    for L in sorted(df_dense['L'].unique()):
        sub_L = df_dense[df_dense['L'] == L].sort_values(by='k')
        if len(sub_L) >= 2:
            ks = sub_L['k'].values
            gaps = sub_L['gap'].values
            rho, pval = spearmanr(ks, gaps)
            ax5.plot(ks, gaps, marker=markers_dense.get(L, 'o'), color=dense_colors.get(L, '#333'), linewidth=1.5, markersize=4, label=f"L = {L} ($\\rho = {rho:.3f}$)")
ax5.legend(loc="lower right", frameon=True, edgecolor='#cccccc', fontsize=8.5)

# (f) Normalized Susceptibility Spectrum
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_title("(f) Normalized Susceptibility Spectrum", fontweight='semibold')
ax6.set_xlabel("Link Density ($p$)")
ax6.set_ylabel("Normalized Susceptibility (KDE)")
for k in target_ks:
    available_files = glob.glob(os.path.join(THERMO_DIR, f"results_Intra_L*_k{k}.csv"))
    if available_files:
        Ls = [int(re.search(r"L(\d+)_", os.path.basename(f)).group(1)) for f in available_files]
        max_L = max(Ls)
        filepath = os.path.join(THERMO_DIR, f"results_Intra_L{max_L}_k{k}.csv")
        try:
            df = pd.read_csv(filepath, usecols=['p_max_susceptibility'], on_bad_lines='skip')
            p_c_arr = pd.to_numeric(df['p_max_susceptibility'], errors='coerce').dropna().values
            if len(p_c_arr) > 10:
                kde = gaussian_kde(p_c_arr)
                p_norm_range = np.linspace(0.1, 1.05, 500)
                ax6.plot(p_norm_range, kde(p_norm_range)/np.max(kde(p_norm_range)), color=colors.get(k, '#333'), linewidth=2.0, label=f"k={k} (L={max_L})")
        except Exception:
            pass
ax6.set_xlim(0.1, 1.05)
ax6.legend(loc="upper left", frameon=True, framealpha=0.8, edgecolor='#cccccc')

plt.suptitle("Statistical Mechanics of 3D Central-Force Rigidity Percolation", fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
output_path = os.path.join(PLOT_DIR, "PRR_Unified_Thermodynamic_Analysis_Calculated.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n[PLOT] Completed live calculations. Figure saved: {output_path}")