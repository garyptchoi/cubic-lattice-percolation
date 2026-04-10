import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr
from scipy.ndimage import gaussian_filter1d
import warnings

# Suppress warnings for clean console output
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. DIRECTORY SETUP 
# ==============================================================================
# Make sure ALL your CSVs (L=1..10 AND L=24..192) are pasted into this folder!
DATA_DIR = r"C:\Users\lautr\Downloads\AWS_Rigidity_Campaign\AWS_Results"
OVERLEAF_DIR = r"C:\Users\lautr\Downloads\Overleaf_Final_Figures"
os.makedirs(OVERLEAF_DIR, exist_ok=True)

print("="*80)
print(f" Scanning directory: {DATA_DIR}")
print("="*80)

all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print(f"Found {len(all_files)} CSV files. Loading master thermodynamic dataset...\n")

# ==============================================================================
# 2. ROBUST DATA LOADING & MERGING
# ==============================================================================
df_list =[]
for file in all_files:
    try:
        df = pd.read_csv(file, on_bad_lines='skip')
        if 'p_crossings' in df.columns:
            df = df.dropna(subset=['p_crossings'])
            
            # Backwards compatibility for older L=24..192 files
            if 'model_type' not in df.columns: df['model_type'] = 'NN'
            if 'p_rigid' not in df.columns: df['p_rigid'] = -1.0
                
            df['p_rigid'] = pd.to_numeric(df['p_rigid'], errors='coerce').fillna(-1.0)
            df['max_susceptibility'] = pd.to_numeric(df['max_susceptibility'], errors='coerce').fillna(-1.0)
            df['max_jump_frac'] = pd.to_numeric(df['max_jump_frac'], errors='coerce').fillna(-1.0)
            
            df_list.append(df)
    except: pass

if not df_list:
    print("No valid data found! Make sure you put the CSV files in the folder.")
    exit()

master_df = pd.concat(df_list, ignore_index=True)

# STRICT FILTER: Drop L=224 for perfect consistency
master_df = master_df[master_df['L'] <= 192] 
master_df['N'] = (master_df['L'] + 1)**3

print(f"Successfully loaded {len(master_df):,} total simulations.")
print(f"System Sizes (L) detected: {sorted(master_df['L'].unique())}")
print(f"Choice parameters (k) detected: {sorted(master_df['k'].unique())}\n")

# ==============================================================================
# 3. REVIEWER 1: 8 INDIVIDUAL THERMODYNAMIC PLOTS (ALL L, k=1,2,8,32)
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
colors = {1: '#1f77b4', 2: '#ff7f0e', 8: '#2ca02c', 32: '#d62728'}
target_ks = [1, 2, 8, 32]

for model in master_df['model_type'].unique():
    model_df = master_df[(master_df['model_type'] == model) & (master_df['k'].isin(target_ks))]
    if len(model_df) == 0: continue
    
    # Find the largest L that has phase curves for all target Ks
    L_k_counts = model_df.groupby('L')['k'].nunique()
    complete_L_values = L_k_counts[L_k_counts == len(target_ks)].index.tolist()
    plot3_L = max(complete_L_values) if complete_L_values else model_df['L'].max()
    
    # --- PLOT 1: FSS (ALL L: 1 to 192) ---
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    for k in target_ks:
        k_data = model_df[(model_df['k'] == k) & (model_df['max_susceptibility'] > 0)]
        agg_data = k_data.groupby('N')['max_susceptibility'].mean().reset_index().dropna()
        if len(agg_data) > 2:
            log_N = np.log(agg_data['N'])
            log_chi = np.log(agg_data['max_susceptibility'])
            slope, intercept, r_val, _, _ = linregress(log_N, log_chi)
            ax1.scatter(log_N, log_chi, color=colors.get(k, '#333'), s=40, label=f'k={k}')
            ax1.plot(log_N, intercept + slope * log_N, color=colors.get(k, '#333'), linestyle='--', 
                     label=r'Fit $\gamma={:.3f}$ ($R^2={:.3f}$)'.format(slope, r_val**2))
    ax1.set_title(f"Finite-Size Scaling of Peak Susceptibility ({model})", fontsize=12, fontweight='bold')
    ax1.set_xlabel(r"$\ln(N)$", fontsize=11)
    ax1.set_ylabel(r"$\ln(\chi'_{\max})$", fontsize=11)
    ax1.legend(fontsize=9)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OVERLEAF_DIR, f"{model}_1_FSS.pdf"), format='pdf', bbox_inches='tight')

    # --- PLOT 2: Macroscopic Jump (ALL L: 1 to 192) ---
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    for k in target_ks:
        k_data = model_df[(model_df['k'] == k) & (model_df['max_jump_frac'] > 0)]
        agg_data = k_data.groupby('L')['max_jump_frac'].mean().reset_index().dropna()
        if len(agg_data) > 0:
            ax2.plot(agg_data['L'], agg_data['max_jump_frac'], marker='o', markersize=6, color=colors.get(k, '#333'), label=f'k={k}')
    ax2.set_title(f"Maximum Single-Step Jump vs System Size ({model})", fontsize=12, fontweight='bold')
    ax2.set_xlabel(r"System Size ($L$)", fontsize=11)
    ax2.set_ylabel(r"$\Delta S_{\max} / N$", fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OVERLEAF_DIR, f"{model}_2_MaxJump.pdf"), format='pdf', bbox_inches='tight')

    # --- PLOTS 3 & 4: Phase Curves and Bimodality (Max L) ---
    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    fig4, ax4 = plt.subplots(figsize=(6, 4.5))
    percentile_y = np.linspace(0.01, 1.0, 100)
    
    for k in target_ks:
        lk_data = model_df[(model_df['L'] == plot3_L) & (model_df['k'] == k)]
        valid_p_arrays = []
        for p_str in lk_data['p_crossings']:
            try:
                p_arr = np.array([float(x) for x in str(p_str).split(';')])
                if len(p_arr) == 100:
                    p_arr[p_arr < 0] = 1.0  
                    p_arr = np.maximum.accumulate(p_arr) 
                    valid_p_arrays.append(p_arr)
            except: continue
        if not valid_p_arrays: continue

        avg_p_curve = np.mean(valid_p_arrays, axis=0)
        ax3.plot(avg_p_curve, percentile_y, color=colors.get(k, '#333'), linewidth=2.5, label=f'k={k}')
        
        pc_ensemble = lk_data['p_max_jump'].median()
        s_values_at_pc =[np.searchsorted(p, pc_ensemble) / 100.0 for p in valid_p_arrays]
            
        if s_values_at_pc:
            bins = np.linspace(-0.005, 1.005, 102)
            counts, bin_edges = np.histogram(s_values_at_pc, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            smoothed_counts = gaussian_filter1d(counts, sigma=1.5)
            ax4.plot(bin_centers, smoothed_counts, color=colors.get(k, '#333'), linewidth=2.5, label=f'k={k}')
            ax4.fill_between(bin_centers, 0, smoothed_counts, color=colors.get(k, '#333'), alpha=0.3)

    ax3.set_title(f"Exact Phase Transition Curve ($L={plot3_L}$, {model})", fontsize=12, fontweight='bold')
    ax3.set_xlabel(r"Edge Density ($p$)", fontsize=11)
    ax3.set_ylabel(r"Giant Component Fraction ($S_{\max} / N$)", fontsize=11)
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    fig3.savefig(os.path.join(OVERLEAF_DIR, f"{model}_3_PhaseCurve.pdf"), format='pdf', bbox_inches='tight')

    ax4.set_title(f"Order Parameter Distribution at $p_c$ ($L={plot3_L}$, {model})", fontsize=12, fontweight='bold')
    ax4.set_xlabel(r"Giant Component Fraction ($S_{\max} / N$)", fontsize=11)
    ax4.set_ylabel("Probability Density", fontsize=11)
    ax4.tick_params(axis='y', labelsize=10)
    ax4.set_xlim(-0.05, 1.05)
    ax4.legend(fontsize=9)
    fig4.tight_layout()
    fig4.savefig(os.path.join(OVERLEAF_DIR, f"{model}_4_Bimodal.pdf"), format='pdf', bbox_inches='tight')

# ==============================================================================
# 4. RIGIDIFICATION EFFICIENCY & MONOTONICITY STATISTICS (All k=1..32)
# ==============================================================================
print("\n" + "="*80)
print(" STATISTICAL TESTS FOR MONOTONICITY (Spearman's Rank Correlation)")
print("="*80)

# Find the largest L that has rigidity data completed for all K (Right now it's L=1, tomorrow L=10)
intra_rigid_df = master_df[(master_df['model_type'] == 'Intra') & (master_df['p_rigid'] > 0)]

if len(intra_rigid_df) > 0:
    max_rigid_L = intra_rigid_df['L'].max()
    print(f"Generating Rigidity Efficiency Plot for L = {max_rigid_L}...\n")
    
    l_df = intra_rigid_df[intra_rigid_df['L'] == max_rigid_L]
    
    gap_data = l_df.groupby('k').agg(
        p_conn=('p_max_susceptibility', 'mean'),
        p_rigid=('p_rigid', 'mean')
    ).reset_index()
    
    gap_data['delta_p'] = gap_data['p_rigid'] - gap_data['p_conn']
    
    # --- SPEARMAN'S RANK CORRELATION TEST ---
    rho_conn, pval_conn = spearmanr(gap_data['k'], gap_data['p_conn'])
    rho_gap, pval_gap = spearmanr(gap_data['k'], gap_data['delta_p'])
    
    print(f"Connectivity Delay (k vs p_conn):  rho = {rho_conn:7.4f}, p-value = {pval_conn:.4e}")
    print(f"Rigidification Gap (k vs Delta p): rho = {rho_gap:7.4f}, p-value = {pval_gap:.4e}")
    if pval_gap < 0.01 and rho_gap < 0:
        print("--> STRONG STATISTICAL PROOF: The Rigidity Gap is Monotonically Shrinking!")
    
    # --- PLOTTING ---
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Rigidification Efficiency & Monotonic Delay (Intra Model, L={max_rigid_L})", fontsize=16, fontweight='bold')
    
    ax_left.plot(gap_data['k'], gap_data['p_conn'], marker='o', color='#1f77b4', linewidth=2, label=r'Connectivity ($p_c^{\text{conn}}$)')
    ax_left.plot(gap_data['k'], gap_data['p_rigid'], marker='s', color='#d62728', linewidth=2, label=r'Rigidity ($p_c^{\text{rigid}}$)')
    ax_left.set_title("Critical Densities vs. Choice Parameter ($k$)", fontsize=12, fontweight='bold')
    ax_left.set_xlabel("Number of Choices ($k$)", fontsize=11)
    ax_left.set_ylabel("Edge Density ($p$)", fontsize=11)
    
    # Stamp the statistics directly onto the plot
    stats_text_left = f"Spearman $\\rho = {rho_conn:.4f}$\n$p$-val $= {pval_conn:.2e}$"
    ax_left.text(0.55, 0.15, stats_text_left, transform=ax_left.transAxes, fontsize=11, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    ax_left.legend(fontsize=10, loc='lower right', bbox_to_anchor=(0.95, 0.3))
    
    ax_right.plot(gap_data['k'], gap_data['delta_p'], marker='D', color='#2ca02c', linewidth=2.5)
    ax_right.set_title(r"Rigidity-Connectivity Gap ($\Delta p_c$) vs. Choice Parameter ($k$)", fontsize=12, fontweight='bold')
    ax_right.set_xlabel("Number of Choices ($k$)", fontsize=11)
    ax_right.set_ylabel(r"Gap ($\Delta p_c$)", fontsize=11)
    ax_right.fill_between(gap_data['k'], 0, gap_data['delta_p'], color='#2ca02c', alpha=0.1)
    ax_right.set_ylim(bottom=0)
    
    stats_text_right = f"Spearman $\\rho = {rho_gap:.4f}$\n$p$-val $= {pval_gap:.2e}$"
    ax_right.text(0.65, 0.80, stats_text_right, transform=ax_right.transAxes, fontsize=11, 
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    fig.savefig(os.path.join(OVERLEAF_DIR, "Intra_Rigidity_Gap_All_K.pdf"), format='pdf', bbox_inches='tight')

print(f"\nSUCCESS! All 8 individual thermodynamic figures + 1 Rigidity figure saved to:")
print(f" -> {OVERLEAF_DIR}")
plt.close('all')