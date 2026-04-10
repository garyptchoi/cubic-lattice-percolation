import os
import pickle
import tarfile
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.stats import linregress, gaussian_kde
from matplotlib.lines import Line2D
import warnings

# Suppress warnings for clean video generation
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
BASE_DIR = r"C:\Users\lautr\Downloads"
CACHE_FILE = os.path.join(BASE_DIR, "optimized_campaign_cache_fast.pkl")
TAR_FILE_1 = os.path.join(BASE_DIR, "V1_Rigidity_20k.tar.gz")
TAR_FILE_2 = os.path.join(BASE_DIR, "PRR resubmission", "V1_Rigidity_20k.tar.gz")

OUTPUT_DIR = os.path.join(BASE_DIR, "Final_Paper_Videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication Aesthetics
plt_sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix'})
SPECTRUM_CMAP = plt.get_cmap('YlOrBr')
NORM = Normalize(vmin=1, vmax=10)

FPS = 4
DPI = 200

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_data():
    print("="*80)
    print(" LOADING DATA FOR VIDEO GENERATION")
    print("="*80)
    
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"Missing {CACHE_FILE}")
        
    with open(CACHE_FILE, 'rb') as f:
        conn_db = pickle.load(f)
    print(f" -> Connectivity data loaded from {os.path.basename(CACHE_FILE)}")

    # Locate Rigidity Tar
    tar_path = TAR_FILE_1 if os.path.exists(TAR_FILE_1) else TAR_FILE_2
    if not os.path.exists(tar_path):
        raise FileNotFoundError("Missing V1_Rigidity_20k.tar.gz")

    rigidity_db = {'NN': {k: {L:[] for L in range(1, 11)} for k in range(1, 33)},
                   'Intra': {k: {L:[] for L in range(1, 11)} for k in range(1, 33)}}

    file_pattern_intra = re.compile(r'results_L(\d+)_k(\d+)\.csv')
    file_pattern_general = re.compile(r'results_(NN|Intra)_L(\d+)_k(\d+)\.csv')

    print(f" -> Parsing Rigidity data from {os.path.basename(tar_path)}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith('.csv'): continue
            
            m_gen = file_pattern_general.search(member.name)
            m_intra = file_pattern_intra.search(member.name)
            
            if m_gen:
                model, L, k = m_gen.group(1), int(m_gen.group(2)), int(m_gen.group(3))
            elif m_intra:
                model, L, k = 'Intra', int(m_intra.group(1)), int(m_intra.group(2))
            else:
                continue

            if L <= 10 and k <= 32:
                f_csv = tar.extractfile(member)
                df = pd.read_csv(f_csv, usecols=['p_rigid'], on_bad_lines='skip')
                valid_rig = pd.to_numeric(df['p_rigid'], errors='coerce').dropna()
                valid_rig = valid_rig[valid_rig > 0]
                rigidity_db[model][k][L].extend(valid_rig.tolist())

    print(" -> Data successfully merged! Starting video rendering...\n")
    return conn_db, rigidity_db

# ==============================================================================
# VIDEO GENERATORS
# ==============================================================================

def make_order_video(model, conn_db, filename):
    print(f"Rendering {filename}...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sm = plt.cm.ScalarMappable(cmap=SPECTRUM_CMAP, norm=NORM)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('System Size (L)', fontsize=14)
    cbar.set_ticks([1, 3, 6, 8, 10])

    y_vals = np.linspace(0.01, 1.0, 100)

    def update(frame):
        k = frame + 1
        ax.clear()
        if model in conn_db and k in conn_db[model]:
            for L in range(1, 11):
                if L in conn_db[model][k]:
                    mean_cross = conn_db[model][k][L]['mean_crossings']
                    color = SPECTRUM_CMAP(0.3 + 0.7 * NORM(L))
                    ax.plot(mean_cross, y_vals, color=color, linewidth=2.0)
                    
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('Edge Density (p)', fontsize=16)
        ax.set_ylabel(r'Order Parameter ($S_{\max}/N$)', fontsize=16)
        ax.set_title(f"{model} Model Connectivity | k = {k}", fontsize=18, fontweight='bold')

    ani = FuncAnimation(fig, update, frames=32, repeat=False)
    writer = FFMpegWriter(fps=FPS, metadata=dict(artist='Trenton Lau'))
    ani.save(os.path.join(OUTPUT_DIR, filename), writer=writer, dpi=DPI)
    plt.close(fig)

def make_chi_scaling_video(model, conn_db, filename):
    print(f"Rendering {filename}...")
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        k = frame + 1
        ax.clear()
        
        if model in conn_db and k in conn_db[model]:
            L_vals, chi_vals = [],[]
            for L in range(1, 11):
                if L in conn_db[model][k]:
                    L_vals.append(L)
                    chi_vals.append(conn_db[model][k][L]['mean_chi'])
            
            if len(L_vals) >= 2:
                N_vals = np.array([(L+1)**3 for L in L_vals])
                ln_N = np.log(N_vals)
                ln_chi = np.log(chi_vals)
                
                slope, intercept, r_val, _, _ = linregress(ln_N, ln_chi)
                
                for i, L in enumerate(L_vals):
                    color = SPECTRUM_CMAP(0.3 + 0.7 * NORM(L))
                    ax.scatter(ln_N[i], ln_chi[i], color=color, s=80, zorder=5, edgecolor='k')
                
                ax.plot(ln_N, intercept + slope * ln_N, color='k', linestyle='--', 
                        label=r'Fit $\gamma={:.3f}$ ($R^2={:.3f}$)'.format(slope, r_val**2))
                ax.legend(loc='upper left', fontsize=14)

        ax.set_xlim(2.0, 7.5) 
        ax.set_ylim(-1.0, 7.5)
        ax.set_xlabel(r'$\ln(N)$', fontsize=16)
        ax.set_ylabel(r'$\ln(\chi_{\max})$', fontsize=16)
        ax.set_title(f"Susceptibility Scaling: {model} Model | k = {k}", fontsize=18, fontweight='bold')

    ani = FuncAnimation(fig, update, frames=32, repeat=False)
    writer = FFMpegWriter(fps=FPS)
    ani.save(os.path.join(OUTPUT_DIR, filename), writer=writer, dpi=DPI)
    plt.close(fig)

# NEW: Macroscopic L=192 Animation
def make_chi_scaling_large_L_video(model, conn_db, filename):
    print(f"Rendering {filename} (Large Scale L up to 192)...")
    
    target_ks =[1, 2, 8, 32]
    
    # Collect all unique Ls across these ks for this model
    unique_Ls = set()
    for k in target_ks:
        if model in conn_db and k in conn_db[model]:
            unique_Ls.update(conn_db[model][k].keys())
    unique_Ls = sorted(list(unique_Ls))
    
    if not unique_Ls:
        print(f"[!] No data for large L video for {model}. Skipping.")
        return
        
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {1: 'tab:blue', 2: 'tab:orange', 8: 'tab:green', 32: 'tab:red'}
    
    def update(frame_idx):
        ax.clear()
        L_max = unique_Ls[frame_idx]
        
        for k in target_ks:
            if model not in conn_db or k not in conn_db[model]: continue
            
            L_vals =[]
            chi_vals = []
            for L in sorted(conn_db[model][k].keys()):
                if L <= L_max:
                    L_vals.append(L)
                    chi_vals.append(conn_db[model][k][L]['mean_chi'])
                    
            if not L_vals: continue
            
            N_vals = np.array([(L+1)**3 for L in L_vals])
            ln_N = np.log(N_vals)
            ln_chi = np.log(chi_vals)
            
            # Scatter Points
            ax.scatter(ln_N, ln_chi, color=colors[k], s=80, zorder=5, edgecolor='k')
            
            # Linear Fit
            if len(L_vals) >= 2:
                # Prioritize large L points for accurate scaling fit if available
                fit_idx =[i for i, L in enumerate(L_vals) if L > 10]
                if len(fit_idx) < 2:
                    fit_idx = list(range(len(L_vals)))
                    
                slope, intercept, r_val, _, _ = linregress(ln_N[fit_idx], ln_chi[fit_idx])
                ax.plot(ln_N, intercept + slope * ln_N, color=colors[k], linestyle='--', linewidth=2.5,
                        label=r'k={} Fit $\gamma={:.3f}$'.format(k, slope))
            else:
                ax.plot([], [], color=colors[k], label=f'k={k}')
                
        ax.set_xlim(1.0, 16.5)
        ax.set_ylim(-1.0, 16.5)
        ax.set_xlabel(r'$\ln(N)$', fontsize=18)
        ax.set_ylabel(r'$\ln(\chi_{\max})$', fontsize=18)
        ax.set_title(f"Macroscopic Susceptibility Scaling: {model} Model up to L={L_max}", fontsize=20, fontweight='bold')
        ax.legend(loc='upper left', fontsize=14)

    # Add hold frames at the very end so the final L=192 result stays on screen
    frame_seq = list(range(len(unique_Ls))) + [len(unique_Ls)-1] * 6 
    
    # 2 FPS ensures we clearly see each scale being injected
    ani = FuncAnimation(fig, update, frames=frame_seq, repeat=False)
    writer = FFMpegWriter(fps=2, metadata=dict(artist='Trenton Lau'))
    ani.save(os.path.join(OUTPUT_DIR, filename), writer=writer, dpi=DPI)
    plt.close(fig)

def make_susceptibility_video(model, conn_db, filename):
    print(f"Rendering {filename}...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sm = plt.cm.ScalarMappable(cmap=SPECTRUM_CMAP, norm=NORM)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('System Size (L)', fontsize=14)
    cbar.set_ticks([1, 3, 6, 8, 10])

    def update(frame):
        k = frame + 1
        ax.clear()
        
        max_chi_overall = 0
        if model in conn_db and k in conn_db[model]:
            for L in range(1, 11):
                if L in conn_db[model][k]:
                    max_chi_overall = max(max_chi_overall, conn_db[model][k][L]['mean_chi'])
                    p_c_arr = np.array(conn_db[model][k][L]['p_c'])
                    mean_chi = conn_db[model][k][L]['mean_chi']
                    
                    if len(p_c_arr) > 10:
                        try:
                            bw = 0.05 / np.log10(L + 2)
                            kde = gaussian_kde(p_c_arr, bw_method=bw)
                            x_range = np.linspace(0.0, 1.0, 300)
                            y_kde = kde(x_range)
                            y_scaled = (y_kde / y_kde.max()) * mean_chi
                            
                            color = SPECTRUM_CMAP(0.3 + 0.7 * NORM(L))
                            ax.plot(x_range, y_scaled, color=color, linewidth=2.0)
                            ax.fill_between(x_range, 0, y_scaled, color=color, alpha=0.1)
                        except:
                            pass

        ax.set_xlim(0.0, 1.0)
        if max_chi_overall > 0:
            ax.set_ylim(0, max_chi_overall * 1.1)
            
        ax.set_xlabel('Edge Density (p)', fontsize=16)
        ax.set_ylabel(r'Susceptibility ($\chi$)', fontsize=16)
        ax.set_title(f"{model} Model Susceptibility Spectrum | k = {k}", fontsize=18, fontweight='bold')

    ani = FuncAnimation(fig, update, frames=32, repeat=False)
    writer = FFMpegWriter(fps=FPS)
    ani.save(os.path.join(OUTPUT_DIR, filename), writer=writer, dpi=DPI)
    plt.close(fig)

def make_rigidity_gap_video(model, conn_db, rigidity_db, filename):
    print(f"Rendering {filename}...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sm = plt.cm.ScalarMappable(cmap=SPECTRUM_CMAP, norm=NORM)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('System Size (L)', fontsize=14)
    cbar.set_ticks([1, 3, 6, 8, 10])

    y_vals = np.linspace(0.01, 1.0, 100)
    custom_lines =[Line2D([0], [0], color='k', linestyle='-', lw=2.5),
                    Line2D([0], [0], color='k', linestyle='--', lw=2.5)]

    def update(frame):
        k = frame + 1
        ax.clear()
        
        if model in conn_db and k in conn_db[model]:
            for L in range(1, 11):
                if L in conn_db[model][k]:
                    color = SPECTRUM_CMAP(0.3 + 0.7 * NORM(L))
                    
                    mean_cross = conn_db[model][k][L]['mean_crossings']
                    ax.plot(mean_cross, y_vals, color=color, linewidth=2.0, linestyle='-')
                    
                    if model in rigidity_db and k in rigidity_db[model] and L in rigidity_db[model][k]:
                        p_rig_arr = np.array(rigidity_db[model][k][L])
                        if len(p_rig_arr) > 5:
                            p_rig_arr.sort()
                            cdf = np.arange(1, len(p_rig_arr) + 1) / len(p_rig_arr)
                            ax.plot(p_rig_arr, cdf, color=color, linewidth=2.5, linestyle='--')

        ax.set_xlim(0.05, 1.0)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('Edge Density (p)', fontsize=16)
        ax.set_ylabel(r'Fraction ($S_{\max}/N$ & $S_{\max}^{\mathrm{rigid}}/N$)', fontsize=16)
        ax.set_title(f"{model} Rigidity-Connectivity Gap | k = {k}", fontsize=18, fontweight='bold')
        ax.legend(custom_lines,['Connectivity', 'Relaxed Rigidity'], loc='lower right', fontsize=14)

    ani = FuncAnimation(fig, update, frames=32, repeat=False)
    writer = FFMpegWriter(fps=FPS)
    ani.save(os.path.join(OUTPUT_DIR, filename), writer=writer, dpi=DPI)
    plt.close(fig)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    conn_db, rigidity_db = load_data()

    # 1-2. Order Parameter Videos
    make_order_video("NN", conn_db, "Video1_nn_order.mp4")
    make_order_video("Intra", conn_db, "Video2_intra_order.mp4")
    
    # 3-4. Standard Chi Scaling Videos (L=1 to 10)
    make_chi_scaling_video("NN", conn_db, "Video3_nn_chi.mp4")
    make_chi_scaling_video("Intra", conn_db, "Video4_intra_chi.mp4")
    
    # 5-6. MACROSCOPIC Chi Scaling Videos (L=1 to 192) - NEW!
    make_chi_scaling_large_L_video("NN", conn_db, "Video5_nn_chi_largeL.mp4")
    make_chi_scaling_large_L_video("Intra", conn_db, "Video6_intra_chi_largeL.mp4")
    
    # 7-8. Susceptibility Density Spectra
    make_susceptibility_video("NN", conn_db, "Video7_nn_sus.mp4")
    make_susceptibility_video("Intra", conn_db, "Video8_intra_sus.mp4")
    
    # 9-10. Rigidity Gap Comparison
    make_rigidity_gap_video("NN", conn_db, rigidity_db, "Video9_nn_rigiditygap.mp4")
    make_rigidity_gap_video("Intra", conn_db, rigidity_db, "Video10_intra_rigiditygap.mp4")

    print("\n" + "="*80)
    print(f" ALL 10 VIDEOS SUCCESSFULLY RENDERED!")
    print(f" Check the folder: {OUTPUT_DIR}")
    print("="*80)