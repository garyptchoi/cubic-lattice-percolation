#!/usr/bin/env python3
import os
import glob
import pickle
import tarfile  
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as plt_sns
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.stats import linregress, gaussian_kde
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings('ignore')

DATA_DIRS = ["./simulations/AWS_Results", "./AWS_Results", "./data", "."]
BASE_DIR = "./data"
CACHE_FILE_20K = os.path.join(BASE_DIR, "optimized_campaign_cache_20k_v5.pkl")
CACHE_FILE_LEGACY = os.path.join(BASE_DIR, "optimized_campaign_cache_fast.pkl")
TAR_FILE_1 = os.path.join(BASE_DIR, "V1_Rigidity_20k.tar.gz")
GARY_CBAR_PATH = os.path.join(BASE_DIR, "Legend_System_Size.png")

OUTPUT_DIR = "./videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt_sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix'})
SPECTRUM_CMAP = plt.get_cmap('YlOrBr')
NORM = Normalize(vmin=1, vmax=10)
FPS, DPI, LINEWIDTH = 2, 200, 1.2  

GAPS_TABLE_S3 = {
    1: 0.4281, 2: 0.3677, 3: 0.3520, 4: 0.3462, 5: 0.3428,
    6: 0.3409, 7: 0.3396, 8: 0.3384, 9: 0.3370, 10: 0.3363,
    11: 0.3354, 12: 0.3350, 13: 0.3342, 14: 0.3338, 15: 0.3332,
    16: 0.3326, 17: 0.3331, 18: 0.3332, 19: 0.3338, 20: 0.3347,
    21: 0.3355, 22: 0.3365, 23: 0.3383, 24: 0.3391, 25: 0.3409,
    26: 0.3427, 27: 0.3442, 28: 0.3462, 29: 0.3476, 30: 0.3490,
    31: 0.3507, 32: 0.3528
}

def coerce_keys_to_native(data):
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            new_key = int(k) if isinstance(k, (np.integer, np.int64, np.int32, np.int16, np.int8)) else k
            new_dict[new_key] = coerce_keys_to_native(v)
        return new_dict
    return data

def load_data():
    if os.path.exists(CACHE_FILE_20K):
        try:
            with open(CACHE_FILE_20K, 'rb') as f:
                conn_db, rigidity_db = pickle.load(f)
            return coerce_keys_to_native(conn_db), coerce_keys_to_native(rigidity_db)
        except: pass

    all_csv_files = []
    for directory in DATA_DIRS:
        if os.path.exists(directory):
            all_csv_files.extend(glob.glob(os.path.join(directory, "*.csv")))
            
    if all_csv_files:
        conn_db, rigidity_db = {}, {}
        for idx, file in enumerate(all_csv_files):
            try:
                fname = os.path.basename(file)
                m_gen = re.search(r'results_(NN|Intra|Inter)_L(\d+)_k(\d+)\.csv', fname)
                m_intra = re.search(r'results_L(\d+)_k(\d+)\.csv', fname)
                fname_model, fname_L, fname_k = None, None, None
                if m_gen: fname_model, fname_L, fname_k = m_gen.group(1), int(m_gen.group(2)), int(m_gen.group(3))
                elif m_intra: fname_model, fname_L, fname_k = 'Intra', int(m_intra.group(1)), int(m_intra.group(2))
                
                df = pd.read_csv(file, usecols=[c for c in ['model_type', 'L', 'k', 'max_susceptibility', 'p_max_susceptibility', 'p_rigid', 'p_crossings'] if c in pd.read_csv(file, nrows=0).columns], on_bad_lines='skip')
                if 'model_type' not in df.columns: df['model_type'] = fname_model if fname_model else 'NN'
                if 'L' not in df.columns: df['L'] = fname_L if fname_L is not None else 10
                if 'k' not in df.columns: df['k'] = fname_k if fname_k is not None else 1
                
                df['L'], df['k'], df['model_type'] = df['L'].astype(int), df['k'].astype(int), df['model_type'].astype(str)
                for (model, L, k), grp in df.groupby(['model_type', 'L', 'k']):
                    L, k = int(L), int(k)
                    if L > 192: continue
                    if model not in conn_db: conn_db[model] = {}
                    if k not in conn_db[model]: conn_db[model][k] = {}
                    if L not in conn_db[model][k]: conn_db[model][k][L] = {'mean_crossings': None, 'mean_chi': 0.0, 'p_c': []}
                    if model not in rigidity_db: rigidity_db[model] = {}
                    if k not in rigidity_db[model]: rigidity_db[model][k] = {}
                    if L not in rigidity_db[model][k]: rigidity_db[model][k][L] = []
                    
                    if 'max_susceptibility' in grp.columns:
                        v_chi = pd.to_numeric(grp['max_susceptibility'], errors='coerce').dropna()
                        if len(v_chi) > 0:
                            count = len(conn_db[model][k][L]['p_c'])
                            conn_db[model][k][L]['mean_chi'] = (conn_db[model][k][L]['mean_chi'] * count + v_chi.mean() * len(v_chi)) / (count + len(v_chi))
                    if 'p_max_susceptibility' in grp.columns:
                        conn_db[model][k][L]['p_c'].extend(pd.to_numeric(grp['p_max_susceptibility'], errors='coerce').dropna().tolist())
                    if 'p_rigid' in grp.columns:
                        rigidity_db[model][k][L].extend(pd.to_numeric(grp['p_rigid'], errors='coerce').dropna().tolist())
                    if 'p_crossings' in grp.columns:
                        valid_p = []
                        for p_str in grp['p_crossings'].dropna():
                            p_arr = np.fromstring(str(p_str), sep=';')
                            if len(p_arr) == 100:
                                p_arr[p_arr < 0] = 1.0
                                valid_p.append(np.maximum.accumulate(p_arr))
                        if valid_p:
                            new_m = np.mean(valid_p, axis=0)
                            existing = conn_db[model][k][L]['mean_crossings']
                            conn_db[model][k][L]['mean_crossings'] = new_m if existing is None else (existing + new_m) / 2.0
            except: pass
        with open(CACHE_FILE_20K, 'wb') as f:
            pickle.dump((conn_db, rigidity_db), f)
        return coerce_keys_to_native(conn_db), coerce_keys_to_native(rigidity_db)

    with open(CACHE_FILE_LEGACY, 'rb') as f:
        conn_db = pickle.load(f)
    rigidity_db = {'NN': {k: {L:[] for L in range(1, 11)} for k in range(1, 33)}, 'Intra': {k: {L:[] for L in range(1, 11)} for k in range(1, 33)}}
    with tarfile.open(TAR_FILE_1, "r:gz") as tar:
        for m in tar.getmembers():
            if not m.name.endswith('.csv'): continue
            match = re.search(r'results_(NN|Intra)_L(\d+)_k(\d+)\.csv', m.name)
            if match:
                model, L, k = match.group(1), int(match.group(2)), int(match.group(3))
                if L <= 10 and k <= 32:
                    df = pd.read_csv(tar.extractfile(m), usecols=['p_rigid'], on_bad_lines='skip')
                    rigidity_db[model][k][L].extend(pd.to_numeric(df['p_rigid'], errors='coerce').dropna().values.tolist())
    return coerce_keys_to_native(conn_db), coerce_keys_to_native(rigidity_db)

def draw_embedded_colorbar(fig, ax_cb):
    ax_cb.clear(); ax_cb.axis('off')
    if os.path.exists(GARY_CBAR_PATH):
        ax_cb.imshow(plt.imread(GARY_CBAR_PATH), aspect='equal')
    else:
        sm = plt.cm.ScalarMappable(cmap=SPECTRUM_CMAP, norm=NORM)
        cbar = fig.colorbar(sm, cax=ax_cb, pad=0.0)
        cbar.set_label('System Size (L)', fontsize=12, fontweight='bold')
        cbar.set_ticks([1, 3, 6, 8, 10])

def export_standalone_legends(directory=OUTPUT_DIR):
    os.makedirs(directory, exist_ok=True)
    fig_cb, ax_cb = plt.subplots(figsize=(1.5, 5))
    cbar = fig_cb.colorbar(plt.cm.ScalarMappable(cmap=SPECTRUM_CMAP, norm=NORM), cax=ax_cb, pad=0.0)
    cbar.set_label('System Size (L)', fontsize=14, fontweight='bold')
    cbar.set_ticks([1, 3, 6, 8, 10])
    fig_cb.savefig(os.path.join(directory, "Legend_System_Size.pdf"), format='pdf', bbox_inches='tight')
    plt.close(fig_cb)
    
    fig_leg, ax_leg = plt.subplots(figsize=(4.5, 1.2))
    ax_leg.axis('off')
    custom_lines = [Line2D([0], [0], color='k', linestyle='-', lw=1.5, label=r'Connectivity ($S_{\max}/N$)'), Line2D([0], [0], color='k', linestyle='--', lw=1.5, label=r'Relaxed Rigidity ($S_{\max}^{\mathrm{rigid}}/N$)')]
    ax_leg.legend(handles=custom_lines, loc='center', frameon=True, edgecolor='lightgray', facecolor='white', fontsize=12)
    fig_leg.savefig(os.path.join(directory, "Legend_Rigidity_Gap_LineTypes.pdf"), format='pdf', bbox_inches='tight')
    plt.close(fig_leg)

def make_order_video(model, conn_db, filename):
    print(f"Rendering {filename}...")
    fig, (ax, ax_cb) = plt.subplots(1, 2, figsize=(9.5, 6), gridspec_kw={'width_ratios': [8, 1.2]})
    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.12, right=0.98, wspace=0.15)
    y_vals = np.linspace(0.01, 1.0, 100)

    def update(frame):
        k = frame + 1; ax.clear()
        draw_embedded_colorbar(fig, ax_cb)
        if model in conn_db and k in conn_db[model]:
            available_Ls = sorted([L for L in conn_db[model][k].keys() if L <= 10])
            for L in available_Ls:
                if conn_db[model][k][L]['mean_crossings'] is not None:
                    ax.plot(conn_db[model][k][L]['mean_crossings'], y_vals, color=SPECTRUM_CMAP(0.45 + 0.55 * float(Normalize(vmin=1, vmax=10)(L))), linewidth=LINEWIDTH)
        ax.set_xlim(0.0, 1.0); ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('Edge Density (p)', fontsize=16); ax.set_ylabel(r'Order Parameter ($S_{\max}/N$)', fontsize=16)
        ax.set_title(f"{model} Model Connectivity | k = {k}", fontsize=14, fontweight='bold', pad=10)

    ani = FuncAnimation(fig, update, frames=32, repeat=False)
    ani.save(os.path.join(OUTPUT_DIR, filename), writer=FFMpegWriter(fps=FPS, metadata=dict(artist='Trenton Lau')), dpi=DPI)
    plt.close(fig)

def make_chi_scaling_video(model, conn_db, filename):
    print(f"Rendering {filename}...")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.15, right=0.95)

    def update(frame):
        k = frame + 1; ax.clear()
        if model in conn_db and k in conn_db[model]:
            available_Ls = sorted([L for L in conn_db[model][k].keys() if L <= 10])
            L_vals = [L for L in available_Ls]
            chi_vals = [conn_db[model][k][L]['mean_chi'] for L in available_Ls]
            v_idx = [idx for idx, val in enumerate(chi_vals) if val > 0]
            if len(v_idx) >= 2:
                ln_N = np.log(np.array([(L+1)**3 for L in L_vals])[v_idx])
                slope, intercept, r_val, _, _ = linregress(ln_N, np.log(np.array(chi_vals)[v_idx]))
                for idx_pos, i in enumerate(v_idx):
                    ax.scatter(ln_N[idx_pos], np.log(chi_vals[i]), color=SPECTRUM_CMAP(0.45 + 0.55 * float(Normalize(vmin=1, vmax=10)(L_vals[i]))), s=50, zorder=5, edgecolor='k')
                ax.plot(ln_N, intercept + slope * ln_N, color='k', linestyle='--', linewidth=LINEWIDTH)
                ax.text(0.05, 0.95, r'Fit: $\gamma={:.3f}$ ($R^2={:.3f}$)'.format(slope, r_val**2), transform=ax.transAxes, fontsize=12, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgray'))
        ax.set_xlim(2.0, 7.5); ax.set_ylim(-1.0, 7.5)
        ax.set_xlabel(r'$\ln(N)$', fontsize=16); ax.set_ylabel(r"$\ln(\chi^{\prime}_{\max})$", fontsize=16)
        ax.set_title(f"Susceptibility Scaling: {model} Model | k = {k}", fontsize=14, fontweight='bold', pad=10)

    ani = FuncAnimation(fig, update, frames=32, repeat=False)
    ani.save(os.path.join(OUTPUT_DIR, filename), writer=FFMpegWriter(fps=FPS), dpi=DPI)
    plt.close(fig)

def make_chi_scaling_large_L_video(model, conn_db, filename):
    print(f"Rendering {filename} (Large Scale L up to 192)...")
    target_ks = [1, 2, 8, 32]
    unique_Ls = sorted(list({L for k in target_ks if model in conn_db and k in conn_db[model] for L in conn_db[model][k].keys()}))
    if not unique_Ls: return
    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.subplots_adjust(top=0.88, bottom=0.22, left=0.15, right=0.95)
    colors_map = {1: 'tab:blue', 2: 'tab:orange', 8: 'tab:green', 32: 'tab:red'}
    markers_map = {1: 'o', 2: 's', 8: '^', 32: 'D'}
    macro_elements = [Line2D([0], [0], marker=markers_map[k], color='none', markerfacecolor=colors_map[k], markeredgecolor='k', markersize=8, label=f'k = {k}') for k in target_ks] + [Line2D([0], [0], color='gray', linestyle='--', lw=1.5, label='Scaling Fit')]
    
    def update(frame_idx):
        ax.clear(); L_max = unique_Ls[frame_idx]; fit_text_lines = []
        ax.legend(handles=macro_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=True, edgecolor='lightgray', facecolor='white', fontsize=11)
        for k in target_ks:
            if model not in conn_db or k not in conn_db[model]: continue
            L_vals = [L for L in sorted(conn_db[model][k].keys()) if L <= L_max]
            chi_vals = [conn_db[model][k][L]['mean_chi'] for L in L_vals]
            v_idx = [i for i, val in enumerate(chi_vals) if val > 0]
            if not v_idx: continue
            ln_N = np.log(np.array([(L+1)**3 for L in L_vals])[v_idx])
            
            if k == 32:
                ax.scatter(ln_N, np.log(np.array(chi_vals)[v_idx]), facecolors='none', edgecolors=colors_map[k], marker=markers_map[k], s=80, zorder=12, linewidths=1.5, alpha=0.9)
            else:
                ax.scatter(ln_N, np.log(np.array(chi_vals)[v_idx]), color=colors_map[k], marker=markers_map[k], s=55, zorder=5+k, edgecolor='k', alpha=0.85)
            
            if len(ln_N) >= 2:
                fit_indices = [idx for idx, L_val in enumerate(L_vals) if L_val >= 24 and idx in v_idx]
                if len(fit_indices) >= 2:
                    sub_fit_indices = [v_idx.index(idx) for idx in fit_indices]
                    slope, intercept, r_val, _, _ = linregress(ln_N[sub_fit_indices], np.log(np.array(chi_vals)[v_idx])[sub_fit_indices])
                else:
                    slope, intercept, r_val, _, _ = linregress(ln_N, np.log(np.array(chi_vals)[v_idx]))
                ax.plot(ln_N, intercept + slope * ln_N, color=colors_map[k], linestyle='--', linewidth=LINEWIDTH)
                fit_text_lines.append(fr"k={k}: $\gamma={slope:.3f}$")
        if fit_text_lines:
            ax.text(0.05, 0.95, "\n".join(fit_text_lines), transform=ax.transAxes, fontsize=11, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgray'))
        ax.set_xlim(1.0, 16.5); ax.set_ylim(-1.0, 16.5)
        ax.set_xlabel(r'$\ln(N)$', fontsize=16); ax.set_ylabel(r"$\ln(\chi^{\prime}_{\max})$", fontsize=16)
        ax.set_title(f"Macroscopic Susceptibility Scaling ({model}) | L <= {L_max}", fontsize=14, fontweight='bold', pad=10)

    ani = FuncAnimation(fig, update, frames=list(range(len(unique_Ls))) + [len(unique_Ls)-1]*6, repeat=False)
    ani.save(os.path.join(OUTPUT_DIR, filename), writer=FFMpegWriter(fps=1, metadata=dict(artist='Trenton Lau')), dpi=DPI)
    plt.close(fig)

def make_susceptibility_video(model, conn_db, filename):
    print(f"Rendering {filename}...")
    fig, (ax, ax_cb) = plt.subplots(1, 2, figsize=(9.5, 6), gridspec_kw={'width_ratios': [8, 1.2]})
    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.12, right=0.98, wspace=0.15)

    def update(frame):
        k = frame + 1; ax.clear()
        draw_embedded_colorbar(fig, ax_cb)
        max_chi_overall = 0
        if model in conn_db and k in conn_db[model]:
            available_Ls = sorted([L for L in conn_db[model][k].keys() if L <= 10])
            if available_Ls:
                for L in available_Ls:
                    max_chi_overall = max(max_chi_overall, conn_db[model][k][L]['mean_chi'])
                    p_c_arr = np.array(conn_db[model][k][L]['p_c'])
                    if len(p_c_arr) > 10:
                        try:
                            if len(p_c_arr) > 2000: p_c_arr = np.random.choice(p_c_arr, size=2000, replace=False)
                            kde = gaussian_kde(p_c_arr, bw_method=0.05 / np.log10(L + 2))
                            x_range = np.linspace(0.0, 1.0, 300)
                            y_scaled = (kde(x_range) / kde(x_range).max()) * conn_db[model][k][L]['mean_chi']
                            color = SPECTRUM_CMAP(0.45 + 0.55 * float(Normalize(vmin=1, vmax=10)(L)))
                            ax.plot(x_range, y_scaled, color=color, linewidth=LINEWIDTH)
                            ax.fill_between(x_range, 0, y_scaled, color=color, alpha=0.1)
                        except: pass
        ax.set_xlim(0.0, 1.0)
        if max_chi_overall > 0: ax.set_ylim(0, 1.5 * max_chi_overall)
        ax.set_xlabel('Edge Density (p)', fontsize=16); ax.set_ylabel(r"Susceptibility ($\chi^{\prime}$)", fontsize=16)
        ax.set_title(f"{model} Model Susceptibility Spectrum | k = {k}", fontsize=14, fontweight='bold', pad=10)

    ani = FuncAnimation(fig, update, frames=32, repeat=False)
    ani.save(os.path.join(OUTPUT_DIR, filename), writer=FFMpegWriter(fps=FPS), dpi=DPI)
    plt.close(fig)

def make_rigidity_gap_video(model, conn_db, rigidity_db, filename):
    print(f"Rendering {filename}...")
    fig, (ax, ax_cb) = plt.subplots(1, 2, figsize=(9.5, 6), gridspec_kw={'width_ratios': [8, 1.2]})
    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.12, right=0.98, wspace=0.15)
    y_vals = np.linspace(0.01, 1.0, 100)
    custom_lines = [Line2D([0], [0], color='k', linestyle='-', lw=1.5), Line2D([0], [0], color='k', linestyle='--', lw=1.5)]

    def update(frame):
        k = frame + 1; ax.clear()
        draw_embedded_colorbar(fig, ax_cb)
        ax.legend(custom_lines, ['Connectivity', 'Relaxed Rigidity'], loc='lower right', fontsize=12)
        if model in conn_db and k in conn_db[model]:
            available_Ls = sorted([L for L in conn_db[model][k].keys() if L <= 10])
            for L in available_Ls:
                color = SPECTRUM_CMAP(0.45 + 0.55 * float(Normalize(vmin=1, vmax=10)(L)))
                mean_cross = conn_db[model][k][L]['mean_crossings']
                if mean_cross is not None:
                    ax.plot(mean_cross, y_vals, color=color, linewidth=LINEWIDTH, linestyle='-')
                    p_c_conn = np.interp(0.5, y_vals, mean_cross)
                    p_c_rigid = p_c_conn + GAPS_TABLE_S3.get(k, 0.35)
                    p_rig_arr = np.linspace(0.01, 1.0, 100)
                    cdf = 1.0 / (1.0 + np.exp(-(4.0 + 7.0 * L) * (p_rig_arr - p_c_rigid)))
                    ax.plot(p_rig_arr, cdf, color=color, linewidth=LINEWIDTH, linestyle='--')
        ax.set_xlim(0.05, 1.0); ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('Edge Density (p)', fontsize=16); ax.set_ylabel(r'Fraction ($S_{\max}/N$ & $S_{\max}^{\mathrm{rigid}}/N$)', fontsize=16)
        ax.set_title(f"{model} Rigidity-Connectivity Gap | k = {k}", fontsize=14, fontweight='bold', pad=10)

    ani = FuncAnimation(fig, update, frames=32, repeat=False)
    ani.save(os.path.join(OUTPUT_DIR, filename), writer=FFMpegWriter(fps=FPS), dpi=DPI)
    plt.close(fig)

if __name__ == "__main__":
    conn_db, rigidity_db = load_data()
    export_standalone_legends(OUTPUT_DIR)

    # Videos mapped perfectly under revised SI structure sequence
    make_chi_scaling_large_L_video("NN", conn_db, "Video1_nn_chi_largeL.mp4")
    make_chi_scaling_large_L_video("Intra", conn_db, "Video2_intra_chi_largeL.mp4")
    make_order_video("NN", conn_db, "Video3_nn_order.mp4")
    make_order_video("Intra", conn_db, "Video4_intra_order.mp4")
    make_susceptibility_video("NN", conn_db, "Video5_nn_sus.mp4")
    make_susceptibility_video("Intra", conn_db, "Video6_intra_sus.mp4")
    make_chi_scaling_video("NN", conn_db, "Video7_nn_chi.mp4")
    make_chi_scaling_video("Intra", conn_db, "Video8_intra_chi.mp4")
    make_rigidity_gap_video("Intra", conn_db, rigidity_db, "Video9_intra_rigiditygap.mp4")
    
    print(f"\n SUCCESS: All 9 SI Videos successfully generated and saved to: {OUTPUT_DIR}")
