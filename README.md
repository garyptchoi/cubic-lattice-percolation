# Explosive Connectivity and Mechanical Rigidity in Cubic Lattice Structures

<img src = "https://github.com/garyptchoi/cubic-lattice-percolation/blob/main/cover.png" height="400"/>

This repository contains the numerical simulation pipeline (designed for massive-scale parallel computing), mathematical validators, data processing scripts, and SI video rendering code for our study on explosive connectivity and mechanical rigidity transitions in 3D cubic lattices under Achlioptas-type product-rule dynamics.

## Repository Structure

*   `simulations/`
    *   `simulate_thermodynamics.py`: Simulates the connectivity phase transition up to macroscopic scales ($L=192, N \approx 7.2 \times 10^6$). Features an ultra-fast `FastDSU` (Disjoint Set Union) class for tracking peak susceptibility, macroscopic jumps, and high-resolution phase crossings.
    *   `simulate_relaxed_rigidity.py`: Simulates the mechanical rigidity transition. Features an exact 3D (3,6)-Pebble Game algorithm modified with relaxed boundary conditions to safely track bulk macroscopic rigidification without being trapped by finite-size surface anomalies.
*   `data_analysis/`
    *   `generate_statistics_tables.py`: Calculates and prints the raw LaTeX rows for the tables shown in our work.
    *   `unified_analysis.py`: Performs calculations directly on raw CSV simulation data to plot the overall thermodynamic mechanics grid.
*   `plotting/`
    *   `plot_thermodynamics.py`: Generates separate figures for phase transitions, jumps, and scaling fits.
    *   `plot_fss_asymptotic.py`: Generates peak susceptibility scaling plots calculated strictly for system sizes $L \ge 24$.
    *   `plot_rigidity_gap.py`: Generates the 8-panel rigidity-connectivity gap transition comparison.
*   `theoretical_validation/`
    *   `plot_legend_only.py`: Standalone editable vector legends.
    *   `plot_ultimate_phase_curves_all_L.py`: Generates the complete spectrum of curves up to $L = 192$.
    *   `validate_lemma.py`: Complete mathematical validation for structural Lemma IV.32.
*   `videos/`
    *   `generate_all_videos.py`: Animates the 9 Supplementary Information movies.

---

## Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/garyptchoi/cubic-lattice-percolation.git
   cd cubic-lattice-percolation
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## Replication Guide

### 1. Dataset Placement
The calculations and plots rely on datasets and visual assets generated from our massive 20,000-repetition campaign. Download `data.zip` and extract the data files into the local `./data/` folder:
```text
cubic-lattice-percolation/
└── data/
    ├── optimized_campaign_cache_fast.pkl     # Connectivity cache for plotting
    ├── optimized_campaign_cache_20k_v5.pkl   # High-statistics video cache
    ├── Legend_System_Size.png                # Custom colorbar legend
    └── V1_Rigidity_20k.tar.gz                # 20k-repetition rigidity dataset
```

### 2. Generating Tables and Correlation Tests
To calculate the statistical rows for the paper tables and perform the Spearman rank correlation proofs, run:
```bash
python data_analysis/generate_statistics_tables.py
```

### 3. Generating Figures
To generate the plotting curves and scale-invariant figures, execute the respective plotting utilities. All outputs will be saved in vector (`.pdf`) and raster (`.png`) formats inside the `./plots` directory:
```bash
# Thermodynamics (Transitions, Jumps, and Bimodality)
python plotting/plot_thermodynamics.py

# Asymptotic FSS scaling (L >= 24)
python plotting/plot_fss_asymptotic.py

# Rigidity-Connectivity Gap
python plotting/plot_rigidity_gap.py

```

### 4. Generating Supplementary Videos
To generate the 9 Supplementary Information animations and stand-alone legends, run:
```bash
python videos/generate_all_videos.py
```
*Note: The generated animations will be exported under the revised SI numbering sequence directly to the `./videos` folder.*

### 5. Running New Simulations
If you wish to execute your own simulation campaign, run the high-performance computing engines directly. The outputs are saved to their respective subdirectories:
```bash
# Run multi-core connectivity simulations
python simulations/simulate_thermodynamics.py --runs 20000

# Run multi-core relaxed rigidity simulations
python simulations/simulate_relaxed_rigidity.py
```

---

## Citation

If you use this code or our datasets in your research, please cite our paper:

```bibtex
@article{lau2025explosive,
    author    = {Lau, Trenton and Choi, Gary P. T.},
    title     = {Explosive connectivity and mechanical rigidity in cubic lattice structures},
    journal   = {arXiv preprint arXiv:2511.01537},
    year      = {2025},
    eprint    = {2511.01537},
    archivePrefix = {arXiv},
    primaryClass = {cond-mat.stat-mech}
}
```

## License

This project is licensed under the Apache-2.0 License.


Copyright (c) 2025-2026, Trenton Lau and Gary P. T. Choi

https://github.com/garyptchoi/cubic-lattice-percolation
