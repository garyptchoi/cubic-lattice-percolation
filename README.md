# Explosive Connectivity and Mechanical Rigidity in Cubic Lattice Structures

This repository contains the high-performance simulation and analysis codebase for the paper **"Explosive Connectivity and Mechanical Rigidity in Cubic Lattice Structures"** by Trenton Lau and Gary P. T. Choi.

It provides a highly optimized suite of Python scripts to study percolation phase transitions and mechanical rigidification in 3D lattice structures (Nearest-Neighbor and Intra-cube models) using a $k$-choice Achlioptas product-rule process.

## Repository Structure

The codebase is organized into five modular directories:

### `simulations/`
Contains the core $O(1)$ physics engines designed for massive-scale parallel computing (e.g., 192-core AWS clusters).
* **`simulate_thermodynamics.py`**: Simulates the connectivity phase transition up to macroscopic scales ($L=192, N \approx 7.2 \times 10^6$). Features an ultra-fast `FastDSU` (Disjoint Set Union) class for tracking peak susceptibility, macroscopic jumps, and high-resolution phase crossings.
* **`simulate_relaxed_rigidity.py`**: Simulates the mechanical rigidity transition. Features an exact 3D (3,6)-Pebble Game algorithm modified with relaxed boundary conditions to safely track bulk macroscopic rigidification without being trapped by finite-size surface anomalies.

### `data_analysis/`
Contains the statistical testing and tabular generation scripts.
* **`generate_statistics_tables.py`**: The central statistical engine. It merges the rigidity `.tar.gz` archives with the thermodynamic `.pkl` cache to calculate the exact rigidity-connectivity gap ($\Delta p_c$). It performs Spearman's rank correlation tests to rigorously prove optimal rigidification efficiency and outputs the pre-formatted LaTeX tables (Tables S2 & S3).

### `plotting/`
Contains the data-parsing and visualization engines that generate the figures used in the manuscript.
* **`plot_fast.py`**: A fast C-engine parser that compiles raw CSVs into an optimized `.pkl` cache and generates the Normalized Susceptibility Spectrum (Fig 5).
* **`plot_thermodynamics.py`**: Generates the finite-size scaling plots, max jump stabilization plots, high-resolution phase curves, and bimodality histograms (Figs 3 & 4).
* **`plot_rigidity_gap.py`**: Generates the definitive 8-panel plot comparing the connectivity curves against the relaxed rigidity cumulative distribution functions (Fig 6).

### `theoretical_validation/`
Contains mathematical validation scripts supporting the paper's proofs.
* **`validate_lemma_IV32.py`**: A dedicated testing script utilizing a fully vectorized edge-generation engine to strictly validate the microscopic mixed-sample selection dynamics (Table S1).

### `videos/`
Contains the animation pipeline for the supplementary materials.
* **`generate_all_videos.py`**: A comprehensive `matplotlib.animation` pipeline leveraging FFmpeg to generate all 10 Supplementary MP4 videos. This includes dynamical transitions across choice parameters, small-scale scaling, and macroscopic finite-size scaling confirming the critical exponent $\gamma=1.000$.

## Requirements & Installation

The codebase requires **Python 3.9+** and standard scientific computing libraries. The simulation scripts are heavily optimized to run on multi-core systems (automatically scaling to the maximum available CPU cores via `multiprocessing`).

```bash
pip install numpy pandas scipy matplotlib seaborn



## Citation

If you use this code or data in your work, please cite our preprint:

Trenton Lau and Gary P. T. Choi, 
"[Explosive connectivity and mechanical rigidity in cubic lattice structures.](https://arxiv.org/abs/2511.01537)"
arXiv preprint arXiv:2511.01537, 2025.

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


Copyright (c) 2025, Trenton Lau and Gary P. T. Choi

https://github.com/garyptchoi/cubic-lattice-percolation