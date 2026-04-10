#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import random
import hashlib
import csv
import multiprocessing as mp
import traceback
import gc

# STRICT HPC COMPLIANCE: Disable multi-threading in backend C-libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    import numpy as np
except ImportError:
    print("FATAL ERROR: 'numpy' is required.", flush=True)
    sys.exit(1)

# ==============================================================================
# 1. CORE ALGORITHMS (O(1) Physics Tracking)
# ==============================================================================
class FastDSU:
    __slots__ = ("parent", "size", "max_size", "max_jump", "sum_sq")
    
    def __init__(self, N: int):
        self.parent = np.arange(N, dtype=np.int32)
        self.size = np.ones(N, dtype=np.int32)
        self.max_size = 1
        self.max_jump = 0
        self.sum_sq = float(N)

    def find(self, i: int) -> int:
        path = i
        while self.parent[path] != path:
            path = self.parent[path]
        curr = i
        while self.parent[curr] != curr:
            nxt = self.parent[curr]
            self.parent[curr] = path
            curr = nxt
        return path

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            
            # O(1) Susceptibility tracking
            si = float(self.size[root_i])
            sj = float(self.size[root_j])
            self.sum_sq -= (si * si + sj * sj)
            
            self.parent[root_j] = root_i
            new_size_int = self.size[root_i] + self.size[root_j]
            self.size[root_i] = new_size_int
            
            new_size_float = float(new_size_int)
            self.sum_sq += (new_size_float * new_size_float)
            
            # Atomic Max Jump tracking
            if new_size_int > self.max_size:
                jump = new_size_int - self.max_size
                if jump > self.max_jump:
                    self.max_jump = jump
                self.max_size = new_size_int
                
            return True
        return False

def generate_edges_fast(L, model_type="NN"):
    """
    ULTRA-MEMORY-SAFE edge generation for NN (Shell 1) and Intra (S1-S3) lattices.
    """
    n = np.int32(L + 1)
    n_sq = n * n
    
    # Nearest-Neighbor (Axis edges) - 3 directions
    offs =[(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    
    if model_type == "Intra":
        # Face diagonals (Shell 2) - 6 directions
        offs.extend([(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)])
        # Body diagonals (Shell 3) - 4 directions
        offs.extend([(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)])
    
    # Calculate exact M to prevent memory overallocation
    M = 0
    for dx, dy, dz in offs:
        x_len = n - abs(dx)
        y_len = n - abs(dy)
        z_len = n - abs(dz)
        M += x_len * y_len * z_len
        
    edges = np.empty((M, 2), dtype=np.int32)
    
    idx = 0
    for dx, dy, dz in offs:
        x_min = max(0, -dx); x_max = min(n, n - dx)
        y_min = max(0, -dy); y_max = min(n, n - dy)
        z_min = max(0, -dz); z_max = min(n, n - dz)
        
        x = np.arange(x_min, x_max, dtype=np.int32)
        y = np.arange(y_min, y_max, dtype=np.int32)
        z = np.arange(z_min, z_max, dtype=np.int32)
        
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        
        u = X + n*Y + n_sq*Z
        v = (X + np.int32(dx)) + n*(Y + np.int32(dy)) + n_sq*(Z + np.int32(dz))
        
        block_size = X.size
        edges[idx:idx+block_size, 0] = u.ravel()
        edges[idx:idx+block_size, 1] = v.ravel()
        idx += block_size
        
    return edges, int(n**3)

# ==============================================================================
# 2. ISOLATED WORKER PROCESS
# ==============================================================================
def run_simulation(task_params):
    model_type, L, k_ep, run_idx = task_params
    t0 = time.time()
    
    try:
        # Cryptographically secure PRNG seeding for exact reproducibility
        seed_str = f"AWS_Campaign_{model_type}_L{L}_k{k_ep}_run{run_idx}"
        seed_int = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16)
        
        random.seed(seed_int)
        sq = np.random.SeedSequence(seed_int)
        rng = np.random.Generator(np.random.PCG64(sq))

        edges, N = generate_edges_fast(L, model_type)
        M = len(edges)
        
        rng.shuffle(edges)
        
        dsu = FastDSU(N)
        current_ptr = M - 1
        step = 0
        
        max_susceptibility = -1.0
        p_max_susceptibility = -1.0
        p_max_jump = -1.0
        last_max_jump = 0
        total_mixed = 0
        violations = 0
        
        p_crossings = [-1.0] * 100
        next_percentile_idx = 1
        
        while current_ptr >= k_ep - 1:
            step += 1
            
            if k_ep == 1:
                best_idx = current_ptr
            else:
                cand_indices = random.sample(range(current_ptr + 1), k_ep)
                min_score = float('inf')
                best_cands =[]
                has_intra = False
                has_inter = False
                
                for idx in cand_indices:
                    u, v = edges[idx]
                    ru, rv = dsu.find(u), dsu.find(v)
                    is_intra = (ru == rv)
                    
                    if is_intra: has_intra = True
                    else: has_inter = True
                    
                    score = float(dsu.size[ru]) * float(dsu.size[rv])
                    
                    if score < min_score:
                        min_score = score
                        best_cands = [(idx, is_intra)]
                    elif score == min_score:
                        best_cands.append((idx, is_intra))
                
                best_idx, chosen_is_intra = random.choice(best_cands)
                
                # Verify Mixed-Sample assumption for the paper
                if has_intra and has_inter:
                    total_mixed += 1
                    if chosen_is_intra:
                        violations += 1

            u, v = edges[best_idx]
            
            if dsu.union(u, v):
                # Calculate True Thermodynamic Susceptibility (excluding Giant Component)
                chi_exclusive = (dsu.sum_sq - float(dsu.max_size)**2) / float(N)
                if chi_exclusive > max_susceptibility:
                    max_susceptibility = chi_exclusive
                    p_max_susceptibility = float(step) / float(M)
                
                if dsu.max_jump > last_max_jump:
                    last_max_jump = dsu.max_jump
                    p_max_jump = float(step) / float(M)
                
                # High-Resolution Transition Width Tracking
                s_frac = float(dsu.max_size) / float(N)
                while next_percentile_idx <= 100 and s_frac >= (next_percentile_idx / 100.0):
                    p_crossings[next_percentile_idx - 1] = float(step) / float(M)
                    next_percentile_idx += 1
            
            edges[best_idx] = edges[current_ptr]
            current_ptr -= 1
            
            if next_percentile_idx > 100:
                break
        
        max_jump_frac = float(dsu.max_jump) / float(N)
        timing = time.time() - t0
        
        p_crossings_str = ";".join(f"{x:.9f}" for x in p_crossings)
        
        return (
            run_idx, model_type, L, k_ep,
            max_susceptibility, p_max_susceptibility, 
            max_jump_frac, p_max_jump, 
            total_mixed, violations, 
            timing, p_crossings_str
        )

    except Exception as e:
        err_str = traceback.format_exc()
        # Atomic debug prints that appear instantly in nohup
        print(f"\n[!] CRITICAL ERROR IN RUN {run_idx} ({model_type}, L={L}, k={k_ep}):\n{err_str}\n", flush=True)
        return (run_idx, model_type, L, k_ep, -1.0, -1.0, -1.0, -1.0, -1, -1, -1.0, "")
    
    finally:
        # ABSOLUTE GUARANTEE: Wipe memory instantly to prevent OOM
        if 'edges' in locals(): del edges
        if 'dsu' in locals(): del dsu
        gc.collect()

# ==============================================================================
# 3. AWS MASTER SCHEDULER (Atomic File Operations + Nohup Tracking)
# ==============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="AWS Thermodynamic Campaign")
    parser.add_argument("--output_dir", type=str, default="AWS_Results", help="Directory for CSV files")
    parser.add_argument("--runs", type=int, default=20000, help="Number of repetitions per config")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    model_types =["NN", "Intra"]
    # REMOVED L=224 TO SAVE AWS BUDGET AND TIME.
    L_values =[24, 32, 48, 64, 96, 128, 160, 192]
    k_values =[1, 2, 8, 32]
    
    EXPECTED_COLS = 12
    HEADERS =[
        "run_idx", "model_type", "L", "k", 
        "max_susceptibility", "p_max_susceptibility", 
        "max_jump_frac", "p_max_jump", 
        "total_mixed_samples", "violations", 
        "compute_time_sec", "p_crossings"
    ]
    
    num_workers = os.cpu_count()
    print(f"==================================================", flush=True)
    print(f" AWS THERMODYNAMIC ENGINE RESTARTED (Max L=192)", flush=True)
    print(f" CPU Cores Detected : {num_workers}", flush=True)
    print(f" Models Scheduled   : {model_types}", flush=True)
    print(f" Target Sizes (L)   : {L_values}", flush=True)
    print(f" Choice Params (k)  : {k_values}", flush=True)
    print(f" Repetitions        : {args.runs:,} per condition", flush=True)
    print(f"==================================================\n", flush=True)
    
    # maxtasksperchild=10 forces the worker to cleanly die and respawn, guaranteeing zero memory rot
    with mp.Pool(processes=num_workers, maxtasksperchild=10) as pool:
        for model_type in model_types:
            for L in L_values:
                for k in k_values:
                    print(f"--> [INIT] Starting Batch: Model={model_type} | L={L} | k={k}", flush=True)
                    
                    csv_path = os.path.join(args.output_dir, f"results_{model_type}_L{L}_k{k}.csv")
                    completed_runs = set()
                    
                    # ATOMIC RESUME LOGIC: Safe against AWS Spot Preemptions
                    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                        valid_rows =[]
                        with open(csv_path, 'r') as f:
                            reader = csv.reader(f)
                            header = next(reader, None)
                            for row_idx, row in enumerate(reader):
                                if row and len(row) == EXPECTED_COLS:
                                    try:
                                        run_id = int(row[0])
                                        completed_runs.add(run_id)
                                        valid_rows.append(row)
                                    except ValueError:
                                        print(f"    [WARNING] Invalid run_idx at line {row_idx + 2}. Skipping.", flush=True)
                                else:
                                    print(f"    [WARNING] Truncating corrupted data at line {row_idx + 2}.", flush=True)
                        
                        # Write to a tmp file, force disk sync, and atomically swap
                        tmp_path = csv_path + ".tmp"
                        with open(tmp_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(HEADERS)
                            writer.writerows(valid_rows)
                            f.flush()
                            os.fsync(f.fileno())
                        os.replace(tmp_path, csv_path)
                    else:
                        with open(csv_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(HEADERS)

                    tasks_to_run =[(model_type, L, k, i) for i in range(args.runs) if i not in completed_runs]
                    
                    if not tasks_to_run:
                        print(f"    [SKIP] Already completed {args.runs} runs. Output safely preserved in: {csv_path}\n", flush=True)
                        continue
                    
                    print(f"    [DISPATCH] Distributing {len(tasks_to_run):,} tasks across {num_workers} cores...", flush=True)
                    
                    t_batch_start = time.time()
                    
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        
                        # chunksize=1 GUARANTEES perfect modular spreading and dynamic load balancing
                        for i, result in enumerate(pool.imap_unordered(run_simulation, tasks_to_run, chunksize=1)):
                            writer.writerow(result)
                            # Immediate flush so data is secure in OS cache, visible to `tail` instantly
                            f.flush()
                            
                            # Clean, nohup-friendly progress dashboard
                            if (i + 1) % max(1, len(tasks_to_run) // 20) == 0 or (i + 1) == len(tasks_to_run):
                                elapsed = time.time() - t_batch_start
                                rate = (i + 1) / elapsed
                                eta = (len(tasks_to_run) - (i + 1)) / rate
                                print(f"    [PROGRESS] {(i+1)/len(tasks_to_run)*100:3.0f}% | "
                                      f"Done: {i+1:,}/{len(tasks_to_run):,} | "
                                      f"Rate: {rate:5.1f} runs/sec | "
                                      f"ETA: {eta/60:5.1f} mins", flush=True)
                    
                    print(f"--> [DONE] Finished Batch: Model={model_type} | L={L} | k={k} in {(time.time() - t_batch_start)/60:.1f} mins.", flush=True)
                    print(f"           Data securely saved to: {csv_path}\n", flush=True)

    print("==================================================", flush=True)
    print(" ALL CAMPAIGN BATCHES COMPLETED SUCCESSFULLY.", flush=True)
    print("==================================================", flush=True)

if __name__ == "__main__":
    mp.freeze_support()
    main()
