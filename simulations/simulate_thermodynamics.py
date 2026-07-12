#!/usr/bin/env python3
import os
import sys
import time
import random
import hashlib
import csv
import multiprocessing as mp
import traceback
import gc

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    import numpy as np
except ImportError:
    print("FATAL ERROR: 'numpy' is required.")
    sys.exit(1)

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
        while self.parent[path] != path: path = self.parent[path]
        curr = i
        while self.parent[curr] != curr:
            nxt = self.parent[curr]
            self.parent[curr] = path
            curr = nxt
        return path

    def union(self, i: int, j: int) -> bool:
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j:
            if self.size[root_i] < self.size[root_j]: root_i, root_j = root_j, root_i
            si, sj = float(self.size[root_i]), float(self.size[root_j])
            self.sum_sq -= (si * si + sj * sj)
            self.parent[root_j] = root_i
            new_size_int = self.size[root_i] + self.size[root_j]
            self.size[root_i] = new_size_int
            new_size_float = float(new_size_int)
            self.sum_sq += (new_size_float * new_size_float)
            
            if new_size_int > self.max_size:
                jump = new_size_int - self.max_size
                if jump > self.max_jump: self.max_jump = jump
                self.max_size = new_size_int
            return True
        return False

def generate_edges_fast(L, model_type="NN"):
    n = np.int32(L + 1)
    n_sq = n * n
    offs = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if model_type == "Intra":
        offs.extend([(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)])
        offs.extend([(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)])
    
    M = 0
    for dx, dy, dz in offs:
        M += (n - abs(dx)) * (n - abs(dy)) * (n - abs(dz))
        
    edges = np.empty((M, 2), dtype=np.int32)
    idx = 0
    for dx, dy, dz in offs:
        x = np.arange(max(0, -dx), min(n, n - dx), dtype=np.int32)
        y = np.arange(max(0, -dy), min(n, n - dy), dtype=np.int32)
        z = np.arange(max(0, -dz), min(n, n - dz), dtype=np.int32)
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        u = X + n*Y + n_sq*Z
        v = (X + np.int32(dx)) + n*(Y + np.int32(dy)) + n_sq*(Z + np.int32(dz))
        block_size = X.size
        edges[idx:idx+block_size, 0] = u.ravel()
        edges[idx:idx+block_size, 1] = v.ravel()
        idx += block_size
    return edges, int(n**3)

def run_simulation(task_params):
    model_type, L, k_ep, run_idx = task_params
    t0 = time.time()
    try:
        seed_str = f"AWS_Campaign_{model_type}_L{L}_k{k_ep}_run{run_idx}"
        seed_int = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16)
        random.seed(seed_int)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed_int)))
        
        edges, N = generate_edges_fast(L, model_type)
        M = len(edges)
        rng.shuffle(edges)
        
        dsu = FastDSU(N)
        current_ptr, step = M - 1, 0
        max_susceptibility = -1.0
        p_max_susceptibility = -1.0
        p_max_jump = -1.0
        last_max_jump, total_mixed, violations = 0, 0, 0
        p_crossings = [-1.0] * 100
        next_percentile_idx = 1
        
        while current_ptr >= k_ep - 1:
            step += 1
            if k_ep == 1:
                best_idx = current_ptr
            else:
                cand_indices = random.sample(range(current_ptr + 1), k_ep)
                min_score = float('inf')
                best_cands = []
                has_intra, has_inter = False, False
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
                if has_intra and has_inter:
                    total_mixed += 1
                    if chosen_is_intra: violations += 1

            u, v = edges[best_idx]
            if dsu.union(u, v):
                chi_exclusive = (dsu.sum_sq - float(dsu.max_size)**2) / float(N)
                if chi_exclusive > max_susceptibility:
                    max_susceptibility = chi_exclusive
                    p_max_susceptibility = float(step) / float(M)
                if dsu.max_jump > last_max_jump:
                    last_max_jump = dsu.max_jump
                    p_max_jump = float(step) / float(M)
                
                s_frac = float(dsu.max_size) / float(N)
                while next_percentile_idx <= 100 and s_frac >= (next_percentile_idx / 100.0):
                    p_crossings[next_percentile_idx - 1] = float(step) / float(M)
                    next_percentile_idx += 1
            
            edges[best_idx] = edges[current_ptr]
            current_ptr -= 1
            if next_percentile_idx > 100: break
            
        p_crossings_str = ";".join(f"{x:.9f}" for x in p_crossings)
        return (run_idx, model_type, L, k_ep, max_susceptibility, p_max_susceptibility, float(dsu.max_jump)/float(N), p_max_jump, total_mixed, violations, time.time() - t0, p_crossings_str)
    except Exception as e:
        return (run_idx, model_type, L, k_ep, -1.0, -1.0, -1.0, -1.0, -1, -1, -1.0, "")
    finally:
        if 'edges' in locals(): del edges
        if 'dsu' in locals(): del dsu
        gc.collect()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AWS Thermodynamic Campaign")
    parser.add_argument("--output_dir", type=str, default="./simulations/AWS_Results", help="Directory for CSV files")
    parser.add_argument("--runs", type=int, default=20000, help="Number of repetitions per config")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_types = ["NN", "Intra"]
    L_values = [24, 32, 48, 64, 96, 128, 160, 192]
    k_values = [1, 2, 8, 32]
    HEADERS = ["run_idx", "model_type", "L", "k", "max_susceptibility", "p_max_susceptibility", "max_jump_frac", "p_max_jump", "total_mixed_samples", "violations", "compute_time_sec", "p_crossings"]
    
    print(f" AWS THERMODYNAMIC ENGINE STARTED\n Cores Detected: {os.cpu_count()}\n")
    with mp.Pool(processes=os.cpu_count(), maxtasksperchild=10) as pool:
        for model_type in model_types:
            for L in L_values:
                for k in k_values:
                    csv_path = os.path.join(args.output_dir, f"results_{model_type}_L{L}_k{k}.csv")
                    completed_runs = set()
                    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                        valid_rows = []
                        with open(csv_path, 'r') as f:
                            reader = csv.reader(f)
                            next(reader, None)
                            for r in reader:
                                if r and len(r) == 12:
                                    completed_runs.add(int(r[0]))
                                    valid_rows.append(r)
                        with open(csv_path + ".tmp", 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(HEADERS)
                            writer.writerows(valid_rows)
                        os.replace(csv_path + ".tmp", csv_path)
                    else:
                        with open(csv_path, 'w', newline='') as f:
                            csv.writer(f).writerow(HEADERS)

                    tasks_to_run = [(model_type, L, k, i) for i in range(args.runs) if i not in completed_runs]
                    if not tasks_to_run: continue
                    
                    t_batch_start = time.time()
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        for i, result in enumerate(pool.imap_unordered(run_simulation, tasks_to_run, chunksize=1)):
                            writer.writerow(result)
                            f.flush()
                            if (i + 1) % max(1, len(tasks_to_run) // 20) == 0 or (i + 1) == len(tasks_to_run):
                                elapsed = time.time() - t_batch_start
                                rate = (i + 1) / elapsed
                                print(f"    [PROGRESS] {(i+1)/len(tasks_to_run)*100:3.0f}% | Done: {i+1:,} | Speed: {rate:5.1f} runs/sec")

if __name__ == "__main__":
    mp.freeze_support()
    main()