#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
All-in-One 3D Cubic Lattice Percolation Simulation Script
Version: 6.0.2 (2025-08-22 - Added resume capability)

- Starts a fresh campaign or resumes an interrupted one.
- Distributes work across multiple machines.
- Uses all available CPU cores minus one for maximum performance on each machine.

Reference:
Trenton Lau and Gary P. T. Choi, 
Explosive connectivity and mechanical rigidity in cubic lattice structures,
arXiv preprint arXiv:2511.01537, 2025.

Copyright (c) 2025, Trenton Lau and Gary P. T. Choi
https://github.com/garyptchoi/cubic-lattice-percolation
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import multiprocessing as mp
import os
import random
import re
import sys
import time

# Limit threaded math libraries for stability in multiprocessing
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"FATAL ERROR: A required library is missing: {e}", file=sys.stderr)
    print("Please run: pip install numpy pandas", file=sys.stderr)
    sys.exit(1)

# --- Core Simulation Components (Unchanged) ---
class DSUConnectivity:
    __slots__ = ("n", "parent", "size", "edges", "m2_sum")
    def __init__(self, n: int):
        self.n = n; self.parent = np.arange(n, dtype=np.int32); self.size = np.ones(n, dtype=np.int32); self.edges = np.zeros(n, dtype=np.int32); self.m2_sum = float(n)
    def find(self, x: int) -> int:
        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union_add_edge(self, u: int, v: int):
        ru, rv = self.find(u), self.find(v)
        if ru == rv: self.edges[ru] += 1; return
        if self.size[ru] < self.size[rv]: ru, rv = rv, ru
        self.m2_sum -= float(self.size[ru]**2) + float(self.size[rv]**2); self.parent[rv] = ru; self.size[ru] += self.size[rv]; self.edges[ru] += self.edges[rv] + 1; self.m2_sum += float(self.size[ru]**2)
    def largest_component_size(self) -> int:
        roots = np.where(self.parent == np.arange(self.n, dtype=np.int32))[0]
        return int(np.max(self.size[roots])) if roots.size > 0 else 0
    def all_roots(self) -> np.ndarray:
        return np.where(self.parent == np.arange(self.n, dtype=np.int32))[0]

class PebbleGame3DGlobal:
    def __init__(self, n_nodes, k=3, l=6): self.n=n_nodes; self.k=k; self.l=l; self.pebbles=np.full(n_nodes,k,dtype=np.int8); self.adj=[[] for _ in range(n_nodes)]; self._visited=np.zeros(n_nodes,dtype=bool)
    def _dfs_find_pebble(self, u):
        path_map,stack,self._visited[u]={u:None},[(u,iter(self.adj[u]))],True
        while stack:
            curr_u,children=stack[-1]
            try:
                v=next(children)
                if not self._visited[v]:
                    path_map[v]=curr_u
                    if self.pebbles[v]>0:
                        self.pebbles[v]-=1; curr_v,parent=v,path_map.get(v)
                        while parent is not None: self.adj[parent].remove(curr_v); self.adj[curr_v].append(parent); curr_v,parent=parent,path_map.get(parent)
                        return True
                    self._visited[v]=True; stack.append((v,iter(self.adj[v])))
            except StopIteration: stack.pop()
        return False
    def is_independent(self, u, v):
        pebbles_on_hand=self.pebbles[u]+self.pebbles[v]
        if pebbles_on_hand>=1:
            if self.pebbles[u]>0:self.pebbles[u]-=1
            else:self.pebbles[v]-=1
            return True
        self._visited.fill(False); self._visited[u]=self._visited[v]=True
        return self._dfs_find_pebble(u) or self._dfs_find_pebble(v)
    def check_rigidity(self, edges):
        if self.n > 1 and len(edges) < self.k*self.n-self.l: return False, 0
        ind_edges = sum(1 for u,v in edges if self.is_independent(u,v) and self.adj[u].append(v) is None)
        return ind_edges >= self.k*self.n-self.l, len(edges)-ind_edges

def generate_potential_edges(L, model_name):
    n = L + 1
    offs = [(1,0,0),(0,1,0),(0,0,1)] if "NN" in model_name else [(dx,dy,dz) for dz in [-1,0,1] for dy in [-1,0,1] for dx in [-1,0,1] if (dx,dy,dz)!=(0,0,0) and (dx*dx+dy*dy+dz*dz<=3) and ((dz>0) or (dz==0 and dy>0) or (dz==0 and dy==0 and dx>0))]
    edges = {tuple(sorted((x+n*(y+n*z), x+dx+n*(y+dy+n*(z+dz))))) for z,y,x in np.ndindex(n,n,n) for dx,dy,dz in offs if 0<=x+dx<n and 0<=y+dy<n and 0<=z+dz<n}
    return sorted(list(edges))

EP_RULES = {"Std K1": {"type": "standard"}, "Min Prod": {"metric": lambda s_u, s_v: float(s_u * s_v)}}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return super().default(obj)

def run_single_simulation(params):
    """Worker function that runs one full simulation based on a dictionary of parameters."""
    L, k_ep, model_name, ep_rule_name, rigidity_on, run_idx = params.values()
    t0 = time.time()
    rng_seed = int(hashlib.sha256(f"{L}-{k_ep}-{model_name}-{ep_rule_name}-{rigidity_on}-{run_idx}".encode()).hexdigest(), 16) & (2**32-1)
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    N = (L + 1)**3
    edges_list = generate_potential_edges(L, model_name)
    M_total = len(edges_list)
    p_h, s_v_h, s_r_h = [], [], []
    result = {"sim_params": params, "status": "Init", "data": {"p_hist": p_h, "smax_v_hist": s_v_h, "smax_rigid_pg_hist": s_r_h}}
    if M_total == 0:
        return {**result, "status": "Success (No edges)", "timing_sec": time.time() - t0}

    rule, dsu, pool, pos, added = EP_RULES[ep_rule_name], DSUConnectivity(N), list(range(M_total)), {i:i for i in range(M_total)}, []

    for step in range(M_total):
        if k_ep == 1:
            best_idx = pool[random.randrange(len(pool))]
        else:
            num_to_sample = min(k_ep, len(pool))
            candidate_indices = random.sample(pool, num_to_sample)
            best_idx = min(candidate_indices, key=lambda i: EP_RULES[ep_rule_name]["metric"](dsu.size[dsu.find(edges_list[i][0])], dsu.size[dsu.find(edges_list[i][1])]))

        last_i, pos_rem = len(pool) - 1, pos.pop(best_idx)
        if pos_rem < last_i:
            swapped = pool[last_i]; pool[pos_rem] = swapped; pos[swapped] = pos_rem
        pool.pop(); added.append(edges_list[best_idx]); dsu.union_add_edge(*edges_list[best_idx])
        p_h.append((step + 1) / M_total); s_v_h.append(dsu.largest_component_size())
        
        max_r = s_r_h[-1] if step > 0 else 0
        if rigidity_on:
            roots_to_check = {r for r in dsu.all_roots() if dsu.size[r] > max_r and dsu.edges[r] >= 3*dsu.size[r]-6}
            if roots_to_check:
                v_roots = np.array([dsu.find(i) for i in range(N)], dtype=np.int32)
                for r in roots_to_check:
                    comp_v = np.where(v_roots == r)[0]
                    v_map = {g: l for l, g in enumerate(comp_v)}
                    comp_e = [(v_map[u], v_map[v]) for u, v in added if u in v_map and v in v_map]
                    is_rigid, _ = PebbleGame3DGlobal(len(comp_v)).check_rigidity(comp_e)
                    if is_rigid: max_r = max(max_r, len(comp_v))
        s_r_h.append(max_r)
        
    return {**result, "status": "Success", "timing_sec": time.time() - t0}

def save_result(result_and_id):
    """Saves a single simulation result to a JSON file."""
    result, task_id_str, output_dir = result_and_id
    fpath = os.path.join(output_dir, f"{task_id_str}.json")
    try:
        with open(fpath, "w") as f:
            json.dump(result, f, cls=NumpyEncoder)
        return True
    except Exception as e:
        print(f"WARNING: Could not save file for task {task_id_str}. Error: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Final All-in-One 3D Percolation Campaign Script (v6.0.2 - Resume capable)")
    parser.add_argument("--batch_name", type=str, default="FULL_CAMPAIGN_L1-10", help="A name for this simulation batch.")
    parser.add_argument("--output_dir", type=str, default=".", help="Root directory to save results. A subfolder with the batch_name will be created here.")
    parser.add_argument("--L_min", type=int, default=1, help="Minimum lattice size L.")
    parser.add_argument("--L_max", type=int, default=10, help="Maximum lattice size L.")
    parser.add_argument("--k_values", type=str, default=",".join(map(str, range(1, 33))), help="Comma-separated list of k values.")
    parser.add_argument("--runs_per_config", type=int, default=1000, help="Number of simulation runs for each unique parameter set.")
    parser.add_argument("--num_machines", type=int, default=20, help="Total number of machines participating in the campaign.")
    parser.add_argument("--machine_id", type=int, required=True, help="The ID of this machine, from 0 to (num_machines - 1).")
    args = parser.parse_args()

    sane_name = re.sub(r'[^\w\.\-]', '_', args.batch_name)
    results_path = os.path.join(args.output_dir, sane_name, f"machine_{args.machine_id}")
    os.makedirs(results_path, exist_ok=True)
    
    print(f"--- 3D PERCOLATION CAMPAIGN (Resume Capable) ---")
    print(f"Batch Name: {sane_name}")
    print(f"This is Machine ID: {args.machine_id} of {args.num_machines-1}")
    print(f"Results will be saved to: {results_path}")

    print("\nGenerating the full list of tasks for the campaign...")
    k_vals = sorted({int(k) for k in args.k_values.split(',')})
    all_tasks = []
    models = ["NN (Shell 1)", "Intra (S1-S3)"]
    rig_configs = [True, False]
    rules = ["Min Prod"]
    
    for L in range(args.L_min, args.L_max + 1):
        for model in models:
            for k in k_vals:
                for rule in (["Std K1"] if k == 1 else rules):
                    for rig in rig_configs:
                        for run in range(args.runs_per_config):
                            params = {'L': L, 'k_ep': k, 'model_name': model, 'ep_rule_name': rule, 'rigidity_on': rig, 'run_idx': run}
                            safe_model = model.replace(' ', '').replace('(', '').replace(')', '')
                            safe_rule = rule.replace(' ', '')
                            task_id = f"L{L}_k{k}_{safe_model}_{safe_rule}_rig{rig}_run{run}"
                            all_tasks.append((task_id, params))
    print(f"Total tasks in full campaign: {len(all_tasks):,}")

    # --- MODIFIED BLOCK START: Filter out completed tasks to allow resuming ---
    all_my_tasks = [task for i, task in enumerate(all_tasks) if i % args.num_machines == args.machine_id]
    
    print(f"\nThis machine is assigned {len(all_my_tasks):,} tasks in total.")
    print("Checking for existing results to resume...")
    
    my_tasks_to_run = []
    for task_id, params in all_my_tasks:
        expected_fpath = os.path.join(results_path, f"{task_id}.json")
        if not os.path.exists(expected_fpath):
            my_tasks_to_run.append((task_id, params))

    completed_count = len(all_my_tasks) - len(my_tasks_to_run)
    if completed_count > 0:
        print(f"Found {completed_count:,} existing results. Resuming with {len(my_tasks_to_run):,} tasks.")
    else:
        print("Starting a fresh run with no existing results found.")

    if not my_tasks_to_run:
        print("All assigned tasks for this machine have already been completed. Exiting.")
        return
    # --- MODIFIED BLOCK END ---
    
    num_workers = max(1, os.cpu_count() - 1)
    print(f"Starting simulation with {num_workers} parallel worker processes.\n")
    
    start_time = time.time()
    with mp.Pool(processes=num_workers) as pool:
        # We only need the params dictionary for the worker function
        params_to_run = [params for _, params in my_tasks_to_run]
        
        results_iterator = pool.imap_unordered(run_single_simulation, params_to_run)
        
        # Iterate through results as they are completed
        for i, result in enumerate(results_iterator):
            # Re-create the task_id from the result parameters to find the correct filename
            params = result['sim_params']
            safe_model = params['model_name'].replace(' ', '').replace('(', '').replace(')', '')
            safe_rule = params['ep_rule_name'].replace(' ', '')
            task_id_str = f"L{params['L']}_k{params['k_ep']}_{safe_model}_{safe_rule}_rig{params['rigidity_on']}_run{params['run_idx']}"

            save_result((result, task_id_str, results_path))

            total_completed = completed_count + i + 1
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1) if i > 0 else elapsed
                eta = avg_time * (len(my_tasks_to_run) - (i + 1))
                print(f"Progress: {i+1}/{len(my_tasks_to_run)} completed ({total_completed:,} of {len(all_my_tasks)} total) | Avg time/sim: {avg_time:.3f}s | Est. remaining: {dt.timedelta(seconds=int(eta))}")
    
    total_time = time.time() - start_time
    print("\n--- CAMPAIGN FINISHED FOR THIS MACHINE ---")
    print(f"Total time taken for this session: {dt.timedelta(seconds=int(total_time))}")
    print(f"All data for Machine {args.machine_id} is saved in {results_path}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
