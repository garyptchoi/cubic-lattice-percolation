#!/usr/bin/env python3
import os, sys, time, csv
import numpy as np
import multiprocessing as mp
from collections import deque

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

class FastDSU:
    def __init__(self, N):
        self.parent = np.arange(N, dtype=np.int32)
        self.size = np.ones(N, dtype=np.int32)
    def find(self, i):
        path = i
        while self.parent[path] != path: path = self.parent[path]
        curr = i
        while self.parent[curr] != curr:
            nxt = self.parent[curr]
            self.parent[curr] = path
            curr = nxt
        return path
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.size[root_i] < self.size[root_j]: root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            return True
        return False

class PebbleGame3D_Relaxed:
    def __init__(self, N: int):
        self.N = N
        self.pebbles = np.full(N, 3, dtype=np.int32)
        self.adj = [[] for _ in range(N)] 
        self.independent_edges = 0

    def search_and_bring(self, target: int, block1: int, block2: int) -> bool:
        visited = bytearray(self.N)
        visited[target] = 1
        queue = deque([target])
        parent = {target: -1}
        found_node = -1
        while queue:
            curr = queue.popleft()
            if self.pebbles[curr] > 0 and curr != block1 and curr != block2:
                found_node = curr
                break
            for nxt in self.adj[curr]:
                if not visited[nxt]:
                    visited[nxt] = 1
                    parent[nxt] = curr
                    queue.append(nxt)
        if found_node != -1:
            curr = found_node
            self.pebbles[curr] -= 1
            while parent[curr] != -1:
                p = parent[curr]
                self.adj[p].remove(curr)
                self.adj[curr].append(p)
                curr = p
            self.pebbles[target] += 1
            return True
        return False

    def add_edge(self, u: int, v: int, dsu_pg: FastDSU) -> bool:
        if u == v: return False
        if dsu_pg.find(u) != dsu_pg.find(v):
            if self.search_and_bring(u, -1, -1):
                self.pebbles[u] -= 1; self.adj[u].append(v); self.independent_edges += 1; dsu_pg.union(u, v); return True
            elif self.search_and_bring(v, -1, -1):
                self.pebbles[v] -= 1; self.adj[v].append(u); self.independent_edges += 1; dsu_pg.union(u, v); return True
            return False
            
        while self.pebbles[u] < 3:
            if not self.search_and_bring(u, u, v): break
        while self.pebbles[v] < 3:
            if not self.search_and_bring(v, u, v): break
            
        if self.pebbles[u] == 3 and self.pebbles[v] == 3:
            if self.search_and_bring(u, u, v):
                self.pebbles[u] -= 1; self.adj[u].append(v); self.independent_edges += 1; return True
            elif self.search_and_bring(v, u, v):
                self.pebbles[v] -= 1; self.adj[v].append(u); self.independent_edges += 1; return True
        return False

def generate_edges_fast(L):
    n = np.int32(L + 1)
    n_sq = n * n
    offs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1), (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)]
    edges = []
    for dx, dy, dz in offs:
        x = np.arange(max(0, -dx), min(n, n - dx), dtype=np.int32)
        y = np.arange(max(0, -dy), min(n, n - dy), dtype=np.int32)
        z = np.arange(max(0, -dz), min(n, n - dz), dtype=np.int32)
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        u = X + n*Y + n_sq*Z
        v = (X + np.int32(dx)) + n*(Y + np.int32(dy)) + n_sq*(Z + np.int32(dz))
        edges.extend(zip(u.ravel(), v.ravel()))
    return np.array(edges, dtype=np.int32), int(n**3)

def run_rigidity_only(task):
    L, k_ep, run_idx = task
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    edges, N = generate_edges_fast(L)
    M = len(edges)
    edges = edges[np.random.permutation(M)]
    
    dsu_graph = FastDSU(N)
    dsu_pg = FastDSU(N)
    pg = PebbleGame3D_Relaxed(N)
    
    current_ptr = M - 1
    step = 0
    target = 3 * N - 6
    
    while current_ptr >= 0:
        step += 1
        actual_k = min(current_ptr + 1, k_ep)
        
        if actual_k == 1:
            best_idx = current_ptr
        else:
            cand_indices = np.random.choice(current_ptr + 1, actual_k, replace=False)
            min_score = float('inf')
            best_cands = []
            for idx in cand_indices:
                u, v = edges[idx]
                ru, rv = dsu_graph.find(u), dsu_graph.find(v)
                score = float(dsu_graph.size[ru]) * float(dsu_graph.size[rv])
                if score < min_score:
                    min_score = score
                    best_cands = [idx]
                elif score == min_score:
                    best_cands.append(idx)
            best_idx = int(np.random.choice(best_cands))

        u, v = edges[best_idx]
        pg.add_edge(u, v, dsu_pg)
        if pg.independent_edges == target:
            return (run_idx, L, k_ep, float(step) / float(M))
            
        dsu_graph.union(u, v)
        edges[best_idx] = edges[current_ptr]
        current_ptr -= 1
        
    return (run_idx, L, k_ep, -1.0)

def main():
    print("==================================================")
    print(" PRE-FLIGHT CHECK (Testing L=10, k=32)")
    print("==================================================")
    test_res = run_rigidity_only((10, 32, 0))
    if test_res[3] == -1.0:
        print("[!] FATAL BUG DETECTED: Rigidity returned -1.0.")
        sys.exit(1)
    else:
        print(f"[SUCCESS] Pre-flight passed! L=10, k=32 rigidified at p = {test_res[3]:.5f}\n")

    REPS = 20000
    L_values = list(range(1, 11))
    k_values = list(range(1, 33))
    
    OUTPUT_DIR = "./simulations/AWS_Rigidity_Only_20k"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    HEADERS = ["run_idx", "L", "k", "p_rigid"]
    
    num_workers = max(1, os.cpu_count() - 1)
    print(f" INITIATING {num_workers}-CORE RIGIDITY CAMPAIGN\n")
    
    start_time_global = time.time()
    with mp.Pool(num_workers) as pool:
        for L in L_values:
            completed = {k: set() for k in k_values}
            files, writers = {}, {}
            for k in k_values:
                csv_path = os.path.join(OUTPUT_DIR, f"results_L{L}_k{k}.csv")
                if os.path.exists(csv_path):
                    with open(csv_path, 'r') as f:
                        reader = csv.reader(f)
                        next(reader, None)
                        for row in reader:
                            if row: completed[k].add(int(row[0]))
                    files[k] = open(csv_path, 'a', newline='')
                else:
                    files[k] = open(csv_path, 'w', newline='')
                    writers[k] = csv.writer(files[k])
                    writers[k].writerow(HEADERS)
                writers[k] = csv.writer(files[k])
            
            tasks = []
            for k in k_values:
                tasks.extend([(L, k, i) for i in range(REPS) if i not in completed[k]])
            if not tasks:
                for f in files.values(): f.close()
                continue
                
            np.random.shuffle(tasks)
            print(f"-> [START] Submitting L={L:2d} | Total Tasks: {len(tasks):,}")
            batch_start = time.time()
            
            for i, res in enumerate(pool.imap_unordered(run_rigidity_only, tasks, chunksize=10)):
                run_idx, L_res, k_res, p_rigid = res
                writers[k_res].writerow([run_idx, L_res, k_res, p_rigid])
                
                if (i + 1) % 1000 == 0 or (i + 1) == len(tasks):
                    for f in files.values(): f.flush()
                    elapsed = time.time() - batch_start
                    print(f"   [ATOMIC LOG] L={L:2d} | Progress: {i+1:6d}/{len(tasks)} | Speed: {(i+1)/elapsed:6.1f} runs/sec")
                            
            for f in files.values(): f.close()
            print(f"-> [DONE] L={L:2d} finished in {(time.time()-batch_start)/60:.1f} minutes.\n")

    print(f" CAMPAIGN COMPLETE IN {(time.time() - start_time_global)/3600:.2f} HOURS!")

if __name__ == "__main__":
    main()
