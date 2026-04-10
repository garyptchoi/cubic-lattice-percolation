"""
GOD-TIER LEMMA IV.32 VALIDATION (All L, All k)
Fully vectorized edge generation + arbitrary k-sampling.
"""

import numpy as np
import random
import time
import sys

class DSU:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x):
        root = int(x)
        while True:
            p = int(self.parent[root])
            if p == root: break
            self.parent[root] = int(self.parent[p])
            root = p
        return root

    def union(self, u, v):
        ru, rv = self.find(u), self.find(v)
        if ru == rv: return False
        if self.size[ru] < self.size[rv]:
            ru, rv = rv, ru
        self.parent[rv] = ru
        self.size[ru] += self.size[rv]
        return True

def generate_edges_vectorized(L):
    n = L + 1
    n_sq = n * n
    total_edges = 3 * L * n_sq
    edges = np.empty((total_edges, 2), dtype=np.int64)
    
    # Vectorized X-edges
    x, y, z = np.mgrid[0:L, 0:n, 0:n]
    u = x + n * y + n_sq * z
    v = (x + 1) + n * y + n_sq * z
    ptr = L * n_sq
    edges[0:ptr, 0] = u.flatten()
    edges[0:ptr, 1] = v.flatten()
    
    # Vectorized Y-edges
    x, y, z = np.mgrid[0:n, 0:L, 0:n]
    u = x + n * y + n_sq * z
    v = x + n * (y + 1) + n_sq * z
    edges[ptr:ptr*2, 0] = u.flatten()
    edges[ptr:ptr*2, 1] = v.flatten()
    
    # Vectorized Z-edges
    x, y, z = np.mgrid[0:n, 0:n, 0:L]
    u = x + n * y + n_sq * z
    v = x + n * y + n_sq * (z + 1)
    edges[ptr*2:ptr*3, 0] = u.flatten()
    edges[ptr*2:ptr*3, 1] = v.flatten()
    
    return edges, n**3

def run_ultimate_test():
    # Selected a clean progression to save you time while showing massive scale
    L_list =[24, 32, 48, 64, 96, 128, 160, 192, 224]
    k_list = [2, 8, 32]
    
    print(f"\n{'L':<5} | {'k':<2} | {'Mixed Samples':<15} | {'Intra Wins (Violations)':<25} | {'Violation Rate (%)'}")
    print("-" * 80)
    
    for L in L_list:
        for k in k_list:
            edges, N = generate_edges_vectorized(L)
            M = len(edges)
            
            np.random.seed(42 + k) # Different seed for different k
            np.random.shuffle(edges)
            
            dsu = DSU(N)
            total_mixed = 0
            lemma_violations = 0
            current_ptr = M - 1
            
            while current_ptr >= k - 1:
                # Sample k edges randomly from available pool
                cand_indices = random.sample(range(current_ptr + 1), k)
                
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
                    
                    # Track minimums (handling ties properly)
                    if score < min_score:
                        min_score = score
                        best_cands = [(idx, is_intra)]
                    elif score == min_score:
                        best_cands.append((idx, is_intra))
                
                # Break ties uniformly at random
                chosen_idx, chosen_is_intra = random.choice(best_cands)
                
                # Check for Lemma IV.32 competition
                if has_intra and has_inter:
                    total_mixed += 1
                    if chosen_is_intra:
                        lemma_violations += 1
                        
                # Execute the chosen edge
                bu, bv = edges[chosen_idx]
                dsu.union(bu, bv)
                
                # Swap with end and decrement
                edges[chosen_idx] = edges[current_ptr]
                current_ptr -= 1
                
                # Stop measuring once the giant component reaches 50% 
                if (M - current_ptr) % 25000 == 0:
                    sample_roots =[dsu.find(random.randint(0, N-1)) for _ in range(50)]
                    if max(dsu.size[r] for r in sample_roots) > 0.5 * N:
                        break
                        
            rate = (lemma_violations / total_mixed) * 100 if total_mixed > 0 else 0
            print(f"{L:<5} | {k:<2} | {total_mixed:<15} | {lemma_violations:<25} | {rate:.4f}%")

if __name__ == "__main__":
    run_ultimate_test()