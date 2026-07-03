"""
    Study of NODE-based suitability sampling (sample hierarchy nodes; an internal node makes its
    whole subtree suitable at once) vs the flat LEAF sampling used so far, at the same 40% budget
    on nurse_instance1_softreq_8nurses.

    Reports:
      1. composition of node-samples: how many nodes drawn, how many internal, cluster sizes;
      2. feasibility rate over 400 seeds: how often does S admit a repair (an MCS inside S)?
      3. oracle runs (first + lookahead) on the first feasible node-samples, vs the known
         leaf-sample numbers.

    Run from experiments/:  python run_node_sampling_study.py
"""
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cpmpy as cp
from oracle import (build_nurse_hierarchy, sample_suitable_set, sample_suitable_nodes,
                    _draw_node_sample, run_hierarchical_oracle)

PCT = 40
N_PROBE = 400

root, hard = build_nurse_hierarchy("nurse_instance1_softreq_8nurses")
leaf_nodes = root.leaves()
k = max(1, round(len(leaf_nodes) * PCT / 100))
name2node = {nd.get_full_name(): nd for nd in root.iter_nodes()}
print(f"instance: {len(leaf_nodes)} leaves, budget k={k} ({PCT}%)\n", flush=True)

# ---- 1. composition ----------------------------------------------------------------------
n_sel, n_internal, max_cluster = [], [], []
for t in range(200):
    S, selected = _draw_node_sample(root, k, random.Random(t))
    assert len(S) == k, (t, len(S))
    internal = [s for s in selected if s in name2node and name2node[s].children]
    n_sel.append(len(selected)); n_internal.append(len(internal))
    max_cluster.append(max((len(name2node[s].leaves()) for s in internal), default=1))
print(f"1. node-sample composition (200 draws): |S|=k exactly in all draws;")
print(f"   nodes drawn per sample: mean {statistics.mean(n_sel):.1f} (leaf-sampling would be {k})")
print(f"   internal nodes per sample: mean {statistics.mean(n_internal):.1f}, "
      f"largest cluster: mean {statistics.mean(max_cluster):.1f} leaves\n", flush=True)

# ---- 2. feasibility rate ------------------------------------------------------------------
def feasible(S):
    kept = [lf.get_grouped_constraint() for lf in leaf_nodes if lf.get_full_name() not in S]
    return cp.Model(hard + [c for c in kept if c is not None]).solve(solver="ortools") is True

leaf_ok = sum(feasible(set(random.Random(t).sample([lf.get_full_name() for lf in leaf_nodes], k)))
              for t in range(N_PROBE))
node_ok = sum(feasible(_draw_node_sample(root, k, random.Random(t))[0]) for t in range(N_PROBE))
print(f"2. feasibility rate over {N_PROBE} seeds (S admits a repair):")
print(f"   leaf-sampling: {leaf_ok}/{N_PROBE} ({100*leaf_ok/N_PROBE:.1f}%)")
print(f"   node-sampling: {node_ok}/{N_PROBE} ({100*node_ok/N_PROBE:.1f}%)\n", flush=True)

# ---- 3. oracle runs on feasible node-samples ---------------------------------------------
print(f"3. oracle on the first 3 feasible node-samples:")
print(f"{'strategy':>10} {'seed':>5} {'result':>9} {'queries':>8} {'backtr':>7} {'|relax|':>8} "
      f"{'#internal-in-sample':>20}", flush=True)
found, t0seed = 0, 0
while found < 3 and t0seed < 400:
    S, sd, selected = sample_suitable_nodes(root, hard, pct=PCT, seed0=t0seed, max_tries=400 - t0seed)
    if S is None:
        break
    n_int = sum(1 for s in selected if s in name2node and name2node[s].children)
    for strat in ("first", "lookahead"):
        t0 = time.perf_counter()
        r = run_hierarchical_oracle(root, hard, S, seed=sd, max_steps=2000, commit_strategy=strat)
        print(f"{strat:>10} {sd:>5} {r['result']:>9} {r['n_decisions']:>8} {r['n_backtrack']:>7} "
              f"{r['n_relaxed']:>8} {n_int:>20} ({time.perf_counter()-t0:.0f}s)", flush=True)
    found += 1
    t0seed = sd + 1
