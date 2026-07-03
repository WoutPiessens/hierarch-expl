"""One (strategy, seed) cell of the strategy suite; run in a subprocess with a timeout.
Usage: python _hybrid_cell.py <strategy> <seed> [rep]
Prints one RESULT line plus one COMMIT line per commit decision, with features:
  size    = |M| (number of group members of the committed GMCS)
  overlap = |leaves(M) ∩ S| (suitable primitives in M's subtrees -- the branch's relaxation budget)
  opts    = number of committable options at that decision
  dead    = 1 if the commit was later undone by a backtrack (dead end), else 0
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle import build_nurse_hierarchy, sample_suitable_set, run_hierarchical_oracle

strat, seed = sys.argv[1], int(sys.argv[2])
rep = sys.argv[3] if len(sys.argv) > 3 else "0"
root, hard = build_nurse_hierarchy("nurse_instance1_softreq_8nurses")
S, sd = sample_suitable_set(root, hard, pct=40, seed0=seed)
name2node = {nd.get_full_name(): nd for nd in root.iter_nodes()}

def overlap(names):
    leaves = set()
    for g in names:
        leaves |= {lf.get_full_name() for lf in name2node[g].leaves()}
    return len(leaves & S)

log = []
t0 = time.perf_counter()
r = run_hierarchical_oracle(root, hard, S, seed=sd, max_steps=2000, commit_strategy=strat, log=log)
dt = time.perf_counter() - t0

# mark each commit dead iff a later backtrack popped it (LIFO)
stack = []
for e in log:
    if e["action"] == "commit":
        e["dead"] = 0
        stack.append(e)
    elif e["action"] == "backtrack" and stack:
        stack.pop()["dead"] = 1

for e in log:
    if e["action"] == "commit":
        print(f"COMMIT {strat} {sd} {rep} size={len(e['mcs'])} overlap={overlap(e['mcs'])} "
              f"opts={e['n_options']} dead={e['dead']}", flush=True)

print(f"RESULT {strat} {sd} {rep} {r['result']} q={r['n_decisions']} c={r['n_commit']} "
      f"r={r['n_refine']} b={r['n_backtrack']} mcs={r['n_gmcs_seen']} mus={r['n_gmus_seen']} "
      f"relax={r['n_relaxed']} t={dt:.0f}s", flush=True)
