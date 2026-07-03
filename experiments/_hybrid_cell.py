"""One (strategy, seed) cell of the strategy suite; run in a subprocess with a timeout.
Usage: python _hybrid_cell.py <strategy> <seed> [rep] [pct] [leaf|node]
  pct  -- suitable-set sampling rate (default 40)
  mode -- 'leaf' (scattered primitives, default) or 'node' (hierarchy-aware clustered sampling)
'selective-relaxation' is the current name of the staged-deletion baseline.
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
import cpmpy as cp
from oracle import (build_nurse_hierarchy, sample_suitable_set, sample_suitable_nodes,
                    run_hierarchical_oracle, run_oracle, run_staged_deletion)
from hierarchy_io import load_flat_instance

strat, seed = sys.argv[1], int(sys.argv[2])
rep = sys.argv[3] if len(sys.argv) > 3 else "0"
pct = float(sys.argv[4]) if len(sys.argv) > 4 else 40
mode = sys.argv[5] if len(sys.argv) > 5 else "leaf"
if strat == "selective-relaxation":                     # current name of the staged-deletion baseline
    strat_impl = "staged-deletion"
else:
    strat_impl = strat
root, hard = build_nurse_hierarchy("nurse_instance1_softreq_8nurses")
if mode == "node":
    S, sd, _sel = sample_suitable_nodes(root, hard, pct=pct, seed0=seed)
else:
    S, sd = sample_suitable_set(root, hard, pct=pct, seed0=seed)
assert sd == seed, f"seed {seed} not feasible at {pct}% ({mode}); first feasible: {sd}"
name2node = {nd.get_full_name(): nd for nd in root.iter_nodes()}


def _to_leaf(n):
    """flat soft-constraint name -> hierarchy leaf full name (same rule as build_nurse_hierarchy)."""
    fam = n.split("__", 1)[0]
    tag = n.split("__", 1)[1].split("_")
    if fam in ("shift_on", "shift_off"):
        return f"{fam} {tag[0]} {tag[1]}"
    return f"{fam} week{int(tag[0][3:]) // 7} {tag[0]}"


class _FixedS:
    """Flat-method oracle over the SAME suitable set S (leaf names)."""
    def __init__(self, S):
        self.S = S
    def evaluate(self, names):                 # baseline: accept an MCS iff fully inside S
        return all(_to_leaf(n) in self.S for n in names)
    def is_suitable(self, n):                  # staged-deletion: per-constraint suitability
        return _to_leaf(n) in self.S
    def stats(self):
        return {}


if strat_impl in ("flat-baseline", "staged-deletion"):
    soft, fhard, soft_names, _ = load_flat_instance("nurse_instance1_softreq_8nurses")
    t0 = time.perf_counter()
    if strat_impl == "flat-baseline":
        name_of = {id(c): n for c, n in zip(soft, soft_names)}
        r = run_oracle("baseline", soft, fhard, name_of, _FixedS(S), max_iterations=5000,
                       solver="exact", map_solver="exact")
        ok = "repaired" if r["accepted"] else "failed"
        print(f"RESULT {strat} {sd} {rep} {ok} q={r['n_mcs_seen']} c=0 r=0 b=0 "
              f"mcs={r['n_mcs_seen']} mus={r['n_mus_seen']} "
              f"relax={len(r['accepted']) if r['accepted'] else 0} t={time.perf_counter()-t0:.0f}s",
              flush=True)
    else:
        r = run_staged_deletion(soft, fhard, soft_names, _FixedS(S),
                                solver="exact", map_solver="exact", max_iterations=5000)
        ok = "repaired" if r["reached_sat"] else "failed"
        # minimality gap (evaluation only, not part of the oracle): greedily re-add relaxed
        # constraints; what survives is a minimal correction subset.
        excess = ""
        if r["reached_sat"]:
            cur = set(r["relaxed_names"])
            def sat(dropped):
                kept = [c for c, n in zip(soft, soft_names) if n not in dropped]
                return cp.Model(fhard + kept).solve(solver="ortools") is True
            for n in list(cur):
                if sat(cur - {n}):
                    cur.discard(n)
            excess = f" min={len(cur)} excess={r['n_relaxed'] - len(cur)}"
        print(f"RESULT {strat} {sd} {rep} {ok} q={r['n_decisions']} c=0 r=0 b=0 "
              f"mcs={r['n_mcs_enumerated']} mus={r['n_mus_seen']} relax={r['n_relaxed']} "
              f"t={time.perf_counter()-t0:.0f}s{excess}", flush=True)
    sys.exit(0)

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
