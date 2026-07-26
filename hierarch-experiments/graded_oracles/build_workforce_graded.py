"""
Build DIVERSE workforce instances from different anon_scenarios of the Explainable-Workforce-
Scheduling repo (github.com/ML-KULeuven/Explainable-Workforce-Scheduling), instead of drop/task
variants of a single base. Each scenario -> AllocationModel; SOFT = task_is_allocated +
same_allocation; made UNSAT by dropping N_DROP teams in coverage-preserving order (seeded random
order). Saves data/workforce-graded/<name>/ and reports UNSAT status + correction-size histogram.

    python build_workforce_graded.py
"""
import sys, json, random, pickle
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
FE = HERE.parent.parent / "final_experiments"
WF = FE / "workforce"
sys.path.insert(0, str(FE))
sys.path.insert(0, str(WF / "allocation_src"))
sys.path.insert(0, str(HERE.parent))
import _bootstrap  # noqa
import cpmpy as cp
from cpmpy.tools.explain import constraint_node_to_dict
from cpmpy.tools.explain.hierarchical import ConstraintNode
from cpmpy.tools.explain.utils import make_assump_model
from models import AllocationModel
from utils import read_instance

SCEN = WF / "allocation_src" / "scenarios"
OUT = HERE.parent / "data" / "workforce-graded2"
# (name, scenario file, N_DROP or "max", order-seed) -- distinct scenarios chosen by task count
# (more tasks -> more contention -> larger corrections) to tile the difficulty bins.
CONFIGS = [
    ("wf-s22", "scenario_22.json", "max", 1),      # 7 tasks   -> <=5
    ("wf-s2", "scenario_2.json", "max", 1),        # 18 tasks  -> <=5 / 6-10
    ("wf-s25", "scenario_25.json", "max", 1),      # 46 tasks  -> 6-10 / 11-15
    ("wf-s3", "scenario_3.json", "max", 1),        # 80 tasks  -> 11-15
    ("wf-s0", "scenario_0.json", "max", 1),        # 105 tasks -> 16-20
    ("wf-s50", "scenario_50.json", "max", 1),      # 107 tasks -> 16-20
    ("wf-s45", "scenario_45.json", "max", 1),      # 82 tasks  (spare / 11-15)
    ("wf-s5", "scenario_5.json", "max", 1),        # 114 tasks (spare / 16-20)
]


def split_buckets(items, levels):
    if levels == 0:
        return [items]
    mid = (len(items) + 1) // 2
    return split_buckets(items[:mid], levels - 1) + split_buckets(items[mid:], levels - 1)


def tree(name, buckets):
    def build(nm, bks):
        node = ConstraintNode(nm)
        if len(bks) == 1:
            for lname, cons in bks[0]:
                ch = node.add_child(lname); ch.parent = node; ch.constraints.extend(cons)
            return node
        mid = len(bks) // 2
        L = build(nm + ".L", bks[:mid]); L.parent = node
        R = build(nm + ".R", bks[mid:]); R.parent = node
        node.children = [L, R]
        return node
    return build(name, buckets)


def build_one(scen_file, n_drop, oseed, levels=3):
    tasks, calendars, same = read_instance(str(SCEN / scen_file))
    m = AllocationModel(tasks, calendars, same, allow_unalloc=False,
                        break_symmetries=True, make_time_worked_vars=True)
    n = len(tasks); TEAMS = m.TEAMS
    ta = m.task_is_allocated(False)
    sa_by_team = {j: [] for j in range(len(TEAMS))}
    for group_ids in same:
        idxs = tasks.index[tasks['task_id'].isin(group_ids)].tolist()
        for j in range(len(TEAMS)):
            sa_by_team[j].append(cp.AllEqual(m.alloc[idxs, j]))
    hard = (m.task_team_compatibility() + m.overlapping_tasks() + m.team_usage()
            + m.time_worked_constraints() + m.get_symmetry_breaking_constraints())
    soft_all = list(ta) + [c for j in range(len(TEAMS)) for c in sa_by_team[j]]
    compat = [set(t['team_ids']) for _, t in tasks.iterrows()]

    def cov_ok(rem):
        return all(len(compat[i] & {TEAMS[j] for j in rem}) >= 1 for i in range(n))
    remaining = set(range(len(TEAMS))); order = []
    rng = random.Random(oseed)
    while True:
        cand = [j for j in remaining if cov_ok(remaining - {j})]
        if not cand:
            break
        j = rng.choice(sorted(cand)); remaining.discard(j); order.append(j)
    if n_drop == "max":
        n_drop = len(order)
    if len(order) < n_drop:
        return None, None, f"only {len(order)} coverage-preserving drops (< {n_drop})"
    hard = list(hard) + [cp.sum(m.alloc[:, order[j]]) == 0 for j in range(n_drop)]
    if cp.Model(hard + soft_all).solve(solver="ortools") is not False:
        return None, None, "not UNSAT at this N_DROP"
    if cp.Model(hard).solve(solver="ortools") is not True:
        return None, None, "hard core UNSAT"
    task_b = split_buckets(list(range(n)), levels)
    ta_b = [[(f"task{i}", [ta[i]]) for i in idxs] for idxs in task_b]
    team_b = split_buckets(list(range(len(TEAMS))), levels)
    sa_b = [[(f"team{j}", [c for c in sa_by_team[j]]) for j in idxs] for idxs in team_b]
    root = ConstraintNode("workforce")
    ia = tree("is-allocated", ta_b); ia.parent = root
    sm = tree("same-allocation", sa_b); sm.parent = root
    root.children = [ia, sm]
    return root, hard, None


def save(name, root, hard):
    hard = [c for c in hard if c is not None]
    leaf_cons = [c for lf in root.leaves() for c in lf.constraints]
    allc = list(hard) + leaf_cons
    idx = {id(c): i for i, c in enumerate(allc)}
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    pickle.dump({"all": allc, "hard": list(range(len(hard)))}, open(d / "constraints.pkl", "wb"))
    json.dump(constraint_node_to_dict(root, idx), open(d / "hierarchy.json", "w"))


def hist(root, hard):
    leaves = root.leaves()
    model, _s, assump = make_assump_model([l.get_grouped_constraint() for l in leaves], list(hard))
    s = cp.SolverLookup.get("ortools", model); sizes = Counter(); seen = set()
    for seed in range(20):
        order = list(range(len(assump))); random.Random(seed).shuffle(order); kept = []
        for i in order:
            if s.solve(assumptions=[assump[j] for j in kept + [i]]) is True:
                kept.append(i)
        C = frozenset(range(len(assump))) - set(kept)
        if C not in seen:
            seen.add(C); sizes[len(C)] += 1
    return dict(sorted(sizes.items()))


def main():
    for name, scen, nd, oseed in CONFIGS:
        try:
            root, hard, err = build_one(scen, nd, oseed)
            if err:
                print(f"  !! {name} ({scen} d{nd}): {err}", flush=True); continue
            save(name, root, hard)
            print(f"  {name} ({scen} d{nd}): {len(root.leaves())} leaves USABLE",
                  flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {name}: ERR {type(e).__name__}: {str(e)[:150]}", flush=True)
    print("BUILD_WF_DONE", flush=True)


if __name__ == "__main__":
    main()
