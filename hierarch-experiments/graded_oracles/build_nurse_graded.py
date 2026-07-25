"""
Build the nurse instances for the graded benchmark, from DIVERSE source website instances
(schedulingbenchmarks.org) via the soft-hard model (shift-on/off + cover = soft, structural = hard,
i.e. objectives turned into constraints). The planning HORIZON is the difficulty knob: short
horizon -> small corrections (fills <=5), long horizon -> large corrections (fills 16-20).

Chosen 6 (diversity: sources 1,2,4,5,6 -- only Instance1 is reused, and only because it is the
one small enough to reach the <=5 bin):

    nurse1-h12 : Instance1, 8 nurses, horizon 12  -> corrections ~3-8   (<=5 bin)
    nurse1-h14 : Instance1, 8 nurses, horizon 14  -> corrections ~5-13  (<=5 / 6-10 / low 11-15)
    nurse2     : Instance2, full,     horizon 14  -> corrections ~11-23 (11-15 / 16-20)
    nurse4     : Instance4, full,     horizon 20  -> (16-20)
    nurse5     : Instance5, full,     horizon 16  -> (mid/high)
    nurse6     : Instance6, full,     horizon 16  -> (mid/high)

Writes hierarch-experiments/data/nurse-graded/<name>/{constraints.pkl,hierarchy.json}. Reports each
instance's UNSAT status + a quick correction-size histogram so bad picks are visible immediately.

    python build_nurse_graded.py
"""
import sys, json, pickle, random
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
FE = HERE.parent.parent / "final_experiments"
sys.path.insert(0, str(FE))
sys.path.insert(0, str(HERE.parent.parent / "experiments"))
sys.path.insert(0, str(HERE.parent.parent / "experiments" / "data" / "models"))
sys.path.insert(0, str(HERE.parent))                        # hierarch-experiments
import _bootstrap  # noqa
import cpmpy as cp
from cpmpy.tools.explain import constraint_node_to_dict
from cpmpy.tools.explain.utils import make_assump_model
from nurserostering import parse_scheduling_period, nurserostering_soft_hard_model
sys.path.insert(0, str(FE / "nurse"))
from build_instance import build_hierarchy         # family->[nurse]->week->day hierarchy
from build_nurse_instance import make_slice

NURSE_DIR = HERE.parent.parent / "examples" / "nurserostering"
OUT = HERE.parent / "data" / "nurse-graded"
# (name, source instance idx, n_nurses or None=all, horizon or None=full)
CONFIGS = [
    ("nurse4", 4, None, None),      # full horizon (h28) -> 16-32
    ("nurse5", 5, None, None),      # full horizon
    ("nurse6", 6, None, None),      # full horizon
]


def save(name, root, hard):
    hard = [c for c in hard if c is not None]
    leaf_cons = [c for leaf in root.leaves() for c in leaf.constraints]
    all_cons = list(hard) + leaf_cons
    index_of = {id(c): i for i, c in enumerate(all_cons)}
    spec = constraint_node_to_dict(root, index_of)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "constraints.pkl", "wb") as f:
        pickle.dump({"all": all_cons, "hard": list(range(len(hard)))}, f)
    with open(d / "hierarchy.json", "w", encoding="utf-8") as f:
        json.dump(spec, f)
    return d


def quick_hist(root, hard):
    leaves = root.leaves()
    soft = [lf.get_grouped_constraint() for lf in leaves]
    model, _s, assump = make_assump_model(soft, list(hard))
    s = cp.SolverLookup.get("ortools", model)
    sizes = Counter(); seen = set()
    for seed in range(30):
        order = list(range(len(assump))); random.Random(seed).shuffle(order); kept = []
        for i in order:
            if s.solve(assumptions=[assump[j] for j in kept + [i]]) is True:
                kept.append(i)
        C = frozenset(range(len(assump))) - set(kept)
        if C not in seen:
            seen.add(C); sizes[len(C)] += 1
    return dict(sorted(sizes.items()))


def main():
    for name, idx, nn, h in CONFIGS:
        data = parse_scheduling_period(str(NURSE_DIR / f"Instance{idx}.txt"))
        n_nurses = len(data["staff"]) if nn is None else nn
        horizon = int(data["horizon"]) if h is None else h
        sliced = make_slice(data, n_nurses, horizon, list(data["shifts"].index))
        hard, soft, soft_names, _ = nurserostering_soft_hard_model(**sliced)
        from cpmpy.tools.explain.hierarchical import ConstraintNode  # noqa
        root = build_hierarchy(soft, soft_names)
        full = cp.Model(list(hard) + soft).solve(solver="ortools")
        hs = cp.Model(list(hard)).solve(solver="ortools")
        if full is not False or hs is not True:
            print(f"  !! {name} (Inst{idx} n={n_nurses} h={horizon}): full={full} hard={hs} "
                  f"-- NOT USABLE (need full UNSAT, hard SAT)", flush=True)
            continue
        d = save(name, root, hard)
        print(f"  {name} (Inst{idx} n={n_nurses} h={horizon}): {len(root.leaves())} leaves, "
              f"USABLE -> {d}", flush=True)
    print("BUILD_NURSE_DONE", flush=True)


if __name__ == "__main__":
    main()
