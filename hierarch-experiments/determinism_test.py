"""
    Reproducibility probe: run a fixed set of (problem, instance, seed, method) cells and print a
    trajectory fingerprint for each, so the SAME cells can be compared across processes, machines
    and hash seeds.

    A cell's fingerprint is the md5 of its full decision script (actions + refined group names +
    committed MCS members), plus the counts. Two runs agree iff they took the identical path.

        python determinism_test.py [budget]      # default 60s
"""
import hashlib
import json
import sys
import time

import _bootstrap  # noqa: F401
from cpmpy.tools.explain import hierarchical_marco

import hierarchy
import oracles as orc
from methods import (HierarchCommitOracle, HierarchPrematureCommitOracle,
                     SOLVER, MAP_SOLVER, ROUND_CAP, _repaired)

CELLS = [   # (problem, instance, oracle seed, oracle class, round_cap)
    ("nurse-suite", "instance1-n6-s1", 0, HierarchCommitOracle, ROUND_CAP),
    ("nurse-suite", "instance4-k70-s1", 210, HierarchCommitOracle, None),
    ("nurse-suite", "instance2-k85-s1", 1, HierarchPrematureCommitOracle, ROUND_CAP),
    ("thesis-suite", "unsat-115-m1-2-3-4", 0, HierarchCommitOracle, ROUND_CAP),
    ("workforce-suite", "ews-d11-o4", 0, HierarchCommitOracle, ROUND_CAP),
    ("workforce-suite", "ews-d12-o2", 214, HierarchCommitOracle, None),
]


def fingerprint(script):
    parts = []
    for s in script:
        a = s["action"]
        if a == "refine":
            parts.append(f"R:{s['group']}")
        elif a == "commit":
            parts.append("C:" + ",".join(sorted(s["mcs"])))
        else:
            parts.append(a[0].upper())
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    import os
    print(f"# host={os.uname().nodename if hasattr(os,'uname') else 'win'} "
          f"hashseed={os.environ.get('PYTHONHASHSEED','random')} budget={budget:.0f}s", flush=True)
    for problem, inst, seed, cls, cap in CELLS:
        root, hard = hierarchy.load_instance(problem, inst)
        hard = [c for c in hard if c is not None]
        o = next(x for x in orc.load_oracles(problem, inst, "mss-20") if x["seed"] == seed)
        oracle = cls(root, hard, set(o["S"]), seed=seed, time_budget=budget)
        oracle.t0 = time.perf_counter()
        for _ in hierarchical_marco(root, list(hard), solver=SOLVER, map_solver=MAP_SOLVER,
                                    decide_step=oracle, deadline=oracle.t0 + budget,
                                    round_cap=cap):
            pass
        el = time.perf_counter() - oracle.t0
        print(json.dumps({
            "cell": f"{problem}/{inst}/s{seed}/{cls.__name__}/cap={cap}",
            "steps": len(oracle.script), "result": oracle.result,
            "repaired": _repaired(root, hard, set(oracle.relaxed)),
            "relaxed": len(set(oracle.relaxed)), "fp": fingerprint(oracle.script),
            "t": round(el, 1)}), flush=True)


if __name__ == "__main__":
    main()
