"""
    Measure ONE benchmark-candidate instance: mean natural-MCS size (N draws) + tractability
    screens. Self-contained on the experiment repo (uses hierarchy.load_instance); candidate
    data is expected under data/<problem>/<instance>/ like any other instance.

        python measure_candidate.py PROBLEM INSTANCE [N_DRAWS] [GATE_CAP]

    Prints one JSON line prefixed 'REC '. Intended to be fanned out with xargs -P.
"""
import json
import random
import sys
import time

import _bootstrap  # noqa: F401
import cpmpy as cp
from cpmpy.tools.explain.utils import make_assump_model

import hierarchy

GATE_SOLVER = "ortools"


def mss_correction_set(root, hard, rng):
    """Random-MSS complement (natural MCS), as in final_experiments/common.py."""
    leaves = root.leaves()
    names = [lf.get_full_name() for lf in leaves]
    soft = [lf.get_grouped_constraint() for lf in leaves]
    soft = [c for c in soft if c is not None]
    model, _s, assump = make_assump_model(soft, list(hard))
    s = cp.SolverLookup.get(GATE_SOLVER, model)
    order = list(range(len(assump)))
    rng.shuffle(order)
    kept = []
    for i in order:
        if s.solve(assumptions=[assump[j] for j in kept + [i]]) is True:
            kept.append(i)
    keptset = set(kept)
    return {names[i] for i in range(len(assump)) if i not in keptset}


def main():
    problem, instance = sys.argv[1], sys.argv[2]
    n_draws = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    gate_cap = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    root, hard = hierarchy.load_instance(problem, instance)
    hard = [c for c in hard if c is not None]
    soft = [lf.get_grouped_constraint() for lf in root.leaves()]
    soft = [c for c in soft if c is not None]
    rec = {"problem": problem, "instance": instance, "L": len(soft)}
    t0 = time.perf_counter()
    fs = cp.Model(list(hard) + soft).solve(solver=GATE_SOLVER)
    rec["t_gate"] = round(time.perf_counter() - t0, 2)
    hs = cp.Model(list(hard)).solve(solver=GATE_SOLVER)
    if fs is not False or hs is not True:
        rec.update(ok=False, reason=f"full={fs} hard={hs}")
    elif rec["t_gate"] > gate_cap:
        rec.update(ok=False, reason=f"gate {rec['t_gate']}s > {gate_cap}s")
    else:
        t0 = time.perf_counter()
        sizes = [len(mss_correction_set(root, hard, random.Random(s)))
                 for s in range(n_draws)]
        rec.update(ok=True, sizes=sorted(sizes), mean=round(sum(sizes) / n_draws, 2),
                   t_measure=round(time.perf_counter() - t0, 1))
    print("REC " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
