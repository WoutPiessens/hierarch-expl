"""
    How NON-minimal are staged-deletion's correction subsets? For every staged-deletion result
    we ask: how many of its deleted constraints are REDUNDANT -- i.e. how many can be put back
    (re-enforced) while the remaining deletions still restore satisfiability? That redundant
    count is exactly "how many constraints to remove to obtain minimality": strip them and the
    correction subset becomes a genuine MCS.

    Method. Staged-deletion returns C = set of soft constraints it deleted to reach SAT (so
    hard + (soft \\ C) is SAT). We shrink C by deletion-based minimization: walk its members and
    for each c try re-enforcing it (removing it from the correction set); if hard + (soft \\ C')
    stays SAT with c back in, c was redundant -> drop it from the correction set. What remains,
    C_min, is a MINIMAL correction subset (an MCS): re-enforcing ANY of its members turns the
    problem UNSAT again. The answer is  removed = |C| - |C_min|.

    We run this on the SAME grid/seeds as run_softreq_size_study.py (so |C| here == that study's
    staged_size), and also compare |C_min| to the true OPTIMUM for the sample (smallest
    acceptable MCS, from the full enumeration) to see whether minimizing staged-deletion's output
    reaches the smallest possible repair or merely a minimal (irreducible) one.

    Run from experiments/:  python run_softreq_minimality_study.py
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cpmpy as cp
from hierarchy_io import load_flat_instance
from oracle import run_staged_deletion, RandomSampleOracle

INSTANCE = "nurse_instance1_softreq_8nurses"
PCTS = [30, 40, 50, 60, 70, 80, 90, 100]
N_SEEDS = 100
SEED0 = 5000                 # same samples as run_softreq_size_study.py
SOLVER = "exact"
MAP_SOLVER = "exact"
CAP = 3000
GATE_SOLVER = "ortools"      # SAT checks for gating + minimization

FLAT_DIR = Path(__file__).resolve().parent / "data" / "flat_instances" / INSTANCE
OUT_JSON = Path(__file__).resolve().parent / "experiment_outputs" / f"softreq_minimality_{INSTANCE}.json"


def fixed_rate_oracle(soft_names, pct, seed):
    return RandomSampleOracle(soft_names, start_pct=pct, escalate_step=0,
                              escalate_after=10**9, seed=seed, max_pct=pct)


def repairs(hard, soft, soft_names, deleted):
    """Is hard + (soft \\ deleted) satisfiable?"""
    kept = [c for c, n in zip(soft, soft_names) if n not in deleted]
    return cp.Model(hard + kept).solve(solver=GATE_SOLVER) is True


def minimize_correction(hard, soft, soft_names, deleted):
    """Deletion-based minimization of a correction subset `deleted` (set of names) down to a
    minimal one (an MCS). Returns the minimized set of names."""
    current = set(deleted)
    for n in list(deleted):
        trial = current - {n}           # try re-enforcing constraint n
        if repairs(hard, soft, soft_names, trial):
            current = trial             # n was redundant -> drop from correction set
    return current


def summarize(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    q = lambda p: xs[min(n - 1, int(p * (n - 1) + 0.5))]
    return {"n": n, "mean": round(statistics.mean(xs), 3),
            "std": round(statistics.pstdev(xs), 3) if n > 1 else 0.0,
            "min": xs[0], "p25": q(0.25), "median": statistics.median(xs), "p75": q(0.75),
            "max": xs[-1]}


def main():
    soft, hard, soft_names, hard_names = load_flat_instance(INSTANCE)
    all_mcses = [frozenset(m) for m in json.loads((FLAT_DIR / "all_mcses.json").read_text())]
    mcs_set = set(all_mcses)
    print(f"{INSTANCE}: {len(soft)} soft, {len(hard)} hard, {len(all_mcses)} enumerated MCSes.\n"
          f"pcts={PCTS}, {N_SEEDS} seeds each (same samples as the size study).\n", flush=True)

    cells = []
    for pct in PCTS:
        for i in range(N_SEEDS):
            seed = SEED0 + i
            allowed = frozenset(fixed_rate_oracle(soft_names, pct, seed).allowed)
            if not repairs(hard, soft, soft_names, allowed):
                continue

            rs = run_staged_deletion(soft, hard, soft_names,
                                     fixed_rate_oracle(soft_names, pct, seed),
                                     solver=SOLVER, map_solver=MAP_SOLVER, max_iterations=CAP)
            if not rs["reached_sat"]:
                continue

            C = set(rs["relaxed_names"])
            C_min = minimize_correction(hard, soft, soft_names, C)
            removed = len(C) - len(C_min)

            acceptable = [m for m in all_mcses if m <= allowed]
            optimum = min((len(m) for m in acceptable), default=None)

            cells.append({
                "pct": pct, "seed": seed,
                "staged_size": len(C),
                "minimized_size": len(C_min),
                "removed_to_minimality": removed,
                "already_minimal": removed == 0,
                "minimized_is_enumerated_mcs": frozenset(C_min) in mcs_set,
                "optimum": optimum,
                "minimized_minus_optimum": (len(C_min) - optimum) if optimum is not None else None,
            })
        done = [c for c in cells if c["pct"] == pct]
        print(f"pct={pct:3d}: {len(done):3d} staged-deletion successes analysed", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"instance": INSTANCE, "pcts": PCTS, "n_seeds": N_SEEDS,
                                    "seed0": SEED0, "cells": cells}, indent=2))

    removed = [c["removed_to_minimality"] for c in cells]
    staged = [c["staged_size"] for c in cells]
    minz = [c["minimized_size"] for c in cells]
    n = len(cells)
    already = sum(c["already_minimal"] for c in cells)
    all_valid_mcs = all(c["minimized_is_enumerated_mcs"] for c in cells)

    print(f"\n===============  STAGED-DELETION: DISTANCE FROM MINIMALITY  ({n} subsets)  ===============")
    print(f"staged size          |C|      : {summarize(staged)}")
    print(f"minimized size       |C_min|  : {summarize(minz)}")
    print(f"removed to minimality |C|-|C_min| : {summarize(removed)}")
    print(f"\nalready minimal (removed==0) : {already}/{n} ({100*already/n:.0f}%)")
    print(f"every minimized subset is an enumerated MCS: {all_valid_mcs}")
    dist = Counter(removed)
    print(f"\nremoved-count distribution (constraints to strip -> minimal):")
    for k in sorted(dist):
        bar = "#" * dist[k]
        print(f"   {k:2d} : {dist[k]:3d}  {bar}")

    print(f"\n--- per sampling rate ---")
    print(f"{'pct':>4} {'n':>4} {'|C|':>6} {'|Cmin|':>7} {'removed':>8} {'%minimal':>9} {'Cmin-opt':>9}")
    for pct in PCTS:
        cs = [c for c in cells if c["pct"] == pct]
        if not cs:
            continue
        mc = statistics.mean(c["staged_size"] for c in cs)
        mm = statistics.mean(c["minimized_size"] for c in cs)
        mr = statistics.mean(c["removed_to_minimality"] for c in cs)
        pm = 100 * sum(c["already_minimal"] for c in cs) / len(cs)
        go = statistics.mean(c["minimized_minus_optimum"] for c in cs)
        print(f"{pct:>4} {len(cs):>4} {mc:>6.2f} {mm:>7.2f} {mr:>8.2f} {pm:>8.0f}% {go:>9.2f}")

    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
