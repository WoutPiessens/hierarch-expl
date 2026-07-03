"""
    Detailed, PAIRED investigation of the CORRECTION-SUBSET SIZE produced by the two oracle
    baselines on nurse_instance1_softreq_8nurses:

        * baseline (`run_oracle` + "baseline"): the accepted MCS -- a MINIMAL (irreducible)
          correction subset lying inside the sampled `allowed` set.
        * staged-deletion (`run_staged_deletion`): the accumulated deleted set -- a generally
          NON-minimal correction subset.

    For every random sample (pct, seed) on which BOTH methods succeed (same allowed set, paired),
    we record both sizes AND the true OPTIMUM for that sample -- the smallest acceptable MCS,
    i.e. min{ |M| : M is a primitive MCS and M subset of allowed }. The optimum is computed
    OFFLINE from the full enumeration in data/flat_instances/<inst>/all_mcses.json (5922 MCSes),
    so it needs no extra solving. `allowed` is reconstructed deterministically from (pct, seed)
    via the same RandomSampleOracle, so it matches exactly what the methods saw.

    This lets us answer, head to head:
      - how much bigger is staged-deletion's subset than baseline's, sample by sample (paired)?
      - how often is each method's subset strictly smaller / equal / larger than the other's?
      - how far is each from the achievable optimum (is baseline actually minimUM, not just
        minimAL)?

    Run from experiments/:  python run_softreq_size_study.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cpmpy as cp
from hierarchy_io import load_flat_instance
from oracle import run_oracle, run_staged_deletion, RandomSampleOracle

INSTANCE = "nurse_instance1_softreq_8nurses"
PCTS = [30, 40, 50, 60, 70, 80, 90, 100]
N_SEEDS = 100
SEED0 = 5000
SOLVER = "exact"
MAP_SOLVER = "exact"
CAP = 3000
GATE_SOLVER = "ortools"

FLAT_DIR = Path(__file__).resolve().parent / "data" / "flat_instances" / INSTANCE
OUT_JSON = Path(__file__).resolve().parent / "experiment_outputs" / f"softreq_size_{INSTANCE}.json"


def fixed_rate_oracle(soft_names, pct, seed):
    return RandomSampleOracle(soft_names, start_pct=pct, escalate_step=0,
                              escalate_after=10**9, seed=seed, max_pct=pct)


def summarize(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    q = lambda p: xs[min(n - 1, int(p * (n - 1) + 0.5))]
    return {"n": n, "mean": statistics.mean(xs), "std": statistics.pstdev(xs) if n > 1 else 0.0,
            "min": xs[0], "p25": q(0.25), "median": statistics.median(xs), "p75": q(0.75),
            "max": xs[-1]}


def main():
    soft, hard, soft_names, hard_names = load_flat_instance(INSTANCE)
    name_of = {id(c): n for c, n in zip(soft, soft_names)}
    all_mcses = [frozenset(m) for m in json.loads((FLAT_DIR / "all_mcses.json").read_text())]
    print(f"{INSTANCE}: {len(soft)} soft, {len(hard)} hard, {len(all_mcses)} enumerated MCSes.\n"
          f"pcts={PCTS}, {N_SEEDS} seeds each.\n", flush=True)

    cells = []
    for pct in PCTS:
        for i in range(N_SEEDS):
            seed = SEED0 + i
            allowed = frozenset(fixed_rate_oracle(soft_names, pct, seed).allowed)

            kept = [c for c, n in zip(soft, soft_names) if n not in allowed]
            if cp.Model(hard + kept).solve(solver=GATE_SOLVER) is not True:
                continue  # sample can't repair -> both fail; skip

            # offline optimum: smallest primitive MCS fully inside `allowed`
            acceptable = [m for m in all_mcses if m <= allowed]
            opt = min((len(m) for m in acceptable), default=None)
            n_acceptable = len(acceptable)

            rb = run_oracle("baseline", soft, hard, name_of,
                            fixed_rate_oracle(soft_names, pct, seed),
                            max_iterations=CAP, solver=SOLVER, map_solver=MAP_SOLVER)
            rs = run_staged_deletion(soft, hard, soft_names,
                                     fixed_rate_oracle(soft_names, pct, seed),
                                     solver=SOLVER, map_solver=MAP_SOLVER, max_iterations=CAP)
            acc = rb["accepted"]
            cells.append({
                "pct": pct, "seed": seed, "n_allowed": len(allowed),
                "n_acceptable_mcs": n_acceptable, "optimum": opt,
                "baseline_size": len(acc) if acc else None,
                "baseline_queries": rb["n_mcs_seen"],
                "staged_size": rs["n_relaxed"] if rs["reached_sat"] else None,
                "staged_queries": rs["n_decisions"],
                "staged_reached_sat": rs["reached_sat"],
            })
        done = [c for c in cells if c["pct"] == pct]
        print(f"pct={pct:3d}: {len(done):3d} paired successes", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"instance": INSTANCE, "pcts": PCTS, "n_seeds": N_SEEDS,
                                    "seed0": SEED0, "cells": cells}, indent=2))

    # ---- paired analysis over all cells where BOTH methods produced a subset ----
    paired = [c for c in cells if c["baseline_size"] is not None and c["staged_size"] is not None]
    b = [c["baseline_size"] for c in paired]
    s = [c["staged_size"] for c in paired]
    o = [c["optimum"] for c in paired]
    diff = [si - bi for si, bi in zip(s, b)]          # staged - baseline
    ratio = [si / bi for si, bi in zip(s, b)]
    b_excess = [bi - oi for bi, oi in zip(b, o)]       # baseline - optimum
    s_excess = [si - oi for si, oi in zip(s, o)]       # staged  - optimum

    staged_bigger = sum(d > 0 for d in diff)
    tie = sum(d == 0 for d in diff)
    staged_smaller = sum(d < 0 for d in diff)
    base_is_optimum = sum(be == 0 for be in b_excess)

    print(f"\n================  PAIRED SIZE ANALYSIS  ({len(paired)} samples)  ================")
    print(f"baseline (minimal MCS) size : {summarize(b)}")
    print(f"staged   (non-min)     size : {summarize(s)}")
    print(f"optimum  (min acc. MCS)     : {summarize(o)}")
    print(f"\nstaged - baseline (paired)  : {summarize(diff)}")
    print(f"   staged strictly LARGER : {staged_bigger}/{len(paired)} ({100*staged_bigger/len(paired):.0f}%)")
    print(f"   tie                    : {tie}/{len(paired)} ({100*tie/len(paired):.0f}%)")
    print(f"   staged strictly SMALLER: {staged_smaller}/{len(paired)} ({100*staged_smaller/len(paired):.0f}%)")
    print(f"   size ratio staged/base : mean={statistics.mean(ratio):.2f}, median={statistics.median(ratio):.2f}, max={max(ratio):.2f}")
    print(f"\nbaseline - optimum          : {summarize(b_excess)}")
    print(f"   baseline IS minimum-card: {base_is_optimum}/{len(paired)} ({100*base_is_optimum/len(paired):.0f}%)")
    print(f"staged   - optimum          : {summarize(s_excess)}")

    # per-pct paired means
    print(f"\n--- per sampling rate (paired means) ---")
    print(f"{'pct':>4} {'n':>4} {'base':>6} {'staged':>7} {'opt':>5} {'st-ba':>6} {'ba-opt':>7} {'st-opt':>7}")
    for pct in PCTS:
        cs = [c for c in paired if c["pct"] == pct]
        if not cs:
            continue
        bb = statistics.mean(c["baseline_size"] for c in cs)
        ss = statistics.mean(c["staged_size"] for c in cs)
        oo = statistics.mean(c["optimum"] for c in cs)
        print(f"{pct:>4} {len(cs):>4} {bb:>6.2f} {ss:>7.2f} {oo:>5.2f} {ss-bb:>6.2f} {bb-oo:>7.2f} {ss-oo:>7.2f}")

    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
