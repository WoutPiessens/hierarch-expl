"""
    Compare the two oracle baselines -- flat `marco` MCS-enumeration (`run_oracle`, method
    "baseline") vs. partial `run_staged_deletion` -- on the nurse_instance1_softreq_8nurses
    instance (structural rules HARD, shift_on/shift_off/cover SOFT), across a grid of SAMPLING
    RATES over the soft constraints.

    ==================================================================================
    Experimental design
    ==================================================================================
    A "sampling rate" p means: the oracle is willing to see a random p% of the 40 soft
    constraints relaxed (RandomSampleOracle's `allowed` set). Escalation is DISABLED here
    (max_pct = start_pct), so the rate is genuinely fixed at p -- the whole point is to sweep p.

    For each (p, seed) we draw one allowed set and run BOTH methods on that SAME set (paired):

      * baseline (`run_oracle` + "baseline"): enumerate MCSes with marco, accept the first MCS
        that is fully inside `allowed`. The accepted MCS is a MINIMAL correction subset.
      * staged-deletion (`run_staged_deletion`): for each MCS marco shows, delete the suitable
        (allowed) constraints in it and continue until the relaxed problem is SAT. The deleted
        set is a (generally NON-minimal) correction subset.

    Feasibility gate (makes the sweep tractable). Restricted to deleting only the `allowed`
    constraints, EITHER method can restore satisfiability iff deleting ALL of them does, i.e.
    iff `hard + (soft constraints NOT in allowed)` is SAT. (=>: an accepted/deleted correction
    subset lies inside `allowed`, so removing all of `allowed` a fortiori repairs. <=: if
    removing all of `allowed` repairs, `allowed` is a correction set and contains a minimal
    one.) So success/failure at a given (p, seed) is a property of the SAMPLE, identical for
    both methods; the methods differ only in HOW they get there (queries, correction-subset
    size). We test this one-solve predicate first, and only run the (potentially slow) marco
    enumeration on samples that can actually succeed -- otherwise baseline would enumerate all
    ~5900 MCSes before giving up.

    ==================================================================================
    Metrics collected per successful (p, seed), averaged per p
    ==================================================================================
      * queries / MCSes shown to the oracle : baseline n_mcs_seen ; staged n_decisions
      * MUSes seen (internal, not shown)     : baseline n_mus_seen ; staged n_mus_seen
      * correction-subset size               : baseline |accepted MCS| (minimal)
                                               ; staged  n_relaxed     (non-minimal)
      * solver calls (staged only; run_oracle doesn't expose them) : staged n_solve_calls
      * wall-clock seconds
    Plus, per p: the success rate (fraction of seeds whose sample admits a repair).

    Run from experiments/:  python run_softreq_sampling_experiments.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for cpmpy

import cpmpy as cp
from hierarchy_io import load_flat_instance
from oracle import run_oracle, run_staged_deletion, RandomSampleOracle

INSTANCE = "nurse_instance1_softreq_8nurses"
PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
N_SEEDS = 40
SEED0 = 1000
SOLVER = "exact"
MAP_SOLVER = "exact"
BASELINE_CAP = 2000          # safety cap; only reached if marco can't find an allowed MCS fast
GATE_SOLVER = "ortools"      # fast SAT check for the feasibility predicate

OUT_JSON = Path(__file__).resolve().parent / "experiment_outputs" / f"softreq_sampling_{INSTANCE}.json"
OUT_CSV = OUT_JSON.with_suffix(".csv")


def fixed_rate_oracle(soft_names, pct, seed):
    """A RandomSampleOracle whose allowed set is a fixed pct% of the soft constraints
    (escalation disabled -> the sampling rate never changes)."""
    return RandomSampleOracle(soft_names, start_pct=pct, escalate_step=0,
                              escalate_after=10**9, seed=seed, max_pct=pct)


def sample_can_repair(hard, soft, soft_names, allowed):
    """Feasibility predicate: can deleting only `allowed` constraints restore SAT?
    True iff hard + (all soft NOT in allowed) is satisfiable."""
    kept = [c for c, n in zip(soft, soft_names) if n not in allowed]
    return cp.Model(hard + kept).solve(solver=GATE_SOLVER) is True


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (None, None)
    return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def main():
    soft, hard, soft_names, hard_names = load_flat_instance(INSTANCE)
    name_of = {id(c): n for c, n in zip(soft, soft_names)}
    print(f"{INSTANCE}: {len(soft)} soft, {len(hard)} hard. "
          f"Sweeping pct={PCTS}, {N_SEEDS} seeds each.\n", flush=True)

    rows = []  # per-pct aggregate
    per_cell = []  # every (pct, seed) record, for the JSON dump

    for pct in PCTS:
        n_success = 0
        b_q, b_mus, b_sz, b_t = [], [], [], []
        s_q, s_mus, s_sz, s_solve, s_t = [], [], [], [], []

        for i in range(N_SEEDS):
            seed = SEED0 + i
            allowed = fixed_rate_oracle(soft_names, pct, seed).allowed
            repairable = sample_can_repair(hard, soft, soft_names, allowed)

            cell = {"pct": pct, "seed": seed, "n_allowed": len(allowed),
                    "repairable": repairable}

            if repairable:
                n_success += 1
                # --- baseline (marco), fresh oracle with the same allowed set ---
                t = time.perf_counter()
                rb = run_oracle("baseline", soft, hard, name_of,
                                fixed_rate_oracle(soft_names, pct, seed),
                                max_iterations=BASELINE_CAP,
                                solver=SOLVER, map_solver=MAP_SOLVER)
                tb = time.perf_counter() - t
                # --- staged-deletion, same allowed set ---
                t = time.perf_counter()
                rs = run_staged_deletion(soft, hard, soft_names,
                                         fixed_rate_oracle(soft_names, pct, seed),
                                         solver=SOLVER, map_solver=MAP_SOLVER,
                                         max_iterations=BASELINE_CAP)
                ts = time.perf_counter() - t

                acc = rb["accepted"]
                b_q.append(rb["n_mcs_seen"]); b_mus.append(rb["n_mus_seen"])
                b_sz.append(len(acc) if acc else None); b_t.append(tb)
                s_q.append(rs["n_decisions"]); s_mus.append(rs["n_mus_seen"])
                s_sz.append(rs["n_relaxed"]); s_solve.append(rs["n_solve_calls"]); s_t.append(ts)

                cell["baseline"] = {"accepted": bool(acc), "n_mcs_seen": rb["n_mcs_seen"],
                                    "n_mus_seen": rb["n_mus_seen"],
                                    "corr_size": len(acc) if acc else None, "seconds": tb}
                cell["staged"] = {"reached_sat": rs["reached_sat"], "n_decisions": rs["n_decisions"],
                                  "n_mus_seen": rs["n_mus_seen"], "corr_size": rs["n_relaxed"],
                                  "n_solve_calls": rs["n_solve_calls"], "seconds": ts}
            per_cell.append(cell)

        agg = {
            "pct": pct,
            "success_rate": n_success / N_SEEDS,
            "n_success": n_success,
            "baseline_queries": mean_std(b_q),
            "baseline_mus": mean_std(b_mus),
            "baseline_corr_size": mean_std(b_sz),
            "baseline_seconds": mean_std(b_t),
            "staged_queries": mean_std(s_q),
            "staged_mus": mean_std(s_mus),
            "staged_corr_size": mean_std(s_sz),
            "staged_solve_calls": mean_std(s_solve),
            "staged_seconds": mean_std(s_t),
        }
        rows.append(agg)

        def fmt(ms):
            m, s = ms
            return "   -  " if m is None else f"{m:5.1f}±{s:4.1f}"
        print(f"pct={pct:3d}  success={n_success:2d}/{N_SEEDS} ({agg['success_rate']*100:3.0f}%) | "
              f"BASE q={fmt(agg['baseline_queries'])} sz={fmt(agg['baseline_corr_size'])} "
              f"mus={fmt(agg['baseline_mus'])} t={fmt(agg['baseline_seconds'])} | "
              f"STAGED q={fmt(agg['staged_queries'])} sz={fmt(agg['staged_corr_size'])} "
              f"mus={fmt(agg['staged_mus'])} solve={fmt(agg['staged_solve_calls'])} "
              f"t={fmt(agg['staged_seconds'])}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"instance": INSTANCE, "n_soft": len(soft), "n_hard": len(hard),
                   "pcts": PCTS, "n_seeds": N_SEEDS, "seed0": SEED0,
                   "aggregate": rows, "cells": per_cell}, f, indent=2)

    # CSV of the per-pct aggregates (means only) for easy plotting
    cols = ["pct", "success_rate", "n_success",
            "baseline_queries", "baseline_corr_size", "baseline_mus", "baseline_seconds",
            "staged_queries", "staged_corr_size", "staged_mus", "staged_solve_calls", "staged_seconds"]
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            def m(key):
                v = r[key]
                return "" if (isinstance(v, tuple) and v[0] is None) else (v[0] if isinstance(v, tuple) else v)
            f.write(",".join(str(m(c)) for c in cols) + "\n")

    print(f"\nWrote {OUT_JSON}\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
