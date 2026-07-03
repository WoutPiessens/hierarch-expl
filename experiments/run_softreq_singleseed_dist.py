"""
    Query counts for baseline vs staged-deletion on ONE sample (seed 48), reported as a stable
    DISTRIBUTION over repeated runs.

    Why a distribution and not a single number: both `exact` and cpmpy's multi-worker `ortools`
    enumerate MCSes in a NON-deterministic order, so the number of MCSes a method shows the
    oracle varies run to run for identical input. A single run is therefore not reproducible;
    the mean over many runs is. We run each (method, rate) K times and report mean / std /
    min / max.

    Run from experiments/:  python run_softreq_singleseed_dist.py
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hierarchy_io import load_flat_instance
from oracle import run_oracle, run_staged_deletion, RandomSampleOracle

INSTANCE = "nurse_instance1_softreq_8nurses"
SEED = 48
PCTS = [40, 50, 60, 70, 80, 90]
K = 15
SOLVER = "exact"
MAP_SOLVER = "exact"
CAP = 5000
OUT = Path(__file__).resolve().parent / "experiment_outputs" / f"softreq_singleseed_dist_{INSTANCE}.json"


def oracle(soft_names, pct):
    return RandomSampleOracle(soft_names, start_pct=pct, escalate_step=0,
                              escalate_after=10**9, seed=SEED, max_pct=pct)


def main():
    soft, hard, soft_names, hard_names = load_flat_instance(INSTANCE)
    name_of = {id(c): n for c, n in zip(soft, soft_names)}
    print(f"{INSTANCE}, seed={SEED}, K={K} repeats per cell (solver={SOLVER}).\n")
    print(f"{'pct':>4} | {'baseline mean±std [min,max]':>30} | {'staged mean±std [min,max]':>28}")

    rows = []
    for pct in PCTS:
        bq, sq = [], []
        for _ in range(K):
            rb = run_oracle("baseline", soft, hard, name_of, oracle(soft_names, pct),
                            max_iterations=CAP, solver=SOLVER, map_solver=MAP_SOLVER)
            rs = run_staged_deletion(soft, hard, soft_names, oracle(soft_names, pct),
                                     solver=SOLVER, map_solver=MAP_SOLVER, max_iterations=CAP)
            bq.append(rb["n_mcs_seen"]); sq.append(rs["n_decisions"])
        bm, bs = statistics.mean(bq), (statistics.pstdev(bq) if K > 1 else 0)
        sm, ss = statistics.mean(sq), (statistics.pstdev(sq) if K > 1 else 0)
        rows.append({"pct": pct, "baseline": bq, "staged": sq,
                     "baseline_mean": bm, "baseline_std": bs, "baseline_min": min(bq), "baseline_max": max(bq),
                     "staged_mean": sm, "staged_std": ss, "staged_min": min(sq), "staged_max": max(sq)})
        print(f"{pct:>4} | {f'{bm:6.1f}±{bs:5.1f} [{min(bq)},{max(bq)}]':>30} | "
              f"{f'{sm:5.1f}±{ss:4.1f} [{min(sq)},{max(sq)}]':>28}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"instance": INSTANCE, "seed": SEED, "K": K, "rows": rows}, indent=2))
    print(f"\nWrote {OUT}")
    print("PYDATA", [(r["pct"], round(r["baseline_mean"], 1), r["baseline_min"], r["baseline_max"],
                     round(r["staged_mean"], 1), r["staged_min"], r["staged_max"]) for r in rows])


if __name__ == "__main__":
    main()
