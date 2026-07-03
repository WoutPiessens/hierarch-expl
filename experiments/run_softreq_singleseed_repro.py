"""
    Reproducible single-sample query counts for baseline vs staged-deletion.

    The problem this fixes: cpmpy names assumption variables from a PROCESS-GLOBAL counter, so
    running one method before another in the same process shifts variable indices and changes
    the map-solver's enumeration order (hence how many MCSes each method needs). To make the
    counts order-independent, we run EVERY (method, rate) in its own fresh subprocess -- each
    starts from the identical counter state, so there is no cross-method contamination.

    Driver mode (no args): sweeps rates for one seed, runs each cell in a fresh process TWICE,
    and checks the two runs agree (reproducibility proof).
    Worker mode (--worker METHOD PCT SEED): runs a single method once, prints "RESULT:<queries>".

    Run from experiments/:  python run_softreq_singleseed_repro.py
"""

import subprocess
import sys
from pathlib import Path

INSTANCE = "nurse_instance1_softreq_8nurses"
SEED = 48
PCTS = [40, 50, 60, 70, 80, 90]
SOLVER = "exact"
MAP_SOLVER = "exact"
CAP = 5000


def worker(method, pct, seed):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hierarchy_io import load_flat_instance
    from oracle import run_oracle, run_staged_deletion, RandomSampleOracle

    soft, hard, soft_names, hard_names = load_flat_instance(INSTANCE)
    oracle = RandomSampleOracle(soft_names, start_pct=pct, escalate_step=0,
                                escalate_after=10**9, seed=seed, max_pct=pct)
    if method == "baseline":
        name_of = {id(c): n for c, n in zip(soft, soft_names)}
        r = run_oracle("baseline", soft, hard, name_of, oracle,
                       max_iterations=CAP, solver=SOLVER, map_solver=MAP_SOLVER)
        print(f"RESULT:{r['n_mcs_seen']}")
    else:
        r = run_staged_deletion(soft, hard, soft_names, oracle,
                                solver=SOLVER, map_solver=MAP_SOLVER, max_iterations=CAP)
        print(f"RESULT:{r['n_decisions']}")


def run_cell(method, pct, seed):
    out = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                          "--worker", method, str(pct), str(seed)],
                         capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("RESULT:"):
            return int(line.split(":")[1])
    raise RuntimeError(f"worker failed ({method},{pct}):\n{out.stdout}\n{out.stderr}")


def driver():
    print(f"{INSTANCE}, seed={SEED}, fresh subprocess per (method,rate), each run twice.\n")
    print(f"{'pct':>4} {'baseline':>9} {'staged':>8}  {'reproducible?':>13}")
    rows = []
    for pct in PCTS:
        b1 = run_cell("baseline", pct, SEED); b2 = run_cell("baseline", pct, SEED)
        s1 = run_cell("staged", pct, SEED);   s2 = run_cell("staged", pct, SEED)
        ok = (b1 == b2) and (s1 == s2)
        rows.append((pct, b1, s1))
        print(f"{pct:>4} {b1:>9} {s1:>8}  {'yes' if ok else f'NO b={b1},{b2} s={s1},{s2}':>13}")
    print("\nPYDATA", rows)


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        driver()
