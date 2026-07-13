"""
    Experiment runner. Sweeps every (problem, instance, oracle, method) cell and writes one CSV row
    per run. The two sampling schemes go to SEPARATE result files:

        results/mss-20.csv        results/random-40.csv

    Each cell runs in its OWN subprocess with a hard timeout (budget + margin), so one slow/hung
    solve can never stall the sweep, and results stream to disk -- a finished cell is never re-run,
    so the sweep is stop/resume safe (ideal for remote servers).

        python run.py                       # everything, both schemes, all methods (resumable)
        python run.py --schemes mss-20
        python run.py --problems nurse thesis --methods hierarch-commit
        python run.py --budget 600
        python run.py --list                # show the cells that would run
        python run.py --cell nurse instance2 mss-20 0 hierarch-commit 600   # one cell (internal)

    CSV columns:
        sampling, problem, instance, seed, method, decisions, judgments, relaxed, pruned,
        excess, commits, backtracks, repaired, timed_out, elapsed_time
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import hierarchy
import oracles as orc

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BUDGET = 600.0
KILL_MARGIN = 120.0                      # subprocess hard-kill grace over the solver budget

COLUMNS = ["sampling", "problem", "instance", "seed", "method", "decisions", "judgments",
           "relaxed", "pruned", "excess", "commits", "backtracks", "repaired",
           "timed_out", "elapsed_time"]


def csv_path(scheme):
    return RESULTS / f"{scheme}.csv"


# ------------------------------------------------------------- one cell -------
def run_cell(problem, instance, scheme, seed, method, budget):
    """Run a single method on a single oracle; return a full CSV-row dict."""
    from methods import METHODS
    root, hard = hierarchy.load_instance(problem, instance)
    oracle = next(o for o in orc.load_oracles(problem, instance, scheme) if o["seed"] == seed)
    S = set(oracle["S"])
    if method.startswith("hierarch"):                        # every hierarch-* variant takes a seed
        m = METHODS[method](root, hard, S, seed=seed, time_budget=budget)
    else:
        m = METHODS[method](root, hard, S, time_budget=budget)
    return {"sampling": scheme, "problem": problem, "instance": instance, "seed": seed,
            "method": method, "decisions": m["decisions"], "judgments": m["judgments"],
            "relaxed": m["relaxed"], "pruned": m["pruned"], "excess": m["excess"],
            "commits": m["commits"], "backtracks": m["backtracks"], "repaired": m["repaired"],
            "timed_out": m["timed_out"], "elapsed_time": m["elapsed"]}


# ------------------------------------------------------------- sweep ----------
def all_cells(problems, instances_of, schemes, methods):
    for problem in problems:
        for instance in instances_of(problem):
            for scheme in schemes:
                for o in orc.load_oracles(problem, instance, scheme):
                    for method in methods:
                        yield (problem, instance, scheme, o["seed"], method)


def _key(row):
    return (row["problem"], row["instance"], row["sampling"], int(row["seed"]), row["method"])


def load_done(scheme):
    p = csv_path(scheme)
    if not p.exists():
        return set()
    with open(p, newline="", encoding="utf-8") as f:
        return {_key(row) for row in csv.DictReader(f)}


def append_row(scheme, row):
    p = csv_path(scheme)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_cell_subprocess(problem, instance, scheme, seed, method, budget):
    """Spawn `python run.py --cell ...` and parse its RESULT line; hard-kill on overrun."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, str(HERE / "run.py"), "--cell", problem, instance, scheme,
           str(seed), method, str(budget)]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=str(HERE), env=env, capture_output=True, text=True,
                           timeout=budget + KILL_MARGIN)
        for line in p.stdout.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[len("RESULT "):])
        # no RESULT: report an error row (kept so the sweep does not retry forever)
        return {"sampling": scheme, "problem": problem, "instance": instance, "seed": seed,
                "method": method, "decisions": None, "judgments": None, "relaxed": None,
                "pruned": None, "excess": None, "commits": None, "backtracks": None,
                "repaired": False, "timed_out": False,
                "elapsed_time": round(time.perf_counter() - t0, 1)}
    except subprocess.TimeoutExpired:
        return {"sampling": scheme, "problem": problem, "instance": instance, "seed": seed,
                "method": method, "decisions": None, "judgments": None, "relaxed": None,
                "pruned": None, "excess": None, "commits": None, "backtracks": None,
                "repaired": False, "timed_out": True,
                "elapsed_time": round(time.perf_counter() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", nargs="+", default=hierarchy.PROBLEMS)
    ap.add_argument("--schemes", nargs="+", default=list(orc.SCHEMES.keys()))
    ap.add_argument("--methods", nargs="+",
                    default=["mcs-enumeration", "selective-relaxation",
                             "hierarch-commit", "hierarch-commit-nocap",
                             "hierarch-explore-backtrack", "hierarch-premature-commit",
                             "hierarch-random-commit", "hierarch-fresh-restart"])
    ap.add_argument("--budget", type=float, default=BUDGET)
    ap.add_argument("--list", action="store_true", help="print the cells that would run and exit")
    ap.add_argument("--cell", nargs=6, metavar=("PROBLEM", "INSTANCE", "SCHEME", "SEED", "METHOD",
                    "BUDGET"), help="internal: run one cell and print a RESULT json line")
    args = ap.parse_args()

    if args.cell:
        problem, instance, scheme, seed, method, budget = args.cell
        row = run_cell(problem, instance, scheme, int(seed), method, float(budget))
        print("RESULT " + json.dumps(row), flush=True)
        return

    cells = list(all_cells(args.problems, hierarchy.list_instances, args.schemes, args.methods))
    if args.list:
        for c in cells:
            print("  " + "/".join(map(str, c)))
        print(f"{len(cells)} cells")
        return

    done = {s: load_done(s) for s in args.schemes}
    todo = [c for c in cells if (c[0], c[1], c[2], c[3], c[4]) not in done[c[2]]]
    print(f"{len(cells)} cells total, {len(cells) - len(todo)} already done, {len(todo)} to run "
          f"(budget {args.budget:.0f}s, hard-kill +{KILL_MARGIN:.0f}s).")

    t_all = time.perf_counter()
    for i, (problem, instance, scheme, seed, method) in enumerate(todo, 1):
        tag = f"{problem}/{instance} {scheme} seed{seed} {method}"
        print(f"[{i}/{len(todo)}] {tag} ...", end=" ", flush=True)
        row = run_cell_subprocess(problem, instance, scheme, seed, method, args.budget)
        append_row(scheme, row)
        print(f"relaxed={row['relaxed']} dec={row['decisions']} judg={row['judgments']} "
              f"repaired={row['repaired']} ({row['elapsed_time']}s)", flush=True)
    print(f"\nall done in {(time.perf_counter() - t_all) / 60:.1f} min. "
          f"Results in {RESULTS}/<scheme>.csv")


if __name__ == "__main__":
    main()
