"""
    Seven-method suite sweep (post-determinism-fix), writing to its OWN results file so the
    pre-fix rows in results/mss-20.csv stay separate.

        python run_suite_v2.py --out results/suite-v2-a.csv --workers 16 --methods A B C

    Cells: 3 problem classes x 6 instances x 20 mss-20 oracles = 360 per method.
    Resumable: rows already present in --out are skipped.
"""
import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hierarchy
import oracles as orc
import run

ALL7 = ["mcs-enumeration", "selective-relaxation", "hierarch-commit",
        "hierarch-premature-commit", "hierarch-commit-nocap",
        "hierarch-premature-commit-nocap", "hierarch-commit-nocap-baseline"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=ALL7)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--budget", type=float, default=600.0)
    args = ap.parse_args()

    cells = [(p, i, "mss-20", o["seed"], m)
             for p in ("nurse-suite", "thesis-suite", "workforce-suite")
             for i in hierarchy.list_instances(p)
             for o in orc.load_oracles(p, i, "mss-20")
             for m in args.methods]

    out = Path(args.out)
    done = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["problem"], r["instance"], r["sampling"], int(r["seed"]), r["method"]))
    todo = [c for c in cells if c not in done]
    print(f"{len(cells)} cells, {len(done)} already done, {len(todo)} to run "
          f"({args.workers}-way, budget {args.budget:.0f}s) -> {out}", flush=True)

    new = not out.exists()
    fh = open(out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=run.COLUMNS)
    if new:
        w.writeheader(); fh.flush()

    t0 = time.perf_counter(); n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run.run_cell_subprocess, *c, args.budget): c for c in todo}
        for fut in as_completed(futs):
            row = fut.result(); w.writerow(row); fh.flush(); n += 1
            if n % 25 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}] {(time.perf_counter()-t0)/60:.0f} min elapsed", flush=True)
    print("SUITE_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
