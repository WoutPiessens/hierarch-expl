"""
    Parallel sweep runner: same cells/CSVs as ``run.py``, but runs WORKERS cells concurrently
    (each cell still in its own hard-killed ``run.py --cell`` subprocess). Only this parent
    process appends to the CSVs, so rows never race *within* one instance.

    To spread a sweep over SEVERAL machines, give each machine a disjoint shard -- e.g. one
    scheme per host, so each host appends to its own results/<scheme>.csv and there is no
    cross-host (NFS) append race:

        hostA$ python run_parallel.py --schemes mss-20    --workers 4
        hostB$ python run_parallel.py --schemes random-40 --workers 4

    Stop/resume-safe exactly like run.py: finished cells are skipped on restart.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import hierarchy
import oracles as orc
import run


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
    ap.add_argument("--budget", type=float, default=run.BUDGET)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cells = list(run.all_cells(args.problems, hierarchy.list_instances, args.schemes,
                               args.methods))
    done = {s: run.load_done(s) for s in args.schemes}
    todo = [c for c in cells if (c[0], c[1], c[2], c[3], c[4]) not in done[c[2]]]
    print(f"{len(cells)} cells total, {len(cells) - len(todo)} already done, {len(todo)} to run "
          f"({args.workers}-way parallel, budget {args.budget:.0f}s, "
          f"hard-kill +{run.KILL_MARGIN:.0f}s).", flush=True)

    t_all = time.perf_counter()
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run.run_cell_subprocess, *c, args.budget): c for c in todo}
        for fut in as_completed(futs):
            row = fut.result()
            run.append_row(row["sampling"], row)
            n += 1
            print(f"[{n}/{len(todo)}] {row['problem']}/{row['instance']} {row['sampling']} "
                  f"seed{row['seed']} {row['method']} relaxed={row['relaxed']} "
                  f"repaired={row['repaired']} to={row['timed_out']} "
                  f"({row['elapsed_time']}s)", flush=True)
    print(f"\nall done in {(time.perf_counter() - t_all) / 60:.1f} min. "
          f"Results in {run.RESULTS}/<scheme>.csv", flush=True)


if __name__ == "__main__":
    main()
