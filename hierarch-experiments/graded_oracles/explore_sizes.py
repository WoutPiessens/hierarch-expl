"""
EXPLORATION: what minimal-correction sizes does each instance naturally admit?

mss-20 builds S around a "natural correction" = the complement of a randomly-grown Maximal
Satisfiable Subset. That complement is itself a MINIMAL correction (MCS), so its size IS a
minimal-MCS size. By sampling many of them we see the range of minimal-MCS sizes each instance
supports -- which tells us which difficulty bins (<=5 / 6-10 / 11-15 / 16-20) are reachable and
whether we can find >=5 distinct corrections per bin.
"""
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # hierarch-experiments on path
import random
import hierarchy
from sampling import _mss_correction

N = 120
BINS = [(1, 5), (6, 10), (11, 15), (16, 20)]
BLAB = ["<=5", "6-10", "11-15", "16-20"]


def explore(problem, inst):
    root, hard = hierarchy.load_instance(problem, inst)
    nl = len(root.leaves())
    seen = set()
    sizes = Counter()
    for seed in range(N):
        C = frozenset(_mss_correction(root, hard, random.Random(seed)))
        if C not in seen:
            seen.add(C)
            sizes[len(C)] += 1
    # distinct corrections per bin
    perbin = [0] * len(BINS)
    for sz, cnt in sizes.items():
        for i, (lo, hi) in enumerate(BINS):
            if lo <= sz <= hi:
                perbin[i] += cnt
    return problem, inst, nl, dict(sizes), perbin


def main():
    jobs = [(p, i) for p in ("nurse-suite", "thesis-suite", "workforce-suite")
            for i in hierarchy.list_instances(p)]
    res = []
    with ProcessPoolExecutor(max_workers=9) as ex:
        futs = [ex.submit(explore, p, i) for p, i in jobs]
        for f in as_completed(futs):
            res.append(f.result())
    res.sort()
    print(f"{'instance':40} {'nleaves':>7} {'sizes(min-max)':>14}   " +
          "  ".join(f"{b:>6}" for b in BLAB))
    for problem, inst, nl, sizes, perbin in res:
        smin, smax = min(sizes), max(sizes)
        tag = f"{problem.split('-')[0]}/{inst}"
        print(f"{tag:40} {nl:>7} {f'{smin}-{smax}':>14}   " +
              "  ".join(f"{x:>6}" for x in perbin))
    print("DONE_EXPLORE", flush=True)


if __name__ == "__main__":
    main()
