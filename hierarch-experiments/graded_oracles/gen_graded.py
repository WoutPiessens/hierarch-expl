"""
GRADED oracle generator -- oracles stratified by minimal-MCS size, balanced per PROBLEM CLASS.

Idea (built on the mss-20 method): the "natural correction" mss-20 samples -- the complement of a
randomly-grown Maximal Satisfiable Subset -- is *itself a minimal correction* (an MCS). So setting
the suitable set S = that correction gives an oracle whose minimal-MCS size is EXACTLY |S|, with no
extra minimisation needed. We sample many corrections per instance, bin them by size into
  <=5 / 6-10 / 11-15 / 16-20,
and then, PER CLASS, fill each bin up to a target by drawing round-robin across whichever instances
can reach that bin (not every instance admits every size -- e.g. some nurse instances only have
size-8 MCSes -- so per-instance quotas are infeasible; per-class balance is the goal).

Writes data/<problem>/<instance>/oracles_graded.json (scheme "graded").

    python gen_graded.py [--maxseed 800] [--cap 14] [--target 30]
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _bootstrap  # noqa: F401  -- puts the cpmpy fork on sys.path; must precede cpmpy
import cpmpy as cp
from cpmpy.tools.explain.utils import make_assump_model
import hierarchy
import oracles as orc
from sampling import _mss_correction, _feasible

GATE = "ortools"
BINS = [(1, 5), (6, 10), (11, 15), (16, 20)]
BLAB = ["<=5", "6-10", "11-15", "16-20"]
CLASSES = ("nurse-suite", "thesis-suite", "workforce-suite")


def bin_of(sz):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= sz <= hi:
            return i
    return None


def _minmcs(relaxable, solver, aof, names):
    """Size of a minimal correction inside `relaxable` (greedy deletion via the assumption solver:
    enforce every leaf not currently relaxed; if still SAT, that leaf is unnecessary)."""
    relaxed = set(relaxable)
    for c in sorted(relaxable):
        trial = relaxed - {c}
        if solver.solve(assumptions=[aof[n] for n in names if n not in trial]) is True:
            relaxed = trial
    return len(relaxed)


def _pad_safe(C, budget, names, solver, aof, rng, tries=3):
    """Pad correction C with distractor leaves up to `budget`, keeping the minimal-MCS = |C| (so the
    padding never opens a shorter correction -- it only enlarges the suitable frontier). Returns S."""
    k = len(C)
    if k >= budget:
        return set(C)
    others = [n for n in names if n not in C]
    for _ in range(tries):                              # a few random pads; accept the first clean one
        rng.shuffle(others)
        S = set(C) | set(others[:budget - k])
        if _minmcs(S, solver, aof, names) == k:
            return S
    # fallback: add leaves one at a time (bounded scan), verifying each keeps the minimal-MCS at k
    S = set(C)
    for x in others[:4 * budget]:
        if len(S) >= budget:
            break
        if _minmcs(S | {x}, solver, aof, names) == k:
            S.add(x)
    return S


def pool_instance(problem, inst, maxseed, cap, pad):
    """Sample corrections; return {bin_index: [sorted-S, ...]} (<= cap distinct per bin). If pad>0,
    each correction is padded with size-preserving distractors up to pad*n_leaves."""
    root, hard = hierarchy.load_instance(problem, inst)
    solver = aof = names = None
    budget = 0
    if pad > 0:
        pairs = [(l.get_full_name(), l.get_grouped_constraint()) for l in root.leaves()]
        pairs = [(n, c) for n, c in pairs if c is not None]
        names = [n for n, _ in pairs]
        model, _sv, assump = make_assump_model([c for _, c in pairs], list(hard))
        solver = cp.SolverLookup.get(GATE, model)
        aof = {names[i]: assump[i] for i in range(len(assump))}
        budget = round(len(root.leaves()) * pad)
    pool = defaultdict(list)
    seen = set()
    for seed in range(maxseed):
        if all(len(pool[b]) >= cap for b in range(len(BINS))):
            break
        C = frozenset(_mss_correction(root, hard, random.Random(seed)))
        b = bin_of(len(C))
        if b is None or C in seen or len(pool[b]) >= cap:
            continue
        seen.add(C)
        S = _pad_safe(C, budget, names, solver, aof, random.Random(10_000 + seed)) if pad > 0 else set(C)
        pool[b].append((len(C), sorted(S)))            # (minmcs = |correction|, padded S)
    return problem, inst, {b: pool[b] for b in range(len(BINS))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxseed", type=int, default=800)
    ap.add_argument("--cap", type=int, default=14)          # per bin per instance in the pool
    ap.add_argument("--target", type=int, default=30)       # per bin per CLASS in the final set
    ap.add_argument("--pad", type=float, default=0.20)      # pad S to this fraction with distractors (0=off)
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()

    jobs = [(p, i) for p in CLASSES for i in hierarchy.list_instances(p)]
    pools = {}                                              # (problem, inst) -> {bin: [S,...]}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(pool_instance, p, i, args.maxseed, args.cap, args.pad) for p, i in jobs]
        for f in as_completed(futs):
            problem, inst, pool = f.result()
            pools[(problem, inst)] = pool
            cov = "  ".join(f"{BLAB[b]}:{len(pool[b])}" for b in range(len(BINS)))
            print(f"  pooled {problem}/{inst}   {cov}", flush=True)

    # ---- per-class balancing: fill each bin up to target, round-robin across instances ----
    selected = defaultdict(list)                            # (problem, inst) -> list of oracle dicts
    summary = {}
    for problem in CLASSES:
        insts = hierarchy.list_instances(problem)
        summary[problem] = {}
        for b in range(len(BINS)):
            queues = {i: list(pools[(problem, i)][b]) for i in insts}
            picked = 0
            while picked < args.target and any(queues.values()):
                for i in insts:
                    if not queues[i]:
                        continue
                    mc, S = queues[i].pop()
                    selected[(problem, i)].append({"bin": BLAB[b], "minmcs": mc, "S": S})
                    picked += 1
                    if picked >= args.target:
                        break
            summary[problem][BLAB[b]] = picked

    # ---- write per-instance oracle files (scheme "graded") ----
    for (problem, inst), oracles in selected.items():
        oracles.sort(key=lambda o: (o["minmcs"]))
        out = [{"scheme": "graded", "seed": k, "bin": o["bin"], "minmcs": o["minmcs"],
                "k": len(o["S"]), "corr_size": o["minmcs"], "S": o["S"]}
               for k, o in enumerate(oracles)]
        orc.save_oracles(problem, inst, "graded", out)

    print("\n=== per-class bin totals (target %d/bin) ===" % args.target)
    print(f"{'class':16} " + "  ".join(f"{b:>7}" for b in BLAB))
    for problem in CLASSES:
        print(f"{problem:16} " + "  ".join(f"{summary[problem][b]:>7}" for b in BLAB))
    Path("graded_summary.json").write_text(json.dumps(summary, indent=2))
    print("GEN_GRADED_DONE", flush=True)


if __name__ == "__main__":
    main()
