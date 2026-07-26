"""
Select the final graded benchmark from a pooled candidate set (produced by
`gen_graded.py --pool-out pools.json`): per problem class, choose EXACTLY 6 instances and an
integer assignment x[i][b] of oracles per (instance, bin) such that

    * every selected instance supplies exactly 20 oracles  (sum_b x[i][b] == 20),
    * every bin gets exactly 30 oracles                    (sum_i x[i][b] == 30),
    * at most MAX_PER_FAMILY selected instances share a source family (diversity),

solved as an ILP (ortools). Then writes data/<class>/<inst>/oracles_graded.json for the selected
instances (20 oracles each, tagged with `bin`), and deletes any stale file on unselected ones.

    python assign_graded.py pools.json [--classes ...] [--per-instance 20] [--per-bin 30]
"""
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _bootstrap  # noqa
import cpmpy as cp
import hierarchy
import oracles as orc

BLAB = ["<=5", "6-10", "11-15", "16-20"]
MAX_PER_FAMILY = 3


def family(inst):
    return inst.split("-")[0] if inst.startswith("nurse") else re.sub(r"-d\d+$|-m[\d-]+$", "", inst)


def solve_class(problem, recs, per_inst, per_bin, n_sel, maxper=None, minper=None):
    maxper = per_inst if maxper is None else maxper
    minper = per_inst if minper is None else minper
    # availability a[inst][bin] and the concrete pooled oracles
    avail = defaultdict(lambda: [0] * len(BLAB))
    pool = defaultdict(lambda: defaultdict(list))          # inst -> bin -> [ (minmcs,S) ]
    for r in recs:
        b = BLAB.index(r["bin"])
        avail[r["inst"]][b] += 1
        pool[r["inst"]][b].append((r["minmcs"], r["S"]))
    insts = sorted(avail)
    B = len(BLAB)
    sel = cp.boolvar(shape=len(insts), name="sel")
    x = cp.intvar(0, maxper, shape=(len(insts), B), name="x")
    m = cp.Model()
    m += cp.sum(sel) == n_sel
    for ii, inst in enumerate(insts):
        for b in range(B):
            m += x[ii, b] <= avail[inst][b]
            m += x[ii, b] <= maxper * sel[ii]
        m += cp.sum(x[ii, :]) <= maxper * sel[ii]
        m += cp.sum(x[ii, :]) >= minper * sel[ii]
    for b in range(B):
        m += cp.sum(x[:, b]) == per_bin
    # diversity: cap selected instances per source family
    fams = defaultdict(list)
    for ii, inst in enumerate(insts):
        fams[family(inst)].append(ii)
    for fam, idxs in fams.items():
        m += cp.sum(sel[ii] for ii in idxs) <= MAX_PER_FAMILY
    if m.solve(solver="ortools") is not True:
        return None
    chosen = {}
    for ii, inst in enumerate(insts):
        if sel[ii].value():
            chosen[inst] = [int(x[ii, b].value()) for b in range(B)]
    return chosen, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pools")
    ap.add_argument("--classes", nargs="+", default=["nurse-graded", "thesis-graded", "workforce-graded"])
    ap.add_argument("--per-instance", type=int, default=20)
    ap.add_argument("--per-bin", type=int, default=30)
    ap.add_argument("--n-sel", type=int, default=6)
    ap.add_argument("--maxper", type=int, default=None, help="max oracles per instance (default=per-instance)")
    ap.add_argument("--minper", type=int, default=None, help="min oracles per selected instance")
    args = ap.parse_args()
    txt = Path(args.pools).read_text()
    recs = json.loads(txt) if txt.lstrip().startswith("[") else \
        [json.loads(l) for l in txt.splitlines() if l.strip()]
    by_class = defaultdict(list)
    for r in recs:
        by_class[r["problem"]].append(r)

    for problem in args.classes:
        print(f"\n=== {problem} ===")
        res = solve_class(problem, by_class.get(problem, []), args.per_instance, args.per_bin, args.n_sel, args.maxper, args.minper)
        if res is None:
            print("  INFEASIBLE with the current candidate pool -- need more/other instances "
                  "(report per-bin availability below)")
            av = defaultdict(lambda: [0] * len(BLAB))
            for r in by_class.get(problem, []):
                av[r["inst"]][BLAB.index(r["bin"])] += 1
            for inst in sorted(av):
                print(f"    {inst:26} avail {dict(zip(BLAB, av[inst]))}")
            continue
        chosen, pool = res
        # clear stale graded files on all candidate instances of this class
        for inst in hierarchy.list_instances(problem):
            p = hierarchy.instance_dir(problem, inst) / "oracles_graded.json"
            if p.exists():
                p.unlink()
        for inst, counts in sorted(chosen.items()):
            out, seed = [], 0
            for b, n in enumerate(counts):
                for mc, S in pool[inst][b][:n]:
                    out.append({"scheme": "graded", "seed": seed, "bin": BLAB[b],
                                "minmcs": mc, "k": len(S), "corr_size": mc, "S": S})
                    seed += 1
            orc.save_oracles(problem, inst, "graded", out)
            print(f"    {inst:26} {dict(zip(BLAB, counts))}  (total {sum(counts)})")
        print(f"  per-bin totals: " + "  ".join(
            f"{BLAB[b]}:{sum(c[b] for c in chosen.values())}" for b in range(len(BLAB))))
    print("ASSIGN_DONE", flush=True)


if __name__ == "__main__":
    main()
