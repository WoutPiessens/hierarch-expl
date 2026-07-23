"""
Generate the HIERARCHY-FRIENDLY oracle set for every suite instance:

    oracles_hiernode.json  --  20 oracles per instance, each S = union of randomly-selected whole
    subtrees of the hierarchy grown until removing S restores satisfiability (see
    sampling.hiernode_oracle). Seeds are aligned with the mss-20 set (0..19) so cells pair up.

    python gen_hiernode.py [--workers 8]
"""
import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import hierarchy
import oracles as orc
from sampling import hiernode_oracle

N = 20


def gen_instance(problem, inst):
    root, hard = hierarchy.load_instance(problem, inst)
    nl = len(root.leaves())
    out = []
    for seed in range(N):
        S, picked = hiernode_oracle(root, hard, random.Random(seed))
        out.append({"scheme": "hiernode", "seed": seed, "pct": round(100 * len(S) / nl, 1),
                    "k": len(S), "n_nodes": len(picked), "corr_size": None, "S": sorted(S)})
    orc.save_oracles(problem, inst, "hiernode", out)
    return problem, inst, nl, sum(o["pct"] for o in out) / len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    jobs = [(p, i) for p in ("nurse-suite", "thesis-suite", "workforce-suite")
            for i in hierarchy.list_instances(p)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(gen_instance, p, i) for p, i in jobs]
        for f in as_completed(futs):
            problem, inst, nl, mean_pct = f.result()
            print(f"  {problem}/{inst}: {N} oracles, mean |S|={mean_pct:.0f}% of {nl} leaves",
                  flush=True)
    print("GEN_HIERNODE_DONE", flush=True)


if __name__ == "__main__":
    main()
