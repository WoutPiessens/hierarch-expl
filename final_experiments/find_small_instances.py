"""
    Search for SIMPLER instances -- natural-MCS size ~5 (instead of ~10) -- one per problem.

    Candidates are generated with the existing builders into ``<problem>/data/`` and measured
    with the random-MSS technique (`common.mss_correction_set`, N_SEEDS draws): the mean draw
    size estimates the instance's typical/average MCS size. Per problem the candidate whose
    mean is closest to TARGET (within [MIN_OK, MAX_OK]) is selected.

    Output: prints one line per candidate + writes ``small_selection.json``.

        python find_small_instances.py            # search all three problems
"""
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common
import cpmpy as cp

TARGET, MIN_OK, MAX_OK = 5.0, 3.0, 7.5
N_SEEDS = 5

NURSE_IDXS = [1, 3, 4, 5]                      # instance2 is the current (harder) one
THESIS_TRANSCRIPTS = [1, 2, 3, 5]              # transcript_4 is the current one
WORKFORCE_DROPS = [2, 3, 4, 6]                 # N_DROP=12 is the current one


def _load_builder(problem):
    spec = importlib.util.spec_from_file_location(
        f"{problem}_build", HERE / problem / "build_instance.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure(problem, instance):
    root, hard = common.load_instance(problem, instance)
    L = len(root.leaves())
    soft = [lf.get_grouped_constraint() for lf in root.leaves()]
    full_sat = cp.Model(list(hard) + soft).solve(solver=common.GATE_SOLVER)
    hard_sat = cp.Model(list(hard)).solve(solver=common.GATE_SOLVER)
    rec = {"problem": problem, "instance": instance, "L": L}
    if full_sat is not False or hard_sat is not True:
        rec.update(ok=False, reason=f"full_sat={full_sat} hard_sat={hard_sat}")
        return rec
    t0 = time.perf_counter()
    sizes = [len(common.mss_correction_set(root, hard, random.Random(s)))
             for s in range(N_SEEDS)]
    rec.update(ok=True, sizes=sorted(sizes), mean=round(sum(sizes) / len(sizes), 2),
               t_measure=round(time.perf_counter() - t0, 1))
    return rec


def log(rec):
    print(json.dumps(rec), flush=True)


def main():
    results = []

    # ---- nurse: other schedulingbenchmarks indices ----------------------------------
    nurse = _load_builder("nurse")
    for idx in NURSE_IDXS:
        inst = f"instance{idx}"
        try:
            if not (common.instance_path("nurse", inst) / "hierarchy.json").exists():
                nurse.main(idx)
            rec = measure("nurse", inst)
        except Exception as e:
            rec = {"problem": "nurse", "instance": inst, "ok": False,
                   "reason": f"{type(e).__name__}: {e}"}
        log(rec); results.append(rec)

    # ---- thesis: other transcripts ---------------------------------------------------
    thesis = _load_builder("thesis")
    for i in THESIS_TRANSCRIPTS:
        inst = f"defense-transcript{i}"
        src = common._ROOT / "experiments" / "data" / "hierarchies" / f"transcript_{i}_hierarchy"
        try:
            if not (common.instance_path("thesis", inst) / "hierarchy.json").exists():
                thesis.main(src=src, instance=inst)
            rec = measure("thesis", inst)
        except Exception as e:
            rec = {"problem": "thesis", "instance": inst, "ok": False,
                   "reason": f"{type(e).__name__}: {e}"}
        log(rec); results.append(rec)

    # ---- workforce: fewer coverage-preserving team drops ------------------------------
    for nd in WORKFORCE_DROPS:
        inst = f"ews-drop{nd}"
        try:
            if not (common.instance_path("workforce", inst) / "hierarchy.json").exists():
                wf = _load_builder("workforce")           # fresh module per drop count
                wf.N_DROP = nd
                wf.INSTANCE_NAME = inst
                wf.main()
            rec = measure("workforce", inst)
        except Exception as e:
            rec = {"problem": "workforce", "instance": inst, "ok": False,
                   "reason": f"{type(e).__name__}: {e}"}
        log(rec); results.append(rec)

    # ---- select per problem ------------------------------------------------------------
    selection = {}
    for problem in ("nurse", "thesis", "workforce"):
        cands = [r for r in results if r["problem"] == problem and r.get("ok")
                 and MIN_OK <= r["mean"] <= MAX_OK]
        if cands:
            best = min(cands, key=lambda r: abs(r["mean"] - TARGET))
            selection[problem] = {"instance": best["instance"], "mean": best["mean"],
                                  "sizes": best["sizes"], "L": best["L"]}
        else:
            selection[problem] = None
    (HERE / "small_selection.json").write_text(json.dumps(
        {"selection": selection, "all": results}, indent=2))
    print("\nSELECTION " + json.dumps(selection), flush=True)


if __name__ == "__main__":
    main()
