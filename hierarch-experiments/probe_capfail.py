"""
    Is the early-fail phenomenon (stop 'failed' with budget left) DIRECTLY caused by the
    per-round conflict cap (ROUND_CAP)?

    Probe: run hierarch-commit on its early-fail cells with an oracle that, at the moment it
    would stop 'failed', records ctx['capped'] (did the enumerator truncate this round, i.e.
    does the CURRENT frontier provably have unseen conflicts?) and, if capped, returns
    {'action': 'continue'} instead of stopping -- fetching the next ROUND_CAP conflicts of the
    same frontier. If continuing surfaces new conflicts/options (or even repairs), the fail was
    cap-induced; if the failed stops occur with capped=False, the frontier's conflict set was
    fully enumerated and the cause lies elsewhere (guard exhaustion), not the cap.

        python probe_capfail.py --cell PROBLEM INSTANCE SCHEME SEED   # one cell
        python probe_capfail.py --list                                # early-fail cells of base
"""
import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from cpmpy.tools.explain import hierarchical_marco

import hierarchy
import oracles as orc
from methods import HierarchCommitOracle, SOLVER, MAP_SOLVER, ROUND_CAP, _repaired

HERE = Path(__file__).resolve().parent
MAX_CONTINUES = 40


class CapFailProbe(HierarchCommitOracle):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fail_events = []
        self._continues = 0

    def __call__(self, ctx):
        act = super().__call__(ctx)
        if act.get("action") == "stop" and self.result == "failed":
            ev = {"round": ctx["round"], "capped": bool(ctx.get("capped")),
                  "new_this_round": len([r for r in ctx["results"] if not r.get("cached")]),
                  "gmcs_known": len(self.gmcs)}
            self.fail_events.append(ev)
            if ev["capped"] and self._continues < MAX_CONTINUES:
                self._continues += 1
                self.result = None
                self.script.pop()                      # undo the recorded stop
                return {"action": "continue"}          # fetch more conflicts, same frontier
        return act


def early_fail_cells():
    rows = []
    for f in glob.glob(str(HERE / "results" / "*.csv")):
        rows += list(csv.DictReader(open(f, newline="", encoding="utf-8")))
    def b(x): return str(x).strip().lower() == "true"
    return sorted({(r["problem"], r["instance"], r["sampling"], int(r["seed"]))
                   for r in rows if r["method"] == "hierarch-commit"
                   and not b(r["repaired"]) and not b(r["timed_out"])})


def run_cell(problem, instance, scheme, seed, budget=600.0):
    root, hard = hierarchy.load_instance(problem, instance)
    o = next(x for x in orc.load_oracles(problem, instance, scheme) if x["seed"] == seed)
    oracle = CapFailProbe(root, hard, set(o["S"]), seed=seed, time_budget=budget)
    oracle.t0 = time.perf_counter()
    for _ in hierarchical_marco(root, list(hard), solver=SOLVER, map_solver=MAP_SOLVER,
                                decide_step=oracle, deadline=oracle.t0 + budget,
                                round_cap=ROUND_CAP):
        pass
    rel = set(oracle.relaxed)
    return {"problem": problem, "instance": instance, "scheme": scheme, "seed": seed,
            "result": oracle.result, "repaired": _repaired(root, hard, rel),
            "continues_used": oracle._continues, "n_fail_events": len(oracle.fail_events),
            "fail_capped": [e["capped"] for e in oracle.fail_events],
            "fail_new_this_round": [e["new_this_round"] for e in oracle.fail_events],
            "elapsed": round(time.perf_counter() - oracle.t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", nargs=4, metavar=("PROBLEM", "INSTANCE", "SCHEME", "SEED"))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for c in early_fail_cells():
            print(" ".join(map(str, c)))
        return
    p, i, s, seed = args.cell
    print("RESULT " + json.dumps(run_cell(p, i, s, int(seed))), flush=True)


if __name__ == "__main__":
    main()
