"""
    Base-vs-incremental MARCO replay on the cells hierarch-premature-commit SOLVED.

    The sweep does not persist decision scripts, but the premature oracle is seeded, so re-running
    a solved (problem, instance, scheme, seed) cell reproduces its run. For each solved cell we:

      1. re-run hierarch-premature-commit to regenerate its decision script, and SAVE the script
         to ``runtime_premature/scripts/`` (the step-by-step log);
      2. replay the script through INCREMENTAL MARCO (persistent hierarchical_marco);
      3. replay it through BASE MARCO (flat marco re-enumerated from scratch at every step);
      both replays run under a per-replay budget and report whether they COMPLETED the script.

        python runtime_premature.py --cell PROBLEM INSTANCE SCHEME SEED BUDGET   # one cell
        python runtime_premature.py --drive [--workers N] [--budget 600]         # all solved cells

    Output: one JSON line per cell in ``runtime_premature/results.jsonl`` (drive mode appends,
    already-done cells are skipped -- stop/resume safe).
"""
import argparse
import csv
import glob
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import _bootstrap  # noqa: F401
from cpmpy.tools.explain import marco, hierarchical_marco

import hierarchy
import oracles as orc
from methods import HierarchPrematureCommitOracle, SOLVER, MAP_SOLVER, ROUND_CAP
from runtime import _ScriptedDecider

HERE = Path(__file__).resolve().parent
OUT = HERE / "runtime_premature"
METHOD = "hierarch-premature-commit"


def solved_cells():
    rows = []
    for f in glob.glob(str(HERE / "results" / "*.csv")):
        rows += list(csv.DictReader(open(f, newline="", encoding="utf-8")))
    return sorted({(r["problem"], r["instance"], r["sampling"], int(r["seed"]))
                   for r in rows
                   if r["method"] == METHOD and r["repaired"].strip().lower() == "true"})


def record(problem, instance, scheme, seed, budget):
    """Re-run the (seeded) premature oracle; return (root, hard, script, result)."""
    root, hard = hierarchy.load_instance(problem, instance)
    o = next(x for x in orc.load_oracles(problem, instance, scheme) if x["seed"] == seed)
    oracle = HierarchPrematureCommitOracle(root, hard, set(o["S"]), seed=seed,
                                           time_budget=budget)
    oracle.t0 = time.perf_counter()
    for _ in hierarchical_marco(root, list(hard), solver=SOLVER, map_solver=MAP_SOLVER,
                                decide_step=oracle, deadline=oracle.t0 + budget,
                                round_cap=ROUND_CAP):
        pass
    return root, hard, oracle.script, oracle.result


def time_incremental_c(root, hard, script, budget):
    """(elapsed, completed) for the persistent hierarchical_marco replay."""
    decider = _ScriptedDecider(script)
    t0 = time.perf_counter()
    for _ in hierarchical_marco(root, list(hard), solver=SOLVER, map_solver=MAP_SOLVER,
                                decide_step=decider, deadline=t0 + budget,
                                round_cap=ROUND_CAP):
        pass
    return time.perf_counter() - t0, decider.i >= len(decider.script)


def time_base_c(root, hard, script, budget):
    """(elapsed, completed) for flat marco re-run from scratch at every step
    (same replay semantics as runtime.time_base, plus a completion flag)."""
    name2node = {nd.get_full_name(): nd for nd in root.iter_nodes()}
    grouped = lambda nm: name2node[nm].get_grouped_constraint()
    frontier = [c.get_full_name() for c in root.children]
    pruned, relaxed, stack = set(), set(), []
    t0 = time.perf_counter()
    completed = True
    for step in script:
        if time.perf_counter() - t0 >= budget:
            completed = False
            break
        soft = [grouped(g) for g in frontier if g not in relaxed and g not in pruned]
        soft = [c for c in soft if c is not None]
        hard_now = list(hard) + [grouped(g) for g in pruned if grouped(g) is not None]
        if soft:
            for _ in marco(soft, hard_now, solver=SOLVER, map_solver=MAP_SOLVER,
                           return_mus=True, return_mcs=True):
                pass
        a = step["action"]
        if a == "refine":
            frontier = [x for x in frontier if x != step["group"]] + list(step["children"])
        elif a == "commit":
            stack.append((list(frontier), set(pruned), set(relaxed)))
            mcs = set(step["mcs"])
            for g in list(frontier):
                if g in relaxed or g in pruned:
                    continue
                if g in mcs:
                    if not name2node[g].children:
                        relaxed.add(g)
                else:
                    pruned.add(g)
        elif a == "backtrack":
            if stack:
                frontier, pruned, relaxed = stack.pop()
        elif a == "stop":
            break
    return time.perf_counter() - t0, completed


def run_cell(problem, instance, scheme, seed, budget):
    root, hard, script, result = record(problem, instance, scheme, seed, budget)
    (OUT / "scripts").mkdir(parents=True, exist_ok=True)
    sp = OUT / "scripts" / f"{scheme}_{problem}_{instance}_seed{seed}.json"
    sp.write_text(json.dumps(script, indent=1), encoding="utf-8")
    t_inc, inc_done = time_incremental_c(root, hard, script, budget)
    t_base, base_done = time_base_c(root, hard, script, budget)
    return {"problem": problem, "instance": instance, "scheme": scheme, "seed": seed,
            "reproduced_result": result, "n_steps": len(script),
            "incremental_s": round(t_inc, 3), "incremental_completed": inc_done,
            "base_s": round(t_base, 3), "base_completed": base_done,
            "speedup": round(t_base / t_inc, 2) if (inc_done and base_done and t_inc > 0)
                       else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", nargs=5, metavar=("PROBLEM", "INSTANCE", "SCHEME", "SEED", "BUDGET"))
    ap.add_argument("--drive", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=float, default=600.0)
    args = ap.parse_args()

    if args.cell:
        p, i, s, seed, budget = args.cell
        row = run_cell(p, i, s, int(seed), float(budget))
        print("RESULT " + json.dumps(row), flush=True)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "results.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["problem"], r["instance"], r["scheme"], r["seed"]))
    todo = [c for c in solved_cells() if c not in done]
    print(f"{len(todo)} solved cells to replay ({args.workers}-way, budget {args.budget:.0f}s)",
          flush=True)

    def work(c):
        p, i, s, seed = c
        cmd = [sys.executable, str(HERE / "runtime_premature.py"), "--cell",
               p, i, s, str(seed), str(args.budget)]
        out = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True,
                             timeout=3 * args.budget + 300)
        for line in out.stdout.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[len("RESULT "):])
        return {"problem": p, "instance": i, "scheme": s, "seed": seed, "error": True,
                "stderr_tail": out.stderr[-400:]}

    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for fut in as_completed(futs):
            row = fut.result()
            with open(res_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            n += 1
            print(f"[{n}/{len(todo)}] {row.get('scheme')} {row.get('problem')} "
                  f"seed{row.get('seed')} steps={row.get('n_steps')} "
                  f"inc={row.get('incremental_s')}s({row.get('incremental_completed')}) "
                  f"base={row.get('base_s')}s({row.get('base_completed')})", flush=True)
    print("REPLAY DONE", flush=True)


if __name__ == "__main__":
    main()
