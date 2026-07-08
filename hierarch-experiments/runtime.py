"""
    Runtime experiment: base MARCO vs incremental MARCO.

    A hierarch-commit run produces a *script* of decisions (commit / refine / backtrack). We replay
    that exact script through two enumeration back-ends and compare total enumeration time:

      * INCREMENTAL : cpmpy's persistent ``hierarchical_marco`` -- the map solver (and, here, the
                      core solver) are built once and kept across every refine/commit/backtrack, so
                      blocking clauses accumulate and no work is redone.

      * BASE        : flat ``marco`` re-run FROM SCRATCH at every step. MARCO is stateless about the
                      interaction, so we track the running frontier by hand: a refined group is
                      replaced by its children, a pruned (backgrounded) group moves into `hard`, and
                      a relaxed leaf is dropped. Each step re-enumerates the current frontier's
                      MUS/MCSes with a fresh solver.

    Only wall-clock enumeration time is compared -- the metric this experiment cares about.

        python runtime.py <problem> <instance> [seed] [scheme] [budget]
        python runtime.py --all
"""
import sys
import time

import _bootstrap  # noqa: F401  -- puts the repo root (cpmpy/) on sys.path; must precede cpmpy
import cpmpy as cp
from cpmpy.tools.explain import marco, hierarchical_marco

import hierarchy
import oracles as orc
from methods import HierarchCommitOracle, SOLVER, MAP_SOLVER, ROUND_CAP


# ------------------------------------------------------------------ script ---
def record_script(root, hard, S, seed=0, time_budget=120.0):
    """Run hierarch-commit once and return its decision script (list of action dicts)."""
    oracle = HierarchCommitOracle(root, hard, S, seed=seed, time_budget=time_budget)
    oracle.t0 = time.perf_counter()
    deadline = oracle.t0 + time_budget
    for _ in hierarchical_marco(root, list(hard), solver=SOLVER, map_solver=MAP_SOLVER,
                                decide_step=oracle, deadline=deadline, round_cap=ROUND_CAP):
        pass
    return oracle.script, oracle.result


# ----------------------------------------------------- incremental back-end ---
class _ScriptedDecider:
    """Replays a recorded script through ``hierarchical_marco``. A backtrack hands back the
    map-solver state snapshot the enumerator provided BEFORE the matching commit (ctx['state'] is
    the current, wrong one), so we stack one state per commit and pop it on backtrack."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.commit_states = []

    def __call__(self, ctx):
        if self.i >= len(self.script):
            return {"action": "stop"}
        step = self.script[self.i]; self.i += 1
        a = step["action"]
        if a == "commit":
            self.commit_states.append(ctx["state"])
            return {"action": "commit", "mcs": list(step["mcs"])}
        if a == "refine":
            return {"action": "refine", "constraints": [step["group"]], "target_level": None}
        if a == "backtrack":
            st = self.commit_states.pop() if self.commit_states else ctx["state"]
            return {"action": "restore", "state": st}
        return {"action": "stop"}


def time_incremental(root, hard, script, budget=None):
    """Total wall time to replay `script` through the persistent hierarchical_marco."""
    decider = _ScriptedDecider(script)
    t0 = time.perf_counter()
    deadline = (t0 + budget) if budget else None
    for _ in hierarchical_marco(root, list(hard), solver=SOLVER, map_solver=MAP_SOLVER,
                                decide_step=decider, deadline=deadline, round_cap=ROUND_CAP):
        pass
    return time.perf_counter() - t0


# ------------------------------------------------------------ base back-end ---
def time_base(root, hard, script, budget=None):
    """Total wall time to replay `script` by re-running flat marco from scratch at every step."""
    name2node = {nd.get_full_name(): nd for nd in root.iter_nodes()}

    def grouped(nm):
        return name2node[nm].get_grouped_constraint()

    frontier = [c.get_full_name() for c in root.children]
    pruned, relaxed, stack = set(), set(), []
    t0 = time.perf_counter()
    for step in script:
        if budget and time.perf_counter() - t0 >= budget:
            break
        soft = [grouped(g) for g in frontier if g not in relaxed and g not in pruned]
        soft = [c for c in soft if c is not None]
        hard_now = list(hard) + [grouped(g) for g in pruned if grouped(g) is not None]
        if soft:                                                 # the work base MARCO redoes each step
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
                    if not name2node[g].children:                # leaf MCS member -> relax (drop)
                        relaxed.add(g)
                else:
                    pruned.add(g)                                # open, not in MCS -> background
        elif a == "backtrack":
            if stack:
                frontier, pruned, relaxed = stack.pop()
        elif a == "stop":
            break
    return time.perf_counter() - t0


# ------------------------------------------------------------------ driver ---
def run_runtime(problem, instance, seed, scheme="mss-20", budget=120.0):
    root, hard = hierarchy.load_instance(problem, instance)
    oracle = next(o for o in orc.load_oracles(problem, instance, scheme) if o["seed"] == seed)
    S = set(oracle["S"])
    script, result = record_script(root, hard, S, seed=seed, time_budget=budget)
    n_steps = len(script)
    t_inc = time_incremental(root, hard, script, budget=budget)
    t_base = time_base(root, hard, script, budget=budget)
    return {"problem": problem, "instance": instance, "scheme": scheme, "seed": seed,
            "result": result, "n_steps": n_steps,
            "incremental_s": round(t_inc, 3), "base_s": round(t_base, 3),
            "speedup": round(t_base / t_inc, 2) if t_inc > 0 else None}


def main():
    if sys.argv[1:2] == ["--all"]:
        cells = [(p, i, o["scheme"], o["seed"])
                 for p in hierarchy.PROBLEMS for i in hierarchy.list_instances(p)
                 for scheme in ("mss-20", "random-40")
                 for o in orc.load_oracles(p, i, scheme)]
    else:
        problem, instance = sys.argv[1], sys.argv[2]
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        scheme = sys.argv[4] if len(sys.argv) > 4 else "mss-20"
        cells = [(problem, instance, scheme, seed)]
    budget = float(sys.argv[5]) if len(sys.argv) > 5 else 120.0
    print(f"{'problem/instance':30s} {'scheme':10s} {'seed':>4s} {'steps':>5s} "
          f"{'incr(s)':>8s} {'base(s)':>8s} {'speedup':>7s}")
    for (p, i, scheme, seed) in cells:
        r = run_runtime(p, i, seed, scheme=scheme, budget=budget)
        print(f"{p + '/' + i:30s} {scheme:10s} {seed:4d} {r['n_steps']:5d} "
              f"{r['incremental_s']:8.3f} {r['base_s']:8.3f} {str(r['speedup']):>7s}", flush=True)


if __name__ == "__main__":
    main()
