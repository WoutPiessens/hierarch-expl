"""
    VERIFIED base-vs-incremental MARCO replay from a decision-script FILE.

    Both back-ends read the same JSON script (as saved by runtime_premature.py: a list of
    {"action": commit|refine|backtrack|stop, ...} steps) and apply IDENTICAL state transitions.
    Per step both enumerate the current frontier's conflicts EXHAUSTIVELY (no round cap), making
    their outputs comparable:

      * BASE        : flat marco, rebuilt from scratch each step -> re-finds old conflicts.
      * INCREMENTAL : persistent hierarchical_marco -> must never yield the same MUS/MCS twice
                      (a conflict found at a coarser granularity stays BLOCKED after refinement,
                      so its finer-grained translation is legitimately never re-yielded).

    Verifications, per step k (conflicts are frozensets of group full-names):
      V1  frontier parity  : incremental's round-k frontier == base's computed frontier.
      V2  no duplicates    : incremental never yields the same (kind, names) twice globally.
      V3  soundness        : every conflict incremental yields at step k is also found by base
                             at step k.
      V4  completeness     : every conflict base finds at step k is either yielded by
                             incremental at step k, or is the refinement of a conflict
                             incremental yielded earlier (each name maps, via ancestor-or-self,
                             onto a previously yielded conflict).

        python replay_compare.py PROBLEM INSTANCE --script FILE [--budget 900] [--per-step]
"""
import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from cpmpy.tools.explain import marco, hierarchical_marco

import hierarchy
from methods import SOLVER, MAP_SOLVER


# ------------------------------------------------------- shared state machine ---
def apply_step(step, st, name2node):
    """Apply one script step to state dict st = {frontier, pruned, relaxed, stack}."""
    a = step["action"]
    if a == "refine":
        st["frontier"] = ([x for x in st["frontier"] if x != step["group"]]
                          + list(step["children"]))
    elif a == "commit":
        st["stack"].append((list(st["frontier"]), set(st["pruned"]), set(st["relaxed"])))
        mcs = set(step["mcs"])
        for g in list(st["frontier"]):
            if g in st["relaxed"] or g in st["pruned"]:
                continue
            if g in mcs:
                if not name2node[g].children:
                    st["relaxed"].add(g)
            else:
                st["pruned"].add(g)
    elif a == "backtrack":
        if st["stack"]:
            st["frontier"], st["pruned"], st["relaxed"] = st["stack"].pop()
    return a


def active(st):
    return [g for g in st["frontier"] if g not in st["relaxed"] and g not in st["pruned"]]


# --------------------------------------------------------------- incremental ---
class VerifyingDecider:
    """Replays the script; records, per round, the frontier and newly yielded conflicts."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.commit_states = []
        self.rounds = []                       # per round: {"frontier": set, "found": [(kind,fs)]}
        self.tprev = None

    def __call__(self, ctx):
        now = time.perf_counter()
        found = [(r["kind"], frozenset(r["names"])) for r in ctx["results"]]
        rel, bg = set(ctx["committed_relaxed"]), set(ctx["committed_background"])
        frontier = {nd.get_full_name() for nd in ctx["frontier_nodes"]
                    if nd.get_full_name() not in rel and nd.get_full_name() not in bg}
        self.rounds.append({"frontier": frontier, "found": found,
                            "seconds": (now - self.tprev) if self.tprev else 0.0})
        self.tprev = time.perf_counter()
        step = self.script[self.i] if self.i < len(self.script) else {"action": "stop"}
        self.i += 1
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


def replay_incremental(root, hard, script, budget):
    dec = VerifyingDecider(script)
    t0 = time.perf_counter()
    dec.tprev = t0
    for _ in hierarchical_marco(root, list(hard), solver=SOLVER, map_solver=MAP_SOLVER,
                                decide_step=dec, deadline=t0 + budget, round_cap=None):
        pass
    return time.perf_counter() - t0, dec.i >= len(dec.script), dec.rounds


# ---------------------------------------------------------------------- base ---
def replay_base(root, hard, script, budget):
    name2node = {nd.get_full_name(): nd for nd in root.iter_nodes()}
    st = {"frontier": [c.get_full_name() for c in root.children],
          "pruned": set(), "relaxed": set(), "stack": []}
    steps = []                                 # per step: {"frontier": set, "found": [(kind,fs)]}
    t0 = time.perf_counter()
    completed = True
    for step in script:
        if time.perf_counter() - t0 >= budget:
            completed = False
            break
        ts = time.perf_counter()
        act = active(st)
        gc = {g: name2node[g].get_grouped_constraint() for g in act}
        soft = [gc[g] for g in act if gc[g] is not None]
        names = {id(gc[g]): g for g in act if gc[g] is not None}
        hard_now = list(hard) + [name2node[g].get_grouped_constraint() for g in st["pruned"]
                                 if name2node[g].get_grouped_constraint() is not None]
        found = []
        if soft:
            for kind, cons in marco(soft, hard_now, solver=SOLVER, map_solver=MAP_SOLVER,
                                    return_mus=True, return_mcs=True):
                found.append((kind, frozenset(names[id(c)] for c in cons)))
        steps.append({"frontier": set(act), "found": found,
                      "seconds": time.perf_counter() - ts})
        if apply_step(step, st, name2node) == "stop":
            break
    return time.perf_counter() - t0, completed, steps


# ---------------------------------------------------------------- verification ---
def ancestors_or_self(name, name2node):
    out, nd = [name], name2node[name].parent
    while nd is not None:
        out.append(nd.get_full_name())
        nd = nd.parent
    return out


def verify(root, inc_rounds, base_steps):
    name2node = {nd.get_full_name(): nd for nd in root.iter_nodes()}
    report = {"frontier_mismatch": [], "duplicates": [], "unsound": [], "unexplained": []}
    seen = set()                               # all (kind, names) incremental yielded so far
    seen_list = []
    n = min(len(inc_rounds), len(base_steps))
    for k in range(n):
        inc, base = inc_rounds[k], base_steps[k]
        # V1 frontier parity
        if inc["frontier"] != base["frontier"]:
            report["frontier_mismatch"].append(
                (k, sorted(inc["frontier"] ^ base["frontier"])))
        base_set = set(base["found"])
        # V2 duplicates + V3 soundness
        for cf in inc["found"]:
            if cf in seen:
                report["duplicates"].append((k, cf))
            if cf not in base_set:
                report["unsound"].append((k, cf))
            seen.add(cf)
            seen_list.append(cf)
        # V4 completeness: base conflicts must be yielded now or refine an earlier one
        inc_now = set(inc["found"])
        for kind, fs in base["found"]:
            if (kind, fs) in inc_now or (kind, fs) in seen:
                continue
            anc = {g: set(ancestors_or_self(g, name2node)) for g in fs}
            covered = any(
                k2 == kind and all(anc[g] & set(fs2) for g in fs)
                for (k2, fs2) in seen_list)
            if not covered:
                report["unexplained"].append((k, (kind, sorted(fs))))
    return report


# --------------------------------------------------------------------- driver ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem"); ap.add_argument("instance")
    ap.add_argument("--script", required=True)
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--per-step", action="store_true")
    args = ap.parse_args()

    script = json.loads(Path(args.script).read_text(encoding="utf-8"))
    root, hard = hierarchy.load_instance(args.problem, args.instance)

    t_inc, inc_done, inc_rounds = replay_incremental(root, hard, script, args.budget)
    t_base, base_done, base_steps = replay_base(root, hard, script, args.budget)

    print(f"script steps: {len(script)}")
    print(f"incremental: {t_inc:8.1f}s completed={inc_done} rounds={len(inc_rounds)} "
          f"conflicts_yielded={sum(len(r['found']) for r in inc_rounds)}")
    print(f"base       : {t_base:8.1f}s completed={base_done} steps={len(base_steps)} "
          f"conflicts_found={sum(len(s['found']) for s in base_steps)}")
    if args.per_step:
        for k in range(max(len(inc_rounds), len(base_steps))):
            i = inc_rounds[k] if k < len(inc_rounds) else None
            b = base_steps[k] if k < len(base_steps) else None
            print(f"  step {k:3d}: inc "
                  f"{'%6.1fs/%3d new' % (i['seconds'], len(i['found'])) if i else '   --'}"
                  f"   base {'%6.1fs/%3d' % (b['seconds'], len(b['found'])) if b else '   --'}"
                  f"   frontier={len(b['frontier']) if b else (len(i['frontier']) if i else 0)}")
    rep = verify(root, inc_rounds, base_steps)
    print("\nverification:")
    for k, v in rep.items():
        print(f"  {k:18}: {len(v)}" + (f"   e.g. {v[:2]}" if v else ""))


if __name__ == "__main__":
    main()
