"""
    Minimal test of the claim: "blocking clauses recorded inside a commit epoch are never
    wrongly enforced after backtracking, because the frontier is restored and the up/down vars
    of epoch-refined nodes are no longer substituted by indicator vars".

    Instance (no hard constraints):
        root -> A (children a1: x>=5, a2: y>=5),  b: x<=3,  c: y<=3
    Conflicts: {a1,b} and {a2,c}.

    Script (drives hierarchical_marco's decide_step directly):
        round 1  frontier {A,b,c}          -> snapshot; COMMIT {A,b}   (relax b, background c)
        round 2  free {A}, bg {c}          -> REFINE A                 (inside the epoch)
        round 3  free {a1,a2}, bg {c}      -> conflicts vs the BACKGROUND get enumerated and
                                              blocked (e.g. MUS {a2}, MCS {a2}); RESTORE snapshot
        round 4  frontier {A,b,c} restored -> REFINE A                 (outer state, NO commits)
        round 5  frontier {a1,a2,b,c}      -> THE TEST: how many NEW fine-granularity MCSes does
                                              the exhaustive round enumerate?

    Ground truth at round 5: the fine MCSes are {b,c}, {a1,a2}, {a1,c}, {a2,b}. Three of them are
    DEDUPLICATED rediscoveries of already-yielded results:
      * {b,c}   = complement of coarse MSS {A}          (round-1 block),
      * {a1,a2} = fine twin of coarse MCS {A}           (round-1 block),
      * {a2,b}  = (inner MCS {a2}) UNION (relaxed {b}): yielding MCS {a2} in round 3, with b
                  relaxed, IS the discovery of correction set {a2,b} -- the round-3 block,
                  augmented with down[b], suppresses exactly this re-discovery and nothing else.
    So a correct round 5 must enumerate exactly ONE genuinely new MCS: {a1,c} -- and must NOT
    be poisoned by the round-3 blocks (naive un-augmented blocks force ind[a2] to be both false
    and true, making round 5 enumerate NOTHING).

    Run from extra/:  python demo_stale_blocks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cpmpy as cp
from cpmpy.tools.explain.hierarchical import ConstraintNode
from cpmpy.tools.explain import hierarchical_marco


def main():
    x, y = cp.intvar(0, 10, name="x"), cp.intvar(0, 10, name="y")
    root = ConstraintNode("r")
    A = root.add_child("A")
    A.add_child("a1").constraints.append(x >= 5)
    A.add_child("a2").constraints.append(y >= 5)
    root.add_child("b").constraints.append(x <= 3)
    root.add_child("c").constraints.append(y <= 3)

    snapshot = {}
    per_round = {}
    step = {"n": 0}

    def decide(ctx):
        n = step["n"]; step["n"] += 1
        per_round[n + 1] = [(r["kind"], sorted(r["names"])) for r in ctx["results"]]
        if n == 0:
            snapshot["s"] = ctx["state"]
            return {"action": "commit", "mcs": ["A", "b"]}
        if n == 1:
            return {"action": "refine", "constraints": ["A"], "target_level": 99}
        if n == 2:
            return {"action": "restore", "state": snapshot["s"]}
        if n == 3:
            return {"action": "refine", "constraints": ["A"], "target_level": 99}
        return {"action": "stop"}

    for _ in hierarchical_marco(root, [], solver="ortools", map_solver="ortools",
                                decide_step=decide):
        pass

    for rnd in sorted(per_round):
        print(f"round {rnd}: {per_round[rnd]}")

    fine_mcses = {frozenset(names) for kind, names in per_round.get(5, []) if kind == "MCS"}
    must_find = {frozenset({"A a1", "c"})}                      # genuinely new at round 5
    dedup = frozenset({"A a2", "b"})                            # == (inner MCS {a2}) UNION (relaxed {b})
    print(f"\nround-5 (outer, refined, no commits) MCSes found: {sorted(map(sorted, fine_mcses))}")
    ok_found = fine_mcses >= must_find
    ok_dedup = dedup not in fine_mcses
    if ok_found and ok_dedup:
        print("=> PASS: the new MCS {a1,c} was found; {a2,b} was correctly deduplicated "
              "(already discovered in round 3 as MCS {a2} with b relaxed); no poisoning.")
    elif not ok_found:
        print("=> FAIL: stale epoch blocks pruned the genuinely new MCS {a1,c}.")
    else:
        print("=> NOTE: {a2,b} was re-enumerated (not deduplicated).")


if __name__ == "__main__":
    main()
