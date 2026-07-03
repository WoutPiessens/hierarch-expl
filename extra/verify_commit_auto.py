"""
    Non-interactive validation of the commit-enabled `hierarchical_marco`, with a *programmed*
    `decide_step` (no terminal input). Checks:

      1. identity: with no commit, small synthetic trees enumerate the expected MUSes/MCSes;
      2. commit on a flat tree: committing to a leaf MCS relaxes exactly that leaf, backgrounds
         the rest, and the run reaches the repaired state;
      3. refine-then-commit on a 2-level tree: refine a group to its leaves, then commit to a
         leaf MCS -> that leaf is relaxed, the rest (incl. the other whole group) is background;
      4. smoke: the real defense (transcript_1) and nurse soft/hard instances load and enumerate
         at least one group-level MCS.

    Run from extra/:  python verify_commit_auto.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for cpmpy

import cpmpy as cp
from cpmpy.tools.explain.hierarchical import ConstraintNode
from cpmpy.tools.explain import hierarchical_marco

from instances import build


def _flat_tree():
    x, y = cp.intvar(0, 10, name="x"), cp.intvar(0, 10, name="y")
    root = ConstraintNode("r")
    for nm, con in [("g1", x >= 5), ("g2", x <= 3), ("g3", y >= 5), ("g4", y <= 8)]:
        root.add_child(nm).constraints.append(con)
    return root  # only conflict: g1 & g2 (over x); g3,g4 compatible


def _two_level_tree():
    x, y = cp.intvar(0, 10, name="x"), cp.intvar(0, 10, name="y")
    root = ConstraintNode("r")
    A = root.add_child("A"); A.add_child("a1").constraints.append(x >= 5); A.add_child("a2").constraints.append(x <= 3)
    B = root.add_child("B"); B.add_child("b1").constraints.append(y >= 5); B.add_child("b2").constraints.append(y <= 8)
    return root  # A is internally UNSAT (x>=5 & x<=3); B is SAT


def _run(root, decide_step):
    got = []
    for kind, cons, names, rnd in hierarchical_marco(
            root, [], solver="ortools", map_solver="ortools", decide_step=decide_step):
        got.append((kind, frozenset(names), rnd))
    return got


def test_identity_flat():
    got = _run(_flat_tree(), decide_step=None)  # None -> default auto-refine (no refinement here)
    mcses = {fs for k, fs, r in got if k == "MCS"}
    muses = {fs for k, fs, r in got if k == "MUS"}
    assert mcses == {frozenset({"g1"}), frozenset({"g2"})}, mcses
    assert muses == {frozenset({"g1", "g2"})}, muses
    print("1. identity (flat, no commit): MCSes {g1},{g2}, MUS {g1,g2}  OK")


def test_commit_flat():
    ctxs = []

    def decide(ctx):
        ctxs.append(ctx)
        mcses = [r["names"] for r in ctx["results"] if r["kind"] == "MCS"]
        if ctx["round"] == 1:
            assert ["g1"] in [list(m) for m in mcses] or ["g2"] in [list(m) for m in mcses], mcses
            return {"action": "commit", "mcs": ["g1"]}   # relax leaf g1
        return {"action": "stop"}

    _run(_flat_tree(), decide)
    last = ctxs[-1]
    assert set(last["committed_relaxed"]) == {"g1"}, last["committed_relaxed"]
    assert set(last["committed_background"]) == {"g2", "g3", "g4"}, last["committed_background"]
    # after committing g1, nothing free should remain unsatisfiable -> round 2 saw no MCS
    assert not [r for r in last["results"] if r["kind"] == "MCS"], last["results"]
    print("2. commit (flat): relaxed={g1}, background={g2,g3,g4}, repaired  OK")


def test_refine_then_commit():
    ctxs = []

    def decide(ctx):
        ctxs.append(ctx)
        if ctx["round"] == 1:
            return {"action": "refine", "constraints": ["A"], "target_level": 99}  # A -> a1,a2
        if ctx["round"] == 2:
            names = {frozenset(r["names"]) for r in ctx["results"] if r["kind"] == "MCS"}
            assert frozenset({"A a1"}) in names or frozenset({"A a2"}) in names, names
            return {"action": "commit", "mcs": ["A a1"]}   # relax leaf a1
        return {"action": "stop"}

    _run(_two_level_tree(), decide)
    last = ctxs[-1]
    assert set(last["committed_relaxed"]) == {"A a1"}, last["committed_relaxed"]
    assert "A a2" in last["committed_background"] and "B" in last["committed_background"], last["committed_background"]
    print("3. refine A then commit: relaxed={A a1}, background>={A a2, B}  OK")


def test_backtrack_refine():
    stack, frontiers = [], []
    step = {"n": 0}

    def decide(ctx):
        frontiers.append(sorted(ctx["frontier"]))
        n = step["n"]; step["n"] += 1
        if n == 0:                                   # frontier [A, B]; refine A -> leaves
            stack.append(ctx["state"])
            return {"action": "refine", "constraints": ["A"], "target_level": 99}
        if n == 1:                                   # frontier has A a1 / A a2; go back
            return {"action": "restore", "state": stack.pop()}
        # n == 2: restored
        assert sorted(ctx["frontier"]) == ["A", "B"], ctx["frontier"]
        return {"action": "stop"}

    _run(_two_level_tree(), decide)
    assert frontiers[0] == ["A", "B"], frontiers
    assert any("A a1" in f for f in frontiers), frontiers          # we did drill in
    assert frontiers[-1] == ["A", "B"], frontiers                  # and came back
    print("5. backtrack refine: frontier A->{a1,a2}-> back to {A,B}  OK")


def test_backtrack_commit():
    stack = []
    committed = []
    step = {"n": 0}

    def decide(ctx):
        committed.append(set(ctx["committed_background"]))
        n = step["n"]; step["n"] += 1
        if n == 0:                                   # commit leaf g1
            stack.append(ctx["state"])
            return {"action": "commit", "mcs": ["g1"]}
        if n == 1:                                   # committed -> now back
            assert ctx["committed_background"], "commit should have set a background"
            return {"action": "restore", "state": stack.pop()}
        # n == 2: restored -> committed sets cleared again
        assert not ctx["committed_background"] and not ctx["committed_relaxed"], ctx
        return {"action": "stop"}

    _run(_flat_tree(), decide)
    assert committed[0] == set() and committed[-1] == set(), committed
    print("6. backtrack commit: background set then cleared on back  OK")


def test_commit_no_double_classify():
    # P (leaf) conflicts with hard on x; M={m1,m2} is internally UNSAT on y. Round-1 MCS {P,M}:
    # commit relaxes P, keeps M open. Refine M, then commit a singleton {m1|m2}. The relaxed leaf
    # P must NOT be re-added to background by the second commit (the bug that produced an
    # "invalid repair").
    x, y = cp.intvar(0, 10, name="x"), cp.intvar(0, 10, name="y")
    root = ConstraintNode("r")
    root.add_child("P").constraints.append(x >= 5)
    M = root.add_child("M")
    M.add_child("m1").constraints.append(y >= 5)
    M.add_child("m2").constraints.append(y <= 3)
    hard = [x <= 3]

    ctxs = []

    def decide(ctx):
        ctxs.append(ctx)
        if ctx["round"] == 1:
            return {"action": "commit", "mcs": ["P", "M"]}
        if ctx["round"] == 2:
            return {"action": "refine", "constraints": ["M"], "target_level": 99}
        if ctx["round"] == 3:
            singleton = next(frozenset(r["names"]) for r in ctx["results"]
                             if r["kind"] == "MCS" and len(r["names"]) == 1)
            return {"action": "commit", "mcs": list(singleton)}
        return {"action": "stop"}

    _run_with_hard(root, hard, decide)
    last = ctxs[-1]
    assert "P" in last["committed_relaxed"], last["committed_relaxed"]
    assert "P" not in last["committed_background"], last["committed_background"]   # the bug
    # the resulting relaxation must genuinely repair
    relaxed = set(last["committed_relaxed"])
    kept = [lf.get_grouped_constraint() for lf in root.leaves() if lf.get_full_name() not in relaxed]
    assert cp.Model(hard + [c for c in kept if c is not None]).solve(solver="ortools"), "not a valid repair"
    print("7. commit no double-classify: relaxed P stays out of background, repair valid  OK")


def test_smoke_real():
    for kind in ("defense", "nurse"):
        root, hard, label = build(kind)

        def stop_after_1(ctx):
            return {"action": "stop"}  # only the initial (coarsest) round

        got = _run_with_hard(root, hard, stop_after_1)
        n_mcs = sum(1 for k, _, _ in got if k == "MCS")
        print(f"4. smoke [{label}]: round-1 yielded {n_mcs} group MCS(es), "
              f"{sum(1 for k,_,_ in got if k=='MUS')} MUS(es)  OK")


def _run_with_hard(root, hard, decide_step):
    got = []
    for kind, cons, names, rnd in hierarchical_marco(
            root, hard, solver="exact", map_solver="exact", decide_step=decide_step):
        got.append((kind, frozenset(names), rnd))
    return got


if __name__ == "__main__":
    test_identity_flat()
    test_commit_flat()
    test_refine_then_commit()
    test_backtrack_refine()
    test_backtrack_commit()
    test_commit_no_double_classify()
    test_smoke_real()
    print("\nAll checks passed.")
