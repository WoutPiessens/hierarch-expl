"""
    One fully-ANNOTATED run of the hierarchical oracle with the `max-overlap` commit strategy
    (pick the committable GMCS M maximizing |leaves(M) ∩ S|). At every decision the script
    prints the oracle's reasoning inputs: the committable options with their overlap scores,
    and for exploration the status of every open group (suitable descendants? relevant?
    refinable?). Uses the node-sampled suitable set of seed 6 (clustered S).

    Run from experiments/:  python run_annotated_maxoverlap.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmpy.tools.explain import hierarchical_marco
from oracle import build_nurse_hierarchy, _draw_node_sample, HierarchicalOracle

root, hard = build_nurse_hierarchy("nurse_instance1_softreq_8nurses")
S, selected = _draw_node_sample(root, 16, random.Random(6))
print("suitable set S (node-sampled, seed 6) came from nodes:", selected)
print("S =", sorted(S), "\n")

orc = HierarchicalOracle(root, hard, S, seed=6, max_steps=2000, commit_strategy="max-overlap")


def ov(names):
    return len(set().union(*(orc._suitable_leaves_under(g) for g in names)))


def annotated(ctx):
    open_names = frozenset(nd.get_full_name() for nd in ctx["frontier_nodes"]
                           if nd.get_full_name() not in set(ctx["committed_relaxed"])
                           and nd.get_full_name() not in set(ctx["committed_background"]))
    n_new = len(ctx["results"])
    print(f"--- round {ctx['round']}: {n_new} new MUS/MCS enumerated; "
          f"{len(open_names)} open group(s) ---")

    # show the commit options the oracle is about to consider
    opts = [M for M in orc.gmcs
            if (open_names, M) not in orc.abandoned and orc._committable(M, open_names)]
    if opts:
        print("  committable GMCS options (need: all members open, >=1 suitable leaf member,")
        print("                            every member has a suitable descendant):")
        for M in opts:
            print(f"    overlap={ov(M):2d}  {sorted(M)}")
    else:
        rejected = []
        for M in orc.gmcs:
            if not (M <= open_names):
                continue                                  # some member no longer open: skip silently
            why = []
            if not any(orc._is_leaf(g) for g in M):
                why.append("no primitive (leaf) member -> committing would relax nothing")
            bad = [g for g in M if not orc._potentially_suitable(g)]
            if bad:
                why.append(f"member(s) with NO suitable descendant: {bad}")
            if (open_names, M) in orc.abandoned:
                why.append("abandoned earlier at this state (commit was backtracked)")
            if why:
                rejected.append((sorted(M), "; ".join(why)))
        print("  no committable GMCS." + (" Rejected candidates:" if rejected else ""))
        for m, w in rejected[:4]:
            print(f"    {m}\n      -> {w}")

    action = orc(ctx)

    if action["action"] == "commit":
        M = frozenset(action["mcs"])
        others = [f"{ov(O)}" for O in opts if O != M]
        print(f"  DECISION: COMMIT {sorted(M)}")
        print(f"    why: max-overlap picks the option with most suitable leaves in its subtrees "
              f"(chosen overlap={ov(M)}, other options' overlaps: {others or 'none'})")
        leaves = [g for g in M if orc._is_leaf(g)]
        pend = [g for g in M if not orc._is_leaf(g)]
        print(f"    effect: relax suitable leaf members {leaves}; keep {pend or 'nothing'} open "
              f"(pending refinement); every other open group -> background")
    elif action["action"] == "refine":
        g = action["constraints"][0]
        node = orc.name2node[g]
        n_suit = len(orc._suitable_leaves_under(g))
        print(f"  DECISION: REFINE {g!r} into {len(node.children)} children")
        reasons = {"pending-committed-member":
                       "it is a non-leaf member of the just-committed GMCS -- must be drilled "
                       "into to locate WHICH of its leaves to relax",
                   "deepen-current-branch":
                       "it is relevant (in an enumerated MUS/MCS), has suitable descendants "
                       f"({n_suit} of its {len(node.leaves())} leaves are in S), and lies in the "
                       "currently explored branch (depth-first)",
                   "random-other-branch":
                       "no refinable group left in the current branch; this one is relevant and "
                       f"has {n_suit} suitable descendant(s) -- picked at random among such"}
        # recover the 'why' from the log record just written
        why = orc.log[-1].get("why", "?")
        print(f"    why: {reasons.get(why, why)}")
    elif action["action"] == "restore":
        print(f"  DECISION: BACKTRACK -- nothing committable (options exhausted/abandoned), "
              f"nothing refinable with suitable descendants, and the relaxed set does not yet "
              f"repair -> undo the most recent commit and mark it abandoned at that state")
    else:
        print(f"  DECISION: STOP ({orc.result}) -- "
              + ("the relaxed set repairs the problem (hard + all non-relaxed leaves is SAT)"
                 if orc.result == "repaired" else "search exhausted"))
    print()
    return action


for _ in hierarchical_marco(root, hard, solver="exact", map_solver="exact", decide_step=annotated):
    pass
print(f"FINAL: {orc.result}; relaxed ({len(orc.relaxed)}): {sorted(orc.relaxed)}")
print(f"queries: {orc.n_commit + orc.n_refine + orc.n_backtrack} "
      f"({orc.n_commit} commits, {orc.n_refine} refines, {orc.n_backtrack} backtracks)")
