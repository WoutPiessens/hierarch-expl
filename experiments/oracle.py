"""
    oracle.py

    Simulates an interactive "oracle" that consumes the MUS/MCS stream produced by a
    constraint-explanation enumeration method and reacts to every proposed MCS: if it's
    *suitable* (currently: a member of a predetermined "acceptable" set, loaded from one of
    the stored random subsets in ``data/flat_instances/<instance>/mcs_subset_<pct>pct.json``),
    the oracle **accepts** it and stops; otherwise it **rejects** it, and the enumeration
    method is simply asked to continue (the generator already produces the next MCS lazily --
    "prompting a new MCS" is just the next iteration of the same loop).

    Extensible by design: :data:`METHODS` maps a method name to a callable with the same
    yield-protocol as :func:`cpmpy.tools.explain.marco` (yields ``(kind, found)`` with
    ``kind in {"MUS", "MCS"}``), so the accept/reject loop in :func:`run_oracle` never needs to
    change when a new method is added -- only :data:`METHODS` does. The only method wired up
    today is ``"baseline"`` (flat ``marco``, rebuilt from scratch); ``"incremental"`` is the
    next one planned (``hierarchical_marco`` / ``map_incremental_marco``, once adapted to take
    a flat instance instead of a :class:`ConstraintNode` hierarchy -- not implemented yet, see
    the placeholder below).

    Usage::

        python oracle.py --instance instance_292_first2unplannable --subset 10 --method baseline
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cpmpy`

import cpmpy as cp
from cpmpy.tools.explain import marco, hierarchical_marco
from cpmpy.tools.explain.hierarchical import ConstraintNode
from cpmpy.tools.explain.utils import make_assump_model
from cpmpy.transformations.get_variables import get_variables

from hierarchy_io import load_flat_instance

FLAT_DIR = Path(__file__).resolve().parent / "data" / "flat_instances"


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
# Every entry is a callable `method(soft, hard, **kwargs) -> generator of (kind, found)`,
# exactly the protocol cpmpy.tools.explain.marco / hierarchical_marco already use (`kind` is
# "MUS" or "MCS", `found` is the list of constraint objects). Add a new method here -- nothing
# else in this file needs to change.

def _baseline_method(soft, hard, *, solver="exact", map_solver="exact", **_ignored):
    """Flat `marco`, rebuilt from scratch -- today's only wired-up method."""
    yield from marco(soft, hard, solver=solver, map_solver=map_solver,
                     return_mus=True, return_mcs=True)


def _incremental_method(soft, hard, *, solver="exact", map_solver="exact", **_ignored):
    """Placeholder for hierarchical_marco / map_incremental_marco. Both currently take a
    ConstraintNode hierarchy (`root`), not a flat (soft, hard) pair like this instance is
    stored as -- wiring this up needs either a trivial one-level hierarchy wrapper around the
    flat soft list, or a flat-input variant of those functions. Left unimplemented on purpose;
    registering it here is what makes `--method incremental` a one-line addition once ready."""
    raise NotImplementedError(
        "the 'incremental' method (hierarchical_marco / map_incremental_marco) is not wired "
        "up yet -- see the docstring above this function")


METHODS = {
    "baseline": _baseline_method,
    "incremental": _incremental_method,
}


# ---------------------------------------------------------------------------
# The oracle itself
# ---------------------------------------------------------------------------

class Oracle:
    """Decides accept/reject for each MCS an enumeration method proposes.

    Today's suitability rule is pure set membership against a fixed "acceptable" collection
    (e.g. one of the stored random subsets of all primitive MCSes) -- swap `evaluate` for a
    different rule (e.g. a cost function, a live human prompt) without touching `run_oracle`.
    """

    def __init__(self, acceptable_mcs_names):
        self.acceptable = set(acceptable_mcs_names)  # set of frozenset[str]

    def evaluate(self, mcs_names: frozenset) -> bool:
        return mcs_names in self.acceptable

    def stats(self) -> dict:
        return {}


class RandomSampleOracle:
    """Alternative suitability rule: instead of knowing which MCSes are acceptable in advance,
    the oracle only knows which *constraints* it's currently willing to see relaxed -- a random
    sample of `pct` percent of all soft constraints (this is "sampling from the constraints,
    whether they are suitable or not", as opposed to `Oracle`'s sampling from the MCSes
    themselves). A proposed MCS is accepted iff EVERY constraint in it is in that allowed set
    (a partially-sampled MCS isn't a usable repair -- the unsampled constraints in it are ones
    the oracle was never asked about, so it can't have approved relaxing them).

    Escalation: if `escalate_after` MCSes in a row are rejected without growing the allowed set
    (i.e. the random sample at the current percentage isn't yielding anything), the percentage
    is increased by `escalate_step` (capped at 100) and MORE constraints are sampled in --
    growing the existing allowed set rather than re-drawing it from scratch, so escalation only
    ever becomes more permissive, never undoes a previous decision.
    """

    def __init__(self, soft_names, start_pct=50, escalate_step=10, escalate_after=100,
                seed=None, max_pct=100):
        self.rng = random.Random(seed)
        self.all_names = list(soft_names)
        self.pct = start_pct
        self.escalate_step = escalate_step
        self.escalate_after = escalate_after
        self.max_pct = max_pct
        self.allowed = set()
        self.rejected_since_resample = 0
        self.n_escalations = 0
        self.pct_history = [start_pct]
        self._resample_to(self.pct)

    def _resample_to(self, pct):
        """Grow `self.allowed` up to `pct`% of all constraints (never shrinks it)."""
        target = max(1, round(len(self.all_names) * pct / 100))
        remaining = [n for n in self.all_names if n not in self.allowed]
        need = target - len(self.allowed)
        if need > 0 and remaining:
            self.allowed.update(self.rng.sample(remaining, min(need, len(remaining))))
        self.rejected_since_resample = 0

    def evaluate(self, mcs_names: frozenset) -> bool:
        accept = mcs_names.issubset(self.allowed)
        if accept:
            return True
        self.rejected_since_resample += 1
        if self.rejected_since_resample >= self.escalate_after and self.pct < self.max_pct:
            self.pct = min(self.max_pct, self.pct + self.escalate_step)
            self.n_escalations += 1
            self.pct_history.append(self.pct)
            self._resample_to(self.pct)
        return False

    def is_suitable(self, constraint_name: str) -> bool:
        """Per-CONSTRAINT suitability (as opposed to `evaluate`, which judges a whole MCS at
        once) -- used by `run_staged_deletion`, which stages relaxations one constraint at a
        time instead of waiting for marco to propose a complete MCS."""
        return constraint_name in self.allowed

    def stats(self) -> dict:
        return {
            "final_pct": self.pct,
            "n_escalations": self.n_escalations,
            "pct_history": self.pct_history,
            "n_allowed_constraints": len(self.allowed),
        }


def run_staged_deletion(soft, hard, soft_names, oracle, solver="exact", map_solver="exact",
                        max_iterations=None, mcs_log=None):
    """
        MCS-enumeration with *partial staging*: a variant of the MARCO loop (it reuses the same
        map-solver / core-solver machinery as `cpmpy.tools.explain.marco`) in which, instead of
        the oracle accepting or rejecting each enumerated MCS *as a whole* (as `run_oracle`
        does), the oracle DELETES the suitable constraints inside every MCS it is shown and
        keeps going. Deleting is exactly the assumption-variable trick the task asks for: when
        constraint i is staged for deletion, its assumption var `assump[i]` is forced FALSE
        (`map_solver += ~assump[i]`), so it is never selected into another seed and never
        enforced in a core solve again -- "the rest remains true". Everything starts true
        (nothing relaxed); deletions accumulate monotonically.

        Because deletions change the assumptions, the MCSes the enumeration yields *after* a
        deletion are no longer MCSes of the original problem -- they are MCSes of the RELAXED
        problem (the original minus everything deleted so far). Stacked on top of the
        already-deleted constraints, each such MCS is a (generally NON-minimal) correction
        subset of the original problem. We count every MCS the oracle is shown
        (`n_mcs_enumerated`), and the final accumulated deleted set (`n_relaxed`) is itself the
        non-minimal correction subset that restored satisfiability.

        The loop stops as soon as the relaxed problem is SAT (the deleted set is now a
        correction subset -> repaired), or when the map solver is exhausted (no correction
        subset reachable using only suitable constraints -> failure), or when
        `max_iterations` MCSes have been shown (cap -> failure, `capped=True`). MUSes the
        enumeration derives are used to steer the map solver exactly as in `marco` (counted in
        `n_mus_seen`, not shown to the oracle -- the oracle only ever judges MCSes).

        One "oracle decision" == one MCS shown to the oracle (it then deletes that MCS's
        suitable part), so `n_decisions == n_mcs_enumerated` -- the same granularity at which
        `run_oracle`'s baseline interacts with the oracle (one decision per proposed MCS),
        which is what makes the two methods' interaction counts comparable.

        :param: oracle: must support `is_suitable(name) -> bool` (e.g. `RandomSampleOracle`)
        :param: max_iterations: give up after this many MCSes shown (default: until exhausted)
        :param: mcs_log: optional list; if given, one record is appended per enumerated MCS (in
            order): ``{"mcs_size": len(mcs), "suitable_in_mcs": <#deleted from it>,
            "relaxed_before": <#already deleted before this MCS>}`` -- lets you trace how the
            MCS size evolves as the relaxed problem shrinks.
        :return: dict with whether SAT was reached, the deleted (relaxed) constraint names, how
            many were deleted (size of the non-minimal correction subset), how many MCSes were
            enumerated/shown, how many MUSes were seen, how many core solve() calls were used,
            the equal oracle-decision count, and whether the iteration cap was hit.
    """
    model, soft2, assump = make_assump_model(soft, hard)
    s = cp.SolverLookup.get(solver, model)
    ms = cp.SolverLookup.get(map_solver)
    ms += cp.any(assump)

    dmap = dict(zip(assump, soft2))
    idx_of = {a: i for i, a in enumerate(assump)}
    name_of_a = {a: soft_names[i] for i, a in enumerate(assump)}
    del_order = {a: -len(get_variables(dmap[a])) for a in assump}  # MUS-shrink order, as in marco

    relaxed = set()  # indices of constraints deleted (assumption forced false)
    n_mcs = 0        # MCSes shown to the oracle == oracle decisions
    n_mus = 0
    n_solve_calls = 0

    def active():
        return [a for i, a in enumerate(assump) if i not in relaxed]

    def core_solve(assumptions):
        nonlocal n_solve_calls
        n_solve_calls += 1
        return s.solve(assumptions=assumptions)

    def relaxed_is_sat():
        # all not-yet-deleted constraints enforced: is the relaxed problem now SAT?
        return core_solve(active()) is True

    def result(reached_sat, capped=False):
        return {"reached_sat": reached_sat, "capped": capped,
                "relaxed_names": [soft_names[i] for i in sorted(relaxed)],
                "n_relaxed": len(relaxed),
                "n_mcs_enumerated": n_mcs, "n_mus_seen": n_mus,
                "n_decisions": n_mcs, "n_solve_calls": n_solve_calls}

    if relaxed_is_sat():  # already SAT with nothing deleted (problem is normally UNSAT)
        return result(True)

    while ms.solve():
        seed = [a for a in assump if a.value() and idx_of[a] not in relaxed]

        if core_solve(seed) is True:
            # SAT seed -> grow to a maximal satisfiable subset over the ACTIVE assumptions,
            # take the complement as the MCS of the relaxed problem (exactly marco's step).
            act = active()
            mss = [a for a in act if a.value() or dmap[a].value()]
            for to_add in frozenset(act) - frozenset(mss):
                if core_solve(mss + [to_add]) is True:
                    mss.append(to_add)
            mcs = [a for a in act if a not in frozenset(mss)]
            ms += cp.any(mcs)  # block this MCS so it is not enumerated again
            n_mcs += 1

            # STAGING: the oracle deletes the suitable constraints inside this MCS.
            suitable = [a for a in mcs if oracle.is_suitable(name_of_a[a])]
            if mcs_log is not None:
                mcs_log.append({"mcs_size": len(mcs), "suitable_in_mcs": len(suitable),
                                "relaxed_before": len(relaxed)})
            for a in suitable:
                relaxed.add(idx_of[a])
                ms += ~a  # force this assumption false from now on (delete the constraint)

            if suitable and relaxed_is_sat():
                return result(True)
            if max_iterations is not None and n_mcs >= max_iterations:
                return result(False, capped=True)

        else:
            # UNSAT seed -> shrink to a MUS, block it in the map solver (steering only).
            core = set(s.get_core())
            for c in sorted(core, key=del_order.get):
                if c not in core:
                    continue
                core.remove(c)
                if core_solve(list(core)):
                    core.add(c)
                else:
                    core = set(s.get_core())
            ms += ~cp.all(core)
            n_mus += 1

    return result(False)  # map solver exhausted: no suitable correction subset exists


def run_oracle(method_name, soft, hard, name_of, oracle, max_iterations=None, **method_kwargs):
    """
        Drive `METHODS[method_name]`'s enumerator and apply `oracle` to every MCS it proposes.
        MUSes are passed through (counted, not evaluated -- the oracle only judges MCSes, per
        the task: accept a *correction set*, i.e. a way to repair the problem).

        Stops as soon as the oracle accepts one MCS (no need to keep searching), or after
        `max_iterations` MCSes if given, or when the method's enumeration is exhausted.

        :return: dict with the accepted MCS's constraint names (or None), how many MCSes were
            rejected first, how many MUSes were seen along the way, and elapsed time.
    """
    enumerate_fn = METHODS[method_name]
    accepted = None
    n_rejected = 0
    n_mus = 0
    n_mcs_seen = 0

    t0 = time.perf_counter()
    for kind, found in enumerate_fn(soft, hard, **method_kwargs):
        if kind == "MUS":
            n_mus += 1
            continue

        n_mcs_seen += 1
        names = frozenset(name_of[id(c)] for c in found)
        if oracle.evaluate(names):
            accepted = sorted(names)
            break
        n_rejected += 1

        if max_iterations is not None and n_mcs_seen >= max_iterations:
            break
    dt = time.perf_counter() - t0

    return {
        "method": method_name,
        "accepted": accepted,
        "n_rejected": n_rejected,
        "n_mcs_seen": n_mcs_seen,
        "n_mus_seen": n_mus,
        "elapsed_seconds": dt,
        **oracle.stats(),
    }


# ---------------------------------------------------------------------------
# Hierarchical oracle  (the automated counterpart of extra/verify_commit.py)
# ---------------------------------------------------------------------------
#
# This is the third "method": instead of enumerating flat MCSes (baseline) or staging deletions
# (staged-deletion), it drives `hierarchical_marco` -- refining a ConstraintNode hierarchy of
# constraint groups and committing to group-level MCSes -- exactly as a human does by hand in
# extra/verify_commit.py, but *automatically*, guided by a fixed set S of "suitable" primitive
# constraints (the sampled subset). It implements the abstraction-state transition policy:
#
#   State A = (frontier grouping, commitment status). At each state the oracle either
#     (1) COMMITS to a "state-relative GMCS" M (group MCS over the OPEN groups) if one exists
#         whose members are all *potentially suitable* (G's leaves meet S) and which contains a
#         *suitable primitive* leaf (so committing relaxes only suitable constraints); or
#     (2) EXPLORES by refining a relevant, potentially-suitable open group -- depth-first in the
#         most recently refined branch, else a random other branch; or
#     (3) BACKTRACKS by undoing the most recent commit when neither is possible.
#
# S is drawn at a fixed sampling rate and re-drawn until it actually admits a repair (an MCS
# M subseteq S), i.e. until relaxing all of S restores satisfiability.


def build_nurse_hierarchy(instance="nurse_instance1_softreq_8nurses"):
    """Wrap a flat nurse soft/hard instance in a ConstraintNode hierarchy (the same shape the
    extra/ driver uses): family (shift_on/shift_off/cover) -> nurse|week -> day(leaf)."""
    soft, hard, names, _ = load_flat_instance(instance)
    root = ConstraintNode("nurse")
    for con, name in zip(soft, names):
        fam = name.split("__", 1)[0]
        tag = name.split("__", 1)[1].split("_")
        if fam in ("shift_on", "shift_off"):
            nurse, day = tag[0], tag[1]                      # request: (nurse, day)
            leaf = root.add_child(fam).add_child(nurse).add_child(day)
        else:                                                # cover: (day) only, grouped by week
            day = tag[0]
            leaf = root.add_child(fam).add_child(f"week{int(day[3:]) // 7}").add_child(day)
        leaf.constraints.append(con)
    return root, hard


def sample_suitable_set(root, hard, pct=40, seed0=0, max_tries=2000, solver="ortools"):
    """Draw S = pct% of the primitive (leaf) constraints, re-drawing with successive seeds until
    S admits a repair, i.e. until there is an MCS M subseteq S. That holds iff relaxing *all* of
    S restores satisfiability, i.e. ``hard + (every leaf NOT in S)`` is SAT (a superset of a valid
    correction set is itself a correction set, and any correction set contains an MCS)."""
    leaf_nodes = root.leaves()
    leaf_names = [lf.get_full_name() for lf in leaf_nodes]
    k = max(1, round(len(leaf_names) * pct / 100))
    for t in range(max_tries):
        S = set(random.Random(seed0 + t).sample(leaf_names, k))
        kept = [lf.get_grouped_constraint() for lf in leaf_nodes if lf.get_full_name() not in S]
        if cp.Model(hard + [c for c in kept if c is not None]).solve(solver=solver) is True:
            return S, seed0 + t
    return None, None


def _draw_node_sample(root, k, rng):
    """One draw of the NODE-sampling scheme: greedy random packing of hierarchy nodes.

    Instead of sampling primitive constraints one by one, sample NODES of the constraint tree:
    selecting an internal node makes its entire subtree of primitives suitable at once. To hit
    the exact budget of k primitives, nodes are visited in random order and a node is selected
    iff its not-yet-covered leaves still FIT in the remaining budget (|leaves(v) \\ S| <= k-|S|);
    when no whole node fits any more, the remainder is topped up with individual uncovered
    leaves (a leaf always fits 1). This guarantees |S| == k exactly, while the selected set is
    CLUSTERED: whole groups become suitable together, mimicking an oracle whose tolerance is
    organised along the hierarchy ("this nurse's requests are negotiable", "week-1 cover is
    flexible") rather than scattered over unrelated primitives.

    :return: (S, selected) -- the suitable leaf names, and the sampled nodes (incl. internal).
    """
    nodes = [nd for nd in root.iter_nodes() if nd is not root]
    rng.shuffle(nodes)
    S, selected = set(), []
    for nd in nodes:
        lv = {lf.get_full_name() for lf in nd.leaves()}
        new = lv - S
        if new and len(new) <= k - len(S):
            S |= lv
            selected.append(nd.get_full_name())
        if len(S) == k:
            break
    if len(S) < k:  # top up with individual uncovered leaves
        uncovered = [lf.get_full_name() for lf in root.leaves() if lf.get_full_name() not in S]
        extra = rng.sample(uncovered, k - len(S))
        S.update(extra)
        selected.extend(extra)
    return S, selected


def sample_suitable_nodes(root, hard, pct=40, seed0=0, max_tries=2000, solver="ortools"):
    """Node-sampling counterpart of :func:`sample_suitable_set`: S is the union of the leaves of
    randomly packed hierarchy NODES (see :func:`_draw_node_sample`), still exactly pct% of the
    primitives, re-drawn with successive seeds until S admits a repair.

    :return: (S, seed, selected_nodes) or (None, None, None)."""
    leaf_nodes = root.leaves()
    k = max(1, round(len(leaf_nodes) * pct / 100))
    for t in range(max_tries):
        S, selected = _draw_node_sample(root, k, random.Random(seed0 + t))
        kept = [lf.get_grouped_constraint() for lf in leaf_nodes if lf.get_full_name() not in S]
        if cp.Model(hard + [c for c in kept if c is not None]).solve(solver=solver) is True:
            return S, seed0 + t, selected
    return None, None, None


class HierarchicalOracle:
    """decide_step policy for `hierarchical_marco` implementing the abstraction-state transitions.

    It is called once per enumeration round with the round's context (the current frontier and
    the group-MCSes/MUSes just enumerated) and returns the next action: ``commit`` / ``refine`` /
    ``restore`` (backtrack) / ``stop``. All decisions are relative to the fixed suitable set S.
    """

    def __init__(self, root, hard, S, seed=0, max_steps=5000, commit_strategy="first",
                 log=None, verbose=False):
        self.S = frozenset(S)                                       # suitable primitive leaf names
        self._hard = list(hard)                                     # for the "already repaired?" test
        self._leaves = root.leaves()
        self.commit_strategy = commit_strategy                      # "first" | "random" | "pure-leaf" | "max-leaf" | "max-overlap" | "lookahead"
        self.log = log if log is not None else []                   # one structured record per decision
        self.verbose = verbose
        self.name2node = {nd.get_full_name(): nd for nd in root.iter_nodes()}
        self.rng = random.Random(seed)                             # only for random branch choice
        self.max_steps = max_steps
        # Per-epoch registry of the group-MCSes / group-MUSes seen since the last commit (as
        # frozensets of group names). Within an epoch the background is fixed, so an MCS found in
        # an earlier round is still a valid state-relative GMCS while its members stay open. It is
        # reset on commit (the background changes) and restored on backtrack -- using a GLOBAL
        # registry instead lets stale GMCSes (valid only under an earlier background) be committed,
        # which mis-guides the search.
        self.gmcs = []
        self.gmus = []
        self.last_refined = None                                   # name of most-recently-refined group == current branch root
        self.pending = []                                          # non-leaf members of the last committed GMCS, still to refine
        self.stack = []                                            # backtrack snapshots (one per commit)
        self.abandoned = set()                                     # (open-frontier signature, M) we backtracked from
        # metrics + outcome
        self.n_commit = self.n_refine = self.n_backtrack = self.n_steps = 0
        self.n_gmcs_seen = self.n_gmus_seen = 0
        self.n_relax_queries = 0                                   # total leaves relaxed by commits (1 query per relaxed constraint)
        self.n_lookahead_solves = 0                                # internal look-ahead SAT checks (not oracle queries)
        self.learned_conflicts = []                                # translated inner MUSes: (frozenset core, frozenset background) pairs
        self.cooldown = 0                                          # explore-after-fail: forced exploration steps before the next commit
        self.result = None
        self.relaxed = []

    def _log(self, round_idx, action, **detail):
        rec = {"step": self.n_steps, "round": round_idx, "action": action, **detail}
        self.log.append(rec)
        if self.verbose:
            extra = "  ".join(f"{k}={v}" for k, v in detail.items())
            print(f"  [step {self.n_steps:3d} | round {round_idx:3d}] {action.upper():9s} {extra}",
                  flush=True)

    # --- suitability predicates ------------------------------------------------------------
    def _is_leaf(self, name):
        return not self.name2node[name].children

    def _potentially_suitable(self, name):
        """G is potentially suitable iff its subtree contains a suitable primitive: G's leaves
        meet S (for a leaf this is just: the leaf itself is in S)."""
        return any(lf.get_full_name() in self.S for lf in self.name2node[name].leaves())

    def _committable(self, M, open_names):
        """A state-relative GMCS M is committable iff (a) all its members are still open, (b) it
        contains at least one primitive (leaf) member -- so committing relaxes something -- and
        (c) every member is potentially suitable. Because committing relaxes exactly the LEAF
        members of M, and a leaf is potentially suitable iff it is in S, (c) guarantees every
        relaxed constraint is suitable."""
        return (M <= open_names
                and any(self._is_leaf(g) for g in M)
                and all(self._potentially_suitable(g) for g in M))

    def _suitable_leaves_under(self, name):
        """The suitable primitive constraints in `name`'s subtree."""
        return {lf.get_full_name() for lf in self.name2node[name].leaves()
                if lf.get_full_name() in self.S}

    def _can_complete(self, M, relaxed):
        """One-solve LOOK-AHEAD: can committing M possibly lead to a repair? After committing M
        every other open group becomes background, so all future relaxations must come from
        within M's own subtrees (and must be suitable). The most that can ever be relaxed down
        that branch is: current relaxed + M's suitable leaf members + every suitable leaf under
        M's non-leaf members. If even relaxing ALL of that leaves the problem UNSAT, committing
        M is a guaranteed dead end -- prune it before paying the commit+backtrack queries."""
        self.n_lookahead_solves += 1
        candidate = set(relaxed)
        for g in M:
            candidate |= self._suitable_leaves_under(g)
        return self._is_repaired(candidate)

    def _is_repaired(self, relaxed):
        """True iff the constraints relaxed so far already form a correction set: hard + (every
        non-relaxed leaf) is SAT. This is the real termination test -- a repair can be reached
        while OPEN groups still remain (they just happen to be jointly satisfiable), so we must
        check it rather than waiting for the open set to empty."""
        kept = [lf.get_grouped_constraint() for lf in self._leaves
                if lf.get_full_name() not in relaxed]
        return cp.Model(self._hard + [c for c in kept if c is not None]).solve(solver="ortools") is True

    def _in_current_branch(self, name):
        """Is `name` inside the most-recently-refined branch (a descendant of last_refined)?"""
        if self.last_refined is None:
            return False
        node = self.name2node[name].parent
        while node is not None:
            if node.get_full_name() == self.last_refined:
                return True
            node = node.parent
        return False

    # --- the transition policy -------------------------------------------------------------
    def __call__(self, ctx):
        self.n_steps += 1
        for r in ctx["results"]:                                   # register this round's GMCSes/GMUSes
            fs = frozenset(r["names"])
            if r["kind"] == "MCS" and fs not in self.gmcs:
                self.gmcs.append(fs); self.n_gmcs_seen += 1
            elif r["kind"] == "MUS" and fs not in self.gmus:
                self.gmus.append(fs); self.n_gmus_seen += 1

        open_names = frozenset(nd.get_full_name() for nd in ctx["frontier_nodes"]
                               if nd.get_full_name() not in set(ctx["committed_relaxed"])
                               and nd.get_full_name() not in set(ctx["committed_background"]))

        self.relaxed = list(ctx["committed_relaxed"])
        if not open_names:                                         # no open groups left
            self.result = "repaired" if self._is_repaired(set(self.relaxed)) else "failed"
            self._log(ctx["round"], "stop", reason=self.result)
            return {"action": "stop"}
        if self.n_steps > self.max_steps:
            self.result = "capped"
            self._log(ctx["round"], "stop", reason="capped")
            return {"action": "stop"}

        # (1) COMMIT: pick a committable state-relative GMCS according to `commit_strategy`.
        options = [M for M in self.gmcs
                   if (open_names, M) not in self.abandoned and self._committable(M, open_names)]
        if self.commit_strategy == "explore-after-fail" and self.cooldown:
            # cooldown after a dead end: skip the commit step entirely for one decision and
            # perform a RANDOM exploration step instead (any relevant, potentially-suitable
            # branch -- not necessarily the current one), so fresh conflict structure is
            # enumerated before the next commit attempt.
            options = []
        if options and self.commit_strategy == "mus-hit":
            # SOLVER-FREE dead-end filter via MARCO duality: every correction set must hit every
            # MUS. After committing M only leaves(M) ∩ S can ever be relaxed, so an enumerated
            # GMUS U with leaves(U) ∩ leaves(M) ∩ S = ∅ can never be defused inside the branch
            # -> M is provably dead by set intersection alone. Among survivors, prefer the one
            # whose WORST-covered known MUS is best covered (bottleneck, not total overlap).
            def _mleaves(M):
                return set().union(*(self._suitable_leaves_under(g) for g in M))

            def _uleaves(U):
                return set().union(*({lf.get_full_name() for lf in self.name2node[u].leaves()}
                                     for u in U))
            ucache = [_uleaves(U) for U in self.gmus]

            def hits_all(M):
                ml = _mleaves(M)
                return all(ul & ml for ul in ucache)

            def bottleneck(M):
                ml = _mleaves(M)
                return min((len(ul & ml) for ul in ucache), default=0)
            viable = [M for M in options if hits_all(M)]
            options = sorted(viable, key=bottleneck, reverse=True)

        if options and self.commit_strategy == "mus-hit-learn":
            # H1 + failure-learning: reject any option M whose commit would re-enforce a LEARNED
            # conflict (C, B_delta) in full -- i.e. no constraint of C ∪ B_delta is already
            # relaxed or relaxable (suitable leaf) inside M. Provably dead (C+B is UNSAT
            # regardless of state). Discovery order kept among survivors (no reordering).
            relaxed_now = set(self.relaxed)

            def learned_dead(M):
                ml = set().union(*(self._suitable_leaves_under(g) for g in M)) | relaxed_now
                return any(not (cl & ml) for cl in self.learned_conflicts)
            rejected = [M for M in options if learned_dead(M)]
            if rejected:
                self._log(ctx["round"], "reject",
                          n_rejected=len(rejected), n_kept=len(options) - len(rejected),
                          examples=[sorted(M) for M in rejected[:2]])
            options = [M for M in options if not learned_dead(M)]

        if options and self.commit_strategy == "lookahead":
            # prune every option that provably cannot complete a repair (one solve each); among
            # the survivors, pick the one whose subtrees hold the fewest suitable leaves (the
            # tightest branch -> fewest relaxations to reach the repair).
            viable = [M for M in options if self._can_complete(M, set(self.relaxed))]
            options = sorted(viable, key=lambda M: sum(len(self._suitable_leaves_under(g)) for g in M))
        if options:
            def leafness(M):        # (all-leaf?, #suitable leaf members) -- pure-leaf MCSes can't dead-end
                leaves = [g for g in M if self._is_leaf(g)]
                return (len(leaves) == len(M), len(leaves))
            if self.commit_strategy in ("first", "lookahead", "mus-hit", "mus-hit-learn",
                                        "explore-after-fail"):
                M = options[0]
            elif self.commit_strategy == "random":
                M = self.rng.choice(options)
            elif self.commit_strategy == "pure-leaf":
                # prefer MCSes made ONLY of suitable leaves (committing one relaxes a complete
                # state-relative correction set -> can never dead-end), smallest first; only if
                # none exists, fall back to the first mixed one.
                pure = [M for M in options if leafness(M)[0]]
                M = min(pure, key=len) if pure else options[0]
            elif self.commit_strategy == "max-leaf":
                M = max(options, key=lambda M: leafness(M)[1])     # most immediate relaxations
            elif self.commit_strategy == "max-overlap":
                # the option whose members' subtrees contain the most suitable primitives:
                # |leaves(M) ∩ S| -- the largest relaxation BUDGET the branch will have. A cheap,
                # solver-free proxy for lookahead's completability check (a big budget is more
                # likely to suffice for a repair).
                M = max(options, key=lambda M: len(set().union(
                        *(self._suitable_leaves_under(g) for g in M))))
            else:
                raise ValueError(f"unknown commit_strategy {self.commit_strategy!r}")
            n_leaves = sum(1 for g in M if self._is_leaf(g))
            self.n_relax_queries += n_leaves
            self._log(ctx["round"], "commit", mcs=sorted(M), n_options=len(options),
                      pure_leaf=leafness(M)[0], relaxes=n_leaves)
            # snapshot everything needed to undo this commit later, then start a new epoch.
            self.stack.append((ctx["state"], list(self.gmcs), list(self.gmus),
                               self.last_refined, list(self.pending), open_names, M))
            self.gmcs, self.gmus, self.last_refined = [], [], None
            # M's NON-LEAF members stay open and become the new "currently explored branch":
            # they must be refined next to locate which of their leaves to relax. (Without this
            # the fresh epoch has no GMCS/GMUS yet -- everything at this granularity is already
            # blocked -- so no group is "relevant" and the oracle would backtrack immediately.)
            self.pending = [g for g in sorted(M) if not self._is_leaf(g)]
            self.n_commit += 1
            return {"action": "commit", "mcs": sorted(M)}

        # (2) EXPLORE: refine a RELEVANT, potentially-suitable, refinable open group. Priority:
        #     (a) a pending non-leaf member of the just-committed GMCS (relevant by construction
        #         -- it IS the currently explored branch right after a commit);
        #     (b) depth-first inside the current refined branch;
        #     (c) a random other open branch.
        relevant = set().union(*self.gmcs, *self.gmus) if (self.gmcs or self.gmus) else set()
        pend = [nm for nm in self.pending
                if nm in open_names and self.name2node[nm].children
                and self._potentially_suitable(nm)]
        cands = [nm for nm in open_names
                 if self.name2node[nm].children                    # refinable (not a leaf)
                 and nm in relevant                                # relevant (appears in a GMCS/GMUS)
                 and self._potentially_suitable(nm)]               # potentially suitable
        branch = [c for c in cands if self._in_current_branch(c)]
        if self.commit_strategy == "explore-after-fail" and self.cooldown and cands:
            pick, why = self.rng.choice(cands), "cooldown-random"  # forced random step after a dead end
            self.cooldown = 0
        elif pend:
            pick, why = pend[0], "pending-committed-member"        # locate the leaves to relax in M
            self.pending = [p for p in self.pending if p != pick]
        elif branch:
            pick, why = branch[0], "deepen-current-branch"         # deepen the current branch (deterministic)
        elif cands:
            pick, why = self.rng.choice(cands), "random-other-branch"  # jump to another open branch
        else:
            self.cooldown = 0    # nothing refinable: don't let the cooldown suppress commits forever
            # Nothing to commit or refine. First: are we already repaired (the relaxed set is a
            # correction set, with the remaining open groups jointly satisfiable)? If so, stop.
            if self._is_repaired(set(self.relaxed)):
                self.result = "repaired"
                self._log(ctx["round"], "stop", reason="repaired")
                return {"action": "stop"}
            # (3) BACKTRACK: otherwise undo the most recent commit.
            if not self.stack:
                self.result = "failed"                            # heuristic could not reach a repair
                self._log(ctx["round"], "stop", reason="failed-no-stack")
                return {"action": "stop"}
            hm_state, gmcs, gmus, last_ref, pending, open_sig, M = self.stack.pop()
            # An inner MCS M' found while the popped commit's leaves R were relaxed IS the
            # discovery of the correction set M' UNION R of the restored state -- the enumerator
            # deliberately never re-yields it (its blocks are augmented with down[R]), so KEEP
            # that knowledge here by translating each inner GMCS before restoring the registry.
            r_delta = frozenset(g for g in M if self._is_leaf(g))  # leaves relaxed by the popped commit
            translated = [fs | r_delta for fs in self.gmcs]
            # LEARN the abandoned epoch's conflicts (for the mus-hit-learn filter): an inner MUS
            # C under the epoch's FULL background B is the state-independent fact "C + B is
            # UNSAT" (cf. the up[B]-augmented blocks). B must be the full background (including
            # enclosing epochs'), else the fact over-claims after deeper backtracks. Store the
            # conflict's LEAF set, precomputed once: a future commit that leaves that whole set
            # enforced (no member relaxed or relaxable) is provably dead.
            b_full = set(ctx["committed_background"])
            for U in self.gmus:
                cl = set()
                for u in (set(U) | b_full):
                    cl.update(lf.get_full_name() for lf in self.name2node[u].leaves())
                fs = frozenset(cl)
                if fs not in self.learned_conflicts:
                    self.learned_conflicts.append(fs)
                    self._log(ctx["round"], "learn", core=sorted(U), n_bg=len(b_full),
                              conflict_leaves=len(fs),
                              suitable_in_conflict=len(fs & self.S))
            # restore the previous epoch's registry: all MUSes/MCSes enumerated between the
            # previous commit and this frontier stay available for refinement & commitment.
            self.gmcs, self.gmus = gmcs, gmus
            for fs in translated:
                if fs not in self.gmcs:
                    self.gmcs.append(fs)
            self.last_refined, self.pending = last_ref, pending
            self.abandoned.add((open_sig, M))                     # don't re-commit M at that state
            self.n_backtrack += 1
            self.cooldown = 1                                     # explore-after-fail: force one random exploration first
            self._log(ctx["round"], "backtrack", undone_mcs=sorted(M), depth=len(self.stack))
            return {"action": "restore", "state": hm_state}

        # Registry translation on refine (the refinement analogue of the M-union-R translation
        # on backtrack): the enumerator deduplicates a refined MCS's finer twin (its blocks
        # reinterpret over the children), so DERIVE it here -- a registered GMCS containing the
        # refined group `pick` yields the correction candidate (M \ {pick}) UNION children(pick)
        # (same leaf set, hence still a correction set; possibly non-minimal). Without this, a
        # non-committable GMCS (no leaf member) can never become committable by refining it: its
        # committable fine form would be suppressed as a duplicate.
        children = [c.get_full_name() for c in self.name2node[pick].children
                    if c.get_grouped_constraint() is not None]
        for fs in list(self.gmcs):
            if pick in fs:
                t = (fs - {pick}) | frozenset(children)
                if t not in self.gmcs:
                    self.gmcs.append(t)
        self.last_refined = pick
        self.n_refine += 1
        self._log(ctx["round"], "refine", group=pick, why=why, n_candidates=len(cands))
        return {"action": "refine", "constraints": [pick], "target_level": None}


def run_hierarchical_oracle(root, hard, S, seed=0, solver="exact", map_solver="exact",
                            max_steps=5000, commit_strategy="first", log=None, verbose=False):
    """Drive `hierarchical_marco` with `HierarchicalOracle`, then verify the resulting relaxation
    is a genuine repair using only suitable constraints, and return the metrics."""
    oracle = HierarchicalOracle(root, hard, S, seed=seed, max_steps=max_steps,
                                commit_strategy=commit_strategy, log=log, verbose=verbose)
    for _ in hierarchical_marco(root, hard, solver=solver, map_solver=map_solver, decide_step=oracle):
        pass
    relaxed = set(oracle.relaxed)
    leaf_nodes = root.leaves()
    kept = [lf.get_grouped_constraint() for lf in leaf_nodes if lf.get_full_name() not in relaxed]
    repair_valid = cp.Model(hard + [c for c in kept if c is not None]).solve(solver="ortools") is True
    return {
        "result": oracle.result,
        "relaxed": sorted(relaxed), "n_relaxed": len(relaxed),
        "relaxed_subset_of_S": relaxed <= set(S),
        "repair_valid": repair_valid,
        "n_commit": oracle.n_commit, "n_refine": oracle.n_refine, "n_backtrack": oracle.n_backtrack,
        "n_decisions": oracle.n_commit + oracle.n_refine + oracle.n_backtrack,
        # query count under "one constraint relaxed per query": each refine and each backtrack is
        # one query, and a commit relaxing k leaves counts as k queries (one per relaxation).
        "n_queries": oracle.n_refine + oracle.n_backtrack + oracle.n_relax_queries,
        "n_lookahead_solves": oracle.n_lookahead_solves,
        "n_gmcs_seen": oracle.n_gmcs_seen, "n_gmus_seen": oracle.n_gmus_seen,
    }


def run_hierarchical_cli(args):
    root, hard = build_nurse_hierarchy(args.hier_instance)
    n_leaves = len(root.leaves())
    S, s_seed = sample_suitable_set(root, hard, pct=args.sample_pct,
                                    seed0=(args.seed if args.seed is not None else 0))
    if S is None:
        print(f"No {args.sample_pct}% suitable set admitting a repair was found -- try a higher rate.")
        return
    print(f"Hierarchical oracle on {args.hier_instance}: {len(hard)} hard, {n_leaves} leaf soft "
          f"constraints.")
    print(f"Suitable set S: {len(S)} of {n_leaves} leaves ({args.sample_pct}%), found at sample "
          f"seed {s_seed} (relaxing all of S restores satisfiability, so an MCS subseteq S exists).")
    r = run_hierarchical_oracle(root, hard, S, seed=s_seed, solver=args.solver,
                                map_solver=args.map_solver)
    print(f"\nresult: {r['result'].upper()}  "
          f"(repair valid={r['repair_valid']}, relaxed subseteq S={r['relaxed_subset_of_S']})")
    print(f"  decisions: {r['n_decisions']}  = {r['n_commit']} commit(s) + {r['n_refine']} "
          f"refine(s) + {r['n_backtrack']} backtrack(s)")
    print(f"  group-MCSes shown: {r['n_gmcs_seen']}, group-MUSes shown: {r['n_gmus_seen']}")
    print(f"  final correction subset (relaxed leaves), size {r['n_relaxed']}:")
    for n in r["relaxed"]:
        print(f"    {n}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_acceptable_subset(instance, pct):
    path = FLAT_DIR / instance / f"mcs_subset_{pct}pct.json"
    return [frozenset(names) for names in json.loads(path.read_text(encoding="utf-8"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default="instance_292_first2unplannable")
    parser.add_argument("--mode", choices=["evaluate", "staged-deletion", "hierarchical"], default="evaluate",
                        help="'evaluate' (default): consume marco's MUS/MCS stream, accept/reject "
                             "each whole MCS (see --accept-mode). 'staged-deletion': skip marco "
                             "entirely -- stage relaxations one constraint at a time via "
                             "assumption variables (using --accept-mode random-sample's suitability "
                             "rule only), stopping as soon as satisfiability is restored. Not "
                             "minimal, but needs no map solver / MUS-MCS enumeration at all.")
    parser.add_argument("--accept-mode", choices=["subset", "random-sample"], default="subset",
                        help="'subset' (default): accept MCSes from a precomputed stored "
                             "5/10/15/20%% MCS subset (--subset). 'random-sample': accept any "
                             "MCS fully covered by a randomly sampled subset of CONSTRAINTS "
                             "(not MCSes), starting at --start-pct and escalating. "
                             "(--mode staged-deletion always uses random-sample's suitability rule.)")
    parser.add_argument("--subset", choices=["5", "10", "15", "20"], default="10",
                        help="[subset mode] which stored random MCS subset (percentage) counts as 'acceptable'")
    parser.add_argument("--start-pct", type=float, default=50,
                        help="[random-sample mode] initial %% of constraints sampled as allowed-to-relax")
    parser.add_argument("--escalate-step", type=float, default=10,
                        help="[random-sample mode] %% to grow the allowed set by on escalation")
    parser.add_argument("--escalate-after", type=int, default=100,
                        help="[random-sample mode] escalate after this many consecutive rejects")
    parser.add_argument("--seed", type=int, default=None,
                        help="[random-sample mode] RNG seed for the constraint sample")
    parser.add_argument("--method", choices=sorted(METHODS.keys()), default="baseline")
    parser.add_argument("--solver", default="exact")
    parser.add_argument("--map-solver", default="exact")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="give up after this many proposed MCSes (default: exhaust the enumeration)")
    parser.add_argument("--hier-instance", default="nurse_instance1_softreq_8nurses",
                        help="[hierarchical mode] flat nurse soft/hard instance to wrap in a hierarchy")
    parser.add_argument("--sample-pct", type=float, default=40,
                        help="[hierarchical mode] %% of primitive constraints sampled as suitable (S)")
    args = parser.parse_args()

    if args.mode == "hierarchical":
        run_hierarchical_cli(args)
        return

    soft, hard, soft_names, hard_names = load_flat_instance(args.instance)
    name_of = {id(c): n for c, n in zip(soft, soft_names)}

    if args.mode == "staged-deletion":
        oracle = RandomSampleOracle(soft_names, start_pct=args.start_pct,
                                    escalate_step=args.escalate_step,
                                    escalate_after=args.escalate_after, seed=args.seed)
        print(f"Staged deletion: {len(oracle.allowed)} of {len(soft)} constraints "
              f"({args.start_pct}%) randomly marked suitable to relax.")
        result = run_staged_deletion(soft, hard, soft_names, oracle, solver=args.solver,
                                      map_solver=args.map_solver,
                                      max_iterations=args.max_iterations)
        if result["reached_sat"]:
            status = "SAT REACHED"
        elif result.get("capped"):
            status = f"iteration cap ({args.max_iterations}) hit, still UNSAT"
        else:
            status = "exhausted reachable MCSes, still UNSAT"
        print(f"\n{status} ({result['n_relaxed']} constraints deleted = non-minimal correction "
              f"subset, {result['n_mcs_enumerated']} MCSes enumerated/shown, "
              f"{result['n_mus_seen']} MUS(es) seen, {result['n_solve_calls']} solve() calls)")
        if result["reached_sat"]:
            print("Relaxed constraints:")
            for n in result["relaxed_names"]:
                print(f"  {n}")
        return

    if args.accept_mode == "subset":
        acceptable = load_acceptable_subset(args.instance, args.subset)
        oracle = Oracle(acceptable)
        print(f"Oracle accepts any of {len(acceptable)} MCSes (the {args.subset}% subset) "
              f"out of {len(soft)} primitive soft constraints.")
    else:
        oracle = RandomSampleOracle(soft_names, start_pct=args.start_pct,
                                    escalate_step=args.escalate_step,
                                    escalate_after=args.escalate_after, seed=args.seed)
        print(f"Oracle starts by randomly allowing {len(oracle.allowed)} of {len(soft)} "
              f"constraints ({args.start_pct}%) to be relaxed, escalating +{args.escalate_step}% "
              f"after every {args.escalate_after} consecutive rejects.")

    result = run_oracle(args.method, soft, hard, name_of, oracle,
                        max_iterations=args.max_iterations,
                        solver=args.solver, map_solver=args.map_solver)

    status = "ACCEPTED" if result["accepted"] else "no acceptable MCS found"
    print(f"\nmethod={result['method']}: {status} "
          f"({result['n_rejected']} rejected, {result['n_mus_seen']} MUS(es) seen, "
          f"{result['elapsed_seconds']:.2f}s)")
    if "final_pct" in result:
        print(f"  random-sample: final allowed %={result['final_pct']}, "
              f"escalations={result['n_escalations']}, pct_history={result['pct_history']}")
    if result["accepted"]:
        print("Accepted MCS:")
        for n in result["accepted"]:
            print(f"  {n}")


if __name__ == "__main__":
    main()
