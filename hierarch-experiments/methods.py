"""
    The three explanation methods, each driven by a suitable set S (leaf names the oracle is
    willing to see relaxed). Every method returns the SAME metrics dict (see ``_metrics``), so the
    runner can write one uniform CSV row per (problem, instance, oracle, method).

    ---------------------------------------------------------------------------------------------
    JUDGMENTS -- how the "number of judgments" metric is defined
    ---------------------------------------------------------------------------------------------
    A *judgment* is one atomic act of the oracle assessing whether a single constraint (flat
    methods) or constraint-group (hierarch-commit) is suitable to relax:

      * mcs-enumeration      : the oracle is shown each enumerated MCS and must judge EVERY
                               constraint in it -> judgments += |MCS|, once per MCS shown.
      * selective-relaxation : same -- shown each MCS, judges every constraint in it (then deletes
                               the suitable ones) -> judgments += |MCS| per MCS shown.
      * hierarch-commit      : at each COMMIT the oracle judges every member of the committed
                               group-MCS M (is this group/leaf suitable?) -> += |M|; at each
                               REFINE it makes ONE judgment (decide to drill into this group).
                               Backtracks are internal bookkeeping and count 0 judgments.

    "Decisions" is coarser: the number of times the oracle makes a *choice* -- one per MCS shown
    (flat methods), and one per commit / refine / backtrack (hierarch-commit).
"""
import random
import time

import _bootstrap  # noqa: F401  -- puts the repo root (cpmpy/) on sys.path; must precede cpmpy
import cpmpy as cp
from cpmpy.tools.explain import marco, hierarchical_marco
from cpmpy.tools.explain.utils import make_assump_model
from cpmpy.transformations.get_variables import get_variables

from hierarchy import flat_soft

SOLVER = "exact"               # core + map solver for enumeration (assumption-based, deterministic)
MAP_SOLVER = "exact"
GATE_SOLVER = "ortools"        # fast plain-SAT feasibility / repair checks
ROUND_CAP = 20                 # per round, show the oracle at most this many group-MCS/MUSes before
                               # it must act. This is an INTERACTION detail (don't enumerate a
                               # frontier's whole -- possibly huge -- conflict set before consulting
                               # the oracle), NOT a bound on the search: the open frontier itself is
                               # never capped. The remaining conflicts are simply found in later
                               # rounds. Without it a single round can enumerate combinatorially.


def _metrics(method, *, decisions, judgments, relaxed, pruned=0, excess=None,
             commits=None, backtracks=None, timed_out=False, elapsed=0.0, repaired=False):
    """Uniform per-run metrics row. Fields not relevant to a method are left None."""
    return {"method": method, "decisions": decisions, "judgments": judgments,
            "relaxed": relaxed, "pruned": pruned, "excess": excess,
            "commits": commits, "backtracks": backtracks,
            "timed_out": timed_out, "elapsed": round(elapsed, 3), "repaired": repaired}


def _repaired(root, hard, relaxed_names):
    """Is `relaxed_names` a correction subset? hard + (every non-relaxed leaf) is SAT."""
    kept = [lf.get_grouped_constraint() for lf in root.leaves()
            if lf.get_full_name() not in relaxed_names]
    return cp.Model(list(hard) + [c for c in kept if c is not None]).solve(solver=GATE_SOLVER) is True


def _greedy_minimal(root, hard, relaxed_names, budget=120.0):
    """Greedily shrink `relaxed_names` to a minimal correction subset; return its size."""
    leaves = root.leaves()
    gc = {lf.get_full_name(): lf.get_grouped_constraint() for lf in leaves}
    names = [lf.get_full_name() for lf in leaves]
    cur = set(relaxed_names)
    t0 = time.perf_counter()
    for n in sorted(cur):
        if time.perf_counter() - t0 >= budget:
            break
        kept = [gc[x] for x in names if x not in (cur - {n})]
        if cp.Model(list(hard) + [c for c in kept if c is not None]).solve(solver=GATE_SOLVER) is True:
            cur.discard(n)
    return len(cur)


# =============================================================================================
#  1. mcs-enumeration  (flat MARCO baseline)
# =============================================================================================
def run_baseline(root, hard, S, time_budget=600.0):
    """Flat MARCO. The oracle accepts the first enumerated MCS that lies entirely inside S."""
    soft, names = flat_soft(root)
    name_of = {id(c): n for c, n in zip(soft, names)}
    S = set(S)
    n_dec = n_judg = 0
    accepted = None
    timed_out = False
    t0 = time.perf_counter()
    for kind, found in marco(soft, list(hard), solver=SOLVER, map_solver=MAP_SOLVER,
                             return_mus=True, return_mcs=True):
        if kind == "MUS":
            continue                                              # MUSes steer marco; oracle judges only MCSes
        n_dec += 1
        mcs_names = frozenset(name_of[id(c)] for c in found)
        n_judg += len(mcs_names)                                  # judged every constraint in the MCS
        if mcs_names <= S:
            accepted = sorted(mcs_names)
            break
        if time.perf_counter() - t0 >= time_budget:
            timed_out = True
            break
    return _metrics("mcs-enumeration", decisions=n_dec, judgments=n_judg,
                    relaxed=(len(accepted) if accepted else 0), pruned=0,
                    timed_out=timed_out, elapsed=time.perf_counter() - t0,
                    repaired=accepted is not None)


# =============================================================================================
#  2. selective-relaxation  (staged deletion)
# =============================================================================================
def run_selective_relaxation(root, hard, S, time_budget=600.0):
    """MARCO with partial staging: instead of accepting/rejecting a whole MCS, the oracle DELETES
    the suitable constraints inside every MCS it is shown and keeps going, until the accumulated
    deleted set restores satisfiability. The result is a (generally non-minimal) correction; the
    excess over a minimal correction is reported."""
    soft, names = flat_soft(root)
    S = set(S)
    model, soft2, assump = make_assump_model(soft, list(hard))
    s = cp.SolverLookup.get(SOLVER, model)
    ms = cp.SolverLookup.get(MAP_SOLVER)
    ms += cp.any(assump)
    dmap = dict(zip(assump, soft2))
    idx_of = {a: i for i, a in enumerate(assump)}
    name_of = {a: names[i] for i, a in enumerate(assump)}
    del_order = {a: -len(get_variables(dmap[a])) for a in assump}     # MUS-shrink order, as in marco

    relaxed = set()                                               # indices deleted (assumption forced false)
    n_dec = n_judg = 0
    timed_out = reached_sat = False
    t0 = time.perf_counter()

    def active():
        return [a for i, a in enumerate(assump) if i not in relaxed]

    def sat(assumptions):
        return s.solve(assumptions=assumptions)

    if sat(active()) is True:                                     # already SAT (shouldn't happen)
        reached_sat = True
    else:
        while ms.solve():
            if time.perf_counter() - t0 >= time_budget:
                timed_out = True
                break
            seed = [a for a in assump if a.value() and idx_of[a] not in relaxed]
            if sat(seed) is True:
                act = active()
                mss = [a for a in act if a.value() or dmap[a].value()]
                for to_add in frozenset(act) - frozenset(mss):
                    if sat(mss + [to_add]) is True:
                        mss.append(to_add)
                mcs = [a for a in act if a not in frozenset(mss)]
                ms += cp.any(mcs)                                 # block this MCS
                n_dec += 1
                n_judg += len(mcs)                                # judged every constraint in the MCS
                for a in mcs:
                    if name_of[a] in S:                           # delete the suitable ones
                        relaxed.add(idx_of[a])
                        ms += ~a
                if sat(active()) is True:
                    reached_sat = True
                    break
            else:                                                 # UNSAT seed -> shrink to a MUS, steer
                core = set(s.get_core())
                for c in sorted(core, key=del_order.get):
                    if c not in core:
                        continue
                    core.remove(c)
                    if sat(list(core)):
                        core.add(c)
                    else:
                        core = set(s.get_core())
                ms += ~cp.all(core)

    relaxed_names = {names[i] for i in relaxed}
    excess = None
    if reached_sat:
        remaining = max(1.0, time_budget - (time.perf_counter() - t0))
        excess = len(relaxed_names) - _greedy_minimal(root, hard, relaxed_names,
                                                       budget=min(120.0, remaining))
    return _metrics("selective-relaxation", decisions=n_dec, judgments=n_judg,
                    relaxed=len(relaxed_names), pruned=0, excess=excess,
                    timed_out=timed_out, elapsed=time.perf_counter() - t0, repaired=reached_sat)


# =============================================================================================
#  3. hierarch-commit  (incremental hierarchical MARCO -- NO bound, NO explore-backtrack)
# =============================================================================================
class HierarchCommitOracle:
    """decide_step for ``hierarchical_marco`` implementing the conceptually-simplest commit policy.
    Each round it picks ONE action, in priority order:

      (1) COMMIT a random *committable* group-MCS M -- all members open + potentially suitable, and
          at least one primitive (leaf) member so committing relaxes something suitable. Committing
          relaxes M's leaf members, keeps its non-leaf members OPEN (to refine further), and
          backgrounds every other open group.
      (2) REFINE (explore) one relevant, potentially-suitable open group to expose committable
          group-MCSes -- a still-pending committed member first, else depth-first in the current
          branch, else any relevant group.
      (3) BACKTRACK the most recent commit when the branch is a dead end; if there is nothing left
          to backtrack, stop.

    There is NO frontier cap and NO explore-backtrack (the latter is obsolete): a refine is only
    ever undone by undoing the commit that opened its branch.
    """

    def __init__(self, root, hard, S, seed=0, time_budget=None):
        self.S = frozenset(S)
        self._hard = list(hard)
        self._leaves = root.leaves()
        self.name2node = {nd.get_full_name(): nd for nd in root.iter_nodes()}
        self.rng = random.Random(seed)
        self.time_budget = time_budget
        self.t0 = None
        # per-epoch registry of group-MCSes/MUSes (reset on commit, restored on backtrack)
        self.gmcs, self.gmus = [], []
        self.pending = []                        # non-leaf members of last commit -> co-refine next
        self.last_refined = None                 # tip of the current free branch (depth-first)
        self.stack = []                          # commit-backtrack snapshots (one per commit)
        self.abandoned = set()                   # (open-signature, M) we backtracked from
        self.abandoned_mcs = set()               # M we backtracked from (signature-independent):
                                                 # never re-commit the SAME group-MCS twice
        # outcome + metrics
        self.n_commit = self.n_refine = self.n_backtrack = self.n_restart = 0
        self.n_continue = 0                      # cap-truncated rounds continued instead of failing
        self.judgments = 0
        self._initial_state = None               # enumerator state at round 1 (for fresh restarts)
        self.result = None
        self.relaxed = []                        # final committed_relaxed (leaf names)
        self.background = set()                  # final committed_background (group names)
        # a minimal replayable script of the decisions taken (for runtime.py's base-vs-incremental
        # comparison): one {"action", ...} per commit / refine / backtrack / stop.
        self.script = []

    # --- predicates ---
    def _is_leaf(self, name):
        return not self.name2node[name].children

    def _potentially_suitable(self, name):
        return any(lf.get_full_name() in self.S for lf in self.name2node[name].leaves())

    def _committable(self, M, open_names):
        return (M <= open_names and any(self._is_leaf(g) for g in M)
                and all(self._potentially_suitable(g) for g in M))

    def _is_repaired(self, relaxed):
        kept = [lf.get_grouped_constraint() for lf in self._leaves
                if lf.get_full_name() not in relaxed]
        return cp.Model(self._hard + [c for c in kept if c is not None]).solve(solver=GATE_SOLVER) is True

    def _in_current_branch(self, name):
        if self.last_refined is None:
            return False
        node = self.name2node[name].parent
        while node is not None:
            if node.get_full_name() == self.last_refined:
                return True
            node = node.parent
        return False

    # --- variant hooks (no-ops in the base policy; overridden by the variant subclasses) ---
    def _pre_commit_action(self, ctx, open_names, rel):
        """Chance to act BEFORE the commit/refine rules (forced explore, fresh restart, ...)."""
        return None

    def _alt_commit_action(self, ctx, open_names):
        """Chance to commit under a WEAKER rule after the strict rule found no option."""
        return None

    def _ps_options(self, open_names):
        """Group-MCSes that are *potentially suitable* (every member holds some S-leaf) and lie
        entirely within the open frontier -- weaker than `_committable` (no leaf-member demand)."""
        committed_active = {snap[-1] for snap in self.stack}
        return [M for M in self.gmcs
                if M <= open_names and M not in self.abandoned_mcs
                and M not in committed_active and (open_names, M) not in self.abandoned
                and all(self._potentially_suitable(g) for g in M)]

    # --- policy ---
    def __call__(self, ctx):
        for r in ctx["results"]:
            fs = frozenset(r["names"])
            if r["kind"] == "MCS" and fs not in self.gmcs:
                self.gmcs.append(fs)
            elif r["kind"] == "MUS" and fs not in self.gmus:
                self.gmus.append(fs)
        bg = set(ctx["committed_background"])
        rel = set(ctx["committed_relaxed"])
        open_names = frozenset(nd.get_full_name() for nd in ctx["frontier_nodes"]
                               if nd.get_full_name() not in rel and nd.get_full_name() not in bg)
        self.relaxed, self.background = list(rel), bg
        if self._initial_state is None:
            self._initial_state = ctx["state"]

        if (self.time_budget is not None and self.t0 is not None
                and time.perf_counter() - self.t0 >= self.time_budget):
            return self._stop("timeout")
        if not open_names:
            return self._stop("repaired" if self._is_repaired(rel) else "failed")

        # variant hook: forced exploration / fresh restart / ... (no-op in the base policy)
        act = self._pre_commit_action(ctx, open_names, rel)
        if act is not None:
            return act

        # (1) COMMIT a random committable group-MCS. Never commit the same MCS twice: skip any M
        # already committed on the current branch (on the stack) or previously backtracked from
        # (abandoned_mcs) -- the plain `abandoned` guard is signature-specific and would otherwise
        # let an unrelated refine re-expose the same M for a redundant re-commit.
        committed_active = {snap[-1] for snap in self.stack}
        options = [M for M in self.gmcs
                   if (open_names, M) not in self.abandoned
                   and M not in self.abandoned_mcs
                   and M not in committed_active
                   and self._committable(M, open_names)]
        if options:
            return self._commit(ctx, self.rng.choice(options), open_names)

        # variant hook: weaker commit rules (premature / random commit; no-op in the base policy)
        act = self._alt_commit_action(ctx, open_names)
        if act is not None:
            return act

        # (2) REFINE toward a committable group-MCS.
        # NOTE: iterate open_names SORTED. It is a frozenset of strings, and Python randomises
        # string hashes per process (PYTHONHASHSEED), so an unsorted iteration makes `cands`
        # -- and hence pend[0]/branch[0]/rng.choice -- order-dependent on the process, not on
        # the seed. That made runs irreproducible across machines/processes (measured: the same
        # cell+seed solving in 2.6s, 8.6s, or timing out, in three consecutive runs).
        relevant = set().union(*self.gmcs, *self.gmus) if (self.gmcs or self.gmus) else set()
        cands = [nm for nm in sorted(open_names) if nm in relevant
                 and self.name2node[nm].children and self._potentially_suitable(nm)]
        pend = [c for c in cands if c in self.pending]
        branch = [c for c in cands if self._in_current_branch(c)]
        if pend:
            return self._refine(ctx, pend[0])
        if branch:
            return self._refine(ctx, branch[0])
        if cands:
            return self._refine(ctx, self.rng.choice(cands))

        # (3) BACKTRACK the last commit, else stop
        if self.stack:
            return self._commit_backtrack(ctx)
        if self._is_repaired(rel):
            return self._stop("repaired")
        # (3b) NEVER fail while the round was cap-truncated: ctx["capped"] means the CURRENT
        # frontier provably has conflicts the oracle has not been shown yet (the per-round cap
        # deferred them), so "no options" only means the shown SAMPLE is exhausted. Fetch the
        # next round_cap conflicts of the same frontier instead of failing early. Every run now
        # ends "repaired" or "timeout" -- the early-fail category is eliminated by construction.
        if ctx.get("capped"):
            self.n_continue += 1
            self.script.append({"action": "continue"})
            return {"action": "continue"}
        return self._stop("failed")

    def _commit(self, ctx, M, open_names):
        self.n_commit += 1
        self.judgments += len(M)                                 # judged each member of M
        self.stack.append((ctx["state"], list(self.gmcs), list(self.gmus),
                           self.last_refined, list(self.pending), open_names, M))
        self.pending = [g for g in sorted(M) if not self._is_leaf(g)]
        # seed the new epoch's registry with the pending members as one GMCS: the enumerator will
        # NOT re-yield this (it is the blocked finer twin of the coarse M just committed), so
        # without the seed, refining a pending member has nothing to translate down to a leaf-MCS.
        self.gmcs = [frozenset(self.pending)] if self.pending else []
        self.gmus, self.last_refined = [], None
        self.script.append({"action": "commit", "mcs": sorted(M)})
        return {"action": "commit", "mcs": sorted(M)}

    def _refine(self, ctx, pick):
        children = [c.get_full_name() for c in self.name2node[pick].children
                    if c.get_grouped_constraint() is not None]
        for fs in list(self.gmcs):                               # translate a registered GMCS one level down
            if pick in fs:
                t = (fs - {pick}) | frozenset(children)
                if t not in self.gmcs:
                    self.gmcs.append(t)
        self.pending = [p for p in self.pending if p != pick]
        self.last_refined = pick
        self.n_refine += 1
        self.judgments += 1                                      # one judgment: drill into this group
        self.script.append({"action": "refine", "group": pick, "children": children})
        return {"action": "refine", "constraints": [pick], "target_level": None}

    def _commit_backtrack(self, ctx):
        state, gmcs, gmus, last_ref, pending, open_sig, M = self.stack.pop()
        r_delta = frozenset(g for g in M if self._is_leaf(g))    # leaves the popped commit relaxed
        translated = [fs | r_delta for fs in self.gmcs]          # keep inner discoveries, lifted up
        self.gmcs, self.gmus = gmcs, gmus
        for fs in translated:
            if fs not in self.gmcs:
                self.gmcs.append(fs)
        self.last_refined, self.pending = last_ref, pending
        self.abandoned.add((open_sig, M))                        # don't re-commit M at that state
        self.abandoned_mcs.add(M)                                # ...nor the same M at any state
        self.n_backtrack += 1
        self.script.append({"action": "backtrack"})
        return {"action": "restore", "state": state}

    def _stop(self, reason):
        self.result = reason
        self.script.append({"action": "stop", "reason": reason})
        return {"action": "stop"}


# ------------------------------------------------------ hierarch-commit variants --------
FRONTIER_CAP = 20              # random-commit: commit randomly once the open frontier exceeds this
RESTART_STEPS = 100            # fresh-restart: restart after this many decisions w/o a NEW relaxation
BRANCH_STEP_CAP = 50           # step-backtrack: force-backtrack a branch after this many decisions
                               # without a commit (~30s at the measured median 0.61 s/decision)


class HierarchExploreBacktrackOracle(HierarchCommitOracle):
    """hierarch-commit + EXPLORE ON BACKTRACK: the moment a commit is backtracked, the next action
    must be a random exploration (refine) of a constraint group that occurs together, in some
    registered group-MCS, with a primitive constraint the just-backtracked commit had relaxed."""

    def __init__(self, root, hard, S, seed=0, time_budget=None):
        super().__init__(root, hard, S, seed=seed, time_budget=time_budget)
        self._explore_seed = None                # leaves relaxed by the commit just backtracked

    def _commit_backtrack(self, ctx):
        M = self.stack[-1][-1]
        act = super()._commit_backtrack(ctx)
        self._explore_seed = frozenset(g for g in M if self._is_leaf(g))
        return act

    def _pre_commit_action(self, ctx, open_names, rel):
        if not self._explore_seed:
            return None
        seed_leaves, self._explore_seed = self._explore_seed, None
        cands = sorted({g for fs in self.gmcs if fs & seed_leaves for g in fs
                        if g in open_names and not self._is_leaf(g)
                        and self.name2node[g].children and self._potentially_suitable(g)})
        if not cands:
            return None                          # nothing refinable co-occurs: fall through
        return self._refine(ctx, self.rng.choice(cands))


class HierarchPrematureCommitOracle(HierarchCommitOracle):
    """hierarch-commit + PREMATURE COMMIT: every round (i.e. after every exploration step), check
    ALL registered group-MCSes; if exactly ONE is potentially suitable -- no other option -- commit
    it, even when it is not yet strictly committable (e.g. it still contains non-leaf groups)."""

    def _alt_commit_action(self, ctx, open_names):
        ps = self._ps_options(open_names)
        if len(ps) == 1:
            return self._commit(ctx, ps[0], open_names)
        return None


class HierarchRandomCommitOracle(HierarchCommitOracle):
    """hierarch-commit + RANDOM COMMIT: once the open frontier grows beyond FRONTIER_CAP groups,
    commit a RANDOM potentially-suitable group-MCS (even if there are several, and even if none is
    strictly committable) to cut the frontier back down."""

    def _alt_commit_action(self, ctx, open_names):
        if len(open_names) <= FRONTIER_CAP:
            return None
        ps = self._ps_options(open_names)
        if not ps:
            return None
        return self._commit(ctx, self.rng.choice(ps), open_names)


class HierarchFreshRestartOracle(HierarchCommitOracle):
    """hierarch-commit + RANDOM FRESH RESTART: if more than RESTART_STEPS decisions pass without
    relaxing a NEW constraint (never relaxed before at ANY point of the run), restore the initial
    abstraction level and start over. Choices are TRULY random (SystemRandom -- the seed argument
    is ignored) so a restart can actually take a different path. The per-attempt duplicate-commit
    guards are cleared on restart: a fresh attempt may re-commit an MCS an earlier attempt
    abandoned (within one attempt the same MCS is still never committed twice)."""

    def __init__(self, root, hard, S, seed=0, time_budget=None):
        super().__init__(root, hard, S, seed=seed, time_budget=time_budget)
        self.rng = random.SystemRandom()         # truly random: restarts must be able to diverge
        self._ever_relaxed = set()
        self._steps_since_new = 0

    def _restart(self):
        self.gmcs, self.gmus = [], []
        self.pending, self.last_refined, self.stack = [], None, []
        self.abandoned, self.abandoned_mcs = set(), set()
        self._steps_since_new = 0
        self.n_restart += 1
        self.script.append({"action": "restart"})
        return {"action": "restore", "state": self._initial_state}

    def _pre_commit_action(self, ctx, open_names, rel):
        new = rel - self._ever_relaxed
        if new:
            self._ever_relaxed |= new
            self._steps_since_new = 0
        else:
            self._steps_since_new += 1
        if self._steps_since_new > RESTART_STEPS:
            return self._restart()
        return None

    def _stop(self, reason):
        # a failed dead-end is not final for this variant: restart until the budget ends the run
        if reason == "failed" and self._initial_state is not None:
            return self._restart()
        return super()._stop(reason)


class HierarchStepBacktrackOracle(HierarchCommitOracle):
    """hierarch-commit + PER-BRANCH STEP CAP: if CAP decisions pass inside the current branch
    without another commit, the branch is presumed stuck -- force-backtrack ONE level (keep the
    commit stack and everything learned; this is NOT a restart). The counter resets on every
    commit and backtrack. Override CAP (class attribute) for other bounds -- see
    ``make_step_backtrack``."""

    CAP = BRANCH_STEP_CAP

    def __init__(self, root, hard, S, seed=0, time_budget=None):
        super().__init__(root, hard, S, seed=seed, time_budget=time_budget)
        self._branch_steps = 0

    def _commit(self, ctx, M, open_names):
        self._branch_steps = 0
        return super()._commit(ctx, M, open_names)

    def _commit_backtrack(self, ctx):
        self._branch_steps = 0
        return super()._commit_backtrack(ctx)

    def _pre_commit_action(self, ctx, open_names, rel):
        self._branch_steps += 1
        if self._branch_steps > type(self).CAP and self.stack:
            return self._commit_backtrack(ctx)
        return None


def run_hierarch_commit(root, hard, S, seed=0, time_budget=600.0, round_cap=ROUND_CAP,
                        method="hierarch-commit", oracle_cls=HierarchCommitOracle):
    """Drive ``hierarchical_marco`` with :class:`HierarchCommitOracle` (no frontier bound, no
    explore-backtrack), verify the resulting relaxation, and return the metrics.

    :param round_cap: per-round conflict cap given to the oracle. Two variants are exposed as
        separate methods (see ``METHODS``):
          * ``hierarch-commit``       -- round_cap=20  : the oracle decides on at most 20 group
                                          MUS/MCSes per round (the rest are found in later rounds).
          * ``hierarch-commit-nocap`` -- round_cap=None: EVERY group MUS/MCS of the current frontier
                                          is enumerated before the oracle decides.
    """
    oracle = oracle_cls(root, hard, S, seed=seed, time_budget=time_budget)
    oracle.t0 = time.perf_counter()
    deadline = oracle.t0 + time_budget if time_budget is not None else None
    for _ in hierarchical_marco(root, list(hard), solver=SOLVER, map_solver=MAP_SOLVER,
                                decide_step=oracle, deadline=deadline, round_cap=round_cap):
        pass
    elapsed = time.perf_counter() - oracle.t0
    relaxed = set(oracle.relaxed)
    repaired = _repaired(root, hard, relaxed)
    if oracle.result is None:                                    # enumerator stopped on the deadline first
        oracle.result = "repaired" if repaired else "timeout"
    # pruned primitives = number of leaf constraints inside the final background (pruned) groups
    pruned = sum(len(oracle.name2node[g].leaves()) for g in oracle.background)
    decisions = (oracle.n_commit + oracle.n_refine + oracle.n_backtrack + oracle.n_restart
                 + oracle.n_continue)
    return _metrics(method, decisions=decisions, judgments=oracle.judgments,
                    relaxed=len(relaxed), pruned=pruned, commits=oracle.n_commit,
                    backtracks=oracle.n_backtrack, timed_out=(oracle.result == "timeout"),
                    elapsed=elapsed, repaired=repaired)


def run_hierarch_commit_nocap(root, hard, S, seed=0, time_budget=600.0):
    """`hierarch-commit` with NO round cap: every group MUS/MCS of a frontier is enumerated before
    the oracle acts (vs the capped variant, which acts on the first 20)."""
    return run_hierarch_commit(root, hard, S, seed=seed, time_budget=time_budget,
                               round_cap=None, method="hierarch-commit-nocap")


def run_explore_backtrack(root, hard, S, seed=0, time_budget=600.0):
    return run_hierarch_commit(root, hard, S, seed=seed, time_budget=time_budget,
                               oracle_cls=HierarchExploreBacktrackOracle,
                               method="hierarch-explore-backtrack")


def run_premature_commit(root, hard, S, seed=0, time_budget=600.0):
    return run_hierarch_commit(root, hard, S, seed=seed, time_budget=time_budget,
                               oracle_cls=HierarchPrematureCommitOracle,
                               method="hierarch-premature-commit")


def run_random_commit(root, hard, S, seed=0, time_budget=600.0):
    return run_hierarch_commit(root, hard, S, seed=seed, time_budget=time_budget,
                               oracle_cls=HierarchRandomCommitOracle,
                               method="hierarch-random-commit")


def run_fresh_restart(root, hard, S, seed=0, time_budget=600.0):
    return run_hierarch_commit(root, hard, S, seed=seed, time_budget=time_budget,
                               oracle_cls=HierarchFreshRestartOracle,
                               method="hierarch-fresh-restart")


def run_hierarch_commit_nocap_baseline(root, hard, S, seed=0, time_budget=600.0):
    """``hierarch-commit-nocap`` driven by BASELINE MARCO instead of the incremental enumerator:
    at every round, flat ``marco`` is re-run FROM SCRATCH (fresh core + map solver) on the
    current frontier's grouped constraints, exhaustively (no cap), and ALL conflicts are shown
    to the oracle. Nothing persists between rounds -- previously found conflicts are re-derived
    and re-shown every round (the oracle's own registry dedupes them). State (frontier / pruned /
    relaxed / snapshots) is tracked by hand, mirroring the enumerator's commit semantics."""
    oracle = HierarchCommitOracle(root, hard, S, seed=seed, time_budget=time_budget)
    oracle.t0 = time.perf_counter()
    deadline = oracle.t0 + time_budget

    name2node = {nd.get_full_name(): nd for nd in root.iter_nodes()}
    frontier = [c.get_full_name() for c in root.children]
    pruned, relaxed = set(), set()
    round_idx = 0
    timed_out_flag = False

    while True:
        round_idx += 1
        if time.perf_counter() > deadline:
            timed_out_flag = True
            break
        active = [g for g in frontier if g not in relaxed and g not in pruned]
        gc = {g: name2node[g].get_grouped_constraint() for g in active}
        soft = [gc[g] for g in active if gc[g] is not None]
        names = {id(gc[g]): g for g in active if gc[g] is not None}
        hard_now = list(hard) + [name2node[g].get_grouped_constraint() for g in pruned
                                 if name2node[g].get_grouped_constraint() is not None]
        results = []
        if soft:
            for kind, cons in marco(soft, hard_now, solver=SOLVER, map_solver=MAP_SOLVER,
                                    return_mus=True, return_mcs=True):
                results.append({"kind": kind,
                                "names": [names[id(c)] for c in cons]})
                if time.perf_counter() > deadline:
                    timed_out_flag = True
                    break
        if timed_out_flag:
            break

        action = oracle({
            "round": round_idx,
            "frontier": list(frontier),
            "frontier_nodes": [name2node[g] for g in frontier],
            "free": list(active),
            "refinable": [g for g in frontier if name2node[g].children],
            "results": results,
            "capped": False,                     # exhaustive per round, like nocap
            "committed_relaxed": sorted(relaxed),
            "committed_background": sorted(pruned),
            "state": {"frontier": list(frontier), "pruned": set(pruned),
                      "relaxed": set(relaxed)},
        })
        a = action.get("action")
        if a == "stop" or not a:
            break
        if a == "continue":
            continue                             # cannot happen (capped=False); harmless
        if a == "restore":
            st = action["state"]
            frontier = list(st["frontier"])
            pruned, relaxed = set(st["pruned"]), set(st["relaxed"])
            continue
        if a == "commit":
            mcs = set(action["mcs"])
            for g in list(frontier):
                if g in relaxed or g in pruned:
                    continue
                if g in mcs:
                    if not name2node[g].children:
                        relaxed.add(g)
                else:
                    pruned.add(g)
            continue
        if a == "refine":
            for nm in action["constraints"]:
                children = [c.get_full_name() for c in name2node[nm].children
                            if c.get_grouped_constraint() is not None]
                frontier = [x for x in frontier if x != nm] + children
            continue
        raise ValueError(f"unknown action {action!r}")

    elapsed = time.perf_counter() - oracle.t0
    rel = set(oracle.relaxed) if oracle.relaxed else set(relaxed)
    repaired = _repaired(root, hard, rel)
    if oracle.result is None:
        oracle.result = "repaired" if repaired else "timeout"
    pruned_leaves = sum(len(name2node[g].leaves()) for g in oracle.background or pruned)
    decisions = (oracle.n_commit + oracle.n_refine + oracle.n_backtrack + oracle.n_restart
                 + oracle.n_continue)
    return _metrics("hierarch-commit-nocap-baseline", decisions=decisions,
                    judgments=oracle.judgments, relaxed=len(rel), pruned=pruned_leaves,
                    commits=oracle.n_commit, backtracks=oracle.n_backtrack,
                    timed_out=(oracle.result == "timeout" or timed_out_flag),
                    elapsed=elapsed, repaired=repaired)


class HierarchCommitNocapEagerOracle(HierarchCommitOracle):
    """hierarch-commit with NO round cap, but EAGER commit: instead of enumerating the whole
    frontier before deciding, commit a group-MCS the moment it is discovered during enumeration,
    provided it is committable (>=1 leaf member, all members potentially suitable, in the open
    frontier) and has not been committed/abandoned before. Realised via the enumerator's
    `early_stop` hook, which cuts the round as soon as `_eager_ok` accepts a freshly-found MCS;
    decide_step then commits it through the normal rule (options is non-empty by construction)."""

    def _eager_ok(self, rec):
        if rec["kind"] != "MCS":
            return False
        M = frozenset(rec["names"])
        committed_active = {snap[-1] for snap in self.stack}
        return (M not in self.abandoned_mcs and M not in committed_active
                and any(self._is_leaf(g) for g in M)
                and all(self._potentially_suitable(g) for g in M))


def run_hierarch_commit_nocap_eager(root, hard, S, seed=0, time_budget=600.0):
    oracle = HierarchCommitNocapEagerOracle(root, hard, S, seed=seed, time_budget=time_budget)
    oracle.t0 = time.perf_counter()
    deadline = oracle.t0 + time_budget
    for _ in hierarchical_marco(root, list(hard), solver=SOLVER, map_solver=MAP_SOLVER,
                                decide_step=oracle, deadline=deadline, round_cap=None,
                                early_stop=oracle._eager_ok):
        pass
    elapsed = time.perf_counter() - oracle.t0
    relaxed = set(oracle.relaxed)
    repaired = _repaired(root, hard, relaxed)
    if oracle.result is None:
        oracle.result = "repaired" if repaired else "timeout"
    pruned = sum(len(oracle.name2node[g].leaves()) for g in oracle.background)
    decisions = (oracle.n_commit + oracle.n_refine + oracle.n_backtrack + oracle.n_restart
                 + oracle.n_continue)
    return _metrics("hierarch-commit-nocap-eager", decisions=decisions, judgments=oracle.judgments,
                    relaxed=len(relaxed), pruned=pruned, commits=oracle.n_commit,
                    backtracks=oracle.n_backtrack, timed_out=(oracle.result == "timeout"),
                    elapsed=elapsed, repaired=repaired)


def run_premature_commit_nocap(root, hard, S, seed=0, time_budget=600.0):
    """PREMATURE COMMIT with NO round cap: every group MUS/MCS of the current frontier is
    enumerated before the oracle decides (cf. hierarch-commit-nocap), combined with the
    premature rule (commit an MCS when it is the ONLY potentially suitable option)."""
    return run_hierarch_commit(root, hard, S, seed=seed, time_budget=time_budget,
                               round_cap=None, oracle_cls=HierarchPrematureCommitOracle,
                               method="hierarch-premature-commit-nocap")


def run_portfolio(root, hard, S, seed=0, time_budget=600.0, round_cap=ROUND_CAP,
                  method="hierarch-portfolio"):
    """PORTFOLIO of the three complementary commit rules (base / premature / random -- overlap
    analysis: Jaccard 0.22-0.37, over half of the solved cells solved by exactly one of them).
    Run them SEQUENTIALLY, each on an equal share of the budget, stopping at the first repair.
    decisions/judgments/commits/backtracks are summed over the attempts (total oracle effort);
    relaxed/pruned come from the successful attempt (else the last one)."""
    t0 = time.perf_counter()
    share = time_budget / 3
    tot = {"decisions": 0, "judgments": 0, "commits": 0, "backtracks": 0}
    m = None
    for cls in (HierarchCommitOracle, HierarchPrematureCommitOracle,
                HierarchRandomCommitOracle):
        m = run_hierarch_commit(root, hard, S, seed=seed, time_budget=share,
                                round_cap=round_cap, method=method, oracle_cls=cls)
        for k in tot:
            tot[k] += m[k] or 0
        if m["repaired"]:
            break
    elapsed = time.perf_counter() - t0
    return _metrics(method, decisions=tot["decisions"], judgments=tot["judgments"],
                    relaxed=m["relaxed"], pruned=m["pruned"],
                    commits=tot["commits"], backtracks=tot["backtracks"],
                    timed_out=(not m["repaired"]) and elapsed >= 0.95 * time_budget,
                    elapsed=elapsed, repaired=m["repaired"])


def run_portfolio_nocap(root, hard, S, seed=0, time_budget=600.0):
    """`hierarch-portfolio` with NO round cap in the constituent runs."""
    return run_portfolio(root, hard, S, seed=seed, time_budget=time_budget,
                         round_cap=None, method="hierarch-portfolio-nocap")


def run_step_backtrack(root, hard, S, seed=0, time_budget=600.0):
    return run_hierarch_commit(root, hard, S, seed=seed, time_budget=time_budget,
                               oracle_cls=HierarchStepBacktrackOracle,
                               method="hierarch-step-backtrack")


def run_step_backtrack_nocap(root, hard, S, seed=0, time_budget=600.0):
    return run_hierarch_commit(root, hard, S, seed=seed, time_budget=time_budget,
                               round_cap=None, oracle_cls=HierarchStepBacktrackOracle,
                               method="hierarch-step-backtrack-nocap")


METHODS = {
    "mcs-enumeration": run_baseline,
    "selective-relaxation": run_selective_relaxation,
    "hierarch-commit": run_hierarch_commit,                  # per-round cap of 20 conflicts
    "hierarch-commit-nocap": run_hierarch_commit_nocap,      # no cap: enumerate ALL conflicts / round
    "hierarch-explore-backtrack": run_explore_backtrack,     # random co-occurring explore on backtrack
    "hierarch-premature-commit": run_premature_commit,       # commit when it is the ONLY ps option
    "hierarch-random-commit": run_random_commit,             # random ps commit once frontier > 20
    "hierarch-fresh-restart": run_fresh_restart,             # truly-random restart after 100 stale steps
    "hierarch-commit-nocap-baseline": run_hierarch_commit_nocap_baseline,
    "hierarch-premature-commit-nocap": run_premature_commit_nocap,
    "hierarch-commit-nocap-eager": run_hierarch_commit_nocap_eager,
    "hierarch-portfolio": run_portfolio,                     # base -> premature -> random, budget/3 each
    "hierarch-portfolio-nocap": run_portfolio_nocap,         # portfolio with no round cap
    "hierarch-step-backtrack": run_step_backtrack,           # force backtrack after 50 stale branch steps
    "hierarch-step-backtrack-nocap": run_step_backtrack_nocap,
}


def make_step_backtrack(cap):
    """Runner for a step-backtrack variant with branch-step cap `cap` (method name carries the
    bound, e.g. ``hierarch-step-backtrack-ub100``)."""
    cls = type(f"HierarchStepBacktrackOracleUb{cap}", (HierarchStepBacktrackOracle,),
               {"CAP": cap})

    def runner(root, hard, S, seed=0, time_budget=600.0):
        return run_hierarch_commit(root, hard, S, seed=seed, time_budget=time_budget,
                                   oracle_cls=cls, method=f"hierarch-step-backtrack-ub{cap}")
    return runner


# cap-sweep variants (ub50 is the plain hierarch-step-backtrack above)
for _cap in (10, 20, 100, 200, 500):
    METHODS[f"hierarch-step-backtrack-ub{_cap}"] = make_step_backtrack(_cap)
