"""
    MARCO-style MUS/MCS enumeration over a hierarchy of :class:`ConstraintNode` groups.

    =================
    List of functions
    =================

    .. autosummary::
        :nosignatures:

        hierarchical_marco
        map_incremental_marco
"""

import time

import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.normalize import toplevel_list

from .hierarchical import activate_descendants_at_level


def _initial_cut(root, initial_level):
    """Partition `root`'s subtree into groups by cutting the tree at `initial_level`."""
    def recurse(node, level):
        if level >= initial_level or not node.children:
            cons = node.get_grouped_constraint()
            return [node] if cons is not None else []
        result = []
        for child in node.children:
            result.extend(recurse(child, level + 1))
        return result

    return recurse(root, 0)


def _collect_candidates(groups):
    """All nodes reachable from `groups` by repeatedly descending into children.

    This is the full set of nodes that :func:`hierarchical_marco` could ever turn into a
    group through refinement, so indicator variables are set up for each of them upfront.
    """
    candidates = []
    seen_ids = set()
    stack = list(groups)
    while stack:
        node = stack.pop()
        if id(node) in seen_ids:
            continue
        seen_ids.add(id(node))
        candidates.append(node)
        for child in node.children:
            if child.get_grouped_constraint() is not None:
                stack.append(child)
    return candidates


def hierarchical_marco(root, hard=[], solver="ortools", map_solver="ortools",
                        initial_level=1, return_mus=True, return_mcs=True, do_solution_hint=True,
                        round_timings=None, scripted_steps=None, log_events=None,
                        core_per_round=False, lazy_map=True, decide_step=None, deadline=None,
                        round_cap=None, direct_commit_literals=False, early_stop=None):
    """
        MARCO-style enumeration of MUSes and MCSes over a hierarchy of constraint groups.

        The hierarchy `root` is first cut into groups at depth `initial_level` (`root` itself
        is at depth 0): every node at that depth, or every leaf found above that depth, becomes
        one group whose equivalent constraint is the conjunction of its whole subtree.

        MARCO is then run over these groups. After all MUSes/MCSes at this granularity have been found,
        any group(s) that was/were involved in at least one MUS or MCS can be refined by replacing the group with
        its children (each again collapsed to the conjunction of its own subtree) and enumeration continues
        on the refined set of groups.
        This way the enumeration progressively zooms into the parts of the hierarchy that are
        considered important by the user.

        In the context of the experimental setting, the constraints to be refined are given in advance.
        In a realistic scenario, this would be decided by user input.

        Both the satisfiability solver and the map solver are constructed once and reused
        for every refinement round, through indicator variables (one set per node that could
        ever become a group, i.e. every node reachable from the initial cut by repeatedly
        descending into children):

        - `i` ("indicator"): the assumption variable passed to the core solver `s`, with
          `i.implies(node.get_grouped_constraint())`.
        - `u` / `d` ("up" / "down"): used by the map solver for MUS/MCS blocking.
        - `a` / `p` ("activated" / "partitioned"): map solver *assumption* variables marking
          whether a node is currently activated (`a`) or a refined/partitioned ancestor of an
          activated constraint (`p`).

        The map solver permanently links these per node via `a.implies(u == i)`,
        `a.implies(d == i)`, `p.implies(d == cp.any(child_d))` and `p.implies(u == cp.all(child_u))`,
        but only enforces the linking relevant to a node's *current* role by asserting `a`/`p`
        as `assumptions=` each round. This lets a MUS/MCS found at one granularity (expressed
        via `u`/`d`) keep blocking the corresponding seeds once its node is refined and `u`/`d`
        get reinterpreted in terms of the (now activated) children, without rebuilding either
        solver.

        Yields `(kind, constraints, names, round)` tuples, where `kind` is `"MUS"` or `"MCS"`,
        `constraints` is the list of grouped constraints (one `cp.all(...)` per involved node),
        `names` is the list of `ConstraintNode.get_full_name()` for those nodes, and `round`
        is the 1-indexed refinement round in which this MUS/MCS was found (incremented every
        time the current set of groups gets refined, mirroring the per-round structure of
        re-running flat `marco` on each successive refinement of `groups`).

        :param: root: root :class:`ConstraintNode` of the constraint hierarchy
        :param: hard: hard constraints, optional, list of expressions
        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
        :param: map_solver: the map solver to use, ideally incremental such as "gurobi", "pysat" or "exact"
        :param: initial_level: depth at which the hierarchy is initially cut into groups
        :param: return_mus: whether the algorithm should return MUSes
        :param: return_mcs: whether the algorithm should return MCSes
        :param: do_solution_hint: when true, will favor large seeds generated by the map-solver, and hence more likely
                                   to return MUSes
        :param: round_timings: optional list; if given, one ``{"round": round_idx, "seconds": ...}`` dict is appended
                               per refinement round, timing the round's enumeration loop (from the first map-solver
                               call of that round until the map solver returns UNSAT for it)
        :param: scripted_steps: optional list of refinement steps; when given, refinement is **scripted** instead of
                               driven by which groups appeared in a MUS/MCS. Each step is a dict
                               ``{"constraints": [<full group name>, ...], "target_level": int}``; after the round's
                               enumeration the named groups are refined to ``target_level`` via
                               :func:`activate_descendants_at_level` (which may skip intermediate levels). The
                               enumeration then runs for exactly ``len(scripted_steps) + 1`` rounds. This mirrors the
                               scripted-scenario refinement of the defense-rostering experiments.
        :param: log_events: optional list; if given, a chronological trace is appended: one
                            ``{"type": "round", "round": int, "frontier": [names]}`` record at the start of each round,
                            and one ``{"type": "iter", "seed": [names], "kind": "MUS"/"MCS", "result": [names]}`` record
                            per map-solver iteration (the seed it returned and the MUS/MCS derived from it).
        :param: core_per_round: experimental setting (see ``map_incremental_marco``). When ``False``
                               (default), the core solver `s` is built once, persistently, exactly like the
                               map solver, with each LEAF's constraint posted once and a group's indicator
                               expanded to its leaves' indicators when used as an assumption (avoids
                               re-posting a leaf's constraint once per ancestor level, at the cost of a
                               longer assumption list per core-solver call). When ``True``, the core solver
                               is instead **rebuilt from scratch every round**, containing only the current
                               round's frontier groups, each with its own indicator directly implying its
                               ``get_grouped_constraint()`` -- exactly like the flat ``marco`` baseline's core
                               solver, and reusing the very same group indicator variables the map solver
                               uses (so no separate leaf-level bookkeeping is needed). Only the map solver
                               stays persistent/incremental across rounds in that case.
        :param: lazy_map: when ``True`` (the default), the map solver's structural linking constraints
                               (``a.implies(u==i)``/``a.implies(d==i)`` and ``p.implies(...)``) are posted
                               **lazily** -- a node's links are added the first round it appears in the
                               frontier (``a``-links) or in ``partitioned`` (``p``-links), instead of eagerly
                               for the whole reachable tree up front. This is output-identical (a node's links
                               are inert until its ``act``/``part`` assumption is asserted, which only ever
                               happens for reached nodes) but keeps the persistent map solver's working set
                               proportional to the *reached* part of the tree rather than all candidates.
                               Since a variable only enters the solver when a constraint references it, this
                               also keeps unreached nodes' indicator/up/down/act/part variables out of the
                               map solver entirely. Empirically this is what removes the per-core-solve
                               slowdown that a large resident map-solver instance otherwise imposes (a
                               process-level effect: the bigger the map solver built alongside the core
                               solver, the slower each core solve runs). Set ``False`` to post all links up
                               front (the original behaviour). The map solver stays fully incremental;
                               nothing is ever removed or reinitialized, and blocking clauses are unaffected.
        :param: decide_step: optional callable driving refinement/commit **interactively** (or
                               programmatically), taking precedence over ``scripted_steps`` and the
                               default auto-refine. After each round's enumeration it is called with a
                               context dict (``round``, ``frontier`` names, ``free`` names, ``refinable``
                               names, ``frontier_nodes`` objects, this round's ``results`` list of
                               ``{"kind", "names"}``, and the current ``committed_relaxed`` /
                               ``committed_background`` names) and must return an action dict:

                                 - ``{"action": "refine", "constraints": [<group full name>...],
                                   "target_level": int|None}`` -- refine those groups; a ``None``
                                   target refines each named group one level below its OWN level
                                   (so several groups at different depths can be refined at once);
                                 - ``{"action": "commit", "mcs": [<group full name>...]}`` -- the user
                                   **commits** to this group-oriented MCS: every active group in it that
                                   is a leaf (no children) is *relaxed* (``ind`` forced false), every
                                   other active group becomes *background* (``ind`` forced true and used
                                   as a core-solver assumption), and any non-leaf group in the MCS stays
                                   free for further refinement;
                                 - ``{"action": "restore", "state": <snapshot>}`` -- backtrack to a
                                   previously seen state. ``<snapshot>`` is the ``"state"`` dict a
                                   prior context provided (frontier ``groups`` + ``partitioned`` /
                                   ``committed_background`` / ``committed_relaxed`` sets); the map
                                   solver keeps all blocking clauses (they are inert without the
                                   matching assumptions), so only these sets are restored;
                                 - ``{"action": "stop"}`` / falsy -- end enumeration.

                               After a commit, every enumerated MUS, MSS and MCS ranges only over
                               the *free* constraints (active, not relaxed, not background); hard and
                               background constraints form the fixed context of every core solve and
                               never appear in a yielded MUS/MCS.

                               The three "commit" map-solver variables per node -- ``background`` (implies
                               ``ind``), ``relaxed`` (primitive/leaf nodes only, implies ``~ind``), both
                               constrained to be true only when the node is active (``act``) -- are inert
                               until asserted, so with no commit the behaviour is identical to before.
    """
    assert hasattr(cp.SolverLookup.get(solver), "get_core"), \
        "hierarchical_marco requires a solver that supports assumption variables"

    hard = toplevel_list(hard)
    groups = _initial_cut(root, initial_level)
    if not groups:
        return

    candidates = _collect_candidates(groups)
    n = len(candidates)

    ind = {id(node): v for node, v in zip(candidates, cp.boolvar(shape=(n,)))}
    up = {id(node): v for node, v in zip(candidates, cp.boolvar(shape=(n,)))}
    down = {id(node): v for node, v in zip(candidates, cp.boolvar(shape=(n,)))}
    act = {id(node): v for node, v in zip(candidates, cp.boolvar(shape=(n,)))}
    part = {id(node): v for node, v in zip(candidates, cp.boolvar(shape=(n,)))}
    # commit machinery: `background` forces a node's constraint on (accepted), `relaxed` forces
    # it off (only defined/used for primitive leaf nodes). Both are asserted (as map-solver
    # assumptions) only after a user commit; inert otherwise.
    background = {id(node): v for node, v in zip(candidates, cp.boolvar(shape=(n,)))}
    relaxed = {id(node): v for node, v in zip(candidates, cp.boolvar(shape=(n,)))}
    dmap = {ind[id(node)]: node for node in candidates}

    def get_leaf_nodes(node):
        """
        Return all leaf nodes descending from `node`.
        A leaf node is a node with no children.
        """
        if not node.children:
            return [node]
        leaf_nodes = []
        for child in node.children:
            leaf_nodes.extend(get_leaf_nodes(child))
        return leaf_nodes

    # leaves_of[id(node)] = the (cached) list of leaf nodes under `node`, for every candidate.
    leaves_of = {id(node): get_leaf_nodes(node) for node in candidates}

    if not core_per_round:
        # one persistent, fully-incremental core solver: each LEAF's own constraint is posted
        # exactly once, gated by its own indicator. Higher-level (group) nodes have no
        # constraint of their own in the core solver; a group's indicator is instead *expanded*
        # to the indicators of all its leaves whenever it is used as a core-solver assumption
        # (see `_expand`, used everywhere `s.solve(assumptions=...)` is called below). This
        # keeps the core model fixed at "one implication per leaf" regardless of which level
        # the enumeration is currently working at, eliminating the constraint duplication
        # across ancestor levels that the naive `ind.implies(node.get_grouped_constraint())`
        # encoding causes (each leaf constraint would otherwise be re-posted once per ancestor).
        model = cp.Model(hard)
        for node in candidates:
            if node.children:
                continue
            cons = node.get_grouped_constraint()
            if cons is not None:
                model += ind[id(node)].implies(cons)
        s = cp.SolverLookup.get(solver, model)

        def _expand(group_vars):
            """Translate a list/set of high-level group indicator vars into the flat list of
            their leaves' indicator vars, for use as core-solver assumptions."""
            out = []
            for v in group_vars:
                out.extend(ind[id(leaf)] for leaf in leaves_of[id(dmap[v])])
            return out

        def _core_to_groups(core_vars):
            """Map a core-solver MUS (in leaf indicators) back to the group indicators it
            belongs to under the current round's grouping."""
            return {leaf_to_group[id(dmap[lv])] for lv in core_vars}
    else:
        # core_per_round: no persistent core solver at all here; `s` is (re)built fresh at
        # the start of every round, below, directly over that round's group indicators --
        # exactly like flat `marco`'s lean core model, just reusing `ind` so the assumptions
        # are the same variables the map solver already uses. No leaf-level expansion needed:
        # a "group" assumption IS the assumption (the round's model is thrown away before any
        # refinement could cause duplication), so expansion/un-expansion are both identities.
        s = None

        def _expand(group_vars):
            return list(group_vars)

        def _core_to_groups(core_vars):
            return set(core_vars)

    # one persistent MAP solver: state-dependent linking between indicator/up/down,
    # only enforced for the currently activated/partitioned nodes (via assumptions)
    map_solver_inst = cp.SolverLookup.get(map_solver)

    # a node's "a-links" (a.implies(u==i)/a.implies(d==i)) and "p-links"
    # (p.implies(d==any(child_d))/p.implies(u==all(child_u))) are only ever enforced when its
    # act/part assumption is asserted -- which only happens for nodes that are reached (a frontier
    # group, or a refined ancestor). `lazy_map` posts them on first reach instead of eagerly for
    # the whole tree, keeping the map solver's working set proportional to the reached part.
    linked_a, linked_p = set(), set()       # ids of nodes whose a-/p-links are posted
    node_by_id = {id(node): node for node in candidates}

    def ensure_a_links(node):
        nonlocal map_solver_inst
        if id(node) in linked_a:
            return
        linked_a.add(id(node))
        i, u, d, a = ind[id(node)], up[id(node)], down[id(node)], act[id(node)]
        map_solver_inst += a.implies(u == i)
        map_solver_inst += a.implies(d == i)
        if direct_commit_literals:
            return       # commits are asserted directly as ind / ~ind literals -- no extra vars
        # commit links: background => ind (accepted), relaxed => ~ind (primitives only);
        # both may only be true when the node is active.
        map_solver_inst += background[id(node)].implies(i)
        map_solver_inst += background[id(node)].implies(a)
        if not node.children:
            map_solver_inst += relaxed[id(node)].implies(~i)
            map_solver_inst += relaxed[id(node)].implies(a)

    def ensure_p_links(node):
        nonlocal map_solver_inst
        if id(node) in linked_p:
            return
        child_ids = [id(child) for child in node.children if id(child) in ind]
        if not child_ids:
            return
        linked_p.add(id(node))
        u, d, p = up[id(node)], down[id(node)], part[id(node)]
        map_solver_inst += p.implies(d == cp.any([down[c] for c in child_ids]))
        map_solver_inst += p.implies(u == cp.all([up[c] for c in child_ids]))

    if not lazy_map:
        for node in candidates:
            ensure_a_links(node)
            ensure_p_links(node)

    do_solution_hint = do_solution_hint and hasattr(map_solver_inst, "solution_hint")
    seen = set()
    partitioned = set()  # ids of nodes refined so far; their "p" var is asserted from then on
    committed_background = set()  # ids: active constraints accepted (background) via a commit
    committed_relaxed = set()     # ids: primitive leaf constraints relaxed via a commit
    round_idx = 0

    # Per-STATE conflict cache. A state (= abstraction level) is characterized by the four
    # assumption-variable sets asserted each round: act(groups), part(partitioned),
    # relaxed(committed_relaxed) and background(committed_background). The map solver is never
    # rewound on restore -- only these assumptions change -- and everything found at a state is
    # remembered here so that (a) backtracking to a state RE-SHOWS its known MUS/MCSes to the
    # oracle (tagged ``"cached": True``) instead of relying on re-derivation, and (b) a conflict
    # re-derived at a state it was already shown in is NOT shown again (state-level no-repeat).
    state_cache = {}          # state_key -> list of {"kind": ..., "names": [...]} in found order
    just_restored = False     # True right after a restore: re-show the target state's cache

    # for scripted refinement: resolve a group's full name to its node
    name_to_node = {node.get_full_name(): node for node in candidates}

    while groups:
        round_idx += 1
        assump = [ind[id(node)] for node in groups]
        deletion_order = {a: -len(get_variables(dmap[a].get_grouped_constraint())) for a in assump}
        # commit: the "free" active constraints are those neither relaxed nor background; seeds,
        # MSS growth and the MCS complement all range only over these. Background constraints are
        # forced on in every core solve. Both reduce to the plain behaviour when nothing is
        # committed (free_assump == assump, bg_core_assump == []).
        free_assump = [ind[id(node)] for node in groups
                       if id(node) not in committed_relaxed and id(node) not in committed_background]
        free_set = frozenset(free_assump)
        bg_core_assump = _expand([ind[bid] for bid in committed_background])
        if core_per_round:
            # rebuild a fresh, lean core solver containing only this round's frontier groups
            # (mirrors flat marco's per-round core model), reusing the same `ind` variables.
            round_model = cp.Model(hard)
            for node in groups:
                round_model += ind[id(node)].implies(node.get_grouped_constraint())
            s = cp.SolverLookup.get(solver, round_model)
        else:
            # current groups partition the active leaves; map each leaf back to its group's
            # indicator var so a core-solver MUS (expressed in leaf indicators) can be folded
            # back up to group granularity.
            leaf_to_group = {id(leaf): ind[id(node)] for node in groups for leaf in leaves_of[id(node)]}

        if lazy_map:
            # state-based: ensure links exist for exactly the nodes asserted this round, right
            # before building the assumptions from the same `groups`/`partitioned` sets.
            for node in groups:
                ensure_a_links(node)
            for pid in partitioned:
                ensure_p_links(node_by_id[pid])

        # `background`/`relaxed` are asserted true only for committed nodes; every other *linked*
        # (reached) node has its var pinned false, so the map solver can never spuriously set them
        # and change enumeration. With nothing committed this pins them all false -> identical to
        # the original behaviour. Unlinked nodes' vars are disconnected, so need no pin.
        if direct_commit_literals:
            # assert the commits DIRECTLY as indicator literals: ind (background/pruned stays
            # on) and ~ind (relaxed leaf forced off). act is already asserted for every group
            # (committed nodes remain in `groups`), so the dedicated background/relaxed vars,
            # their linking constraints AND the pins are all unnecessary.
            map_solver_assump = ([part[i] for i in partitioned]
                                 + [act[id(node)] for node in groups]
                                 + [ind[i] for i in committed_background]
                                 + [~ind[i] for i in committed_relaxed])
        else:
            pins = []
            for i in linked_a:
                if i not in committed_background:
                    pins.append(~background[i])
                if not node_by_id[i].children and i not in committed_relaxed:
                    pins.append(~relaxed[i])
            map_solver_assump = ([part[i] for i in partitioned]
                                 + [act[id(node)] for node in groups]
                                 + [background[i] for i in committed_background]
                                 + [relaxed[i] for i in committed_relaxed] + pins)

        # state key = the assumption signature asserted this round (act/part/relaxed/background)
        state_key = (frozenset(id(node) for node in groups), frozenset(partitioned),
                     frozenset(committed_relaxed), frozenset(committed_background))
        state_known = state_cache.setdefault(state_key, [])
        known_keys = {(r["kind"], frozenset(r["names"])) for r in state_known}
        # after a backtrack, re-show this state's previously found conflicts to the oracle
        cached_results = ([dict(r, cached=True) for r in state_known] if just_restored else [])
        just_restored = False

        # Blocking clauses recorded while commits are active are made sound across states by
        # augmenting them with the SEMANTIC vars of the committed nodes (no assumption vars):
        #  * MUS blocks + `up` of the background nodes: the clause becomes the state-
        #    INDEPENDENT truth "core + background is jointly UNSAT" -- within the epoch the
        #    added literals are true (no blocking power lost), outside they are free.
        #  * MCS blocks + `down` of the relaxed nodes: an inner MCS M found under relaxed set R
        #    IS the discovery of the correction set M UNION R of the outer problem (every MCS
        #    over relaxed+open constraints must contain the relaxed ones). Blocking seeds that
        #    avoid M and R therefore only DEDUPLICATES that already-found result -- exactly as
        #    a coarse MCS legitimately blocks its finer twin after refinement. Within the epoch
        #    the added literals are false (down of a relaxed node is off), so blocking power is
        #    unchanged; the only outer discovery the clause suppresses is M UNION R itself.
        _bg_up = [up[i] for i in committed_background]
        _rel_down = [down[i] for i in committed_relaxed]

        map_solver_inst += cp.any([down[id(node)] for node in groups] + _rel_down)
        if do_solution_hint:
            hint = [1] * len(assump)
            map_solver_inst.solution_hint(assump, hint)

        appeared = set()
        round_results = []  # (kind, names) found this round, for decide_step

        if log_events is not None:
            log_events.append({"type": "round", "round": round_idx,
                               "frontier": [node.get_full_name() for node in groups]})

        _t_round_start = time.perf_counter()
        _round_capped = False        # True iff this round stopped early due to round_cap
        while map_solver_inst.solve(assumptions=map_solver_assump):
            if deadline is not None and time.perf_counter() > deadline:
                break   # wall-clock budget hit mid-enumeration -- stop this round early
            if round_cap is not None and len(round_results) >= round_cap:
                # INTERACTIVE cap: don't exhaust the frontier's (possibly huge) conflict set
                # before consulting the oracle -- present at most `round_cap` new results per
                # round. Sound: all blocking clauses persist, so the remaining conflicts are
                # simply found in LATER rounds instead of this one (deferred, never lost).
                _round_capped = True
                break

            # seeds may only contain active constraints that are not relaxed and not background
            seed = [a for a in free_assump if a.value()]
            seed_names = [dmap[a].get_full_name() for a in seed] if log_events is not None else None

            if s.solve(assumptions=_expand(seed) + bg_core_assump) is True:
                # SAT, grow to a full MSS -- only free (not relaxed, not background) constraints
                # can be added, and the MCS is the complement over the free constraints.
                mss = [a for a in free_assump if a.value() or dmap[a].get_grouped_constraint().value()]
                _hit_deadline = False
                for to_add in set(free_assump) - set(mss):
                    if deadline is not None and time.perf_counter() > deadline:
                        _hit_deadline = True           # grow loop can be O(|frontier|) solves --
                        break                          # bound the overshoot to a single solve
                    if s.solve(assumptions=_expand(mss + [to_add]) + bg_core_assump) is True:
                        mss.append(to_add)
                if _hit_deadline:
                    break       # abandon this (partial) grow: add no block, yield nothing
                mcs = [a for a in free_assump if a not in frozenset(mss)]
                if not mcs:
                    # every free constraint is satisfiable alongside the background: the committed
                    # relaxation already repairs the problem, nothing left to enumerate this round.
                    break
                map_solver_inst += cp.any([down[id(dmap[a])] for a in mcs] + _rel_down)

                kind, found = "MCS", mcs
                do_yield = return_mcs

            else:  # UNSAT, shrink to a MUS over the FREE constraints only (hard + background are
                   # the fixed context, always supplied via bg_core_assump, so they are excluded
                   # from the MUS just as they are from seeds / the MCS).
                core = {c for c in _core_to_groups(s.get_core()) if c in free_set}
                if not core:
                    # the conflict lies entirely in hard + background: no free constraint to
                    # blame, nothing free to enumerate this round.
                    break
                _hit_deadline = False
                for c in sorted(core, key=deletion_order.get):
                    if deadline is not None and time.perf_counter() > deadline:
                        _hit_deadline = True           # shrink loop is also O(|core|) solves
                        break
                    if c not in core:
                        continue
                    core.remove(c)
                    if s.solve(assumptions=_expand(core) + bg_core_assump) is True:
                        core.add(c)
                    else:
                        core = {c for c in _core_to_groups(s.get_core()) if c in free_set}
                if _hit_deadline:
                    break       # abandon the partial shrink: add no block, yield nothing

                # augmented with up of the background constraints: within the epoch these are
                # true (ind forced on) so the block is exactly ~all(up[core]); outside it states
                # the state-INDEPENDENT truth "core+background is jointly UNSAT".
                map_solver_inst += ~cp.all([up[id(dmap[a])] for a in core] + _bg_up)

                kind, found = "MUS", core
                do_yield = return_mus

            if do_solution_hint:
                map_solver_inst.solution_hint(assump, hint)

            found_nodes = [dmap[a] for a in found]
            appeared.update(id(node) for node in found_nodes)
            if found_nodes:
                rec = {"kind": kind, "names": [node.get_full_name() for node in found_nodes]}
                rkey = (kind, frozenset(rec["names"]))
                if rkey not in known_keys:
                    # new to THIS state: remember it and show it. A re-derivation of a conflict
                    # already shown at this state (possible after backtracks, since inner-epoch
                    # blocking clauses are augmented to be inert outside their epoch) is blocked
                    # again above but NOT re-shown to the oracle.
                    known_keys.add(rkey)
                    state_known.append(rec)
                    round_results.append(rec)
                    # early-stop hook: let the driver act on a conflict THE MOMENT it is
                    # discovered, without exhausting the (uncapped) frontier first. Used by the
                    # eager-commit variant: stop the round as soon as a committable MCS appears,
                    # so decide_step can commit it immediately instead of after full enumeration.
                    if early_stop is not None and early_stop(rec):
                        _round_capped = True
                        if do_yield:
                            key = (kind, frozenset(id(node) for node in found_nodes))
                            if key not in seen:
                                seen.add(key)
                                yield (kind, [nd.get_grouped_constraint() for nd in found_nodes],
                                       [nd.get_full_name() for nd in found_nodes], round_idx)
                        break

            if log_events is not None:
                log_events.append({"type": "iter", "seed": seed_names, "kind": kind,
                                   "result": [node.get_full_name() for node in found_nodes]})

            if do_yield:
                key = (kind, frozenset(id(node) for node in found_nodes))
                if key not in seen:
                    seen.add(key)
                    yield (kind, [node.get_grouped_constraint() for node in found_nodes],
                           [node.get_full_name() for node in found_nodes], round_idx)

        # map solver returned UNSAT for this round: its enumeration is complete
        if round_timings is not None:
            round_timings.append({"round": round_idx, "seconds": time.perf_counter() - _t_round_start})

        if deadline is not None and time.perf_counter() > deadline:
            # let the driver observe this round's (partial) results once, then stop the sweep so
            # the oracle can record a timeout instead of the enumerator spinning unbounded.
            if decide_step is not None:
                decide_step({
                    "round": round_idx,
                    "frontier": [nd.get_full_name() for nd in groups],
                    "frontier_nodes": list(groups),
                    "free": [nd.get_full_name() for nd in groups
                             if id(nd) not in committed_relaxed and id(nd) not in committed_background],
                    "refinable": [nd.get_full_name() for nd in groups if nd.children],
                    "results": cached_results + round_results,
                    "capped": False,       # deadline stop: the driver must wrap up, not continue
                    "committed_relaxed": [node_by_id[i].get_full_name() for i in committed_relaxed],
                    "committed_background": [node_by_id[i].get_full_name() for i in committed_background],
                    "state": {"groups": list(groups), "partitioned": set(partitioned),
                              "committed_background": set(committed_background),
                              "committed_relaxed": set(committed_relaxed)},
                })
            return

        if decide_step is not None:
            # interactive / programmatic driving: ask for the next refine-or-commit action.
            free_now = [nd for nd in groups
                        if id(nd) not in committed_relaxed and id(nd) not in committed_background]
            action = decide_step({
                "round": round_idx,
                "frontier": [nd.get_full_name() for nd in groups],
                "frontier_nodes": list(groups),
                "free": [nd.get_full_name() for nd in free_now],
                "refinable": [nd.get_full_name() for nd in groups if nd.children],
                "results": cached_results + round_results,
                "capped": _round_capped,   # True: this round was cut short by round_cap, so the
                                           # frontier has MORE conflicts -- 'continue' fetches them
                "committed_relaxed": [node_by_id[i].get_full_name() for i in committed_relaxed],
                "committed_background": [node_by_id[i].get_full_name() for i in committed_background],
                # a snapshot sufficient to restore the map-solver assumptions on backtrack; the
                # persistent blocking clauses are inert on their own (they only block in concert
                # with the assumptions), so restoring these sets is enough -- see `restore` below.
                "state": {"groups": list(groups), "partitioned": set(partitioned),
                          "committed_background": set(committed_background),
                          "committed_relaxed": set(committed_relaxed)},
            })
            if not action or action.get("action") == "stop":
                break
            if action["action"] == "continue":
                # same frontier, next round: the persistent blocks make the inner loop resume
                # exactly where the cap stopped it, yielding the next `round_cap` conflicts.
                continue
            if action["action"] == "restore":
                # backtrack: restore a previously-snapshotted frontier / committed sets. The map
                # solver keeps all its blocking clauses; those referencing now-inactive nodes are
                # inert (their up/down vars are unlinked, hence free), so only the assumptions,
                # rebuilt from these sets next round, change.
                st = action["state"]
                groups = list(st["groups"])
                partitioned.clear(); partitioned.update(st["partitioned"])
                committed_background.clear(); committed_background.update(st["committed_background"])
                committed_relaxed.clear(); committed_relaxed.update(st["committed_relaxed"])
                just_restored = True   # next round: re-show the restored state's cached conflicts
                continue
            if action["action"] == "commit":
                mcs_ids = {id(name_to_node[nm]) for nm in action["mcs"]}
                for node in groups:  # groups == the currently active constraints
                    if id(node) in committed_relaxed or id(node) in committed_background:
                        continue                        # already committed earlier -- leave as-is
                                                        # (a relaxed node must NOT become background)
                    if id(node) in mcs_ids:
                        if not node.children:          # leaf member of the MCS -> relax
                            committed_relaxed.add(id(node))
                        # non-leaf MCS member stays free for further refinement
                    else:                               # active, not in MCS -> background
                        committed_background.add(id(node))
                continue
            if action["action"] == "refine":
                new_groups = list(groups)
                for name in action["constraints"]:
                    node = name_to_node.get(name)
                    if node is None:
                        raise ValueError(f"decide_step refine references unknown group {name!r}")
                    # target_level None -> refine THIS node one level down (per-node), so several
                    # nodes at different depths can each be refined by one level in one action.
                    tl = action.get("target_level")
                    active, parts = activate_descendants_at_level(node, node.level() + 1 if tl is None else tl)
                    new_groups = [g for g in new_groups if g is not node]
                    new_groups.extend(active)
                    for p_node in parts:
                        partitioned.add(id(p_node))
                seen_ids, deduped = set(), []
                for g in new_groups:
                    if id(g) not in seen_ids:
                        seen_ids.add(id(g))
                        deduped.append(g)
                groups = deduped
                continue
            raise ValueError(f"unknown decide_step action {action!r}")

        if scripted_steps is not None:
            # scripted refinement: apply the next predefined step, run len(steps)+1 rounds
            if round_idx - 1 >= len(scripted_steps):
                break
            step = scripted_steps[round_idx - 1]
            new_groups = list(groups)
            for name in step["constraints"]:
                node = name_to_node.get(name)
                if node is None:
                    raise ValueError(f"scripted step references unknown group {name!r}")
                active, parts = activate_descendants_at_level(node, step["target_level"])
                new_groups = [g for g in new_groups if g is not node]
                new_groups.extend(active)
                for p_node in parts:
                    partitioned.add(id(p_node))
            # de-duplicate while preserving order
            seen_ids, deduped = set(), []
            for g in new_groups:
                if id(g) not in seen_ids:
                    seen_ids.add(id(g))
                    deduped.append(g)
            groups = deduped
            continue

        refined = False
        new_groups = []
        for g in groups:
            if g.children and id(g) in appeared:
                refined = True
                partitioned.add(id(g))
                for child in g.children:
                    if id(child) in ind:
                        new_groups.append(child)
            else:
                new_groups.append(g)

        if not refined:
            break
        groups = new_groups


def map_incremental_marco(root, hard=[], solver="ortools", map_solver="ortools",
                           initial_level=1, return_mus=True, return_mcs=True, do_solution_hint=True,
                           round_timings=None, scripted_steps=None, log_events=None, lazy_map=True,
                           decide_step=None):
    """
        Experimental variant of :func:`hierarchical_marco` that keeps only the **map**
        solver persistent/incremental across refinement rounds. The core (SAT) solver is
        rebuilt from scratch every round, containing just that round's frontier groups --
        exactly like flat ``marco``'s core solver -- but reusing the very same group
        indicator variables the map solver uses, so no extra leaf-level bookkeeping is
        needed and no constraint is ever duplicated.

        This isolates the cost/benefit of core-solver persistence from that of map-solver
        persistence: compare against the flat per-round baseline (neither persistent) and
        :func:`hierarchical_marco` (both persistent) to see which one drives the runtime
        difference between hierarchical and flat enumeration on a given instance.

        See :func:`hierarchical_marco` for all parameters; this is a thin wrapper around it
        with ``core_per_round=True``.
    """
    yield from hierarchical_marco(root, hard, solver=solver, map_solver=map_solver,
                                   initial_level=initial_level, return_mus=return_mus,
                                   return_mcs=return_mcs, do_solution_hint=do_solution_hint,
                                   round_timings=round_timings, scripted_steps=scripted_steps,
                                   log_events=log_events, core_per_round=True, lazy_map=lazy_map,
                                   decide_step=decide_step)
