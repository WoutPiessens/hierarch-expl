"""
    MARCO-style MUS/MCS enumeration over a hierarchy of :class:`ConstraintNode` groups.

    =================
    List of functions
    =================

    .. autosummary::
        :nosignatures:

        hierarchical_marco
"""

import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.normalize import toplevel_list


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
                        initial_level=1, return_mus=True, return_mcs=True, do_solution_hint=True):
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

        Yields `(kind, constraints, names)` tuples, where `kind` is `"MUS"` or `"MCS"`,
        `constraints` is the list of grouped constraints (one `cp.all(...)` per involved node),
        and `names` is the list of `ConstraintNode.get_full_name()` for those nodes.

        :param: root: root :class:`ConstraintNode` of the constraint hierarchy
        :param: hard: hard constraints, optional, list of expressions
        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
        :param: map_solver: the map solver to use, ideally incremental such as "gurobi", "pysat" or "exact"
        :param: initial_level: depth at which the hierarchy is initially cut into groups
        :param: return_mus: whether the algorithm should return MUSes
        :param: return_mcs: whether the algorithm should return MCSes
        :param: do_solution_hint: when true, will favor large seeds generated by the map-solver, and hence more likely
                                   to return MUSes
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
    dmap = {ind[id(node)]: node for node in candidates}

    # one persistent core solver: "indicator" implies the node's grouped constraint
    model = cp.Model(hard)
    for node in candidates:
        model += ind[id(node)].implies(node.get_grouped_constraint())
    s = cp.SolverLookup.get(solver, model)

    # one persistent MAP solver: state-dependent linking between indicator/up/down,
    # only enforced for the currently activated/partitioned nodes (via assumptions)
    map_solver_inst = cp.SolverLookup.get(map_solver)
    for node in candidates:
        i, u, d, a, p = ind[id(node)], up[id(node)], down[id(node)], act[id(node)], part[id(node)]
        map_solver_inst += a.implies(u == i)
        map_solver_inst += a.implies(d == i)
        child_ids = [id(child) for child in node.children if id(child) in ind]
        if child_ids:
            map_solver_inst += p.implies(d == cp.any([down[c] for c in child_ids]))
            map_solver_inst += p.implies(u == cp.all([up[c] for c in child_ids]))

    do_solution_hint = do_solution_hint and hasattr(map_solver_inst, "solution_hint")
    seen = set()
    partitioned = set()  # ids of nodes refined so far; their "p" var is asserted from then on

    while groups:
        assump = [ind[id(node)] for node in groups]
        deletion_order = {a: -len(get_variables(dmap[a].get_grouped_constraint())) for a in assump}

        map_solver_assump = [part[i] for i in partitioned] + [act[id(node)] for node in groups]
        map_solver_inst += cp.any([down[id(node)] for node in groups])
        if do_solution_hint:
            hint = [1] * len(assump)
            map_solver_inst.solution_hint(assump, hint)

        appeared = set()

        while map_solver_inst.solve(assumptions=map_solver_assump):

            seed = [a for a in assump if a.value()]

            if s.solve(assumptions=seed) is True:
                # SAT, grow to a full MSS
                mss = [a for a in assump if a.value() or dmap[a].get_grouped_constraint().value()]
                for to_add in set(assump) - set(mss):
                    if s.solve(assumptions=mss + [to_add]) is True:
                        mss.append(to_add)
                mcs = [a for a in assump if a not in frozenset(mss)]
                map_solver_inst += cp.any([down[id(dmap[a])] for a in mcs])

                kind, found = "MCS", mcs
                do_yield = return_mcs

            else:  # UNSAT, shrink to a MUS
                core = set(s.get_core())
                for c in sorted(core, key=deletion_order.get):
                    if c not in core:
                        continue
                    core.remove(c)
                    if s.solve(assumptions=list(core)) is True:
                        core.add(c)
                    else:
                        core = set(s.get_core())

                map_solver_inst += ~cp.all([up[id(dmap[a])] for a in core])

                kind, found = "MUS", core
                do_yield = return_mus

            if do_solution_hint:
                map_solver_inst.solution_hint(assump, hint)

            found_nodes = [dmap[a] for a in found]
            appeared.update(id(node) for node in found_nodes)

            if do_yield:
                key = (kind, frozenset(id(node) for node in found_nodes))
                if key not in seen:
                    seen.add(key)
                    yield kind, [node.get_grouped_constraint() for node in found_nodes], [node.get_full_name() for node in found_nodes]

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
