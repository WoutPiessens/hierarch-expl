"""
    UMUS: union of all minimal unsatisfiable subsets ("relevant" constraints)

    =================
    List of functions
    =================

    .. autosummary::
        :nosignatures:

        umus
"""
import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables

from .utils import make_assump_model


def _umus_recurse(CUMU, F, F0, nec, forced, solver, dmap, callback=None):
    """
        Recursive helper for `umus`, operating purely on sets of assumption variables.

        :param: CUMU: running union of constraints (assumption vars) known to be part of some MUS
        :param: F: remaining candidate assumption vars to consider
        :param: F0: the full set of assumption vars (soft constraints)
        :param: nec: assumption vars proven necessary (always part of the conflict) in this branch
        :param: forced: assumption vars already processed in this branch
        :param: solver: solver instance with hard + implied-soft constraints already loaded
        :param: dmap: mapping from assumption var to original constraint
        :param: callback: optional, called with `len(CUMU)` whenever CUMU grows
    """
    if not F:
        return False, CUMU
    if F <= CUMU or not (forced <= F):
        return False, CUMU

    # find a MUS M of F \ nec, with nec forced true as background
    assert not solver.solve(assumptions=list(F | nec)), "umus: subproblem must be UNSAT"
    candidates = F - nec
    core = set(solver.get_core()) & candidates
    for a in set(core):
        if a not in core:
            continue
        core.remove(a)
        if solver.solve(assumptions=list(core | nec)) is True:
            core.add(a)
        else:  # still UNSAT, refine using new core
            core = set(solver.get_core()) & candidates
    M = core

    CUMU = CUMU | M
    if callback is not None:
        callback(len(CUMU))

    if CUMU == F0:
        return True, CUMU
    if M == F or F <= CUMU:
        return False, CUMU

    for c in M - forced:
        if solver.solve(assumptions=list((F - {c}) | nec)) is True:
            # F\{c} (+nec) is SAT again, so c is necessary for this conflict
            nec = nec | {c}
        else:
            done, CUMU = _umus_recurse(CUMU, F - {c}, F0, nec, forced, solver, dmap, callback)
            if done:
                return True, CUMU
            if F <= CUMU:
                return False, CUMU
        forced = forced | {c}
        if not solver.solve(assumptions=list(forced)):
            return False, CUMU

    return False, CUMU


def umus(soft, hard=[], solver="ortools", callback=None):
    """
        Compute CUMU, the union of all soft constraints that participate in at least one MUS of (soft, hard).

        Repeatedly extracts a MUS, then for each of its constraints checks whether it is necessary
        for the current conflict; non-necessary constraints are recursively removed and the search
        continues on the remainder. The union of all MUSes found this way (CUMU) is returned.

        This algorithm originates from the following paper:


        Assumption-based implementation for solvers that support s.solve(assumptions=...) and s.get_core().

        :param: soft: soft constraints, list of expressions
        :param: hard: hard constraints, optional, list of expressions
        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
        :param: callback: optional, called with `len(CUMU)` every time CUMU grows (e.g. to log progress over time)
    """

    assert hasattr(cp.SolverLookup.get(solver), "get_core"), f"umus requires a solver that supports assumption variables, got {solver}"

    (m, soft, assump) = make_assump_model(soft, hard=hard)
    s = cp.SolverLookup.get(solver, m)
    dmap = dict(zip(assump, soft))

    F0 = set(assump)
    assert not s.solve(assumptions=list(F0)), "umus: model must be UNSAT"

    # find an initial MUS (deletion-based shrinking of the solver's unsat core)
    core = set(s.get_core())
    for a in sorted(core, key=lambda a: -len(get_variables(dmap[a]))):
        if a not in core:
            continue
        core.remove(a)
        if s.solve(assumptions=list(core)) is True:
            core.add(a)
        else:
            core = set(s.get_core())

    CUMU = set(core)
    if callback is not None:
        callback(len(CUMU))
    if CUMU != F0:
        _, CUMU = _umus_recurse(CUMU, F0, F0, set(), set(), s, dmap, callback)

    return [dmap[a] for a in CUMU]
