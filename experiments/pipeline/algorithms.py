"""
    Event-stream wrappers around the :mod:`cpmpy.tools.explain` methods.

    Each wrapper has the same shape::

        def <name>_events(instance, settings) -> list[event dict]

    where every event dict has ``elapsed_seconds``, ``kind`` and ``metric`` (and
    optionally ``round`` and ``detail``). A *series* in an experiment is just one of
    these wrappers; the runner times nothing itself, it only consumes the events.

    ``settings`` is a plain dict carrying run-wide knobs: ``solver``, ``map_solver``,
    ``initial_level``, ``return_mus``, ``return_mcs``. Each wrapper reads what it needs.

    The ``cpmpy.tools.explain`` functions are used through their public APIs only —
    no changes to ``umus.py`` or ``hierarchical_marco.py`` are required.
"""

from __future__ import annotations

import time

from cpmpy.tools.explain import marco, umus, hierarchical_marco, map_incremental_marco
from cpmpy.tools.explain.hierarchical_marco import _initial_cut
from cpmpy.tools.explain.hierarchical import activate_descendants_at_level


def _refinable_children(node):
    return [c for c in node.children if c.get_grouped_constraint() is not None]


def _scripted_steps(instance, settings):
    """The scripted refinement steps for this instance, or None for auto-refinement."""
    scenarios = settings.get("scenarios")
    if not scenarios:
        return None
    return scenarios.get(instance.name)


# ---------------------------------------------------------------------------
# Flat algorithms (operate on instance.flat())
# ---------------------------------------------------------------------------

def umus_events(instance, settings):
    """Single call to `umus`; one event per growth of CUMU (via the callback)."""
    soft, hard = instance.flat()
    events = []
    t0 = time.perf_counter()

    def callback(n):
        events.append({
            "elapsed_seconds": time.perf_counter() - t0,
            "kind": "CUMU",
            "metric": n,
        })

    umus(soft, hard, solver=settings["solver"], callback=callback)
    return events


def marco_events(instance, settings):
    """Flat `marco`; metric = running union of constraint indices seen in any MUS/MCS."""
    soft, hard = instance.flat()
    index_of = {id(c): i for i, c in enumerate(soft)}
    events = []
    seen = set()
    t0 = time.perf_counter()

    for kind, found in marco(soft, hard, solver=settings["solver"],
                             map_solver=settings["map_solver"],
                             return_mus=settings.get("return_mus", True),
                             return_mcs=settings.get("return_mcs", True)):
        seen.update(index_of[id(c)] for c in found)
        events.append({
            "elapsed_seconds": time.perf_counter() - t0,
            "kind": kind,
            "metric": len(seen),
        })
    return events


# ---------------------------------------------------------------------------
# Hierarchical algorithms (operate on instance.hierarchy())
# ---------------------------------------------------------------------------

def _initial_level(instance, settings):
    return instance.meta.get("initial_level", settings.get("initial_level", 1))


def hierarchical_marco_events(instance, settings):
    """
        One call to `hierarchical_marco`. One event per refinement round, whose
        ``metric`` is that round's *enumeration time* in seconds: the wall-clock from
        the first map-solver call of the round until the map solver returns UNSAT
        (measured inside `hierarchical_marco` via its ``round_timings`` hook).
        ``elapsed_seconds`` is the cumulative enumeration time over the rounds so far.
    """
    return _hier_events(hierarchical_marco, instance, settings)


def map_incremental_marco_events(instance, settings):
    """
        One call to `map_incremental_marco` (only the MAP solver is persistent across rounds;
        the core solver is rebuilt per round like the flat baseline). Same per-round
        enumeration-time metric as :func:`hierarchical_marco_events`.
    """
    return _hier_events(map_incremental_marco, instance, settings)


def _hier_events(fn, instance, settings):
    """Shared driver for the single-call hierarchical enumerators (`fn` yields the same
    tuples and accepts `round_timings`/`scripted_steps`/`log_events`/`lazy_map`)."""
    root, hard = instance.hierarchy()
    round_timings = []
    for _ in fn(
            root, hard, solver=settings["solver"], map_solver=settings["map_solver"],
            initial_level=_initial_level(instance, settings),
            return_mus=settings.get("return_mus", True),
            return_mcs=settings.get("return_mcs", True),
            round_timings=round_timings,
            scripted_steps=_scripted_steps(instance, settings),
            log_events=settings.get("_log_sink"),
            lazy_map=settings.get("lazy_map", False)):
        pass

    events = []
    cumulative = 0.0
    for rt in round_timings:
        cumulative += rt["seconds"]
        events.append({
            "elapsed_seconds": cumulative,
            "kind": "ENUM",
            "metric": rt["seconds"],
            "round": rt["round"],
        })
    return events


def _marco_round_event(groups, hard, settings, round_idx, cumulative):
    """Run flat `marco` over `groups`, returning (event, appeared, round_seconds)."""
    soft = [g.get_grouped_constraint() for g in groups]
    group_of = {id(c): g for c, g in zip(soft, groups)}
    log = settings.get("_log_sink")
    if log is not None:
        log.append({"type": "round", "round": round_idx,
                    "frontier": [g.get_full_name() for g in groups]})
    name_of = {id(c): g.get_full_name() for c, g in zip(soft, groups)} if log is not None else None
    appeared = set()
    enum_timing = []
    for kind, found in marco(soft, hard, solver=settings["solver"],
                             map_solver=settings["map_solver"],
                             return_mus=settings.get("return_mus", True),
                             return_mcs=settings.get("return_mcs", True),
                             enum_timing=enum_timing, log_events=log, name_of=name_of):
        appeared.update(id(group_of[id(c)]) for c in found)
    round_seconds = enum_timing[0] if enum_timing else 0.0
    event = {
        "elapsed_seconds": cumulative + round_seconds,
        "kind": "ENUM",
        "metric": round_seconds,
        "round": round_idx,
    }
    return event, appeared, round_seconds


def _apply_scripted_to_frontier(groups, step):
    """Refine the named groups of `step` to its target level (scripted baseline)."""
    name_to_node = {g.get_full_name(): g for g in groups}
    new_groups = list(groups)
    for name in step["constraints"]:
        node = name_to_node.get(name)
        if node is None:
            raise ValueError(f"scripted step references unknown group {name!r}")
        active, _ = activate_descendants_at_level(node, step["target_level"])
        new_groups = [g for g in new_groups if g is not node]
        new_groups.extend(active)
    seen_ids, deduped = set(), []
    for g in new_groups:
        if id(g) not in seen_ids:
            seen_ids.add(id(g))
            deduped.append(g)
    return deduped


def baseline_marco_events(instance, settings):
    """
        Flat `marco` rebuilt from scratch for every refinement round (the baseline the
        single-call `hierarchical_marco` is compared against). One event per round,
        whose ``metric`` is that round's *enumeration time* in seconds: the wall-clock
        of that round's whole `marco` enumeration loop (first map-solver call until the
        map solver returns UNSAT), via `marco`'s ``enum_timing`` hook.

        Refinement follows the scripted steps for this instance if any (matching
        defense-rostering: ``len(steps) + 1`` rounds, named groups refined to a target
        level); otherwise it auto-refines every group that appeared in a MUS/MCS.
    """
    root, hard = instance.hierarchy()
    groups = _initial_cut(root, _initial_level(instance, settings))
    steps = _scripted_steps(instance, settings)
    events = []
    cumulative = 0.0

    if steps is not None:
        for it in range(len(steps) + 1):
            event, _appeared, secs = _marco_round_event(groups, hard, settings, it + 1, cumulative)
            cumulative += secs
            events.append(event)
            if it < len(steps):
                groups = _apply_scripted_to_frontier(groups, steps[it])
        return events

    round_idx = 0
    while groups:
        round_idx += 1
        event, appeared, secs = _marco_round_event(groups, hard, settings, round_idx, cumulative)
        cumulative += secs
        events.append(event)

        refined = False
        new_groups = []
        for g in groups:
            children = _refinable_children(g)
            if children and id(g) in appeared:
                refined = True
                new_groups.extend(children)
            else:
                new_groups.append(g)

        if not refined:
            break
        groups = new_groups

    return events
