"""
    Experiment: hierarchical MUS/MCS enumeration, baseline vs incremental.

    Compares, on the hierarchical defense-rostering transcript instances:
    - **baseline**: flat `marco` rebuilt from scratch for every refinement round.
    - **incremental**: a single call to `hierarchical_marco` that builds both solvers
      once and reuses them across rounds.

    Metric: per-round **enumeration time** (seconds) — for each refinement round, the
    wall-clock time from the first map-solver call of that round until the map solver
    returns UNSAT (i.e. the round's full MUS/MCS enumeration). The output includes a
    per-round timing table (round x series) per transcript.
"""

from __future__ import annotations

from pipeline import (
    ExperimentSpec, Series,
    transcript_instances, baseline_marco_events, hierarchical_marco_events,
    map_incremental_marco_events,
    TRANSCRIPTS,
)
from scripted_scenarios import SCENARIOS

NAME = "hierarchical_runtime"

# the available algorithm series (name -> (event wrapper, plot colour, doc))
ALGORITHMS = {
    "baseline": (baseline_marco_events, "#54A24B",
                 "flat `marco`, rebuilt from scratch (new core + MAP solver) for every "
                 "refinement round"),
    "incremental": (hierarchical_marco_events, "#F58518",
                    "a single call to `hierarchical_marco`: both the core (persistent, "
                    "leaf-level) and MAP solver are built once and reused across rounds"),
    "map_incremental": (map_incremental_marco_events, "#4C78A8",
                        "`map_incremental_marco`: only the MAP solver is persistent; the core "
                        "solver is rebuilt per round exactly like the baseline (isolates the "
                        "effect of map-solver persistence alone)"),
}


def add_args(parser):
    parser.add_argument("--instances", nargs="+", choices=sorted(TRANSCRIPTS.keys()),
                        default=sorted(TRANSCRIPTS.keys()),
                        help="transcript instances to run")
    parser.add_argument("--algorithms", nargs="+", choices=list(ALGORITHMS.keys()),
                        default=["baseline", "incremental"],
                        help="which algorithm series to compare (default: baseline incremental)")
    import argparse as _argparse
    parser.add_argument("--lazy-map", action=_argparse.BooleanOptionalAction, default=True,
                        help="for the incremental/map_incremental series, post the map solver's "
                             "structural linking constraints lazily (only for reached nodes); "
                             "output-identical, shrinks the persistent map solver (default: on; "
                             "pass --no-lazy-map for the original eager behaviour)")
    parser.add_argument("--solver", default="exact")
    parser.add_argument("--map-solver", default="exact")
    parser.add_argument("--initial-level", type=int, default=1)
    parser.add_argument("--refinement", choices=["scripted", "auto"], default="scripted",
                        help="'scripted' replays the per-transcript refinement scenario "
                             "(matches defense-rostering: len(steps)+1 iterations, level-skipping); "
                             "'auto' refines every group that appeared in a MUS/MCS, one level at a time")


def build_spec(args):
    return ExperimentSpec(
        name=NAME,
        title="Hierarchical MUS/MCS enumeration: baseline vs incremental",
        description=(
            "Compares two ways of enumerating MUSes/MCSes over the hierarchical "
            "constraint groups of the defense-rostering transcript benchmark instances. "
            "For each refinement round, the measured quantity is the round's "
            "**enumeration time**: the wall-clock from the first map-solver call of that "
            "round until the map solver returns UNSAT (the round's complete MUS/MCS "
            "enumeration). A \"round\" is one pass over the current set of constraint "
            "groups, before the next refinement. Refinement defaults to the per-transcript "
            "**scripted scenario** (ported from defense-rostering: a fixed sequence of "
            "named-group refinements to target levels, which may skip intermediate levels, "
            "giving len(steps)+1 iterations); pass --refinement auto for appearance-driven, "
            "one-level-at-a-time refinement. See ROUND_TIMES.md for the per-round breakdown."
        ),
        instances=transcript_instances(args.instances),
        series=[
            Series(name, ALGORITHMS[name][0], ALGORITHMS[name][1])
            for name in args.algorithms
        ],
        settings={
            "solver": args.solver,
            "map_solver": args.map_solver,
            "initial_level": args.initial_level,
            "scenarios": SCENARIOS if args.refinement == "scripted" else None,
            "lazy_map": args.lazy_map,
        },
        algorithms_doc=[
            (name, ALGORITHMS[name][2]) for name in args.algorithms
        ],
        metric_label="enum time (s)",
        plot_ylabel="enumeration time per round (s)",
        plot_title="{instance}: per-round enumeration time",
        has_rounds=True,
        metric_is_round_time=True,
    )
