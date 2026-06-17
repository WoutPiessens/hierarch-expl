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
    TRANSCRIPTS,
)
from scripted_scenarios import SCENARIOS

NAME = "hierarchical_runtime"


def add_args(parser):
    parser.add_argument("--instances", nargs="+", choices=sorted(TRANSCRIPTS.keys()),
                        default=sorted(TRANSCRIPTS.keys()),
                        help="transcript instances to run")
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
            Series("baseline", baseline_marco_events, "#54A24B"),
            Series("incremental", hierarchical_marco_events, "#F58518"),
        ],
        settings={
            "solver": args.solver,
            "map_solver": args.map_solver,
            "initial_level": args.initial_level,
            "scenarios": SCENARIOS if args.refinement == "scripted" else None,
        },
        algorithms_doc=[
            ("baseline", "flat `marco`, rebuilt from scratch (new core + MAP solver) "
                         "for every refinement round"),
            ("incremental", "a single call to `hierarchical_marco`, which builds both "
                            "solvers once and reuses them across all refinement rounds"),
        ],
        metric_label="enum time (s)",
        plot_ylabel="enumeration time per round (s)",
        plot_title="{instance}: per-round enumeration time",
        has_rounds=True,
        metric_is_round_time=True,
    )
