"""
    Experiment: relevant constraints discovered over time, umus vs marco.

    On large-scale flat UNSAT instances (synthetic graph-colouring and/or cached
    XCSP3 UNSAT instances), compares:
    - **umus**: a single call to `umus`, which incrementally grows a set CUMU of
      constraints known to be part of some MUS (one event per growth of CUMU).
    - **marco**: repeatedly calling flat `marco`; the metric is the running union of
      constraints seen in any yielded MUS/MCS.
"""

from __future__ import annotations

from pipeline import (
    ExperimentSpec, Series,
    synthetic_instances, xcsp3_unsat_instances, umus_events, marco_events,
    BENCHMARKS,
)

NAME = "relevant_constraints"


def add_args(parser):
    parser.add_argument("--instances", nargs="*", choices=sorted(BENCHMARKS.keys()),
                        default=sorted(BENCHMARKS.keys()),
                        help="synthetic graph-colouring instances to run")
    parser.add_argument("--xcsp3-instances", nargs="*", default=None,
                        help="explicit cached UNSAT XCSP3 instance keys to add; if "
                             "omitted, all cached UNSAT instances within the size range "
                             "are included automatically")
    parser.add_argument("--xcsp3-min-constraints", type=int, default=0)
    parser.add_argument("--xcsp3-max-constraints", type=int, default=200)
    parser.add_argument("--solver", default="ortools")
    parser.add_argument("--map-solver", default="ortools")


def build_spec(args):
    instances = synthetic_instances(args.instances)
    instances += xcsp3_unsat_instances(
        min_constraints=args.xcsp3_min_constraints,
        max_constraints=args.xcsp3_max_constraints,
        keys=args.xcsp3_instances,
    )
    return ExperimentSpec(
        name=NAME,
        title="Relevant constraints over time: umus vs marco",
        description=(
            "Tracks how many \"relevant\" constraints (constraints appearing in at least "
            "one MUS/MCS) are discovered over time on large-scale flat UNSAT instances "
            "(synthetic graph-colouring and/or real XCSP3 instances)."
        ),
        instances=instances,
        series=[
            Series("umus", umus_events, "#4C78A8"),
            Series("marco", marco_events, "#F58518"),
        ],
        settings={"solver": args.solver, "map_solver": args.map_solver},
        algorithms_doc=[
            ("umus", "single call to `umus`, which incrementally grows a set CUMU of "
                     "constraints known to be part of some MUS; each growth of CUMU is "
                     "one event"),
            ("marco", "repeatedly calling flat `marco`, which yields one MUS or MCS at a "
                      "time; the metric is the size of the running union of constraints "
                      "seen in any yielded MUS/MCS so far"),
        ],
        metric_label="relevant",
        plot_ylabel="# relevant constraints",
        plot_title="{instance}: relevant constraints discovered over time",
        has_rounds=False,
    )
