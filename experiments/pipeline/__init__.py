"""
    Reusable experiment-pipeline framework for benchmarking the methods in
    :mod:`cpmpy.tools.explain`.

    See ``experiments/docs/pipeline.md`` for the design. The pieces:

    - :mod:`pipeline.instances`  - the uniform ``Instance`` (flat + hierarchy views)
    - :mod:`pipeline.algorithms` - event-stream wrappers around explain methods
    - :mod:`pipeline.rows`       - canonical row schema + CSV I/O
    - :mod:`pipeline.plotting`   - generic over-time / per-round plots
    - :mod:`pipeline.reporting`  - SUMMARY.md + INDEX.md
    - :mod:`pipeline.runner`     - the ``run_experiment(spec)`` driver
"""

from .instances import (
    Instance,
    transcript_instances,
    synthetic_instances,
    xcsp3_unsat_instances,
    TRANSCRIPTS,
    BENCHMARKS,
)
from .algorithms import (
    umus_events,
    marco_events,
    hierarchical_marco_events,
    baseline_marco_events,
)
from .runner import ExperimentSpec, Series, run_experiment

__all__ = [
    "Instance", "transcript_instances", "synthetic_instances", "xcsp3_unsat_instances",
    "TRANSCRIPTS", "BENCHMARKS",
    "umus_events", "marco_events", "hierarchical_marco_events", "baseline_marco_events",
    "ExperimentSpec", "Series", "run_experiment",
]
