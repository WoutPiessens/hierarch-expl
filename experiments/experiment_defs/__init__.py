"""
    Experiment definitions. Each module here describes one experiment as a thin
    layer over :mod:`pipeline`: it exposes ``NAME``, ``add_args(parser)`` and
    ``build_spec(args)``, and is registered in the :data:`REGISTRY` below so
    ``run.py`` can dispatch to it by name.

    To add an experiment: create ``experiment_defs/<name>.py`` with those three
    symbols and add it to ``REGISTRY``. See ``experiments/docs/experiments.md``.
"""

from . import hierarchical_runtime, relevant_constraints

REGISTRY = {
    hierarchical_runtime.NAME: hierarchical_runtime,
    relevant_constraints.NAME: relevant_constraints,
}
