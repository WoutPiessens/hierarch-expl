"""
    The uniform :class:`Instance` abstraction and the provider functions that build
    instances from the three benchmark sources.

    An :class:`Instance` always exposes a **flat** view ``(soft, hard)`` and, when the
    source is hierarchical, also a **hierarchy** view ``(root, hard)``. Because
    ``ConstraintNode.all_constraints()`` flattens any hierarchy, a hierarchical
    instance can feed *both* hierarchical experiments (via :meth:`Instance.hierarchy`)
    and flat ones (via :meth:`Instance.flat`). This is what lets the transcript
    benchmarks and the (flat) XCSP3 benchmarks flow through the same pipeline.

    Sources (each wraps an existing module, unchanged):
    - ``transcript_instances``  -> :func:`hierarchy_io.load_hierarchy`
    - ``synthetic_instances``   -> :func:`large_unsat_benchmark.build_graph_coloring_instance`
    - ``xcsp3_unsat_instances`` -> :func:`xcsp3_unsat_finder.get_unsat_instances`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from hierarchy_io import load_hierarchy
from large_unsat_benchmark import build_graph_coloring_instance
from xcsp3_unsat_finder import get_unsat_instances

from cpmpy.tools.xcsp3 import read_xcsp3
from cpmpy.transformations.normalize import toplevel_list


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    """A benchmark instance with a flat view and, optionally, a hierarchy view."""
    name: str
    source: str
    flat_loader: Callable[[], tuple]
    hier_loader: Optional[Callable[[], tuple]] = None
    meta: dict = field(default_factory=dict)

    def flat(self):
        """Return ``(soft, hard)`` lists of constraints."""
        return self.flat_loader()

    def has_hierarchy(self):
        return self.hier_loader is not None

    def hierarchy(self):
        """Return ``(root: ConstraintNode, hard)``; only for hierarchical sources."""
        if self.hier_loader is None:
            raise ValueError(f"instance {self.name!r} has no hierarchy view")
        return self.hier_loader()

    def describe(self):
        return self.source


# ---------------------------------------------------------------------------
# Catalogues
# ---------------------------------------------------------------------------

# Hierarchical transcript benchmarks (exported from defense-rostering).
# transcript_1..3 are curated interactive-driver snapshots (planned defenses pinned via
# `already-allocated` + a few defenses to plan). transcript_4..7 are auto-generated from
# defense-rostering's `input_data/instances_unsat/*` "plan all defenses in physical rooms"
# (planned_defenses=[], defenses_to_plan=all): the explanation model has no virtual
# "online" room, so a full physical schedule is UNSAT and is explained by the soft
# person/room-unavailability constraints. See data/hierarchies/<t>_hierarchy/_manifest.json
# for each one's source instance, defense split, and refinement scenario.
TRANSCRIPTS = {
    "transcript_1": {"hierarchy_dir": "transcript_1_hierarchy", "initial_level": 1},
    "transcript_2": {"hierarchy_dir": "transcript_2_hierarchy", "initial_level": 1},
    "transcript_3": {"hierarchy_dir": "transcript_3_hierarchy", "initial_level": 1},
    "transcript_4": {"hierarchy_dir": "transcript_4_hierarchy", "initial_level": 1},
    "transcript_5": {"hierarchy_dir": "transcript_5_hierarchy", "initial_level": 1},
    "transcript_6": {"hierarchy_dir": "transcript_6_hierarchy", "initial_level": 1},
    "transcript_7": {"hierarchy_dir": "transcript_7_hierarchy", "initial_level": 1},
    # anonymized copies of transcript_1..3 (person names -> pseudo-random ids); same pickle,
    # only the person node labels differ. See experiments/anonymize_transcripts.py and the
    # mapping in data/hierarchies/anonymization_mapping.json.
    "transcript_1_anon": {"hierarchy_dir": "transcript_1_anon_hierarchy", "initial_level": 1},
    "transcript_2_anon": {"hierarchy_dir": "transcript_2_anon_hierarchy", "initial_level": 1},
    "transcript_3_anon": {"hierarchy_dir": "transcript_3_anon_hierarchy", "initial_level": 1},
}

# Synthetic graph-colouring benchmarks.
BENCHMARKS = {
    "small": dict(n_nodes=16, n_colors=3, n_cliques=2, clique_size=4,
                  extra_edge_prob=0.08, seed=1),
    "medium": dict(n_nodes=24, n_colors=3, n_cliques=3, clique_size=4,
                   extra_edge_prob=0.06, seed=1),
}


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def transcript_instances(names):
    """Build hierarchical transcript instances (lazily loaded, cached per instance)."""
    out = []
    for name in names:
        cfg = TRANSCRIPTS[name]
        cache = {}

        def hier(cfg=cfg, cache=cache):
            if "loaded" not in cache:
                cache["loaded"] = load_hierarchy(cfg["hierarchy_dir"])
            return cache["loaded"]

        def flat(hier=hier):
            root, hard = hier()
            return root.all_constraints(), hard

        out.append(Instance(
            name=name,
            source=f"defense-rostering hierarchy ({cfg['hierarchy_dir']})",
            flat_loader=flat, hier_loader=hier,
            meta=dict(cfg),
        ))
    return out


def synthetic_instances(keys):
    """Build synthetic graph-colouring instances."""
    out = []
    for key in keys:
        params = BENCHMARKS[key]

        def flat(params=params):
            return build_graph_coloring_instance(**params)

        out.append(Instance(
            name=key,
            source=f"synthetic graph-colouring ({params})",
            flat_loader=flat,
            meta=dict(params),
        ))
    return out


def xcsp3_unsat_instances(min_constraints=0, max_constraints=200, keys=None):
    """
        Build flat instances from the cached UNSAT XCSP3 instances.

        If `keys` is None, auto-select all cached UNSAT instances within
        ``[min_constraints, max_constraints]``; otherwise load exactly those keys
        (even if outside the size range).
    """
    entries = {e["key"]: e for e in get_unsat_instances(min_constraints, max_constraints)}
    if keys is None:
        keys = sorted(entries.keys())
    else:
        full = {e["key"]: e for e in get_unsat_instances(max_constraints=None)}
        for k in keys:
            entries[k] = full[k]

    out = []
    for key in keys:
        entry = entries[key]

        def flat(entry=entry):
            model = read_xcsp3(entry["path"])
            return toplevel_list(model.constraints, merge_and=False), []

        out.append(Instance(
            name=key,
            source=f"XCSP3 UNSAT ({entry['num_constraints']} constraints, {entry['path']})",
            flat_loader=flat,
            meta=dict(entry),
        ))
    return out
