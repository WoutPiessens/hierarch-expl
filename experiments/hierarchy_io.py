"""
    Load a benchmark instance exported by defense-rostering's ``export_hierarchy.py``
    (a ``constraints.pkl`` + ``hierarchy.json`` pair) into a :class:`ConstraintNode`
    tree plus a list of hard constraints, ready to pass to :func:`hierarchical_marco`,
    :func:`marco` or :func:`umus`.

    Resolution order for a (relative) hierarchy name, so the repo can run standalone on
    an external machine without a defense-rostering checkout:

      1. ``experiments/data/hierarchies/<name>/``         (self-contained, committed)
      2. ``experiments/data/hierarchies/<name>_hierarchy/``
      3. ``$DEFENSE_ROSTERING_DIR/input_data/<name>/``    (sibling repo fallback)

    Use ``import_hierarchies.py`` to populate ``data/hierarchies/`` from defense-rostering.
"""

import json
import os
import pickle
import re
from pathlib import Path

from cpmpy.tools.explain import constraint_node_from_dict
from cpmpy.expressions import variables as _vars
from cpmpy.transformations.get_variables import get_variables


def _advance_var_counters(constraints):
    """
        Advance cpmpy's global auto-naming counters past the unpickled variables.

        cpmpy auto-names variables ``BV<n>`` / ``IV<n>`` from a process-global counter and
        identifies variables *by name*. Unpickling restores the variables' names but NOT the
        counter, which resets to 0 in a fresh process. Without this bump, freshly-created
        variables (e.g. the assumption/indicator boolvars built by ``marco`` /
        ``hierarchical_marco`` / ``umus``) would reuse names like ``BV0`` and silently
        **alias** real model variables, corrupting every solve. See the investigation in the
        project notes: this manifested as ``already-allocated`` appearing UNSAT-on-its-own.
    """
    max_bv = _vars._BoolVarImpl.counter
    max_iv = _vars._IntVarImpl.counter
    for v in get_variables(constraints):
        m = re.fullmatch(re.escape(_vars._BV_PREFIX) + r"(\d+)", v.name or "")
        if m:
            max_bv = max(max_bv, int(m.group(1)) + 1)
            continue
        m = re.fullmatch(re.escape(_vars._IV_PREFIX) + r"(\d+)", v.name or "")
        if m:
            max_iv = max(max_iv, int(m.group(1)) + 1)
    _vars._BoolVarImpl.counter = max_bv
    _vars._IntVarImpl.counter = max_iv

# In-repo, self-contained copies of the exported hierarchies (preferred).
LOCAL_HIERARCHY_DIR = Path(__file__).resolve().parent / "data" / "hierarchies"

# defense-rostering checkout, used only as a fallback. Overridable via the
# DEFENSE_ROSTERING_DIR env var; defaults to a sibling of this repo's parent directory:
#   PycharmProjects/
#     defense-rostering/
#     hierarch-expl/hierarch-expl/experiments/hierarchy_io.py
DEFENSE_ROSTERING_DIR = Path(
    os.environ.get("DEFENSE_ROSTERING_DIR",
                   Path(__file__).resolve().parents[3] / "defense-rostering"))


def _resolve_hierarchy_dir(hierarchy_dir):
    hierarchy_dir = Path(hierarchy_dir)
    if hierarchy_dir.is_absolute():
        return hierarchy_dir

    name = hierarchy_dir.name
    candidates = [
        LOCAL_HIERARCHY_DIR / name,
        LOCAL_HIERARCHY_DIR / f"{name}_hierarchy",
        DEFENSE_ROSTERING_DIR / "input_data" / name,
    ]
    for cand in candidates:
        if (cand / "constraints.pkl").exists():
            return cand
    # nothing found: return the first (local) candidate so the error message is clear
    return candidates[0]


def load_hierarchy(hierarchy_dir):
    """
        Load a ``constraints.pkl`` / ``hierarchy.json`` pair written by
        ``export_hierarchy.py`` in defense-rostering.

        :param: hierarchy_dir: a hierarchy name (e.g. ``transcript_2`` or
            ``transcript_2_hierarchy``) resolved via :func:`_resolve_hierarchy_dir`, or
            an absolute path to a directory containing the two files
        :return: ``(root, hard)`` where `root` is the :class:`ConstraintNode` tree of soft
            constraints and `hard` is the list of hard constraints
    """
    hierarchy_dir = _resolve_hierarchy_dir(hierarchy_dir)

    with open(hierarchy_dir / "constraints.pkl", "rb") as f:
        data = pickle.load(f)
    all_constraints = data["all"]
    # critical: stop freshly-created assumption vars from aliasing these unpickled vars
    _advance_var_counters(all_constraints)
    hard = [all_constraints[i] for i in data["hard"]]

    with open(hierarchy_dir / "hierarchy.json", encoding="utf-8") as f:
        spec = json.load(f)
    root = constraint_node_from_dict(spec, all_constraints)

    return root, hard


# Flat (no-hierarchy) instances: experiments/data/flat_instances/<name>/constraints.pkl, a dict
# {"soft": [...], "hard": [...], "soft_names": [...], "hard_names": [...]} -- no hierarchy.json,
# no grouping. See experiments/data/flat_instances/*/_manifest.json for provenance.
LOCAL_FLAT_DIR = Path(__file__).resolve().parent / "data" / "flat_instances"


def load_flat_instance(name):
    """
        Load a flat (ungrouped) ``constraints.pkl`` -- a plain ``{"soft", "hard", "soft_names",
        "hard_names"}`` dict, as written for e.g. ``instance_292_first2unplannable``.

        :param: name: subdirectory name under ``data/flat_instances/``
        :return: ``(soft, hard, soft_names, hard_names)``
    """
    path = LOCAL_FLAT_DIR / name / "constraints.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    soft, hard = data["soft"], data["hard"]
    _advance_var_counters(soft + hard)
    return soft, hard, data.get("soft_names"), data.get("hard_names")
