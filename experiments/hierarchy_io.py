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
from pathlib import Path

from cpmpy.tools.explain import constraint_node_from_dict

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
    hard = [all_constraints[i] for i in data["hard"]]

    with open(hierarchy_dir / "hierarchy.json", encoding="utf-8") as f:
        spec = json.load(f)
    root = constraint_node_from_dict(spec, all_constraints)

    return root, hard
