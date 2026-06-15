"""
    Load a benchmark instance exported by defense-rostering's ``export_hierarchy.py``
    (a ``constraints.pkl`` + ``hierarchy.json`` pair) into a :class:`ConstraintNode`
    tree plus a list of hard constraints, ready to pass to :func:`hierarchical_marco`,
    :func:`marco` or :func:`umus`.
"""

import json
import pickle
from pathlib import Path

from cpmpy.tools.explain import constraint_node_from_dict

# defense-rostering checkout, assumed to be a sibling of this repo's parent directory:
#   PycharmProjects/
#     defense-rostering/
#     hierarch-expl/hierarch-expl/experiments/hierarchy_io.py
DEFENSE_ROSTERING_DIR = Path(__file__).resolve().parents[3] / "defense-rostering"


def load_hierarchy(hierarchy_dir):
    """
        Load a ``constraints.pkl`` / ``hierarchy.json`` pair written by
        ``export_hierarchy.py`` in defense-rostering.

        :param: hierarchy_dir: directory containing ``constraints.pkl`` and ``hierarchy.json``,
            either absolute or relative to :data:`DEFENSE_ROSTERING_DIR`'s ``input_data``
        :return: ``(root, hard)`` where `root` is the :class:`ConstraintNode` tree of soft
            constraints and `hard` is the list of hard constraints
    """
    hierarchy_dir = Path(hierarchy_dir)
    if not hierarchy_dir.is_absolute():
        hierarchy_dir = DEFENSE_ROSTERING_DIR / "input_data" / hierarchy_dir

    with open(hierarchy_dir / "constraints.pkl", "rb") as f:
        data = pickle.load(f)
    all_constraints = data["all"]
    hard = [all_constraints[i] for i in data["hard"]]

    with open(hierarchy_dir / "hierarchy.json", encoding="utf-8") as f:
        spec = json.load(f)
    root = constraint_node_from_dict(spec, all_constraints)

    return root, hard
