"""
    Instance loading for the hierarchical-explanation experiments.

    An instance lives in ``data/<problem>/<instance>/`` as two files written once by
    ``build_data.py``:

      * ``constraints.pkl`` : ``{"all": [<every cpmpy constraint>], "hard": [<indices of hard>]}``
      * ``hierarchy.json``  : the ConstraintNode tree (a dict produced by cpmpy's
        ``constraint_node_to_dict``), whose LEAVES hold the soft/primitive constraints.

    ``load_instance`` returns ``(root, hard)`` ready for cpmpy's ``marco`` / ``hierarchical_marco``.
    The soft (primitive) constraints are exactly the leaves of ``root``; every method here is
    keyed on a leaf's ``get_full_name()``.
"""
import json
import pickle
import re
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root (cpmpy/) on sys.path; must precede cpmpy
from cpmpy.tools.explain import constraint_node_from_dict
from cpmpy.expressions import variables as _vars
from cpmpy.transformations.get_variables import get_variables

DATA = Path(__file__).resolve().parent / "data"
PROBLEMS = ["nurse", "thesis", "workforce"]


def _advance_var_counters(constraints):
    """Advance cpmpy's global BV/IV auto-naming counters past the unpickled variables.

    cpmpy names variables ``BV<n>``/``IV<n>`` from a process-global counter and identifies them
    BY NAME. Unpickling restores the names but not the counter (which resets to 0 in a fresh
    process), so freshly-created assumption/indicator vars (built by marco / hierarchical_marco)
    would reuse ``BV0`` and silently ALIAS real model variables, corrupting every solve. Bumping
    the counter past the loaded variables prevents that. (Must be called right after unpickling.)
    """
    max_bv, max_iv = _vars._BoolVarImpl.counter, _vars._IntVarImpl.counter
    for v in get_variables(constraints):
        m = re.fullmatch(re.escape(_vars._BV_PREFIX) + r"(\d+)", v.name or "")
        if m:
            max_bv = max(max_bv, int(m.group(1)) + 1); continue
        m = re.fullmatch(re.escape(_vars._IV_PREFIX) + r"(\d+)", v.name or "")
        if m:
            max_iv = max(max_iv, int(m.group(1)) + 1)
    _vars._BoolVarImpl.counter, _vars._IntVarImpl.counter = max_bv, max_iv


def instance_dir(problem, instance):
    return DATA / problem / instance


def list_instances(problem):
    d = DATA / problem
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if (p / "hierarchy.json").exists())


def load_instance(problem, instance):
    """Return ``(root, hard)`` for a stored instance."""
    d = instance_dir(problem, instance)
    with open(d / "constraints.pkl", "rb") as f:
        data = pickle.load(f)
    all_constraints = data["all"]
    _advance_var_counters(all_constraints)                         # critical: see docstring above
    hard = [all_constraints[i] for i in data["hard"]]
    with open(d / "hierarchy.json", encoding="utf-8") as f:
        spec = json.load(f)
    root = constraint_node_from_dict(spec, all_constraints)
    return root, hard


def leaf_names(root):
    """Full names of the primitive (leaf) constraints, in tree order."""
    return [lf.get_full_name() for lf in root.leaves()]


def flat_soft(root):
    """``(soft_constraints, soft_names)`` -- the leaves as a flat instance (names == leaf full
    names). This is what the flat baselines (marco / staged deletion) operate on."""
    leaves = root.leaves()
    return ([lf.get_grouped_constraint() for lf in leaves],
            [lf.get_full_name() for lf in leaves])
