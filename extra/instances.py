"""
    Instance builders for the `extra/` verification of the commit-enabled `hierarchical_marco`.

    Two instances, selectable by a config option:

      * ``"defense"`` -- the first defense-scheduling instance (``transcript_1``), loaded with its
        real committed hierarchy via ``hierarchy_io.load_hierarchy``.
      * ``"nurse"``   -- the LARGER, MULTI-SHIFT nurse-rostering SOFT/HARD model
        (``nurse_instance7_softreq_multishift``: 20 nurses, 28 days, shifts E/D/L; structural
        rules HARD, shift_on/shift_off/cover SOFT). It ships as a *flat* soft list, so here we
        wrap those soft constraints in a 5-level ``ConstraintNode`` hierarchy, split by SHIFT
        first:

            level 1: shift                  -- shiftE / shiftD / shiftL
            level 2: family                 -- shift_on / shift_off / cover
            level 3: nurse (shift_on/off)   -- per-nurse requests
                     week  (cover)          -- cover has no nurse, so grouped by 7-day week
            level 4: week  (shift_on/off)   -- per-week within a nurse
                     day (leaf) (cover)     -- the primitive cover constraint
            level 5: day (leaf) (shift_on/off) -- the primitive request constraint

        So shift_on/shift_off leaves sit at level 5 (shift->family->nurse->week->day) and cover
        leaves at level 4 (shift->family->week->day). Splitting by shift first makes the coarsest
        (level-1) granularity ask "which shifts cannot be planned?": here shifts E and L are
        individually infeasible while D is fine, so the shift-level correction set is {E, L}.

        Name decoding (kept in the constraint names for provenance): ``w<n>`` on a request is its
        objective Weight; on a cover, ``req<n>`` is the Requirement (number of nurses that must be
        on that shift/day), ``wo<n>`` / ``wu<n>`` are the original objective "Weight for over" /
        "Weight for under" penalties. In the SOFT/HARD model cover is posted as ``Count == req``
        (no slack), so wo/wu are informational only.
"""

import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(_EXPERIMENTS))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for cpmpy

from cpmpy.tools.explain.hierarchical import ConstraintNode
from hierarchy_io import load_hierarchy, load_flat_instance

NURSE_INSTANCE = "nurse_instance7_softreq_multishift"
DEFENSE_INSTANCE = "transcript_1"


def build_defense():
    """The first defense-scheduling instance, with its real hierarchy."""
    root, hard = load_hierarchy(DEFENSE_INSTANCE)
    return root, hard, f"defense: {DEFENSE_INSTANCE}"


def _family_of(soft_name):
    """cover__day4... -> 'cover'; shift_on__nurseA... -> 'shift_on'; else 'other'."""
    return soft_name.split("__", 1)[0] if "__" in soft_name else "other"


def _leaf_name(soft_name):
    """The part after the family prefix, e.g. 'nurseA_day2_shiftD_w2' or 'day0_shiftD_req5_...'."""
    return soft_name.split("__", 1)[1] if "__" in soft_name else soft_name


def _day_num(day_tag):
    """'day10' -> 10."""
    return int(day_tag[3:])


def build_nurse():
    """The multi-shift nurse soft/hard model, wrapped in a 5-level hierarchy split by SHIFT first:
    shift_on/off: shift -> family -> nurse -> week -> day(leaf);  cover: shift -> family -> week -> day(leaf)."""
    soft, hard, soft_names, _ = load_flat_instance(NURSE_INSTANCE)
    root = ConstraintNode("nurse")
    for con, name in zip(soft, soft_names):
        fam = _family_of(name)
        tag = _leaf_name(name).split("_")
        if fam in ("shift_on", "shift_off"):
            nurse, day, shift = tag[0], tag[1], tag[2]            # 'nurseA','day2','shiftE'
            shift_node = root.add_child(shift)                    # L1: shift
            fam_node = shift_node.add_child(fam)                  # L2: family
            nurse_node = fam_node.add_child(nurse)                # L3: nurse
            week_node = nurse_node.add_child(f"week{_day_num(day) // 7}")  # L4: week
            leaf = week_node.add_child(day)                       # L5: day (leaf)
        else:                                                     # cover: 'day0','shiftE',...
            day, shift = tag[0], tag[1]
            shift_node = root.add_child(shift)                    # L1: shift
            fam_node = shift_node.add_child(fam)                  # L2: family (cover)
            week_node = fam_node.add_child(f"week{_day_num(day) // 7}")  # L3: week
            leaf = week_node.add_child(day)                       # L4: day (leaf)
        leaf.constraints.append(con)
    return root, hard, f"nurse soft/hard: {NURSE_INSTANCE}"


BUILDERS = {"defense": build_defense, "nurse": build_nurse}


def build(kind):
    if kind not in BUILDERS:
        raise ValueError(f"unknown instance {kind!r}; choose from {sorted(BUILDERS)}")
    return BUILDERS[kind]()
