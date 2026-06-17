"""
    scripted_scenarios.py

    Scripted refinement scenarios for the hierarchical (transcript) benchmarks, ported
    from defense-rostering's ``SCRIPTED_SCENARIOS``
    (``defense_rostering_explanation_simplified.py``).

    Each scenario is a list of refinement steps. A step refines the named groups to a
    ``target_level`` (depth in the constraint tree; root = level 0, its children = 1, ...),
    via :func:`cpmpy.tools.explain.hierarchical.activate_descendants_at_level`, which can
    skip intermediate levels (e.g. a single-child "renaming" level). Refinement is fully
    script-driven, **not** based on which groups appeared in a MUS/MCS, so an instance with
    ``k`` steps runs exactly ``k + 1`` iterations — matching defense-rostering.

    Group names use the hierarch-expl convention (space-joined ``get_full_name()``,
    no ``<angle brackets>``); the defense-rostering originals used ``<...>``.
"""

SCENARIOS = {
    "transcript_1": [
        {"constraints": ["person-unavailable"], "target_level": 2},
        {"constraints": ["person-unavailable Arthur Offermans"], "target_level": 3},
        {"constraints": ["already-allocated"], "target_level": 2},
    ],
    "transcript_2": [
        {"constraints": ["person-unavailable", "room-unavailable"], "target_level": 2},
        {"constraints": [
            "person-unavailable Wannes Meert",
            "person-unavailable Jesse Davis",
            "person-unavailable Hendrik Blockeel",
        ], "target_level": 4},
        {"constraints": ["already-allocated"], "target_level": 2},
    ],
    "transcript_3": [
        {"constraints": ["person-unavailable"], "target_level": 2},
        {"constraints": ["person-unavailable Adalberto Simeone"], "target_level": 3},
        {"constraints": [
            "person-unavailable Adalberto Simeone 2025-02-25",
            "person-unavailable Adalberto Simeone 2025-02-26",
        ], "target_level": 4},
    ],
}
