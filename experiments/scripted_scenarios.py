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
    # transcript_{1,2,3}_anon: identical instances to transcript_{1,2,3} but with every person
    # name replaced by a pseudo-random id (see anonymize_transcripts.py + the mapping in
    # data/hierarchies/anonymization_mapping.json). Same scenario, only the person references
    # are the anonymized ids, so results are bit-identical to the un-anonymized versions.
    "transcript_1_anon": [
        {"constraints": ["person-unavailable"], "target_level": 2},
        {"constraints": ["person-unavailable Person_4728"], "target_level": 3},   # Arthur Offermans
        {"constraints": ["already-allocated"], "target_level": 2},
    ],
    "transcript_2_anon": [
        {"constraints": ["person-unavailable", "room-unavailable"], "target_level": 2},
        {"constraints": [
            "person-unavailable Person_8994",   # Wannes Meert
            "person-unavailable Person_8179",   # Jesse Davis
            "person-unavailable Person_4925",   # Hendrik Blockeel
        ], "target_level": 4},
        {"constraints": ["already-allocated"], "target_level": 2},
    ],
    "transcript_3_anon": [
        {"constraints": ["person-unavailable"], "target_level": 2},
        {"constraints": ["person-unavailable Person_5877"], "target_level": 3},   # Adalberto Simeone
        {"constraints": [
            "person-unavailable Person_5877 2025-02-25",
            "person-unavailable Person_5877 2025-02-26",
        ], "target_level": 4},
    ],
    # transcript_4..7: auto-generated from input_data/instances_unsat/* ("plan all defenses
    # in physical rooms"). The refinement zooms the person-unavailability conflict from
    # all-people (lvl1) -> one conflict-relevant person (lvl2) -> that person on one date
    # (lvl3) -> that date's timeslots (lvl4). The person/date were picked by probing the
    # enumeration (they actually appear in a MUS/MCS), so every iteration is non-empty.
    "transcript_4": [  # instance_117
        {"constraints": ["person-unavailable", "room-unavailable"], "target_level": 2},
        {"constraints": ["person-unavailable Julie Birkholz"], "target_level": 3},
        {"constraints": ["person-unavailable Julie Birkholz 2025-09-01"], "target_level": 4},
    ],
    "transcript_5": [  # instance_224
        {"constraints": ["person-unavailable", "room-unavailable"], "target_level": 2},
        {"constraints": ["person-unavailable Stef Verreydt"], "target_level": 3},
        {"constraints": ["person-unavailable Stef Verreydt 2025-02-28"], "target_level": 4},
    ],
    "transcript_6": [  # instance_209
        {"constraints": ["person-unavailable", "room-unavailable"], "target_level": 2},
        {"constraints": ["person-unavailable Danny Hughes"], "target_level": 3},
        {"constraints": ["person-unavailable Danny Hughes 2025-02-24"], "target_level": 4},
    ],
    "transcript_7": [  # instance_284
        {"constraints": ["person-unavailable", "room-unavailable"], "target_level": 2},
        {"constraints": ["person-unavailable Mathias Verbeke"], "target_level": 3},
        {"constraints": ["person-unavailable Mathias Verbeke 2025-02-27"], "target_level": 4},
    ],
}
