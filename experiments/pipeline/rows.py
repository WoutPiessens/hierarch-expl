"""
    Canonical row schema shared by every experiment, plus CSV (de)serialization.

    Every experiment, whatever it compares, writes one CSV with the same columns.
    A "row" is one timed event produced by running one *series* (the thing being
    compared, e.g. ``baseline`` vs ``incremental``, or ``umus`` vs ``marco``) on one
    *instance*. Keeping a single schema means the plotting and reporting code never
    has to know which experiment produced the data.

    Columns
    -------
    - ``experiment``      : experiment name (e.g. ``hierarchical_runtime``)
    - ``instance``        : instance name (e.g. ``transcript_2`` or an XCSP3 key)
    - ``series``          : the compared method (e.g. ``baseline`` / ``umus``)
    - ``solver``          : core/SAT solver used
    - ``map_solver``      : MAP/hitting-set solver used (blank if not applicable)
    - ``event_index``     : 1-based index of this event within (instance, series)
    - ``round``           : refinement round (blank for non-hierarchical experiments)
    - ``elapsed_seconds`` : wall-clock seconds since this series started on this instance
    - ``kind``            : ``MUS`` / ``MCS`` / ``CUMU`` (event type)
    - ``metric``          : the primary tracked quantity at this event (e.g. number of
                            relevant constraints/groups discovered so far)
    - ``detail``          : free-form (e.g. ``;``-joined node names)
"""

from __future__ import annotations

import csv

CSV_FIELDNAMES = [
    "experiment", "instance", "series", "solver", "map_solver",
    "event_index", "round", "elapsed_seconds", "kind", "metric", "detail",
]


def sanitize_name(name):
    """Make `name` safe to embed in a filename."""
    return name.replace("/", "_").replace("\\", "_")


def events_to_rows(events, experiment, instance, series, solver, map_solver):
    """
        Turn a list of event dicts (as returned by the wrappers in
        :mod:`pipeline.algorithms`) into canonical rows.

        Each event must have ``elapsed_seconds``, ``kind`` and ``metric``; it may have
        ``round`` and ``detail``.
    """
    rows = []
    for i, ev in enumerate(events):
        rows.append({
            "experiment": experiment,
            "instance": instance,
            "series": series,
            "solver": solver,
            "map_solver": map_solver,
            "event_index": i + 1,
            "round": ev.get("round"),
            "elapsed_seconds": ev["elapsed_seconds"],
            "kind": ev["kind"],
            "metric": ev["metric"],
            "detail": ev.get("detail", ""),
        })
    return rows


def write_csv(rows, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in CSV_FIELDNAMES})


def read_csv(csv_path):
    """Read a canonical CSV back, coercing the numeric columns (for ``--replot-only``)."""
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["event_index"] = int(r["event_index"])
        r["elapsed_seconds"] = float(r["elapsed_seconds"])
        r["metric"] = float(r["metric"]) if "." in r["metric"] else int(r["metric"])
        r["round"] = int(r["round"]) if r["round"] not in ("", None) else None
    return rows
