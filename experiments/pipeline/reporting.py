"""
    Human-readable reporting alongside the CSV/plots:

    - :func:`write_markdown_summary` writes a per-run ``SUMMARY.md`` (which benchmark,
      which algorithms, which settings, headline results). This is the function that
      used to live in ``report_utils.py``, extended with a run-metadata block.
    - :func:`run_metadata` captures the git commit + solver list so a result can be
      traced back to the exact code/configuration that produced it.
    - :func:`update_index` appends one line per run to ``experiment_outputs/INDEX.md``,
      giving a single scannable overview of every run.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path


def git_short_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def solver_list():
    try:
        import cpmpy as cp
        return ", ".join(cp.SolverLookup.solvernames())
    except Exception:
        return "unknown"


def run_metadata():
    """A dict of run-wide provenance, rendered into SUMMARY.md."""
    return {
        "git commit": git_short_hash(),
        "available solvers": solver_list(),
    }


def write_markdown_summary(path, title, description, algorithms, settings,
                           table_headers, table_rows, metadata=None):
    """
        Write a Markdown summary to `path`.

        :param: title: top-level heading
        :param: description: one or more paragraphs (str, or list of str)
        :param: algorithms: list of (name, description) pairs, rendered as a bullet list
        :param: settings: dict of run-wide settings (e.g. solver names)
        :param: table_headers: column headers for the per-instance results table
        :param: table_rows: rows (list of str/numbers) matching `table_headers`
        :param: metadata: optional dict of provenance (git commit, solvers, ...)
    """
    lines = [f"# {title}", "", f"_Generated: {datetime.now().isoformat(timespec='seconds')}_", ""]

    if isinstance(description, str):
        description = [description]
    for para in description:
        lines.append(para)
        lines.append("")

    if metadata:
        lines.append("## Run metadata")
        lines.append("")
        for key, value in metadata.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    if algorithms:
        lines.append("## Algorithms compared")
        lines.append("")
        for name, desc in algorithms:
            lines.append(f"- **{name}**: {desc}")
        lines.append("")

    if settings:
        lines.append("## Settings")
        lines.append("")
        for key, value in settings.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    if table_rows:
        lines.append("## Results")
        lines.append("")
        lines.append("| " + " | ".join(table_headers) + " |")
        lines.append("|" + "|".join(["---"] * len(table_headers)) + "|")
        for row in table_rows:
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def round_time_table(rows, series_order):
    """
        Pivot per-round timing rows (``round`` set, ``metric`` = enumeration seconds)
        into a Markdown table with one row per ``(instance, round)`` and one column per
        series. Returns the Markdown text.
    """
    by = {}
    for r in rows:
        if r.get("round") in (None, ""):
            continue
        by.setdefault((r["instance"], int(r["round"])), {})[r["series"]] = float(r["metric"])

    headers = ["Instance", "Round"] + [f"{s} enum time (s)" for s in series_order]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for inst, rd in sorted(by):
        cells = [inst, str(rd)] + [
            (f"{by[(inst, rd)][s]:.4f}" if s in by[(inst, rd)] else "") for s in series_order]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


INDEX_HEADERS = ["When", "Experiment", "Instances", "Git", "Headline", "Run dir"]


def update_index(index_path, run_info):
    """
        Append one row about a finished run to ``experiment_outputs/INDEX.md``
        (creating the file with a header if it does not exist yet).

        `run_info` keys: ``experiment``, ``instances``, ``git``, ``headline``, ``run_dir``.
    """
    index_path = Path(index_path)
    if not index_path.exists():
        header = ["# Experiment runs index", "",
                  "One row per finished run, newest at the bottom.", "",
                  "| " + " | ".join(INDEX_HEADERS) + " |",
                  "|" + "|".join(["---"] * len(INDEX_HEADERS)) + "|"]
        index_path.write_text("\n".join(header) + "\n", encoding="utf-8")

    row = [
        datetime.now().isoformat(timespec="seconds"),
        run_info.get("experiment", ""),
        run_info.get("instances", ""),
        run_info.get("git", ""),
        run_info.get("headline", ""),
        run_info.get("run_dir", ""),
    ]
    with open(index_path, "a", encoding="utf-8") as f:
        f.write("| " + " | ".join(str(v) for v in row) + " |\n")
