"""
    The experiment driver.

    An experiment is fully described by an :class:`ExperimentSpec`: which instances to
    run, which series (algorithm wrappers) to compare, the run settings, and a few
    labels for plots/markdown. :func:`run_experiment` then does the same thing for
    every experiment:

        for each instance:
            for each active series:
                events = series.fn(instance, settings)   # timed
                rows  += events_to_rows(...)
            plot(instance)
        write CSV, SUMMARY.md, update INDEX.md

    Nothing here is experiment-specific, so adding a new experiment never means
    touching this file.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .plotting import plot_over_time, plot_per_round, plot_round_metric, sanitize_name
from .reporting import round_time_table, run_metadata, update_index, write_markdown_summary
from .rows import events_to_rows, read_csv, write_csv


@dataclass
class Series:
    """One compared method: a name, an event-stream wrapper, and a plot colour."""
    name: str
    fn: Callable           # (instance, settings) -> list[event dict]
    color: str


@dataclass
class ExperimentSpec:
    name: str                       # short id, e.g. "hierarchical_runtime"
    title: str                      # human title for SUMMARY.md
    description: str                # paragraph(s) for SUMMARY.md
    instances: list                 # list[Instance]
    series: list                    # list[Series]
    settings: dict                  # solver, map_solver, initial_level, ...
    algorithms_doc: list            # [(name, description)] for SUMMARY.md
    metric_label: str = "relevant"  # used in summary column headers
    plot_ylabel: str = "# relevant constraints"
    plot_title: str = "{instance}: relevant constraints discovered over time"
    has_rounds: bool = False        # hierarchical experiments split events into rounds
    round_title: str = "{instance}: round {round} - MUS/MCS found over time"
    round_ylabel: str = "# MUS/MCS found this round"
    # When True, the per-event `metric` is that round's enumeration time (seconds): the
    # plot becomes per-round time vs round number, the summary reports total enumeration
    # time, and a per-round timing table (round x series) is written and printed.
    metric_is_round_time: bool = False


def _colors(spec):
    return {s.name: s.color for s in spec.series}


def _plot_instance(spec, instance_name, rows, plots_dir):
    colors = _colors(spec)
    series_order = [s.name for s in spec.series]
    base = sanitize_name(instance_name)

    if spec.metric_is_round_time:
        plot_round_metric(
            rows, instance_name, series_order, colors,
            title=spec.plot_title.format(instance=instance_name),
            xlabel="Refinement round", ylabel=spec.plot_ylabel,
            path=plots_dir / f"{spec.name}_{base}.png",
        )
        return

    plot_over_time(
        rows, instance_name, series_order, colors,
        title=spec.plot_title.format(instance=instance_name),
        xlabel="Elapsed time (s)", ylabel=spec.plot_ylabel,
        path=plots_dir / f"{spec.name}_{base}.png",
    )
    if spec.has_rounds:
        plot_per_round(
            rows, instance_name, series_order, colors,
            title_fmt=spec.round_title, xlabel="Elapsed time since round start (s)",
            ylabel=spec.round_ylabel,
            path_fmt=plots_dir / f"{spec.name}_{base}_round{{round}}.png",
        )


def _write_enumeration_log(path, spec, series_logs):
    """
        Write a chronological enumeration trace: for each (instance, series), the
        map-solver seed and the MUS/MCS derived from it at every iteration, grouped by
        refinement round (with that round's frontier of active groups).
    """
    lines = [
        f"# Enumeration log: {spec.name}", "",
        "Chronological trace of the MUS/MCS enumeration. For every map-solver iteration "
        "the **seed** (the subset of active groups the map solver returned) is shown "
        "together with the **MUS** or **MCS** derived from it. Iterations are grouped by "
        "refinement round; each round header lists the round's frontier of active groups.",
        "",
    ]
    by_inst = {}
    for inst, series, records in series_logs:
        by_inst.setdefault(inst, []).append((series, records))

    for inst, series_list in by_inst.items():
        lines.append(f"## Instance: {inst}")
        lines.append("")
        for series, records in series_list:
            lines.append(f"### Series: {series}")
            lines.append("")
            it = n_mus = n_mcs = 0
            for rec in records:
                if rec["type"] == "round":
                    front = rec["frontier"]
                    lines.append("")
                    lines.append(f"**Round {rec['round']}** — frontier ({len(front)} groups): "
                                 + ", ".join(f"`{name}`" for name in front))
                    lines.append("")
                else:
                    it += 1
                    if rec["kind"] == "MUS":
                        n_mus += 1
                    else:
                        n_mcs += 1
                    seed = ", ".join(rec["seed"]) if rec["seed"] else "(empty)"
                    result = ", ".join(rec["result"]) if rec["result"] else "(empty)"
                    lines.append(f"- iter {it}: seed = {{{seed}}}  →  **{rec['kind']}** {{{result}}}")
            lines.append("")
            lines.append(f"_Totals for {series}: {n_mus} MUS, {n_mcs} MCS over {it} "
                         f"map-solver iterations._")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(spec, output_dir, *, replot_only=False, skip=()):
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{spec.name}_summary.csv"
    md_path = output_dir / "SUMMARY.md"

    if replot_only:
        rows = read_csv(csv_path)
        for inst in spec.instances:
            _plot_instance(spec, inst.name, rows, plots_dir)
        return

    active_series = [s for s in spec.series if s.name not in skip]
    solver = spec.settings.get("solver", "")
    map_solver = spec.settings.get("map_solver", "")

    all_rows = []
    summary_rows = []
    series_logs = []  # (instance, series, [log records]) for ENUMERATION_LOG.md
    for inst in spec.instances:
        print(f"[Load] {inst.name} <- {inst.source}", flush=True)
        srow = [inst.name, inst.describe()]

        for s in active_series:
            print(f"[{s.name}] {inst.name}", flush=True)
            log_records = []
            spec.settings["_log_sink"] = log_records
            t0 = time.perf_counter()
            events = s.fn(inst, spec.settings)
            elapsed = time.perf_counter() - t0
            spec.settings.pop("_log_sink", None)
            if log_records:
                series_logs.append((inst.name, s.name, log_records))
            print(f"[{s.name}] {inst.name} done in {elapsed:.2f}s, {len(events)} events", flush=True)

            rows = events_to_rows(events, spec.name, inst.name, s.name, solver, map_solver)
            all_rows.extend(rows)

            if spec.metric_is_round_time:
                total_enum = sum(r["metric"] for r in rows)
                srow += [f"{total_enum:.4f}", f"{elapsed:.2f}"]
            else:
                final_metric = rows[-1]["metric"] if rows else 0
                srow += [final_metric, f"{elapsed:.2f}"]
            if spec.has_rounds:
                n_rounds = max((r["round"] for r in rows if r["round"] is not None), default=0)
                srow.append(n_rounds)

        summary_rows.append(srow)
        _plot_instance(spec, inst.name, all_rows, plots_dir)

    write_csv(all_rows, csv_path)
    print(f"[Done] CSV written to {csv_path}", flush=True)

    headers = ["Instance", "Source"]
    metric_col = "total enum time (s)" if spec.metric_is_round_time else f"final {spec.metric_label}"
    for s in active_series:
        headers += [f"{s.name} {metric_col}", f"{s.name} time (s)"]
        if spec.has_rounds:
            headers.append(f"{s.name} #rounds")

    meta = run_metadata()
    write_markdown_summary(
        md_path, title=spec.title, description=spec.description,
        algorithms=spec.algorithms_doc, settings=spec.settings,
        table_headers=headers, table_rows=summary_rows, metadata=meta,
    )
    print(f"[Done] Markdown summary written to {md_path}", flush=True)

    if spec.metric_is_round_time:
        series_order = [s.name for s in active_series]
        table_md = round_time_table(all_rows, series_order)
        (output_dir / "ROUND_TIMES.md").write_text(
            f"# Per-round enumeration time ({spec.name})\n\n"
            "Each cell is one round's enumeration time in seconds: from the first "
            "map-solver call of that round until the map solver returns UNSAT.\n\n"
            + table_md, encoding="utf-8")
        print(f"[Done] Per-round timing table written to {output_dir / 'ROUND_TIMES.md'}", flush=True)
        print("\n=== Per-round enumeration time (seconds) ===\n" + table_md, flush=True)

    if series_logs:
        log_path = output_dir / "ENUMERATION_LOG.md"
        _write_enumeration_log(log_path, spec, series_logs)
        print(f"[Done] Enumeration log written to {log_path}", flush=True)

    headline = "; ".join(
        f"{s.name}={summary_rows[0][2 + i * (3 if spec.has_rounds else 2)]}"
        for i, s in enumerate(active_series)
    ) if summary_rows else ""
    update_index(output_dir.parent / "INDEX.md", {
        "experiment": spec.name,
        "instances": ",".join(inst.name for inst in spec.instances),
        "git": meta["git commit"],
        "headline": headline,
        "run_dir": output_dir.name,
    })
    return all_rows
