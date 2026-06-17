"""
    Generic, series-agnostic plotting over canonical rows (see :mod:`pipeline.rows`).

    Two plot kinds, used by every experiment:

    - :func:`plot_over_time` : the primary ``metric`` vs ``elapsed_seconds`` step plot,
      one line per series.
    - :func:`plot_per_round` : for hierarchical experiments only, one plot per
      refinement ``round`` showing how many MUS/MCS were found within that round
      (cumulative count vs time-since-round-start).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .rows import sanitize_name


def _series_rows(rows, instance, series):
    return sorted((r for r in rows
                   if r["instance"] == instance and r["series"] == series),
                  key=lambda r: r["event_index"])


def plot_over_time(rows, instance, series_order, colors, *, title, xlabel, ylabel, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for series in series_order:
        srows = _series_rows(rows, instance, series)
        if not srows:
            continue
        xs = [r["elapsed_seconds"] for r in srows]
        ys = [r["metric"] for r in srows]
        ax.step(xs, ys, where="post", color=colors[series], linewidth=2, label=series)
        ax.scatter(xs, ys, color=colors[series], s=20)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved {path}", flush=True)


def plot_per_round(rows, instance, series_order, colors, *, title_fmt, xlabel, ylabel, path_fmt):
    inst_rows = [r for r in rows if r["instance"] == instance]
    rounds = sorted({r["round"] for r in inst_rows if r["round"] is not None})

    for round_idx in rounds:
        round_rows = [r for r in inst_rows if r["round"] == round_idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        for series in series_order:
            srows = sorted((r for r in round_rows if r["series"] == series),
                           key=lambda r: r["event_index"])
            if not srows:
                continue
            t0 = srows[0]["elapsed_seconds"]
            xs = [r["elapsed_seconds"] - t0 for r in srows]
            ys = list(range(1, len(srows) + 1))
            ax.step(xs, ys, where="post", color=colors[series], linewidth=2, label=series)
            ax.scatter(xs, ys, color=colors[series], s=20)

        ax.set_title(title_fmt.format(instance=instance, round=round_idx))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = str(path_fmt).format(round=round_idx)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Plot] Saved {path}", flush=True)


def plot_round_metric(rows, instance, series_order, colors, *, title, xlabel, ylabel, path):
    """Per-round ``metric`` vs round number, one marked line per series (e.g. per-round
    enumeration time)."""
    inst_rows = [r for r in rows if r["instance"] == instance and r["round"] is not None]

    fig, ax = plt.subplots(figsize=(8, 5))
    for series in series_order:
        srows = sorted((r for r in inst_rows if r["series"] == series), key=lambda r: r["round"])
        if not srows:
            continue
        xs = [r["round"] for r in srows]
        ys = [r["metric"] for r in srows]
        ax.plot(xs, ys, marker="o", color=colors[series], linewidth=2, label=series)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved {path}", flush=True)


__all__ = ["plot_over_time", "plot_per_round", "plot_round_metric", "sanitize_name"]
