"""
    compare_runs.py

    Overlay the same instance's series from two experiment runs, to see how a code or
    configuration change affected the results. Reads two canonical
    ``*_summary.csv`` files (see :mod:`pipeline.rows`) and, for each instance present in
    both, draws ``metric`` vs ``elapsed_seconds`` with the first run dashed and the
    second run solid.

    Usage::

        python compare_runs.py OLD_summary.csv NEW_summary.csv [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cpmpy`

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.rows import read_csv, sanitize_name

COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]


def _series_xy(rows, instance, series):
    srows = sorted((r for r in rows if r["instance"] == instance and r["series"] == series),
                   key=lambda r: r["event_index"])
    return [r["elapsed_seconds"] for r in srows], [r["metric"] for r in srows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old_csv")
    parser.add_argument("new_csv")
    parser.add_argument("--out-dir", default=None,
                        help="where to write comparison PNGs (default: next to new_csv)")
    args = parser.parse_args(argv)

    old = read_csv(args.old_csv)
    new = read_csv(args.new_csv)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.new_csv).resolve().parent / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = sorted({r["instance"] for r in new} & {r["instance"] for r in old})
    series = sorted({r["series"] for r in new})
    color_of = {s: COLORS[i % len(COLORS)] for i, s in enumerate(series)}

    for instance in instances:
        fig, ax = plt.subplots(figsize=(8, 5))
        for s in series:
            ox, oy = _series_xy(old, instance, s)
            nx, ny = _series_xy(new, instance, s)
            if ox:
                ax.step(ox, oy, where="post", color=color_of[s], linestyle="--",
                        linewidth=1.5, alpha=0.7, label=f"{s} (old)")
            if nx:
                ax.step(nx, ny, where="post", color=color_of[s], linewidth=2, label=f"{s} (new)")
        ax.set_title(f"{instance}: old vs new")
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("metric")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"compare_{sanitize_name(instance)}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Plot] Saved {path}", flush=True)


if __name__ == "__main__":
    main()
