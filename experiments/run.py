"""
    run.py - single entry point for running experiments on the cpmpy/tools/explain
    methods.

    Usage::

        python run.py --list
        python run.py hierarchical_runtime [--instances transcript_2 ...] [--solver exact]
        python run.py relevant_constraints [--instances small ...] [--solver ortools]

    Shared flags (all experiments): --output-dir, --tag, --replot-only, --skip.
    Experiment-specific flags are added by each experiment's ``add_args`` (see
    ``experiment_defs/``). See ``experiments/docs/experiments.md`` to add a new one.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cpmpy`

from experiment_defs import REGISTRY  # noqa: E402
from pipeline import run_experiment  # noqa: E402

OUTPUT_ROOT = Path(__file__).resolve().parent / "experiment_outputs"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list available experiments and exit")

    sub = parser.add_subparsers(dest="experiment")
    for name, mod in sorted(REGISTRY.items()):
        p = sub.add_parser(name, help=(mod.__doc__ or "").strip().splitlines()[0] if mod.__doc__ else name)
        mod.add_args(p)
        p.add_argument("--output-dir", default=None,
                       help="output directory (default: experiment_outputs/<name>_<tag-or-timestamp>)")
        p.add_argument("--tag", default=None,
                       help="stable name for the output dir instead of a timestamp")
        p.add_argument("--replot-only", action="store_true",
                       help="regenerate plots from an existing CSV without rerunning")
        p.add_argument("--skip", nargs="*", default=[],
                       help="series names to skip (e.g. --skip baseline)")

    args = parser.parse_args(argv)

    if args.list or not args.experiment:
        print("Available experiments:")
        for name, mod in sorted(REGISTRY.items()):
            first = (mod.__doc__ or name).strip().splitlines()[0]
            print(f"  {name:24s} {first}")
        return

    mod = REGISTRY[args.experiment]
    spec = mod.build_spec(args)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_ROOT / f"{args.experiment}_{suffix}"

    run_experiment(spec, output_dir, replot_only=args.replot_only, skip=set(args.skip))


if __name__ == "__main__":
    main()
