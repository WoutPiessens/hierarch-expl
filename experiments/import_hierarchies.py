"""
    import_hierarchies.py

    One-time helper: copy the exported hierarchical benchmark instances
    (``constraints.pkl`` + ``hierarchy.json``) out of a defense-rostering checkout and
    into this repo under ``experiments/data/hierarchies/``, so the experiment pipeline
    can run standalone on another machine without needing defense-rostering present.

    Run this on a machine that *does* have defense-rostering, then commit / rsync the
    repo (including ``experiments/data/hierarchies/``) to the target machine.

    Usage::

        python import_hierarchies.py                       # all transcript_* instances
        python import_hierarchies.py transcript_2_hierarchy # specific dirs
        DEFENSE_ROSTERING_DIR=/path/to/defense-rostering python import_hierarchies.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cpmpy`

from hierarchy_io import DEFENSE_ROSTERING_DIR, LOCAL_HIERARCHY_DIR

FILES = ("constraints.pkl", "hierarchy.json")
DEFAULT_DIRS = ["transcript_1_hierarchy", "transcript_2_hierarchy", "transcript_3_hierarchy"]


def import_one(name):
    src = DEFENSE_ROSTERING_DIR / "input_data" / name
    if not (src / "constraints.pkl").exists():
        print(f"[skip] {name}: not found at {src}")
        return False
    dest = LOCAL_HIERARCHY_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    for fn in FILES:
        shutil.copy2(src / fn, dest / fn)
    size_mb = sum((dest / fn).stat().st_size for fn in FILES) / 1e6
    print(f"[copy] {name} -> {dest}  ({size_mb:.1f} MB)")
    return True


def main(argv=None):
    names = (argv if argv else sys.argv[1:]) or DEFAULT_DIRS
    print(f"[source] {DEFENSE_ROSTERING_DIR / 'input_data'}")
    print(f"[target] {LOCAL_HIERARCHY_DIR}")
    n = sum(import_one(name) for name in names)
    print(f"[done] imported {n}/{len(names)} hierarchy instances")


if __name__ == "__main__":
    main()
