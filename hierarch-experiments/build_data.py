"""
    One-off data builder: populate ``data/`` so the experiments are self-contained.

    For each of the three instances it (1) copies the instance files (constraints.pkl +
    hierarchy.json + _info.json) from the exploratory ``final_experiments`` tree, and (2) samples
    the two oracle sets (20 x mss-20 and 20 x random-40) into separate JSON files.

    Run once:  python build_data.py
    Re-run with --resample to only redraw oracles (keeps the copied instance files).
"""
import shutil
import sys
from pathlib import Path

import hierarchy
import oracles as orc
from sampling import sample_oracles

# the one instance per problem, and where to copy its files from
INSTANCES = [
    ("nurse",     "instance2",           "nurse/data/instance2"),
    ("thesis",    "defense-transcript4", "thesis/data/defense-transcript4"),
    ("workforce", "ews-instance103",     "workforce/data/ews-instance103"),
]
_SRC_ROOT = Path(__file__).resolve().parents[1] / "final_experiments"
_FILES = ["constraints.pkl", "hierarchy.json", "_info.json"]


def copy_instance(problem, instance, src_rel):
    src = _SRC_ROOT / src_rel
    dst = hierarchy.instance_dir(problem, instance)
    dst.mkdir(parents=True, exist_ok=True)
    for f in _FILES:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    print(f"copied {problem}/{instance} <- {src}")


def build(resample_only=False):
    for problem, instance, src_rel in INSTANCES:
        if not resample_only:
            copy_instance(problem, instance, src_rel)
        root, hard = hierarchy.load_instance(problem, instance)
        print(f"{problem}/{instance}: {len(root.leaves())} soft leaves, {len(hard)} hard")
        for scheme, cfg in orc.SCHEMES.items():
            # distinct base seeds per scheme so the two sets never share draws
            seed0 = 0 if scheme.startswith("mss") else 10_000
            ora = sample_oracles(root, hard, scheme, cfg["pct"], cfg["n"], seed0=seed0)
            orc.save_oracles(problem, instance, scheme, ora)


if __name__ == "__main__":
    build(resample_only="--resample" in sys.argv)
