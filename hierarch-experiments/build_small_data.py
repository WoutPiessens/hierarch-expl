"""
    Build the 'smaller-problems' data: copy the SELECTED simpler instances (from
    ``final_experiments/small_selection.json``, produced by find_small_instances.py) into
    ``data/<problem>-small/<instance>/`` -- kept fully SEPARATE from the current problems --
    and sample 5 mss-20 oracles each (no random-40 set for now).

    Run once:  python build_small_data.py
"""
import json
import shutil
from pathlib import Path

import hierarchy
import oracles as orc
from sampling import sample_oracles

HERE = Path(__file__).resolve().parent
SEL = json.loads((HERE.parent / "final_experiments" / "small_selection.json").read_text())
_SRC_ROOT = HERE.parent / "final_experiments"
_FILES = ["constraints.pkl", "hierarchy.json", "_info.json"]
N_ORACLES = 5


def main():
    for problem, pick in SEL["selection"].items():
        if pick is None:
            print(f"!! no selection for {problem}, skipping")
            continue
        instance = pick["instance"]
        small = f"{problem}-small"
        src = _SRC_ROOT / problem / "data" / instance
        dst = hierarchy.instance_dir(small, instance)
        dst.mkdir(parents=True, exist_ok=True)
        for f in _FILES:
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        root, hard = hierarchy.load_instance(small, instance)
        print(f"{small}/{instance}: {len(root.leaves())} leaves, {len(hard)} hard, "
              f"measured natural-MCS sizes {pick['sizes']} (mean {pick['mean']})")
        ora = sample_oracles(root, hard, "mss-20", 20, N_ORACLES, seed0=0)
        orc.save_oracles(small, instance, "mss-20", ora)


if __name__ == "__main__":
    main()
