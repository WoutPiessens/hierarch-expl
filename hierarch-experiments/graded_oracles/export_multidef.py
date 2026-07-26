"""
Export thesis instances that require MANY previously-unplanned defenses to be planned at once
(defense_to_plan = a long list) -> a large joint conflict -> corrections in the 16-20 bin (the
single-defense variants give <=5; the current -m sets top out ~15). Runs the defense-rostering
exporter in its own venv, saves the hierarchy into data/thesis-graded/<name>/, and reports the
hard-SAT status + a quick correction-size histogram.

    python export_multidef.py
"""
import sys, subprocess, shutil, random
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
FE = HERE.parent.parent / "final_experiments"
sys.path.insert(0, str(FE))
sys.path.insert(0, str(HERE.parent))
import _bootstrap  # noqa
import common
import cpmpy as cp
from cpmpy.tools.explain.utils import make_assump_model

DR = Path(r"C:\Users\Wout\PycharmProjects\defense-rostering")
DR_PY = DR / ".venv-export" / "Scripts" / "python.exe"
PLANNED = {115: [0, 6, 8, 11, 14, 15, 17], 117: [1, 6, 8, 13], 172: [1, 2, 6, 10, 12, 13]}
OUTDIR = {115: "output_data/20260204_125624_0", 117: "output_data/20260204_125521_0",
          172: "output_data/20260204_125814_0"}
DEST = HERE.parent / "data" / "thesis-graded"
# (name, instance id, list of unplanned defenses to require -- push high to reach the 16-20 bin)
CONFIGS = [
    ("unsat-115-mult9", 115, [1, 2, 3, 4, 5, 7, 9, 10, 12]),
    ("unsat-115-mult10", 115, [1, 2, 3, 4, 5, 7, 9, 10, 12, 13]),
    ("unsat-117-mult9", 117, [0, 2, 3, 4, 5, 7, 9, 10, 11]),
    ("unsat-117-mult10", 117, [0, 2, 3, 4, 5, 7, 9, 10, 11, 12]),
]


def export_one(iid, dlist):
    import pandas as pd
    src = DR / "input_data" / "instances_unsat" / f"instance_{iid}"
    clean = DR / "input_data" / "instances_unsat_clean" / f"instance_{iid}"
    clean.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix == ".csv":
            pd.read_csv(f).fillna("").to_csv(clean / f.name, index=False)
        elif not f.name.endswith(".bak"):
            shutil.copy2(f, clean / f.name)
    base = (DR / "example-configs" / "config-2026-without-obj.yaml").read_text(encoding="utf-8")
    lines = []
    for line in base.splitlines():
        if line.startswith("input_data:"):
            line = f'input_data: "instances_unsat/instance_{iid}"'
        elif line.startswith("adjacency_objective:"):
            line = "adjacency_objective: false"
        lines.append(line)
    lines += ['', f'output_data: "{OUTDIR[iid]}"', f'planned_defenses: {PLANNED[iid]}',
              f'defense_to_plan: {dlist}']
    cfg = DR / f"cfg_multidef_{iid}.yaml"
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DR / "_export_test" / "dummy_out").mkdir(parents=True, exist_ok=True)
    (DR / "_export_test" / "dummy_out" / "output.csv").write_text("day,start_time,room\n")
    dst = DEST / f"_tmp_{iid}"
    env = dict(__import__("os").environ, PYTHONPATH=str(HERE.parent.parent))
    p = subprocess.run([str(DR_PY), "export_hierarchy.py", "--config", str(cfg), "--output", str(dst)],
                       cwd=str(DR), capture_output=True, text=True, timeout=900, env=env)
    if not (dst / "hierarchy.json").exists():
        raise RuntimeError("export failed: " + (p.stderr or p.stdout).strip()[-300:])
    return dst


def measure(root, hard):
    leaves = root.leaves()
    hs = cp.Model(list(hard)).solve(solver="ortools")
    full = cp.Model(list(hard) + [l.get_grouped_constraint() for l in leaves]).solve(solver="ortools")
    if full is not False or hs is not True:
        return f"full={full} hard={hs} (unusable)"
    soft = [l.get_grouped_constraint() for l in leaves]
    model, _s, assump = make_assump_model(soft, list(hard))
    s = cp.SolverLookup.get("ortools", model)
    sizes = Counter(); seen = set()
    for seed in range(8):
        order = list(range(len(assump))); random.Random(seed).shuffle(order); kept = []
        for i in order:
            if s.solve(assumptions=[assump[j] for j in kept + [i]]) is True:
                kept.append(i)
        C = frozenset(range(len(assump))) - set(kept)
        if C not in seen:
            seen.add(C); sizes[len(C)] += 1
    return f"{len(leaves)} leaves, sizes={dict(sorted(sizes.items()))}"


def main():
    for name, iid, dlist in CONFIGS:
        try:
            tmp = export_one(iid, dlist)
            dst = DEST / name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp / "constraints.pkl", dst / "constraints.pkl")
            shutil.copy2(tmp / "hierarchy.json", dst / "hierarchy.json")
            shutil.rmtree(tmp, ignore_errors=True)
            root, hard = common.load_hierarchy(str(dst))
            print(f"  {name} (inst {iid}, {len(dlist)} defenses): {measure(root, hard)}", flush=True)
        except Exception as e:
            print(f"  {name}: ERR {type(e).__name__}: {str(e)[:200]}", flush=True)
    print("EXPORT_MULTIDEF_DONE", flush=True)


if __name__ == "__main__":
    main()
