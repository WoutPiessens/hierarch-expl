"""Enumerate ALL primitive (flat, leaf-level) MCSes via baseline marco, WITH incremental
progress logging (flushed every N iterations) so progress is never silently lost. Stores
all-MCSes + 4 random subsets (5/10/15/20%) keyed by soft-constraint NAME."""
import sys, json, random, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpmpy.tools.explain import marco
from hierarchy_io import load_flat_instance

INST = sys.argv[1]
LOG_EVERY = 20

soft, hard, soft_names, hard_names = load_flat_instance(INST)
name_of = {id(c): n for c, n in zip(soft, soft_names)}
print(f"{INST}: {len(soft)} soft, {len(hard)} hard constraints. Enumerating...", flush=True)

out_dir = Path("data/flat_instances") / INST
mcses = []
mus_count = 0
t0 = time.perf_counter()

def flush_partial():
    with open(out_dir / "all_mcses.json", "w", encoding="utf-8") as f:
        json.dump([sorted(m) for m in mcses], f, indent=2)

for kind, found in marco(soft, hard, solver="exact", map_solver="exact", return_mus=True, return_mcs=True):
    if kind == "MCS":
        mcses.append(frozenset(name_of[id(c)] for c in found))
    else:
        mus_count += 1
    total = len(mcses) + mus_count
    if total % LOG_EVERY == 0:
        dt = time.perf_counter() - t0
        print(f"  [{dt:7.1f}s] {len(mcses)} MCS, {mus_count} MUS so far ({total} total iterations)", flush=True)
        flush_partial()

dt = time.perf_counter() - t0
print(f"DONE {INST}: {len(mcses)} primitive MCSes, {mus_count} primitive MUSes, enumerated in {dt:.2f}s", flush=True)
flush_partial()

random.seed(20260629)
n = len(mcses)
for pct in (5, 10, 15, 20):
    k = max(1, round(n * pct / 100))
    subset = random.sample(mcses, k)
    with open(out_dir / f"mcs_subset_{pct}pct.json", "w", encoding="utf-8") as f:
        json.dump([sorted(m) for m in subset], f, indent=2)
    print(f"  {pct}% subset: {k} MCSes -> mcs_subset_{pct}pct.json", flush=True)
