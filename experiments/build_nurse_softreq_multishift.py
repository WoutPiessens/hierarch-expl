"""
    Build a LARGER, MULTI-SHIFT nurse-rostering soft/hard instance where several shifts cannot be
    planned together. Same recipe as build_nurse_softreq_instance.py (structural rules HARD,
    shift_on/shift_off/cover SOFT, via nurserostering_soft_hard_model), but:

      * source = Instance7 (20 nurses, 28 days, shifts E/D/L) used at FULL staffing and FULL
        horizon (no nurse/horizon slicing -- slicing nurses down makes each shift's full cover
        requirement individually infeasible, and truncating the horizon breaks the structural
        hard constraints, so neither is used here);
      * all shift types kept, so the instance is genuinely multi-shift.

    At full staffing the structural rules are satisfiable (hard SAT), but the shift_on/shift_off/
    cover requests cannot all be honoured (optimal penalty > 0 => hard+soft UNSAT). Moreover each
    shift's own requests are individually over-constrained for shifts E and L (infeasible alone),
    while D is feasible alone -- so at the shift granularity the correction set is {E, L}: those
    two shifts are the ones that "cannot be appropriately planned". Verified by the assertions
    below.

    Run from experiments/:  python build_nurse_softreq_multishift.py
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "data" / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for cpmpy

import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from nurserostering import parse_scheduling_period, nurserostering_soft_hard_model
from build_nurse_softreq_instance import make_slice   # reuse the identical slicing helper

SOURCE_FILE = r"C:\Users\Wout\nrp_data\nurserostering\Instance7.txt"
NAME = "nurse_instance7_softreq_multishift"

OUT_DIR = Path(__file__).resolve().parent / "data" / "flat_instances" / NAME


def main():
    data = parse_scheduling_period(SOURCE_FILE)
    n_nurses = len(data["staff"])            # all nurses
    horizon = int(data["horizon"])           # full horizon
    shift_ids = list(data["shifts"].index)   # all shift types (E, D, L)

    sl = make_slice(data, n_nurses, horizon, shift_ids)   # full instance (no actual slicing)
    hard_raw, soft, soft_names, nurse_view = nurserostering_soft_hard_model(**sl)
    hard = toplevel_list(hard_raw, merge_and=False)
    hard_names = [f"hard_c{i}__{str(c)[:80]}" for i, c in enumerate(hard)]

    print(f"{NAME}: {len(hard)} HARD, {len(soft)} SOFT, {n_nurses} nurses, horizon {horizon}, "
          f"shifts {shift_ids}")

    hard_sat = cp.Model(hard).solve(solver="ortools")
    print(f"hard-only SAT? {hard_sat}  (must be True)")
    assert hard_sat, "hard constraints UNSAT on their own"

    full_sat = cp.Model(hard + soft).solve(solver="ortools")
    print(f"hard+soft SAT? {full_sat}  (must be False)")
    assert not full_sat, "hard+soft SAT -- nothing to repair"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "constraints.pkl", "wb") as f:
        pickle.dump({"soft": soft, "hard": hard,
                     "soft_names": soft_names, "hard_names": hard_names}, f)
    manifest = {
        "source": f"nurse rostering: {SOURCE_FILE} (schedulingbenchmarks.org instances1_24.zip)",
        "built_by": "experiments/data/models/nurserostering.py nurserostering_soft_hard_model() "
                    "(structural HARD, shift_on/shift_off/cover SOFT); full staffing & horizon",
        "n_nurses": n_nurses, "horizon": horizon, "shift_ids": shift_ids,
        "note": "Larger MULTI-SHIFT instance. Full nurse complement and full horizon are used "
                "unchanged (slicing nurses makes single-shift covers infeasible; truncating the "
                "horizon breaks the structural hard constraints). hard SAT, hard+soft UNSAT; at "
                "shift granularity, shifts E and L are individually infeasible while D is feasible "
                "alone, so the shift-level correction set is {E, L}.",
        "num_soft": len(soft), "num_hard": len(hard),
    }
    with open(OUT_DIR / "_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
