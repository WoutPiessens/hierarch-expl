"""
    Build a nurse-rostering flat instance in which the STRUCTURAL rostering rules are HARD
    (always enforced, never part of an MCS) and the shift-on / shift-off / cover requirements
    are SOFT (relaxable -- the MCS candidates). This is the mirror image of
    build_nurse_instance.py (which makes *everything* soft).

    It is the counterpart of build_nurse_instance.py, and reuses the very same slicing helper,
    but calls the NEW model function `nurserostering_soft_hard_model` (added to the copied model
    at experiments/data/models/nurserostering.py) instead of `nurserostering_model`.

    ====================================================================
    Why a different source instance than build_nurse_instance.py?
    ====================================================================
    build_nurse_instance.py deliberately picks Instance13, because Instance13 is one of the only
    3/24 real instances whose *hard* (structural) constraints are ALREADY infeasible on their
    own -- perfect when you want everything soft. That is exactly wrong here: if the structural
    constraints are to be HARD (must always hold), the instance's hard part must be SATISFIABLE,
    otherwise there is no feasible roster to repair towards and no soft MCS can ever restore
    satisfiability.

    We therefore use Instance1 (8 nurses / 14 days / a single shift type "D"), a "normal"
    instance whose hard constraints are satisfiable but whose optimal penalty is strictly
    positive (the original objective optimum for Instance1 is 607 > 0) -- i.e. the shift-on /
    shift-off / cover requests CANNOT all be met at once. Using all 8 nurses of Instance1 (the
    per-day cover requirements then genuinely conflict with each nurse's own structural limits),
    this yields:
        - hard (structural) alone: SAT
        - hard + soft (with all requests/cover): UNSAT
    with 40 soft constraints and ~5900 primitive MCSes. (A 2-nurse slice yields only 1 MCS and a
    6-nurse slice ~118; the full 8-nurse instance is the rich setting used for the experiments.)
    which is precisely the setting the oracle's baselines need: MCSes are minimal subsets of the
    SOFT requests/cover whose removal restores feasibility, and no hard constraint can ever
    appear in one.

    ====================================================================
    Changes relative to the source model
    ====================================================================
    The model function itself (`nurserostering_soft_hard_model` in data/models/nurserostering.py)
    is a faithful copy of `nurserostering_model` with only the hard/soft split changed (see that
    function's docstring). Here, as in build_nurse_instance.py, we only (a) restrict the INPUT
    DATA and (b) post-process the OUTPUT:

    1. NURSE SUBSET. First N_NURSES nurses in source order (no cherry-picking).
    2. HORIZON. Kept at the instance's FULL horizon (14 days). Unlike the all-soft build we do
       NOT shrink the horizon: truncating it makes even the structural constraints (min total
       minutes, min/max consecutive) infeasible, which would wrongly turn the HARD part UNSAT.
    3. SHIFT TYPES. Kept as all shift types of the instance (Instance1 has just one, "D").
    4. FLATTENING. The genuinely vectorized "cannot follow" HARD constraints are split into
       individual scalar constraints via toplevel_list(..., merge_and=False), same as the
       all-soft build, so "one hard constraint" is one concrete rule. The SOFT constraints are
       already scalar (one == / != / Count==) and are kept 1:1 with their names.
    5. HARD vs SOFT. Per the task: all structural rules are HARD; shift-on, shift-off and cover
       are SOFT (posted as plain constraints, NOT folded into an objective).

    Run from experiments/:  python build_nurse_softreq_instance.py
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "data" / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cpmpy`

import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from nurserostering import parse_scheduling_period, nurserostering_soft_hard_model

SOURCE_FILE = r"C:\Users\Wout\nrp_data\nurserostering\Instance1.txt"
N_NURSES = 8                 # all 8 nurses of Instance1
HORIZON = 14                 # Instance1's full horizon -- do NOT shrink (see docstring pt 2)
SHIFT_IDS = ["D"]            # Instance1's only shift type
NAME = "nurse_instance1_softreq_8nurses"

OUT_DIR = Path(__file__).resolve().parent / "data" / "flat_instances" / NAME


def make_slice(data, n_nurses, horizon, shift_ids):
    """Restrict nurses / horizon / shift types and drop now-dangling
    days_off/shift_on/shift_off/cover rows -- identical to build_nurse_instance.make_slice."""
    shifts = data["shifts"].loc[shift_ids].copy()
    shifts["cannot follow"] = shifts["cannot follow"].apply(
        lambda lst: [s for s in lst if s in shift_ids])

    staff = data["staff"].iloc[:n_nurses].reset_index(drop=True)
    eids = set(staff["ID"])

    do = data["days_off"]
    do = do[(do["EmployeeID"].isin(eids)) & (do["DayIndex"] < horizon)].reset_index(drop=True)
    so = data["shift_on"]
    so = so[(so["EmployeeID"].isin(eids)) & (so["Day"] < horizon) & (so["ShiftID"].isin(shift_ids))].reset_index(drop=True)
    sf = data["shift_off"]
    sf = sf[(sf["EmployeeID"].isin(eids)) & (sf["Day"] < horizon) & (sf["ShiftID"].isin(shift_ids))].reset_index(drop=True)
    cov = data["cover"]
    cov = cov[(cov["Day"] < horizon) & (cov["ShiftID"].isin(shift_ids))].reset_index(drop=True)

    return dict(horizon=horizon, shifts=shifts, staff=staff,
                days_off=do, shift_on=so, shift_off=sf, cover=cov)


def main():
    data = parse_scheduling_period(SOURCE_FILE)
    sl = make_slice(data, N_NURSES, HORIZON, SHIFT_IDS)
    nurse_ids = list(sl["staff"]["ID"])

    hard_raw, soft, soft_names, nurse_view = nurserostering_soft_hard_model(**sl)
    # step 4: flatten the vectorized cannot-follow HARD constraints into scalar atoms.
    hard = toplevel_list(hard_raw, merge_and=False)
    hard_names = [f"hard_c{i}__{str(c)[:80]}" for i, c in enumerate(hard)]

    print(f"{NAME}: {len(hard)} HARD (structural), {len(soft)} SOFT (shift_on/off + cover), "
          f"nurses={nurse_ids}, horizon={HORIZON}, shifts={SHIFT_IDS}")

    # Sanity: the hard part MUST be feasible on its own (else no roster to repair towards),
    # and hard+soft MUST be infeasible (else there is nothing to explain / repair).
    hard_sat = cp.Model(hard).solve(solver="ortools")
    print(f"hard-only SAT? {hard_sat}  (must be True -- structural rules must be satisfiable)")
    assert hard_sat, "hard constraints are UNSAT on their own -- pick a different instance/slice"

    full_sat = cp.Model(hard + soft).solve(solver="ortools")
    print(f"hard+soft SAT? {full_sat}  (must be False -- otherwise nothing to repair)")
    assert not full_sat, "hard+soft is SAT -- requests don't conflict; pick a bigger slice"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "constraints.pkl", "wb") as f:
        pickle.dump({"soft": soft, "hard": hard,
                     "soft_names": soft_names, "hard_names": hard_names}, f)
    manifest = {
        "source": f"nurse rostering: {SOURCE_FILE} (schedulingbenchmarks.org instances1_24.zip)",
        "built_by": "examples/nurserostering.py -> copied & extended to "
                    "experiments/data/models/nurserostering.py: nurserostering_soft_hard_model() "
                    "(structural rules HARD, shift_on/shift_off/cover SOFT -- no objective)",
        "nurse_ids": nurse_ids,
        "n_nurses": N_NURSES,
        "horizon": HORIZON,
        "shift_ids": SHIFT_IDS,
        "hard_soft_split": {
            "hard": "structural rostering rules (cannot-follow, per-type max shifts, min/max "
                    "total minutes, max/min consecutive shifts, min consecutive days off, max "
                    "weekends, fixed days-off) -- always enforced, never in an MCS",
            "soft": "shift-on requests (nurse works shift S on day d), shift-off requests (nurse "
                    "avoids shift S on day d), and cover requirements (Count(day,shift)==req, "
                    "posted WITHOUT slack vars) -- relaxable, the MCS candidates. Originally "
                    "these were weighted penalty terms in nurserostering_model's objective.",
        },
        "transformations": [
            "1. nurse subset: first N_NURSES nurses in source order (no cherry-picking)",
            "2. horizon: kept at the instance's FULL horizon (not shrunk -- truncating it makes "
            "the structural HARD constraints infeasible)",
            "3. shift types: all shift types of the instance",
            "4. flattening: toplevel_list(hard, merge_and=False) splits vectorized cannot-follow "
            "HARD constraints into scalar atoms; soft constraints are already scalar",
            "5. hard/soft: structural rules HARD, shift_on/shift_off/cover SOFT (no objective) "
            "-- per task instructions",
        ],
        "note": "Instance1 chosen (not Instance13, used by build_nurse_instance.py) BECAUSE its "
                "hard part is satisfiable while its optimal penalty is > 0 (607), so hard+soft "
                "is UNSAT. Both properties are re-verified by assertions in this builder.",
        "num_soft": len(soft),
        "num_hard": len(hard),
    }
    with open(OUT_DIR / "_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
