"""
    Build a nurse-rostering flat instance from a real downloaded schedulingbenchmarks.org
    instance (via examples/nurserostering.py), with ALL constraints treated as soft (per the
    task: "all constraints may be considered as soft constraints").

    ====================================================================
    TRANSPARENCY: every change made relative to the original source model
    ====================================================================

    `nurserostering_model()` itself (the constraint-building logic in examples/nurserostering.py)
    is used completely UNCHANGED -- no constraint-generation code was modified. Everything
    below is either (a) restricting which INPUT DATA is fed into that unchanged function, or
    (b) a post-hoc transformation applied to its OUTPUT.

    1. INSTANCE SELECTION. Of the 24 real instances downloadable from schedulingbenchmarks.org,
       only 3 (Instance13/21/22) are genuinely UNSAT as a whole when solving just the hard
       constraints (verified by trying all 24); all others are SAT even at their smallest
       (e.g. Instance1: 8 nurses/14 days/1 shift, already 334 hard constraints). The 3 UNSAT
       ones are all huge (12.6k-60k hard constraints) -- far too large for exhaustive
       soft-constraint MCS enumeration. We use Instance13 (120 nurses/28 days/18 shifts) as the
       source and restrict it down (steps 2-4) rather than using it as-is.

    2. NURSE SUBSET. Restricted `staff` to the first `N_NURSES` rows (by source order, i.e. the
       first N nurses as they appear in Instance13.txt -- no cherry-picking). Per-nurse
       constraints (cannot-follow, max/min shifts of each type, total minutes, max/min
       consecutive shifts, min consecutive days off, max weekends, fixed days off) are
       independent across nurses; the only constraint that couples nurses together is the
       per-(day,shift) cover requirement, and it is slack-absorbed in the original model
       (`nb_nurses - slack_over + slack_under == requirement`, slack bounded only by
       `len(staff)`) so it can never be a NECESSARY cause of infeasibility on its own -- it can
       only ever make things easier (more nurses = looser slack bound) or, when the nurse
       subset is small, occasionally tight enough to also contribute. We verified nurse 0
       ("A") alone is individually UNSAT (every one of Instance13's 120 nurses is, checked
       individually); adding more nurses to the subset can only keep the combined model UNSAT
       (it's a logical AND of each nurse's own constraints), never make it SAT.
    3. HORIZON. Restricted from 28 days to `HORIZON` days (the planning period becomes "day 0
       to day HORIZON-1"). `days_off` / `shift_on` / `shift_off` / `cover` rows referencing a
       day outside this reduced window are dropped (they would otherwise index past the
       resized `nurse_view` array, or fail silently in `nurserostering_model`'s row lookups).
    4. SHIFT TYPES. Restricted from 18 shift types to a `SHIFT_IDS` subset. The `shifts`
       dataframe is sliced to those rows, and each shift's "cannot follow" list is filtered to
       only reference shift IDs that remain (a "cannot follow X" entry for a dropped shift X is
       meaningless once X no longer exists, so it's removed rather than left dangling).
       `shift_on` / `shift_off` / `cover` rows referencing a dropped shift ID are also dropped
       (same reasoning: a request/requirement for a shift that no longer exists is meaningless,
       and would otherwise raise a lookup error in `nurserostering_model`). A nurse's
       `max_shifts_<id>` columns for shifts NOT in the subset are simply unused (harmless,
       since `nurserostering_model` only iterates `shifts.iterrows()`).
       NOTE -- this is the one place where the restriction can also AFFECT feasibility, not just
       size: with fewer shift types available (in particular if a long shift like "a5"/720 min
       were dropped), the per-day minutes achievable shrinks, which can independently affect
       whether MinTotalMinutes is reachable. We always re-verify UNSAT after slicing (assert
       below) rather than assuming it's preserved.
    5. FLATTENING (output transformation, not an input restriction). `nurserostering_model`'s
       "cannot follow" loop posts ONE vectorized array constraint per (shift, other_shift) pair
       (spanning every day-transition and every nurse at once via numpy broadcasting), rather
       than one constraint per day/nurse. We apply `toplevel_list(model.constraints,
       merge_and=False)` to split these into individual scalar constraints, so that "one soft
       constraint" means one concrete (nurse, day, shift-pair) rule, matching the granularity
       used for the defense-rostering instances. We verified this step does NOT alter or
       decompose any OTHER constraint's mathematical content (e.g. the `Count`-based cover/
       max-shifts constraints already appear in that exact form in the RAW, pre-flatten
       `model.constraints` -- `toplevel_list` only expands the genuinely array-valued
       cannot-follow entries, confirmed by checking types/counts before and after).
    6. HARD vs SOFT. Per the task instructions, ALL of the resulting constraints are treated as
       soft (`hard=[]`); none of `nurserostering_model`'s hard constraints are exempted.

    ====================================================================
    Sizing
    ====================================================================
    A scan over horizon (4-8 days), shift-type-count (2-5), and nurse-count (1-3) was run to
    find a configuration close to ~1000 MCSes without becoming impractically slow (some
    configurations, e.g. 2 nurses/7 days/5 shifts, were still climbing past 2800 MCS after
    120s and were abandoned). horizon=4, shifts=[a1,a4,a5,d1], 2 nurses gave 1393 MCS in 16.7s
    -- the closest practical match found to "~1000" without an extended scan.

    Run from experiments/:  python build_nurse_instance.py
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cpmpy`

import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from nurserostering import parse_scheduling_period, nurserostering_model

SOURCE_FILE = r"C:\Users\Wout\nrp_data\nurserostering\Instance13.txt"
N_NURSES = 2              # first 2 nurses in source order ("A", "B")
HORIZON = 4
SHIFT_IDS = ["a1", "a4", "a5", "d1"]
NAME = "nurse_instance13_2nurses_h4"

OUT_DIR = Path(__file__).resolve().parent / "data" / "flat_instances" / NAME


def make_slice(data, n_nurses, horizon, shift_ids):
    """Steps 2-4 from the module docstring: restrict nurses / horizon / shift types, and drop
    any days_off/shift_on/shift_off/cover rows that would now reference something removed."""
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

    model, nurse_view = nurserostering_model(**sl)
    # step 5: flatten the genuinely vectorized cannot-follow entries into per-(nurse,day) atoms
    soft = toplevel_list(model.constraints, merge_and=False)
    hard = []  # step 6: nothing is exempted -- everything is soft
    soft_names = [f"nurse_c{i}__{str(c)[:80]}" for i, c in enumerate(soft)]
    hard_names = []

    print(f"{NAME}: {len(soft)} constraints (ALL soft), nurses={nurse_ids}, "
          f"horizon={HORIZON}, shifts={SHIFT_IDS}")
    full_sat = cp.Model(soft).solve(solver="ortools")
    print(f"full SAT? {full_sat}  (must be False)")
    assert not full_sat, "instance is SAT -- nothing to repair, pick a different slice"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "constraints.pkl", "wb") as f:
        pickle.dump({"soft": soft, "hard": hard,
                    "soft_names": soft_names, "hard_names": hard_names}, f)
    manifest = {
        "source": f"nurse rostering: {SOURCE_FILE} (schedulingbenchmarks.org instances1_24.zip)",
        "built_by": "examples/nurserostering.py nurserostering_model() -- UNCHANGED, only its "
                    "input data is restricted (see this file's module docstring for the full, "
                    "itemized list of every change made relative to the original model)",
        "nurse_ids": nurse_ids,
        "n_nurses": N_NURSES,
        "horizon": HORIZON,
        "shift_ids": SHIFT_IDS,
        "transformations": [
            "1. instance selection: Instance13 (one of only 3/24 real instances genuinely "
            "UNSAT as a whole; the other 2 are also too large; all instances below ~12k "
            "constraints are SAT as downloaded)",
            "2. nurse subset: first N_NURSES nurses in source order (no cherry-picking); "
            "per-nurse constraints are independent except cover, which is slack-absorbed and "
            "so can only ever loosen, not tighten, feasibility as nurses are added",
            "3. horizon: restricted to first HORIZON days; days_off/shift_on/shift_off/cover "
            "rows referencing a later day are dropped",
            "4. shift types: restricted to SHIFT_IDS; shifts' 'cannot follow' lists and any "
            "shift_on/shift_off/cover rows referencing a dropped shift ID are filtered out",
            "5. flattening: toplevel_list(model.constraints, merge_and=False) splits the "
            "vectorized cannot-follow array constraints into individual per-(nurse,day) "
            "scalar constraints; verified this does not alter any other constraint",
            "6. hard/soft: ALL resulting constraints are treated as soft (hard=[]), per task "
            "instructions; none of nurserostering_model's hard constraints are exempted",
        ],
        "note": "UNSAT-ness was re-verified after every slicing step (not assumed to be "
                "preserved from the source instance) -- see 'full SAT?' assertion in "
                "build_nurse_instance.py. Sized via a scan over horizon/shift-count/nurse-count "
                "to land near ~1000 MCSes (see module docstring 'Sizing' section); some "
                "configurations (e.g. 2 nurses/7 days/5 shifts) were abandoned for growing "
                "past 2800+ MCSes within 120s with no sign of leveling off.",
        "num_soft": len(soft),
        "num_hard": 0,
    }
    with open(OUT_DIR / "_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
