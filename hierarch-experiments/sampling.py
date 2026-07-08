"""
    Oracle sampling -- two SEPARATE schemes, kept apart on purpose so their results can be
    compared independently:

      * ``mss-20``    : 20 oracles at a 20% sampling rate, built with the **MSS strategy**.
                        S is a suitable set constructed AROUND a "natural" minimal correction set
                        (the complement of a randomly-grown Maximal Satisfiable Subset), padded to
                        the 20% budget with scattered leaves. Because S is built around a real
                        correction, it is feasible by construction (removing S restores SAT) even
                        though a purely random 20% subset almost never is.

      * ``random-40`` : 20 oracles at a 40% sampling rate, built with **random sampling**.
                        S is just k = 40% of the leaves drawn uniformly at random, re-drawn with
                        the next seed until it happens to be feasible (contains some correction).

    An "oracle" is a dict ``{"scheme","seed","pct","k","corr_size","S":[leaf names]}``. A method's
    oracle is defined entirely by its suitable set S: an MCS/correction is acceptable iff every
    constraint in it is in S.
"""
import random

import _bootstrap  # noqa: F401  -- puts the repo root (cpmpy/) on sys.path; must precede cpmpy
import cpmpy as cp
from cpmpy.tools.explain.utils import make_assump_model

GATE_SOLVER = "ortools"        # fast plain-SAT feasibility gate (the methods themselves use exact)


# --------------------------------------------------------------- feasibility ---
def _feasible(root, hard, S):
    """Does removing all of S restore satisfiability? (i.e. S contains a correction subset)."""
    kept = [lf.get_grouped_constraint() for lf in root.leaves()
            if lf.get_full_name() not in S]
    return cp.Model(list(hard) + [c for c in kept if c is not None]).solve(solver=GATE_SOLVER) is True


# ------------------------------------------------------- MSS-strategy sampling ---
def _mss_correction(root, hard, rng):
    """A 'natural' minimal correction set: grow a Maximal Satisfiable Subset of the soft leaves in
    random order (keep a leaf while the kept set stays SAT), then take the COMPLEMENT. This is
    exactly how MCS-enumeration would discover an MCS from a random seed, so it is a *typical*
    correction (not the globally-smallest one). SAT solves only."""
    leaves = root.leaves()
    names = [lf.get_full_name() for lf in leaves]
    soft = [lf.get_grouped_constraint() for lf in leaves]
    model, _soft2, assump = make_assump_model(soft, list(hard))
    s = cp.SolverLookup.get(GATE_SOLVER, model)
    order = list(range(len(assump)))
    rng.shuffle(order)
    kept = []
    for i in order:
        if s.solve(assumptions=[assump[j] for j in kept + [i]]) is True:
            kept.append(i)
    keptset = set(kept)
    return {names[i] for i in range(len(assump)) if i not in keptset}      # complement = MCS


def _mss_oracle(root, hard, k, rng):
    """Build one MSS-strategy suitable set of exactly k leaves: a natural correction C padded with
    random other leaves up to k. Returns ``(S, corr_size)`` or raises if C already exceeds k."""
    C = _mss_correction(root, hard, rng)
    if len(C) > k:
        raise ValueError(f"natural correction ({len(C)}) exceeds the {k}-leaf budget")
    others = [lf.get_full_name() for lf in root.leaves() if lf.get_full_name() not in C]
    pad = set(rng.sample(others, min(k - len(C), len(others))))
    return (C | pad), len(C)


# ------------------------------------------------------------ random sampling ---
def _random_oracle(root, hard, k, rng):
    """Draw k leaves uniformly at random. Returns ``S`` (feasibility checked by the caller)."""
    names = [lf.get_full_name() for lf in root.leaves()]
    return set(rng.sample(names, k))


# ---------------------------------------------------------------- public API ---
def sample_oracles(root, hard, scheme, pct, n, seed0=0, max_tries=500_000,
                   escalate_step=10, escalate_after=2000, max_pct=100, verbose=True):
    """Produce ``n`` feasible oracles for one scheme.

    :param scheme: ``"mss-20"`` (MSS strategy) or ``"random-40"`` (random sampling)
    :param pct:    STARTING sampling rate in percent (20 for mss, 40 for random)
    :param n:      how many oracles to produce

    RATE ESCALATION: some instances have large minimal corrections, so a random draw at the target
    rate is (almost) never feasible. If ``escalate_after`` consecutive draws fail to yield a
    feasible oracle, the rate is bumped by ``escalate_step`` (40 -> 50 -> 60 -> ...) and sampling
    continues -- so an easy instance stays at the target rate while a hard one climbs until it can
    be filled. Each oracle records the ACTUAL ``pct`` / ``k`` it was drawn at (so mixed rates within
    a set are visible), and the scheme label (e.g. ``random-40``) still names the *target*.

    :return: list of oracle dicts, each with a distinct ``seed``
    """
    n_leaves = len(root.leaves())
    oracles, seed, tries, misses, cur_pct = [], seed0, 0, 0, pct
    while len(oracles) < n and tries < max_tries:
        k = max(1, round(n_leaves * cur_pct / 100))
        rng = random.Random(seed); tries += 1; seed += 1
        ok, corr, S = False, None, None
        if scheme.startswith("mss"):
            try:
                S, corr = _mss_oracle(root, hard, k, rng)
                ok = _feasible(root, hard, S)                     # feasible by construction
            except ValueError:
                ok = False                                        # natural correction overshot k
        else:                                                     # random sampling
            S = _random_oracle(root, hard, k, rng)
            ok = _feasible(root, hard, S)
        if ok:
            oracles.append({"scheme": scheme, "seed": seed - 1, "pct": float(cur_pct), "k": k,
                            "corr_size": corr, "S": sorted(S)})
            misses = 0
            if verbose:
                print(f"  [{scheme}] {len(oracles)}/{n}  seed={seed - 1}  pct={cur_pct}  "
                      f"|S|={len(S)}  corr={corr}", flush=True)
        else:
            misses += 1
            if misses >= escalate_after and cur_pct + escalate_step <= max_pct:
                cur_pct += escalate_step
                misses = 0
                print(f"  [{scheme}] rate too hard -> escalating to {cur_pct}% "
                      f"(have {len(oracles)}/{n})", flush=True)
    if len(oracles) < n:
        print(f"  [WARN] only {len(oracles)}/{n} feasible {scheme} oracles found in {tries} tries "
              f"(reached {cur_pct}%)")
    return oracles
