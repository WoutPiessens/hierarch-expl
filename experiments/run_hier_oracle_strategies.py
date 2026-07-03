"""
    Log the hierarchical oracle's decisions and compare COMMIT STRATEGIES on the nurse
    soft/hard instance (nurse_instance1_softreq_8nurses), sampling rate 40%.

    Terminology (used in the output table):
      query        -- one oracle DECISION: a commit, a refine, or a backtrack. queries ==
                      n_commit + n_refine + n_backtrack. (The final "stop" is not counted.)
      mcs_enum     -- number of distinct group-level MCSes (GMCSes) hierarchical_marco enumerated
                      and showed to the oracle from the start of the run until full repair
                      (the map solver blocks each one globally, so each is enumerated exactly once).
      mus_enum     -- same for group-level MUSes (enumerated to steer the map solver; the oracle
                      also uses them for its relevance filter).
      relax_q      -- total primitive constraints relaxed by commits (one per suitable leaf in a
                      committed GMCS); the "one constraint relaxed per query" weighted count is
                      n_refine + n_backtrack + relax_q.
      lookahead    -- internal SAT checks the lookahead strategy performs to validate a commit
                      candidate (oracle-side reasoning, not interaction queries).
      |relax|      -- size of the final correction subset (relaxed suitable leaves).

    Strategies (which committable state-relative GMCS to commit):
      first      -- first-fit in discovery order (the original policy)
      random     -- uniform among the committable options (randomness as a tie-breaker)
      pure-leaf  -- prefer the smallest GMCS consisting ONLY of suitable leaves; committing such
                    an M relaxes a complete state-relative correction set at once, so it can
                    never dead-end into a backtrack. Falls back to first-fit if none is pure.
      max-leaf   -- the option with the most suitable leaf members (max immediate progress).
      lookahead  -- prune options that provably cannot complete a repair (after committing M all
                    other open groups become background, so every future relaxation must come
                    from M's subtrees: one SAT check on "current relaxed + all suitable leaves
                    under M" decides it); among survivors pick the tightest branch. The internal
                    look-ahead solves are counted separately (they are oracle-side reasoning,
                    not queries).

    Run from experiments/:  python run_hier_oracle_strategies.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmpy.tools.explain import hierarchical_marco
from oracle import build_nurse_hierarchy, sample_suitable_set, run_hierarchical_oracle

INSTANCE = "nurse_instance1_softreq_8nurses"
PCT = 40
SAMPLE_SEEDS = [48, 53, 73, 82, 95]        # known-feasible 40% samples (searched earlier)
STRATEGIES = ["first", "random", "pure-leaf", "max-leaf", "lookahead"]
MAX_STEPS = 2000
OUT = Path(__file__).resolve().parent / "experiment_outputs" / f"hier_oracle_strategies_{INSTANCE}.json"


def main():
    root, hard = build_nurse_hierarchy(INSTANCE)
    print(f"{INSTANCE}: {len(hard)} hard, {len(root.leaves())} leaf soft; sampling rate {PCT}%\n",
          flush=True)

    # ---- 1. one fully-logged run so every choice is visible --------------------------------
    S, sd = sample_suitable_set(root, hard, pct=PCT, seed0=SAMPLE_SEEDS[0])
    print(f"=== DETAILED DECISION LOG  (sample seed {sd}, |S|={len(S)}, strategy=lookahead) ===",
          flush=True)
    print("S =", sorted(S), "\n", flush=True)
    r = run_hierarchical_oracle(root, hard, S, seed=sd, max_steps=MAX_STEPS,
                                commit_strategy="lookahead", verbose=True)
    print(f"\n--> {r['result'].upper()}: relaxed {r['n_relaxed']} {r['relaxed']}\n"
          f"    queries={r['n_queries']} decisions={r['n_decisions']} "
          f"(c={r['n_commit']} r={r['n_refine']} b={r['n_backtrack']}) "
          f"lookahead_solves={r['n_lookahead_solves']}\n", flush=True)

    # ---- 2. strategy comparison over the feasible samples ----------------------------------
    print("=== STRATEGY COMPARISON  (query = one oracle decision: commit/refine/backtrack) ===",
          flush=True)
    print(f"{'strategy':>10} {'seed':>5} {'result':>9} {'queries':>8} {'commit':>7} {'refine':>7} "
          f"{'backtr':>7} {'mcs_enum':>9} {'mus_enum':>9} {'relax_q':>8} {'|relax|':>8} {'time':>6}",
          flush=True)
    rows = []
    for strat in STRATEGIES:
        for s0 in SAMPLE_SEEDS:
            S, sd = sample_suitable_set(root, hard, pct=PCT, seed0=s0)
            if sd != s0:      # only use the exact pre-verified seeds (comparability)
                continue
            log = []
            t0 = time.perf_counter()
            r = run_hierarchical_oracle(root, hard, S, seed=sd, max_steps=MAX_STEPS,
                                        commit_strategy=strat, log=log)
            dt = time.perf_counter() - t0
            rows.append({"strategy": strat, "seed": sd, **r, "seconds": dt, "log": log})
            print(f"{strat:>10} {sd:>5} {r['result']:>9} {r['n_decisions']:>8} {r['n_commit']:>7} "
                  f"{r['n_refine']:>7} {r['n_backtrack']:>7} {r['n_gmcs_seen']:>9} "
                  f"{r['n_gmus_seen']:>9} {r['n_queries']:>8} {r['n_relaxed']:>8} {dt:>5.0f}s",
                  flush=True)

    # ---- 3. aggregate ----------------------------------------------------------------------
    print("\n=== MEAN over seeds (repaired runs only) ===", flush=True)
    print(f"{'strategy':>10} {'n_ok':>5} {'queries':>9} {'mcs_enum':>9} {'mus_enum':>9} "
          f"{'backtracks':>11} {'|relax|':>8} {'time':>6}", flush=True)
    for strat in STRATEGIES:
        ok = [x for x in rows if x["strategy"] == strat and x["result"] == "repaired"]
        if not ok:
            print(f"{strat:>10}     0        --", flush=True)
            continue
        m = lambda k: statistics.mean(x[k] for x in ok)
        print(f"{strat:>10} {len(ok):>5} {m('n_decisions'):>9.1f} {m('n_gmcs_seen'):>9.1f} "
              f"{m('n_gmus_seen'):>9.1f} {m('n_backtrack'):>11.1f} {m('n_relaxed'):>8.1f} "
              f"{m('seconds'):>5.0f}s", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"instance": INSTANCE, "pct": PCT, "sample_seeds": SAMPLE_SEEDS,
                               "strategies": STRATEGIES, "rows": rows}, indent=2))
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
