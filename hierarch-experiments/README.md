# hierarch-experiments

Self-contained experiments for interactive constraint-relaxation via MUS/MCS explanation. This
folder holds the **finalized, good-to-go** code; use `../final_experiments` for scratch work.

## What it does

Two independent experiments:

1. **Method comparison** (`run.py`) — three methods, each driven by a *suitable set* `S` (the
   leaf constraints an oracle is willing to see relaxed):
   - **`mcs-enumeration`** — flat MARCO baseline; accept the first enumerated MCS ⊆ `S`.
   - **`selective-relaxation`** — staged deletion; delete the suitable constraints in every shown
     MCS until satisfiable (reports *excess* over a minimal correction).
   - **`hierarch-commit`** / **`hierarch-commit-nocap`** — incremental hierarchical MARCO: refine
     constraint groups and commit to group-level MCSes. Conceptually the simplest hierarchical
     strategy — **no frontier bound, no explore-backtrack** (a refine is only undone by undoing the
     commit that opened it). Two variants, evaluated separately, differing only in how many
     conflicts the oracle sees per round before it acts:
       - `hierarch-commit` — per-round cap of **20** group MUS/MCSes (the rest are found in later
         rounds); acts on the first 20.
       - `hierarch-commit-nocap` — **no cap**: every group MUS/MCS of the current frontier is
         enumerated before the oracle decides (thorough, but the per-round enumeration can explode
         on wide frontiers).

2. **Runtime experiment** (`runtime.py`) — replays a hierarch-commit decision script through
   **base MARCO** (re-enumerate from scratch each step) vs **incremental MARCO** (persistent map +
   core solver) and reports the speedup.

## Instances & oracles

One instance per problem: `nurse/instance2`, `thesis/defense-transcript4`,
`workforce/ews-instance103`. For each instance, **two separate oracle sets**:

| set | file | rate | how sampled |
|---|---|---|---|
| `mss-20` | `oracles_mss20.json` | 20% | **MSS strategy** — `S` built around a natural minimal correction (MSS-complement), padded to 20% |
| `random-40` | `oracles_random40.json` | 40% | **random sampling** — 40% of leaves drawn uniformly, re-drawn until feasible |

20 oracles each. Results are written to **separate CSVs** (`results/mss-20.csv`,
`results/random-40.csv`).

**Rate escalation:** if a target rate can't yield 20 feasible oracles for a hard instance (large
minimal correction ⇒ a random draw almost never contains a correction), the sampler bumps the rate
(40 → 50 → 60 → …) until it can fill the set. Each oracle records the actual `pct`/`k` it was drawn
at (so mixed rates are visible in the oracle files), while the scheme name keeps the target rate.

## Metrics (per CSV row)

`sampling, problem, instance, seed, method, decisions, judgments, relaxed, pruned, excess,
commits, backtracks, repaired, timed_out, elapsed_time`

- **decisions** — number of times the oracle makes a choice (one per MCS shown for the flat
  methods; one per commit / refine / backtrack for hierarch-commit).
- **judgments** — number of single-constraint suitability judgments (see `methods.py` header):
  flat methods judge every constraint in every shown MCS (`+= |MCS|`); hierarch-commit judges
  every member of a committed group-MCS (`+= |M|`) plus one per refine.
- **relaxed** — size of the correction found (primitive constraints relaxed).
- **pruned** — (hierarch-commit) number of primitive constraints inside the backgrounded groups.
- **excess** — (selective-relaxation) relaxed size minus a greedy-minimal correction.
- **commits / backtracks** — (hierarch-commit) counts.

## Running

```bash
# 0. one-off: build instances + sample the two oracle sets into data/
python build_data.py

# 1. method comparison (resumable; each cell in its own subprocess with a hard timeout)
python run.py                                  # everything, both schemes
python run.py --schemes mss-20                 # just the 20% MSS set
python run.py --problems nurse --methods hierarch-commit
python run.py --budget 600                     # per-cell solver budget (seconds)
python run.py --list                           # preview the cells

# 2. runtime experiment (base vs incremental MARCO)
python runtime.py --all
python runtime.py nurse instance2 0 mss-20 120
```

### On a remote server

Everything uses only this repo (Python + cpmpy, bundled at `../cpmpy`) and the `exact` solver.

```bash
cd hierarch-experiments
python build_data.py            # once (or copy a prebuilt data/ over)
python run.py --budget 600      # long sweep; stop/resume-safe -- rerun to continue
```

Each cell runs in an isolated subprocess and is hard-killed at `budget + 120s`, so a hung solve
never stalls the run. Finished cells are skipped on restart (results stream to the CSVs), so you
can `nohup`/`tmux` it and reconnect. To parallelize, run disjoint `--problems` / `--schemes` in
separate processes — they append to different rows/files safely.

## Files

| file | purpose |
|---|---|
| `hierarchy.py` | load `(root, hard)` from `data/`; leaf helpers |
| `sampling.py` | the two oracle-sampling schemes |
| `oracles.py` | load/save the two per-instance oracle files |
| `methods.py` | the three methods + metrics (the hierarch-commit oracle lives here) |
| `runtime.py` | base-vs-incremental MARCO timing |
| `run.py` | the sweep runner → per-scheme CSVs |
| `build_data.py` | one-off: copy instances + sample oracles into `data/` |
