# Explain-method experiments

Benchmarks for the MUS/MCS/MSS methods in [`cpmpy/tools/explain`](../cpmpy/tools/explain):
`umus`, `marco`, `hierarchical_marco`, and friends. This directory is a small,
modular pipeline so that running an experiment, adding a new one, and comparing results
across code changes are all easy — including on a bigger external machine.

> **Start here**, then see [`docs/pipeline.md`](docs/pipeline.md) (design),
> [`docs/instances.md`](docs/instances.md) (benchmark sources & formats) and
> [`docs/experiments.md`](docs/experiments.md) (how to add an experiment + ideas).

## The pipeline in one picture

```
(1) ACQUIRE        (2) LOAD               (3) RUN              (4) AGGREGATE   (5) REPORT
raw instance  -->  uniform Instance  -->  algorithm event  -->  canonical  -->  plots +
(xcsp3 /           .flat() (always)       stream (timed)        CSV rows       tables +
 synthetic /       .hierarchy() (when                                          SUMMARY.md +
 transcript)        meaningful)                                                INDEX.md
```

Every instance exposes a **flat** view `(soft, hard)`; hierarchical instances also
expose a **hierarchy** view `(root, hard)`. Because `ConstraintNode.all_constraints()`
flattens any hierarchy, the hierarchical transcripts and the flat XCSP3 instances move
through the same machinery (see [`docs/instances.md`](docs/instances.md)).

## Quick start

```bash
# 1. one-time setup
pip install -r requirements.txt           # or: make setup
python import_hierarchies.py              # copy transcript_* data into data/hierarchies/
                                          # (only needed where defense-rostering exists)

# 2. see what's available
python run.py --list                      # or: make list

# 3. run an experiment
python run.py hierarchical_runtime --instances transcript_2 --solver exact --map-solver exact
python run.py relevant_constraints --instances small medium --xcsp3-max-constraints 200
```

Outputs land in `experiment_outputs/<experiment>_<tag-or-timestamp>/`:

```
<run dir>/
  <experiment>_summary.csv      # one row per timed event (canonical schema)
  SUMMARY.md                    # benchmark, algorithms, settings, git commit, results table
  plots/
    <experiment>_<instance>.png            # metric vs time, one line per series
    <experiment>_<instance>_round<n>.png   # per refinement round (hierarchical only)
experiment_outputs/INDEX.md     # one appended line per run (overview of all runs)
```

## Running on an external machine

- **Self-contained data.** The transcript benchmarks are committed under
  `data/hierarchies/`, so no defense-rostering checkout is needed there.
  (Regenerate with `python import_hierarchies.py` where defense-rostering exists; the
  loader also falls back to `$DEFENSE_ROSTERING_DIR/input_data`.)
- **XCSP3 instances** are cached under `xcsp3_data/` (status in `sat_status_cache.json`,
  UNSAT files mirrored under `xcsp3_data/unsat/`). Classification is resumable:
  `python xcsp3_unsat_finder.py --year 2024 --track CSP`.
- **Long runs**: use `--tag NAME` for a stable output dir (clean re-run diffs), and
  background with `nohup make all > run.log 2>&1 &`. All progress prints are flushed.

## Tips & tricks

- **`exact` is much faster** than `ortools` as both solver and map_solver for the
  hierarchical experiment — it is the default there.
- **`--replot-only`** regenerates the plots from an existing run's CSV without rerunning.
- **`--skip <series>`** skips one side of a comparison (e.g. `--skip baseline`).
- **Comparing two runs**: `python compare_runs.py OLD/…_summary.csv NEW/…_summary.csv`
  overlays each instance's series (old dashed, new solid) — handy for seeing how a
  change moved the curves.
- **Provenance**: every `SUMMARY.md` records the git commit and available solvers, and
  `experiment_outputs/INDEX.md` keeps a one-line-per-run log, so a result can always be
  traced back to the code/config that produced it.

## Layout

| Path | What |
|---|---|
| `run.py` | single CLI entry point / dispatcher |
| `pipeline/` | the reusable framework (instances, algorithms, rows, plotting, reporting, runner) |
| `experiment_defs/` | one thin file per experiment (registered in `REGISTRY`) |
| `hierarchy_io.py`, `large_unsat_benchmark.py`, `xcsp3_unsat_finder.py` | benchmark sources |
| `import_hierarchies.py`, `compare_runs.py` | data import / run comparison utilities |
| `data/hierarchies/` | committed transcript exports (self-contained) |
| `xcsp3_data/` | downloaded + classified XCSP3 instances |
| `experiment_outputs/` | all run outputs + `INDEX.md` |
| `Makefile`, `requirements.txt` | convenience + dependencies |
| `docs/` | design, instance formats, how to add an experiment |
