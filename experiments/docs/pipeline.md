# Pipeline design

The pipeline turns a raw benchmark instance into plots/tables, in five stages. Each
stage is a data transformation, and each "axis that changes independently" lives in its
own module so a change in one rarely touches the others.

## Stages

```
(1) ACQUIRE raw instance
      XCSP3:        download + classify SAT/UNSAT  -> xcsp3_data/ + sat_status_cache.json
      synthetic:    build_graph_coloring_instance(params)
      hierarchical: defense-rostering export       -> constraints.pkl + hierarchy.json
(2) LOAD into a uniform Instance
      flat view:        (soft, hard)                       [always available]
      hierarchy view:   (root: ConstraintNode, hard)       [hierarchical sources only]
(3) RUN an algorithm  ->  timed event stream
      each event: {elapsed_seconds, kind, round?, metric, detail?}
(4) AGGREGATE events  ->  canonical rows  ->  CSV
(5) REPORT            ->  over-time plot + per-round plots + results table
                          + SUMMARY.md + append to experiment_outputs/INDEX.md
```

## Module map

| Module | Stage | Responsibility |
|---|---|---|
| `pipeline/instances.py` | 1–2 | `Instance` (with `.flat()` / `.hierarchy()`) + providers `transcript_instances`, `synthetic_instances`, `xcsp3_unsat_instances`. Wraps `hierarchy_io`, `large_unsat_benchmark`, `xcsp3_unsat_finder`. |
| `pipeline/algorithms.py` | 3 | Event-stream wrappers: `umus_events`, `marco_events`, `hierarchical_marco_events`, `baseline_marco_events`. Each `(instance, settings) -> list[event]`. |
| `pipeline/rows.py` | 4 | Canonical row schema, `events_to_rows`, `write_csv` / `read_csv`. |
| `pipeline/plotting.py` | 5 | Generic `plot_over_time` + `plot_per_round` (series-agnostic). |
| `pipeline/reporting.py` | 5 | `write_markdown_summary` (SUMMARY.md, with git/solver metadata) + `update_index` (INDEX.md). |
| `pipeline/runner.py` | all | `ExperimentSpec`, `Series`, and `run_experiment(spec)` — the instance×series driver. Nothing experiment-specific lives here. |
| `experiment_defs/<name>.py` | — | One thin file per experiment: builds an `ExperimentSpec`. |
| `run.py` | — | CLI dispatcher over `experiment_defs.REGISTRY`. |

## Why this shape

- **One canonical CSV schema** (see `pipeline/rows.py`) means plotting/reporting never
  branch on which experiment produced the data.
- **Algorithms are event streams.** A "series" is just a function that returns timed
  events; comparison is "run several series on the same instance." Adding a method to
  compare = adding one wrapper + listing it in a `Series`.
- **The runner is generic.** Instances × series, timing, CSV, plots, markdown, index —
  all driven by the `ExperimentSpec`. Adding an experiment never edits the framework.
- **`umus.py` / `hierarchical_marco.py` are untouched.** Their public APIs already
  provide what the wrappers need (`hierarchical_marco` yields the 1-indexed `round`,
  `umus(..., callback=)` reports CUMU growth).

## Event → metric conventions

- `relevant_constraints` (flat `marco`): `metric` is the size of the running union of
  constraint indices seen in any MUS/MCS so far. `umus`: `metric` is `len(CUMU)`
  straight from the callback.
- `hierarchical_runtime` (`metric_is_round_time=True`): one event per refinement round,
  whose `metric` is the round's enumeration time in seconds — from the first map-solver
  call of the round until the map solver returns UNSAT. Measured inside the algorithms
  via `hierarchical_marco(..., round_timings=...)` and `marco(..., enum_timing=...)`,
  so wall-clock noise from solver *construction* and instance loading is excluded. The
  runner then writes a per-round table (`ROUND_TIMES.md`).
- `round` is set only by the hierarchical wrappers; flat experiments leave it blank,
  which automatically disables per-round plots and the `#rounds` summary column.
