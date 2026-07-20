# Benchmark suite (360 cells)

18 instances = 3 problem classes x 6 instances; 20 mss-20 oracles each -> **360 cells**.
Difficulty spans the easy..hard band (mean natural-MCS in parentheses); the 6th instance
per class is the original hard benchmark.

```
data/
  <problem>-suite/            problem class: nurse | thesis | workforce
    <instance>/               one benchmark instance
      hierarchy.json          the constraint hierarchy (thesis: anonymised Person1..N)
      constraints.pkl         flat constraints + hard indices
      oracles_mss20.json      <-- the 20 oracles (canonical; what the loader reads)
      oracles/mss-20/         <-- browsable: one file PER oracle
        seed<N>.json          one oracle: {scheme, seed, pct, k, corr_size, S}
  suite_index.json            machine-readable manifest of all 360 cells
```

## nurse-suite
| instance | mean MCS | #oracles | oracle path |
|---|--:|--:|---|
| `instance4-k70-s1` | 7.2 | 20 | `data/nurse-suite/instance4-k70-s1/oracles_mss20.json` |
| `instance1-n6-s1` | 8.4 | 20 | `data/nurse-suite/instance1-n6-s1/oracles_mss20.json` |
| `instance1-n6-s2` | 10.0 | 20 | `data/nurse-suite/instance1-n6-s2/oracles_mss20.json` |
| `instance2-k90-s1` | 11.8 | 20 | `data/nurse-suite/instance2-k90-s1/oracles_mss20.json` |
| `instance2-k85-s1` | 14.2 | 20 | `data/nurse-suite/instance2-k85-s1/oracles_mss20.json` |
| `instance2` | 16.0 | 20 | `data/nurse-suite/instance2/oracles_mss20.json` |

## thesis-suite
| instance | mean MCS | #oracles | oracle path |
|---|--:|--:|---|
| `unsat-115-m1-2-3-4` | 7.2 | 20 | `data/thesis-suite/unsat-115-m1-2-3-4/oracles_mss20.json` |
| `unsat-117-m0-2-3-4-5-7` | 7.6 | 20 | `data/thesis-suite/unsat-117-m0-2-3-4-5-7/oracles_mss20.json` |
| `unsat-117-m0-2-3-4-5-7-9` | 8.6 | 20 | `data/thesis-suite/unsat-117-m0-2-3-4-5-7-9/oracles_mss20.json` |
| `unsat-115-m1-2-3-4-5` | 9.2 | 20 | `data/thesis-suite/unsat-115-m1-2-3-4-5/oracles_mss20.json` |
| `unsat-115-m1-2-3-4-5-7-9` | 11.2 | 20 | `data/thesis-suite/unsat-115-m1-2-3-4-5-7-9/oracles_mss20.json` |
| `defense-transcript4` | 15.4 | 20 | `data/thesis-suite/defense-transcript4/oracles_mss20.json` |

## workforce-suite
| instance | mean MCS | #oracles | oracle path |
|---|--:|--:|---|
| `ews-d11-o4` | 8.0 | 20 | `data/workforce-suite/ews-d11-o4/oracles_mss20.json` |
| `ews-d12-t90-s2` | 9.2 | 20 | `data/workforce-suite/ews-d12-t90-s2/oracles_mss20.json` |
| `ews-d12-t90-s1` | 12.8 | 20 | `data/workforce-suite/ews-d12-t90-s1/oracles_mss20.json` |
| `ews-d12-o1` | 16.4 | 20 | `data/workforce-suite/ews-d12-o1/oracles_mss20.json` |
| `ews-d12-o2` | 16.8 | 20 | `data/workforce-suite/ews-d12-o2/oracles_mss20.json` |
| `ews-instance103` | 16.8 | 20 | `data/workforce-suite/ews-instance103/oracles_mss20.json` |
