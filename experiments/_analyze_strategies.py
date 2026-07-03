"""Aggregate /tmp/analysis.log: per-strategy query stats, dead-end rates, and the feature
comparison (size / suitable-leaf overlap of chosen commits) that explains first-vs-random.
Usage: python _analyze_strategies.py <logfile>"""
import re, statistics, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/analysis.log"
results, commits = [], []
for line in open(path, encoding="utf-8", errors="replace"):
    if line.startswith("RESULT"):
        p = line.split()
        if p[4] == "TIMEOUT" or len(p) < 6:
            results.append({"strat": p[1], "seed": p[2], "rep": p[3], "result": "timeout"})
            continue
        kv = dict(x.split("=") for x in p[5:] if "=" in x)
        results.append({"strat": p[1], "seed": p[2], "rep": p[3], "result": p[4],
                        **{k: float(v.rstrip("s")) for k, v in kv.items()}})
    elif line.startswith("COMMIT"):
        p = line.split()
        kv = dict(x.split("=") for x in p[4:])
        commits.append({"strat": p[1], "seed": p[2], "rep": p[3],
                        **{k: int(v) for k, v in kv.items()}})

STRATS = ["first", "random", "max-overlap", "lookahead"]

print("=== per-strategy over all (seed, rep) runs ===")
print(f"{'strategy':>12} {'n':>3} {'ok':>3} {'queries mean±sd':>17} {'median':>7} {'backtr':>7} {'|relax|':>8}")
for s in STRATS:
    rs = [r for r in results if r["strat"] == s]
    ok = [r for r in rs if r["result"] == "repaired"]
    if not ok:
        print(f"{s:>12} {len(rs):>3}   0"); continue
    q = [r["q"] for r in ok]
    print(f"{s:>12} {len(rs):>3} {len(ok):>3} {statistics.mean(q):>9.1f}±{statistics.pstdev(q):<6.1f} "
          f"{statistics.median(q):>7.1f} {statistics.mean(r['b'] for r in ok):>7.2f} "
          f"{statistics.mean(r['relax'] for r in ok):>8.2f}")

print("\n=== commit decisions: features of the CHOSEN option, and dead-end rate ===")
print(f"{'strategy':>12} {'#commits':>9} {'dead rate':>10} {'mean |M|':>9} {'mean overlap':>13} {'mean opts':>10}")
for s in STRATS:
    cs = [c for c in commits if c["strat"] == s]
    if not cs:
        continue
    print(f"{s:>12} {len(cs):>9} {sum(c['dead'] for c in cs)/len(cs):>10.2f} "
          f"{statistics.mean(c['size'] for c in cs):>9.2f} "
          f"{statistics.mean(c['overlap'] for c in cs):>13.2f} "
          f"{statistics.mean(c['opts'] for c in cs):>10.2f}")

print("\n=== dead-end rate by overlap (all strategies pooled; only decisions with >1 option) ===")
pool = [c for c in commits if c["opts"] > 1]
buckets = defaultdict(list)
for c in pool:
    b = "1-2" if c["overlap"] <= 2 else "3-4" if c["overlap"] <= 4 else "5-7" if c["overlap"] <= 7 else "8+"
    buckets[b].append(c["dead"])
for b in ["1-2", "3-4", "5-7", "8+"]:
    if buckets[b]:
        print(f"  overlap {b:>4}: dead rate {sum(buckets[b])/len(buckets[b]):.2f}  (n={len(buckets[b])})")

print("\n=== first vs random: when multiple options existed, what did they choose? ===")
for s in ["first", "random"]:
    cs = [c for c in commits if c["strat"] == s and c["opts"] > 1]
    if cs:
        print(f"  {s:>7}: n={len(cs):>3}  mean overlap of chosen M = "
              f"{statistics.mean(c['overlap'] for c in cs):.2f}  dead rate = "
              f"{sum(c['dead'] for c in cs)/len(cs):.2f}")
