"""
    Oracle storage. The two sampling schemes are kept in SEPARATE files per instance so their
    experimental results stay separate:

        data/<problem>/<instance>/oracles_mss20.json     -- 20 oracles, 20% MSS strategy
        data/<problem>/<instance>/oracles_random40.json  -- 20 oracles, 40% random sampling
"""
import json

from hierarchy import instance_dir

# scheme -> (sampling rate %, number of oracles, on-disk filename)
SCHEMES = {
    "mss-20":    {"pct": 20, "n": 20, "file": "oracles_mss20.json"},
    "random-40": {"pct": 40, "n": 20, "file": "oracles_random40.json"},
    # hierarchy-friendly: S = union of randomly-selected whole subtrees, grown until feasible
    "hiernode":  {"pct": None, "n": 20, "file": "oracles_hiernode.json"},
    # graded: S = a natural minimal correction, stratified by size (minimal-MCS) into difficulty bins
    "graded":    {"pct": None, "n": None, "file": "oracles_graded.json"},
}


def oracles_path(problem, instance, scheme):
    return instance_dir(problem, instance) / SCHEMES[scheme]["file"]


def load_oracles(problem, instance, scheme):
    p = oracles_path(problem, instance, scheme)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def save_oracles(problem, instance, scheme, oracle_list):
    p = oracles_path(problem, instance, scheme)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(oracle_list, indent=2), encoding="utf-8")
    print(f"  saved {len(oracle_list)} '{scheme}' oracles -> {p}")
