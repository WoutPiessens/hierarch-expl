"""
    Hierarchy-shape ablation data for nurse/instance2:

      * nurse-hier1/instance2 : verbatim copy of nurse/instance2 (family -> nurse -> week -> day)
        -- rerun under a fresh label so both cases run on identical, current code;
      * nurse-hier2/instance2 : same constraints, REGROUPED hierarchy: for shift_on/shift_off,
        week and day take priority over nurse (family -> week -> day -> nurse-leaf); cover
        (already week -> day) unchanged.

    The oracle files are copied with every S entry renamed to the new leaf full names
    ("shift_on nurseA week0 day5" -> "shift_on week0 day5 nurseA"), so the SAME suitable sets
    drive both cases.

    Run once:  python make_nurse_hier2.py
"""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "nurse" / "instance2"


def leaves_with_path(nd, path=()):
    if not nd.get("children"):
        yield path + (nd["name"],), nd
    else:
        for c in nd["children"]:
            yield from leaves_with_path(c, path + (nd["name"],))


def regroup(spec):
    """family -> week -> day -> nurse(leaf) for shift_on/shift_off; cover unchanged.
    Returns (new_spec, {old_full_name: new_full_name})."""
    name_map = {}
    new_children = []
    for fam in spec["children"]:
        if fam["name"] not in ("shift_on", "shift_off"):
            new_children.append(fam)
            for path, _ in leaves_with_path(fam, (spec["name"],)):
                full = " ".join(path[1:])          # get_full_name skips the root
                name_map[full] = full
            continue
        # collect (week, day, nurse, leaf constraints); original: fam -> nurse -> week -> day
        entries = []
        for path, leaf in leaves_with_path(fam):
            _, nurse, week, day = path             # (fam, nurse, week, day)
            entries.append((week, day, nurse, leaf["constraints"]))
        weeks = {}
        for week, day, nurse, cons in entries:
            weeks.setdefault(week, {}).setdefault(day, []).append((nurse, cons))
        fam_node = {"name": fam["name"], "constraints": [], "children": []}
        for week in sorted(weeks):
            wk_node = {"name": week, "constraints": [], "children": []}
            for day in sorted(weeks[week], key=lambda d: int(d.replace("day", ""))):
                day_node = {"name": day, "constraints": [], "children": []}
                for nurse, cons in sorted(weeks[week][day]):
                    day_node["children"].append(
                        {"name": nurse, "constraints": list(cons), "children": []})
                    old = f'{fam["name"]} {nurse} {week} {day}'
                    new = f'{fam["name"]} {week} {day} {nurse}'
                    name_map[old] = new
                wk_node["children"].append(day_node)
            fam_node["children"].append(wk_node)
        new_children.append(fam_node)
    return {"name": spec["name"], "constraints": [], "children": new_children}, name_map


def main():
    spec = json.loads((SRC / "hierarchy.json").read_text(encoding="utf-8"))
    new_spec, name_map = regroup(spec)

    for label, out_spec, mapper in (("nurse-hier1", spec, None),
                                    ("nurse-hier2", new_spec, name_map)):
        dst = HERE / "data" / label / "instance2"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / "constraints.pkl", dst / "constraints.pkl")
        (dst / "hierarchy.json").write_text(json.dumps(out_spec), encoding="utf-8")
        for ofile in ("oracles_mss20.json", "oracles_random40.json"):
            oracles = json.loads((SRC / ofile).read_text(encoding="utf-8"))
            if mapper is not None:
                for o in oracles:
                    o["S"] = [mapper[nm] for nm in o["S"]]
            (dst / ofile).write_text(json.dumps(oracles, indent=2), encoding="utf-8")
        info = {"problem": label, "instance": "instance2",
                "source": "nurse/instance2" + ("" if mapper is None else
                          " regrouped: family->week->day->nurse")}
        (dst / "_info.json").write_text(json.dumps(info, indent=2))
        print(f"built {label}/instance2")

    # sanity: hier2 loads, leaf count matches, oracle names resolve
    import hierarchy
    root2, hard2 = hierarchy.load_instance("nurse-hier2", "instance2")
    names2 = set(hierarchy.leaf_names(root2))
    root1, _ = hierarchy.load_instance("nurse-hier1", "instance2")
    assert len(names2) == len(hierarchy.leaf_names(root1)), "leaf count mismatch"
    import oracles as orc
    for scheme in ("mss-20", "random-40"):
        for o in orc.load_oracles("nurse-hier2", "instance2", scheme):
            missing = [nm for nm in o["S"] if nm not in names2]
            assert not missing, f"unresolved S names: {missing[:3]}"
    print(f"sanity OK: {len(names2)} leaves, all oracle S names resolve in hier2")


if __name__ == "__main__":
    main()
