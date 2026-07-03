"""
    Interactive terminal driver to verify the commit-enabled `hierarchical_marco` by hand.

    Pick an instance with ``--instance`` (config option): ``defense`` (the first defense-scheduling
    instance, transcript_1) or ``nurse`` (the nurse rostering soft/hard model).

    MUSes, MCSes and relevant open groups are numbered SEPARATELY (each category counts from 0),
    since you commit an MCS by its number and refine a constraint by its number. Numbers are
    persistent: they accumulate across refinement rounds within a commitment epoch -- so an
    MCS/MUS found earlier (now blocked from re-appearing) stays listed and selectable by its
    number -- and reset to 0 after a commit; `back` restores the epoch's numbering. `*` marks
    items newly found in the current round. Commands:

        refine <#>              refine that open group (by its number) down ONE level (the default)
        refine <#>,<#>,...      refine several open groups at once (each one level down)
        refine <#>[,...] <L>    ...or send them all to level L
        commit <#>              commit to that MCS (by its number)
        commit <name; name>     commit to an explicit set of open group full-names
        open                    list ALL open groups (incl. those not in any MUS/MCS)
        bg                      list the (hidden) background groups
        back                    undo the last COMMIT (and any refines after it)
        stop                    end

    On commit, every active group in the chosen MCS that is a *leaf* (no children) is relaxed,
    every other active group becomes background, and non-leaf MCS members stay free to refine.

    Backtracking. We snapshot the state (frontier + committed sets) before each COMMIT. ``back``
    restores the last snapshot in-place -- no re-enumeration -- undoing the last commit and any
    refines made after it. The persistent map solver keeps all its blocking clauses; those are
    inert on their own (they only block together with the map-solver assumptions), and any clause
    from an abandoned finer branch references now-inactive nodes whose up/down variables are
    unlinked (free), so it can never wrongly block. Only the assumptions -- rebuilt from the
    restored sets -- change.

    Run from extra/:
        python verify_commit.py --instance nurse
        python verify_commit.py --instance defense
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for cpmpy

import cpmpy as cp
from cpmpy.tools.explain import hierarchical_marco

from instances import build


def _fmt_set(names):
    return "{" + ", ".join(names) + "}"


def _open_nodes(ctx):
    """Active groups that are neither relaxed nor background -- the ones you can refine/commit."""
    hidden = set(ctx["committed_relaxed"]) | set(ctx["committed_background"])
    return [nd for nd in ctx["frontier_nodes"] if nd.get_full_name() not in hidden]


def _report_repair(ctx, hard, name2node):
    """When no OPEN groups remain (every group is relaxed or background), the repair is done.
    Validity is checked at the LEAF level: hard + (every soft leaf that is NOT relaxed). Using
    leaves (not background-group conjunctions) is robust even if a relaxed leaf also lies under
    some background group."""
    relaxed_leaves = set()
    for n in ctx["committed_relaxed"]:
        relaxed_leaves.update(lf.get_full_name() for lf in name2node[n].leaves())
    kept = [nd.get_grouped_constraint() for nd in name2node.values()
            if not nd.children and nd.get_full_name() not in relaxed_leaves]
    kept = [c for c in kept if c is not None]
    sat = cp.Model(hard + kept).solve(solver="ortools")
    print("\n" + "=" * 60)
    if sat:
        print("  ==>  REPAIR COMPLETE  <==")
        print("  Every constraint group is now relaxed or background, and the accepted")
        print("  constraints (hard + all non-relaxed soft) are satisfiable.")
        print(f"  Relaxed to repair ({len(ctx['committed_relaxed'])}): "
              f"{_fmt_set(sorted(ctx['committed_relaxed']))}")
    else:
        print("  ==>  no OPEN groups remain, but relaxing only these leaves does NOT restore")
        print("       satisfiability -- NOT a valid repair.")
    print("=" * 60)


def make_decide(state_stack, hard, name2node):
    """decide_step callback with a persistently-numbered registry of MUSes/MCSes/open groups.

    Numbers are unique and stable WITHIN a commitment epoch: MUSes, MCSes and relevant open
    groups accumulate across refinement rounds (so an MCS/MUS that was found earlier and is now
    blocked stays listed and selectable by its number). Numbering resets to 0 on a commit; `back`
    undoes the last commit and restores that epoch's registry, so its options are selectable
    again. `commit <n>` takes an MCS number, `refine <n>` an open-group number.
    """
    # per-epoch registry with SEPARATE numbering per category (you commit an MCS by its number
    # and refine a relevant constraint by its number, so the two number spaces are independent).
    reg = {"mcs_n": 0, "mus_n": 0, "grp_n": 0, "mcs": [], "mus": [], "groupnum": {}}

    def reg_snapshot():
        return {"mcs_n": reg["mcs_n"], "mus_n": reg["mus_n"], "grp_n": reg["grp_n"],
                "mcs": list(reg["mcs"]), "mus": list(reg["mus"]), "groupnum": dict(reg["groupnum"])}

    def reg_restore(s):
        reg["mcs_n"], reg["mus_n"], reg["grp_n"] = s["mcs_n"], s["mus_n"], s["grp_n"]
        reg["mcs"] = list(s["mcs"]); reg["mus"] = list(s["mus"]); reg["groupnum"] = dict(s["groupnum"])

    def _next(key):
        n = reg[key]; reg[key] += 1; return n

    def register(ctx):
        """Assign per-category numbers to newly-seen MCSes/MUSes/open groups this round. Groups in
        a MUS/MCS are numbered first; then every remaining open group is numbered too, so any open
        group -- e.g. a just-committed non-leaf member awaiting refinement -- is selectable even if
        it isn't in this epoch's MUS/MCS list."""
        open_names = {nd.get_full_name() for nd in _open_nodes(ctx)}
        mcs_keys = {fs for _, fs in reg["mcs"]}
        mus_keys = {fs for _, fs in reg["mus"]}
        for r in ctx["results"]:
            fs = frozenset(r["names"])
            if r["kind"] == "MCS" and fs not in mcs_keys:
                reg["mcs"].append((_next("mcs_n"), fs)); mcs_keys.add(fs)
            elif r["kind"] == "MUS" and fs not in mus_keys:
                reg["mus"].append((_next("mus_n"), fs)); mus_keys.add(fs)
            for name in r["names"]:                       # number the groups involved (first)
                if name in open_names and name not in reg["groupnum"]:
                    reg["groupnum"][name] = _next("grp_n")
        for nd in _open_nodes(ctx):                       # then any remaining open group
            name = nd.get_full_name()
            if name not in reg["groupnum"]:
                reg["groupnum"][name] = _next("grp_n")

    def display(ctx):
        register(ctx)
        relaxed = sorted(ctx["committed_relaxed"])
        background = ctx["committed_background"]
        open_lookup = {nd.get_full_name(): nd for nd in _open_nodes(ctx)}
        new_mcs = {frozenset(r["names"]) for r in ctx["results"] if r["kind"] == "MCS"}
        new_mus = {frozenset(r["names"]) for r in ctx["results"] if r["kind"] == "MUS"}

        print(f"\n================  round {ctx['round']}  ================")
        if relaxed:
            print(f"relaxed ({len(relaxed)}): {_fmt_set(relaxed)}")
        if background:
            print(f"background: {len(background)} group(s) hidden  (type 'bg' to list them)")

        print(f"MCSes (numbered, this epoch):")
        for num, fs in reg["mcs"]:
            mark = " *" if fs in new_mcs else "  "
            note = "" if fs <= set(open_lookup) else "   (some members refined -- refine/commit finer)"
            print(f"  [{num}]{mark}{_fmt_set(sorted(fs))}{note}")
        if reg["mus"]:
            print(f"MUSes (numbered, this epoch):")
            for num, fs in reg["mus"]:
                mark = " *" if fs in new_mus else "  "
                print(f"  [{num}]{mark}{_fmt_set(sorted(fs))}")

        # show the RELEVANT open groups (those in some MUS/MCS this epoch) prominently; if none
        # are relevant but groups are open (e.g. right after committing a non-leaf member), show
        # all open. Non-relevant open groups are still numbered/selectable -- see 'open'.
        relevant_names = set()
        for _, fs in reg["mcs"]:
            relevant_names |= fs
        for _, fs in reg["mus"]:
            relevant_names |= fs
        rel = sorted((reg["groupnum"][n], n) for n in open_lookup if n in relevant_names)
        if rel:
            shown, omitted = rel, len(open_lookup) - len(rel)
        else:
            shown, omitted = sorted((reg["groupnum"][n], n) for n in open_lookup), 0
        print(f"\n>>> OPEN constraint groups ({len(shown)})  --  refine/commit target these:")
        if not shown:
            print("    (none open -- nothing left to refine or commit)")
        for num, name in shown:
            nd = open_lookup[name]
            print(f"  [{num}] {name}   (level {nd.level()}, {'leaf' if not nd.children else 'group'})")
        if omitted:
            print(f"  (+{omitted} more open group(s) not in any MUS/MCS -- type 'open' to list)")
        print("  (* = newly found this round)")
        return open_lookup

    def decide(ctx):
        open_lookup = display(ctx)
        if not open_lookup:
            _report_repair(ctx, hard, name2node)
        num_to_mcs = {num: fs for num, fs in reg["mcs"]}
        num_to_name = {num: name for name, num in reg["groupnum"].items()}
        while True:
            try:
                raw = input("action (refine <#>[,#...] | commit <#>|<names> | open | bg | back | stop) > ").strip()
            except EOFError:
                return {"action": "stop"}
            if not raw:
                continue
            parts = raw.split()
            cmd = parts[0].lower()

            if cmd in ("stop", "s", "quit", "q"):
                return {"action": "stop"}

            if cmd in ("open", "o"):
                print(f"  all open groups ({len(open_lookup)}):")
                for num, name in sorted((reg["groupnum"][n], n) for n in open_lookup):
                    nd = open_lookup[name]
                    print(f"    [{num}] {name}   (level {nd.level()}, "
                          f"{'leaf' if not nd.children else 'group'})")
                continue

            if cmd in ("bg", "background"):
                bg = sorted(ctx["committed_background"])
                print(f"  background ({len(bg)}): {_fmt_set(bg)}" if bg else "  (no background groups)")
                continue

            if cmd in ("back", "b", "undo"):
                if not state_stack:
                    print("  ! nothing to undo (no commit to revert)"); continue
                hm_state, reg_snap = state_stack.pop()
                reg_restore(reg_snap)
                print("  <- undoing the last commit")
                return {"action": "restore", "state": hm_state}

            if cmd in ("refine", "r"):
                if len(parts) < 2:
                    print("  ! usage: refine <#>[,<#>...] [target_level]"); continue
                try:
                    nums = [int(x) for x in parts[1].split(",") if x != ""]
                except ValueError:
                    print("  ! usage: refine <#>[,<#>...] [target_level]"); continue
                names, bad = [], []
                for n in nums:
                    nm = num_to_name.get(n)
                    if nm is None or nm not in open_lookup:
                        bad.append(str(n))
                    elif not open_lookup[nm].children:
                        bad.append(f"{n}(leaf)")
                    elif nm not in names:
                        names.append(nm)
                if bad or not names:
                    print(f"  ! cannot refine: {bad}" if bad else "  ! nothing to refine"); continue
                target = int(parts[2]) if len(parts) > 2 else None   # None -> one level per node
                # refine does NOT push a backtrack point -- only commits do
                return {"action": "refine", "constraints": names, "target_level": target}

            if cmd in ("commit", "c"):
                rest = raw[len(parts[0]):].strip()
                if rest.isdigit():
                    fs = num_to_mcs.get(int(rest))
                    if fs is None:
                        print("  ! not an MCS number"); continue
                    if not fs <= set(open_lookup):
                        print("  ! that MCS has members that were since refined -- commit a finer "
                              "MCS instead"); continue
                    names = sorted(fs)
                else:
                    names = [n.strip() for n in rest.split(";") if n.strip()]
                    unknown = [n for n in names if n not in open_lookup]
                    if not names or unknown:
                        print(f"  ! commit needs an MCS number or ';'-separated OPEN group names"
                              f"{'; unknown: ' + str(unknown) if unknown else ''}"); continue
                state_stack.append((ctx["state"], reg_snapshot()))   # backtrack point (+ registry)
                reg_restore({"mcs_n": 0, "mus_n": 0, "grp_n": 0,
                             "mcs": [], "mus": [], "groupnum": {}})    # reset epoch
                return {"action": "commit", "mcs": names}

            print("  ! unrecognised command")

    return decide


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", choices=["defense", "nurse"], default="nurse",
                    help="which instance to explore (config option)")
    ap.add_argument("--initial-level", type=int, default=1)
    ap.add_argument("--solver", default="exact")
    ap.add_argument("--map-solver", default="exact")
    args = ap.parse_args()

    root, hard, label = build(args.instance)
    print(f"Loaded {label}")
    print("Enumerating -- you drive refine/commit/back after each round.")

    name2node = {nd.get_full_name(): nd for nd in root.iter_nodes()}
    state_stack = []        # state snapshots for in-place backtracking (no re-enumeration)
    for _ in hierarchical_marco(root, hard, solver=args.solver, map_solver=args.map_solver,
                                initial_level=args.initial_level,
                                decide_step=make_decide(state_stack, hard, name2node)):
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
