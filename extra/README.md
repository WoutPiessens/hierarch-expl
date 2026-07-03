# extra/ — verifying the commit extension to `hierarchical_marco`

This directory lets you exercise the new **commit** capability added to
`cpmpy/tools/explain/hierarchical_marco.py`.

Besides refining a group to more detail, you can now **commit to a group-oriented MCS**: every
active group in that MCS which is a *leaf* (no children) is **relaxed** (`ind` forced false),
every other active group becomes **background** (`ind` forced true, and added as an assumption to
every core solve), and any non-leaf group in the MCS stays free to refine further. Seeds, MSS
growth and the MCS complement range only over the still-*free* active constraints. With no commit,
behaviour is identical to before.

The extension is driven by a new `decide_step` callback on `hierarchical_marco` (takes precedence
over `scripted_steps`/auto-refine); after each round it returns
`{"action": "refine"|"commit"|"stop", ...}`.

## Files
- `instances.py` — builds the two test instances (config option):
  - `defense` → the first defense-scheduling instance (`transcript_1`), with its real hierarchy;
  - `nurse` → the larger, multi-shift nurse soft/hard model (`nurse_instance7_softreq_multishift`:
    20 nurses, 28 days, shifts E/D/L), wrapped in a 5-level hierarchy split by SHIFT first:
    `shift → family (shift_on/shift_off/cover) → nurse|week → week|day → day`. At the coarsest
    (shift) level, shifts E and L cannot be planned (each an MUS; together the MCS `{E, L}`) while
    D is fine. (Built by `experiments/build_nurse_softreq_multishift.py`.)
- `verify_commit.py` — **interactive** terminal driver; you do the refine/commit steps yourself.
- `verify_commit_auto.py` — non-interactive checks (identity, commit, refine-then-commit, smoke).

## Run

Interactive (pick the instance with `--instance`):

```
cd extra
python verify_commit.py --instance nurse
python verify_commit.py --instance defense
```

MUSes, MCSes and relevant open groups are numbered SEPARATELY (each category counts from 0),
since you commit an MCS by its number and refine a constraint by its number. Numbers are
PERSISTENT: they accumulate across refinement rounds within a commitment epoch — so an MCS/MUS
found earlier (now blocked from re-appearing) stays listed and selectable by its number — and
reset to 0 only after a commit; `back` restores the epoch's numbering. `*` marks items newly found
this round; an MCS whose members have since been refined is flagged and must be committed finer.

```
refine <#>                  # refine that open group (by number) down ONE level (default)
refine <#>,<#>,...          # refine several open groups at once (each one level down)
refine <#>[,...] <L>        # ...or send them all to level L
commit <#>                  # commit to that MCS (by number)
commit <name; name; ...>    # commit to an explicit set of open group full-names
open                        # list ALL open groups with numbers (incl. those not in a MUS/MCS)
bg                          # list the (hidden) background groups
back                        # undo the last COMMIT (and any refines made after it)
stop
```

`back` restores the state **in place** (no re-enumeration) to just before the last commit: it
reverts the frontier and committed sets via a `restore` action. The persistent map solver keeps
all its blocking clauses, which is safe — they are inert on their own (they only block together
with the map-solver assumptions), and a clause from an abandoned finer branch references
now-inactive nodes whose up/down variables are unlinked, so it can never wrongly block. When no
OPEN groups remain, the driver checks `hard + (all non-relaxed soft leaves)` and reports whether
the repair is complete. The nurse hierarchy is 5 levels (`shift -> family -> nurse -> week -> day`
for requests, `shift -> family -> week -> day` for cover).

Automated sanity checks:

```
cd extra
python verify_commit_auto.py
```
