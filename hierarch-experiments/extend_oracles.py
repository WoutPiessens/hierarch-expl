"""Extend each 5-oracle suite instance to 20 mss-20 oracles (append 15, distinct seed space)."""
import sys
import hierarchy, oracles as orc
from sampling import sample_oracles
problem, inst = sys.argv[1], sys.argv[2]
existing = orc.load_oracles(problem, inst, "mss-20")
if len(existing) >= 20:
    print(f"OK {problem}/{inst}: already {len(existing)}"); sys.exit()
root, hard = hierarchy.load_instance(problem, inst)
hard = [c for c in hard if c is not None]
extra = sample_oracles(root, hard, "mss-20", 20, 20 - len(existing), seed0=200)
have = {o["seed"] for o in existing}
extra = [o for o in extra if o["seed"] not in have]
orc.save_oracles(problem, inst, "mss-20", existing + extra)
print(f"OK {problem}/{inst}: {len(existing)}+{len(extra)}")
