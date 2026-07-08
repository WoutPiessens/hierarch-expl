"""Make the repo root (which contains ``cpmpy/``) importable.

Import this module BEFORE importing ``cpmpy`` from any script here, so that running a module
directly (``python runtime.py``) works regardless of the current working directory:

    import _bootstrap  # noqa: F401
    import cpmpy as cp
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]        # hierarch-expl/hierarch-expl (contains cpmpy/)
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
