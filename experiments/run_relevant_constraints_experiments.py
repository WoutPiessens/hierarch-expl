"""
    DEPRECATED. This experiment is now the ``relevant_constraints`` experiment in the
    modular pipeline. Use::

        python run.py relevant_constraints [--instances small medium ...] \\
            [--xcsp3-max-constraints 200] [--solver ortools] [--tag NAME]

    See experiments/README.md and experiments/docs/experiments.md.
"""

import sys

if __name__ == "__main__":
    sys.exit(__doc__)
