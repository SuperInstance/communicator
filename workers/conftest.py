"""
Test-time path setup for workers/.

budget.py and batching.py use flat imports (`from budget import Costs`) so they
can be run directly -- `python workers/hotpath/batching.py` -- without the
package machinery that a Workers directory has no use for. That only resolves
if hotpath/ itself is on sys.path, which is what this does.

Deliberately no __init__.py: hotpath/ is a Workers directory that happens to
carry two pure-Python modules, not a Python package.
"""

import os
import sys

HOTPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hotpath")

if HOTPATH not in sys.path:
    sys.path.insert(0, HOTPATH)