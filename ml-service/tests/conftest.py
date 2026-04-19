"""Shared fixtures. Keeping it tiny: most tests operate on pure functions."""
import os
import sys

# Make `ml-service/` importable whether pytest is run from the repo root or
# from inside ml-service/.
_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
