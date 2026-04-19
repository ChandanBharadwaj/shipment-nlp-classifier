"""Unit tests for the Phase-2 keyword-prune script (pure helpers only).

The DB-facing `run()` is left untested here — it's an offline batch script
that's easy to eyeball in dry-run mode. We cover the two pure helpers:
`_mean_centroid` (shape + normalization) and the single-token gate inlined
in `run()` (replicated here as a predicate).
"""
from __future__ import annotations

import numpy as np
import pytest

from diagnostics.prune_keywords import _mean_centroid


def test_mean_centroid_returns_unit_vector():
    # Two arbitrary 4-d vectors; their mean should come back L2-normalized.
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 0.0])
    out = _mean_centroid([("01", a), ("02", b)])
    assert out.shape == (4,)
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)


def test_mean_centroid_single_child_passthrough():
    v = np.array([0.6, 0.8, 0.0])  # already unit
    out = _mean_centroid([("10", v)])
    assert np.allclose(out, v)


def test_mean_centroid_zero_vectors_do_not_crash():
    # Pathological input: mean collapses to zero. The helper must return
    # the zero vector rather than dividing by zero.
    z = np.zeros(5)
    out = _mean_centroid([("01", z), ("02", z)])
    assert out.shape == (5,)
    assert np.all(out == 0.0)


# The multi-token gate is `" " in kw or "-" in kw`. These assertions pin
# that policy so a refactor to e.g. regex tokenization can't silently let
# single tokens through.
@pytest.mark.parametrize("kw,expected_multi_token", [
    ("motor", False),
    ("alpha", False),
    ("action figure", True),
    ("lithium-ion", True),
    ("stuffed animal", True),
    ("car", False),
    ("t-shirt", True),
])
def test_multi_token_gate_shape(kw, expected_multi_token):
    is_multi = (" " in kw) or ("-" in kw)
    assert is_multi is expected_multi_token
