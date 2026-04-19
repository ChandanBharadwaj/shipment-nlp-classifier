"""Unit tests for the top-margin rule in classifier._apply_bands.

Pure-function tests: no DB, no embedding. We fabricate the `scores` dict
directly — that's the only input `_apply_bands` reads.
"""
from __future__ import annotations

import pytest

from classifier import _apply_bands, get_metrics, reset_metrics


def _mk(scores: dict[str, float]) -> dict:
    """Shape a flat {cat: final_score} map into the dict _apply_bands expects."""
    return {
        cat: {"final_score": v, "matched": False}
        for cat, v in scores.items()
    }


def test_top_only_when_runner_outside_margin():
    s = _mk({"A": 0.52, "B": 0.39})
    matched, state, reason = _apply_bands(s, 0.52, 0.45, 0.35, margin_delta=0.06)
    assert matched == ["A"]
    assert state == "classified"
    assert reason is None
    assert s["A"]["matched"] is True
    assert s["B"]["matched"] is False


def test_both_when_runner_within_margin():
    s = _mk({"A": 0.52, "B": 0.48})
    matched, state, _ = _apply_bands(s, 0.52, 0.45, 0.35, margin_delta=0.06)
    assert set(matched) == {"A", "B"}
    assert state == "classified"
    assert s["A"]["matched"] is True
    assert s["B"]["matched"] is True


def test_unclassified_when_below_floor():
    s = _mk({"A": 0.30})
    matched, state, reason = _apply_bands(s, 0.30, 0.45, 0.35)
    assert matched == []
    assert state == "unclassified"
    assert reason == "low_similarity"


def test_low_confidence_below_threshold_but_above_floor():
    s = _mk({"A": 0.40, "B": 0.37})
    matched, state, reason = _apply_bands(s, 0.40, 0.45, 0.35)
    # Margin rule is scoped to the `classified` band; below threshold we
    # still return exactly one top label regardless of runner-up proximity.
    assert matched == ["A"]
    assert state == "low_confidence"
    assert reason is None


def test_wider_margin_admits_more_labels():
    s = _mk({"A": 0.60, "B": 0.48, "C": 0.47})
    matched, _, _ = _apply_bands(s, 0.60, 0.45, 0.35, margin_delta=0.15)
    assert set(matched) == {"A", "B", "C"}


def test_tighter_margin_rejects_close_runners():
    s = _mk({"A": 0.60, "B": 0.58})
    matched, _, _ = _apply_bands(s, 0.60, 0.45, 0.35, margin_delta=0.01)
    assert matched == ["A"]


def test_runner_below_threshold_never_fires_even_within_margin():
    # B is within δ of A, but B itself is below threshold. It must not fire.
    s = _mk({"A": 0.46, "B": 0.44})
    matched, state, _ = _apply_bands(s, 0.46, 0.45, 0.35, margin_delta=0.10)
    assert matched == ["A"]
    assert state == "classified"


def test_margin_suppression_counter_increments():
    reset_metrics()
    s = _mk({"A": 0.60, "B": 0.50, "C": 0.48})  # B, C above thr but > δ from A
    _apply_bands(s, 0.60, 0.45, 0.35, margin_delta=0.05)
    assert get_metrics()["margin_suppressions_total"] == 2


def test_no_suppression_when_all_within_margin():
    reset_metrics()
    s = _mk({"A": 0.52, "B": 0.50})
    _apply_bands(s, 0.52, 0.45, 0.35, margin_delta=0.05)
    assert get_metrics()["margin_suppressions_total"] == 0


def test_no_suppression_when_only_top_above_threshold():
    reset_metrics()
    s = _mk({"A": 0.52, "B": 0.40})  # only A above threshold
    _apply_bands(s, 0.52, 0.45, 0.35, margin_delta=0.05)
    assert get_metrics()["margin_suppressions_total"] == 0
