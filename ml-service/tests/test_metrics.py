"""Unit tests for the in-process metrics counters exposed via /metrics."""
from __future__ import annotations

from classifier import _apply_bands, get_metrics, reset_metrics


def _mk(scores):
    return {cat: {"final_score": v, "matched": False} for cat, v in scores.items()}


def test_reset_zeros_all_counters():
    # Poke every counter from a known starting point.
    reset_metrics()
    _apply_bands(_mk({"A": 0.60, "B": 0.50}), 0.60, 0.45, 0.35, margin_delta=0.05)
    m = get_metrics()
    assert m["margin_suppressions_total"] == 1
    reset_metrics()
    m = get_metrics()
    assert m["margin_suppressions_total"] == 0
    assert m["predictions_total"] == 0
    assert m["rerank_fires_total"] == 0
    assert m["rerank_drops_total"] == 0


def test_get_metrics_returns_snapshot_not_live_reference():
    reset_metrics()
    snap = get_metrics()
    _apply_bands(_mk({"A": 0.60, "B": 0.50}), 0.60, 0.45, 0.35, margin_delta=0.05)
    # Old snapshot must not reflect the change — it was a copy.
    assert snap["margin_suppressions_total"] == 0
    assert get_metrics()["margin_suppressions_total"] == 1


def test_expected_metric_keys_present():
    m = get_metrics()
    for k in ("predictions_total", "margin_suppressions_total",
              "rerank_fires_total", "rerank_drops_total"):
        assert k in m
