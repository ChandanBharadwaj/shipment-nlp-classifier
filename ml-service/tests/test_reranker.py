"""Unit tests for the cross-encoder reranker path.

We don't load a real cross-encoder — the reranker short-circuits on
`_cross_encoder is None`, which covers the default (disabled) code path.
For the "reranker ran" path we monkey-patch `_cross_encoder` with a stub
that returns canned logits, so the test runs in milliseconds with no
model download.
"""
from __future__ import annotations

import math

import pytest

import classifier
from classifier import (
    _maybe_rerank,
    _rerank_candidates,
    _sigmoid,
    get_metrics,
    reset_metrics,
)


# ── _sigmoid ───────────────────────────────────────────────────────────────────

def test_sigmoid_zero_is_half():
    assert _sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_bounded_and_monotonic():
    for x in [-1000.0, -10.0, -1.0, 0.0, 1.0, 10.0, 1000.0]:
        y = _sigmoid(x)
        assert 0.0 <= y <= 1.0
    xs = [-5.0, -1.0, 0.0, 0.5, 3.0]
    ys = [_sigmoid(x) for x in xs]
    assert ys == sorted(ys)


def test_sigmoid_stable_for_large_magnitudes():
    # Naive `1/(1+exp(-x))` overflows for large positive x; large negatives
    # underflow. Our helper branches to stay finite on both sides.
    assert _sigmoid(1000.0) == pytest.approx(1.0)
    assert _sigmoid(-1000.0) == pytest.approx(0.0)


# ── _rerank_candidates short-circuits ──────────────────────────────────────────

def _scores_for(cats: list[str]) -> dict:
    return {c: {"final_score": 0.5, "matched": True} for c in cats}


def test_rerank_short_circuits_when_ce_disabled():
    assert classifier._cross_encoder is None  # default state
    out = _rerank_candidates("any text", ["A", "B"], _scores_for(["A", "B"]),
                             category_descriptions={"A": "a", "B": "b"})
    assert out == ["A", "B"]


def test_rerank_short_circuits_on_single_candidate(monkeypatch):
    class Dummy:
        def predict(self, pairs):
            raise AssertionError("should not be called")
    monkeypatch.setattr(classifier, "_cross_encoder", Dummy())
    out = _rerank_candidates("x", ["A"], _scores_for(["A"]),
                             category_descriptions={"A": "a"})
    assert out == ["A"]


def test_rerank_short_circuits_on_missing_descriptions(monkeypatch):
    class Dummy:
        def predict(self, pairs):
            raise AssertionError("should not be called")
    monkeypatch.setattr(classifier, "_cross_encoder", Dummy())
    # Empty description map → nothing to send to the CE.
    out = _rerank_candidates("x", ["A", "B"], _scores_for(["A", "B"]),
                             category_descriptions=None)
    assert out == ["A", "B"]
    out = _rerank_candidates("x", ["A", "B"], _scores_for(["A", "B"]),
                             category_descriptions={})
    assert out == ["A", "B"]


# ── _rerank_candidates actually running ────────────────────────────────────────

class _StubCE:
    def __init__(self, logit_map: dict[str, float]):
        self.logit_map = logit_map

    def predict(self, pairs):
        # pairs: list of (text, description). We encoded the category name into
        # the description string, so we fish it back out for the stub mapping.
        out = []
        for _text, desc in pairs:
            for cat, logit in self.logit_map.items():
                if cat in desc:
                    out.append(logit)
                    break
            else:  # pragma: no cover - defensive
                out.append(0.0)
        return out


def test_rerank_drops_candidate_outside_sigmoid_margin(monkeypatch):
    # In sigmoid space: σ(5)=0.993, σ(0)=0.5 — gap of 0.493, far more than the
    # default margin of 0.15, so B must drop.
    monkeypatch.setattr(classifier, "_cross_encoder",
                        _StubCE({"A": 5.0, "B": 0.0}))
    scores = _scores_for(["A", "B"])
    descs = {"A": "desc for A", "B": "desc for B"}
    out = _rerank_candidates("shipment text", ["A", "B"], scores, descs, margin=0.15)
    assert out == ["A"]
    # Annotations on both: raw logit + squashed prob.
    assert scores["A"]["cross_encoder_score"] == pytest.approx(5.0)
    assert scores["A"]["cross_encoder_prob"] == pytest.approx(_sigmoid(5.0), abs=1e-4)
    assert scores["B"]["cross_encoder_score"] == pytest.approx(0.0)
    assert scores["B"]["cross_encoder_prob"] == pytest.approx(0.5, abs=1e-4)


def test_rerank_keeps_both_when_close_in_sigmoid_space(monkeypatch):
    # σ(1.2)≈0.769, σ(1.0)≈0.731 — gap 0.038, inside the 0.15 margin.
    monkeypatch.setattr(classifier, "_cross_encoder",
                        _StubCE({"A": 1.2, "B": 1.0}))
    out = _rerank_candidates("text", ["A", "B"], _scores_for(["A", "B"]),
                             {"A": "desc A", "B": "desc B"}, margin=0.15)
    assert set(out) == {"A", "B"}


def test_rerank_preserves_input_order(monkeypatch):
    # CE ranks B > A, but the returned list follows the original `matched`
    # order for stable downstream display.
    monkeypatch.setattr(classifier, "_cross_encoder",
                        _StubCE({"A": 1.0, "B": 1.1}))
    out = _rerank_candidates("t", ["A", "B"], _scores_for(["A", "B"]),
                             {"A": "desc A", "B": "desc B"}, margin=0.5)
    assert out == ["A", "B"]


# ── _maybe_rerank gating + metrics ─────────────────────────────────────────────

def test_maybe_rerank_skips_when_state_not_classified(monkeypatch):
    class Trap:
        def predict(self, pairs):
            raise AssertionError("should not run")
    monkeypatch.setattr(classifier, "_cross_encoder", Trap())
    reset_metrics()
    matched, reranked = _maybe_rerank("t", ["A", "B"], "low_confidence",
                                      _scores_for(["A", "B"]),
                                      {"A": "a", "B": "b"})
    assert reranked is False
    assert matched == ["A", "B"]
    assert get_metrics()["rerank_fires_total"] == 0


def test_maybe_rerank_skips_on_single_matched():
    reset_metrics()
    matched, reranked = _maybe_rerank("t", ["A"], "classified",
                                      _scores_for(["A"]), {"A": "a"})
    assert reranked is False
    assert matched == ["A"]
    assert get_metrics()["rerank_fires_total"] == 0


def test_maybe_rerank_counts_fires_and_drops(monkeypatch):
    monkeypatch.setattr(classifier, "_cross_encoder",
                        _StubCE({"A": 5.0, "B": 0.0, "C": -2.0}))
    reset_metrics()
    scores = _scores_for(["A", "B", "C"])
    matched, reranked = _maybe_rerank(
        "t", ["A", "B", "C"], "classified", scores,
        {"A": "desc A", "B": "desc B", "C": "desc C"},
    )
    assert reranked is True
    assert matched == ["A"]
    m = get_metrics()
    assert m["rerank_fires_total"] == 1
    assert m["rerank_drops_total"] == 2
    # Dropped categories have their `matched` flag cleared.
    assert scores["A"]["matched"] is True
    assert scores["B"]["matched"] is False
    assert scores["C"]["matched"] is False


def test_maybe_rerank_fires_without_dropping_when_all_close(monkeypatch):
    monkeypatch.setattr(classifier, "_cross_encoder",
                        _StubCE({"A": 1.0, "B": 1.05}))
    reset_metrics()
    scores = _scores_for(["A", "B"])
    matched, reranked = _maybe_rerank(
        "t", ["A", "B"], "classified", scores,
        {"A": "desc A", "B": "desc B"},
    )
    assert reranked is True
    assert set(matched) == {"A", "B"}
    m = get_metrics()
    assert m["rerank_fires_total"] == 1
    assert m["rerank_drops_total"] == 0
