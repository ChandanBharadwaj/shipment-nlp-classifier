"""
Unit tests for the CCTR signal-class scoring pipeline.

These tests exercise the typed scoring path end-to-end *without* a database —
keywords_typed and centroids are constructed directly. That keeps the test
fast and isolates it from seed data churn.

Coverage:
  - anchor isolation: an anchor wins its chapter regardless of suppressors elsewhere
  - signal redirection: a modifier retypes signals to its target chapter
  - suppressor effect: a suppressor reduces its chapter's score
  - per-chapter granularity: same keyword in two chapters of one category votes per-chapter
  - modifier without target list: no-op (doesn't accidentally retype)

The scorer is `_score_one_typed`. Centroids are mocked as constant unit
vectors so the cosine path is deterministic and the keyword pipeline is
the only thing under test.
"""
from __future__ import annotations

import numpy as np
import pytest

import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from classifier import _score_one_typed, _modifier_retype_targets, _per_chapter_score


def _vec(seed: int, dim: int = 384) -> np.ndarray:
    """Deterministic L2-normalized vector. Different seeds give different
    centroids so the test can verify per-chapter resolution."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


# ── Modifier retype targets ──────────────────────────────────────────────────

def test_modifier_with_target_list_activates_chapters() -> None:
    rows = [
        ("toy",     1.0, "95", "modifier",   ["95"]),
        ("battery", 0.5, "85", "signal",     []),
    ]
    targets = _modifier_retype_targets("toy gun", rows)
    assert targets == {"95"}


def test_modifier_without_target_list_is_noop() -> None:
    rows = [
        ("toy",     1.0, "95", "modifier", []),    # no targets → no retype
        ("battery", 0.5, "85", "signal",   []),
    ]
    targets = _modifier_retype_targets("toy gun", rows)
    assert targets == set()


def test_modifier_not_in_text_does_not_fire() -> None:
    rows = [
        ("toy",     1.0, "95", "modifier", ["95"]),
        ("battery", 0.5, "85", "signal",   []),
    ]
    targets = _modifier_retype_targets("alkaline battery pack", rows)
    assert targets == set()


# ── Per-chapter score composition ────────────────────────────────────────────

def test_anchor_dominates_its_chapter() -> None:
    rows = [
        ("m16 rifle", 1.0, "93", "anchor", []),
        ("rifle",     0.5, "93", "signal", []),
    ]
    score, hits = _per_chapter_score("M16 rifle 5.56mm", rows, retyped_chapters=set())
    assert "93" in score
    # alpha=0.5; anchor(1.0) + signal(0.5) = 1.5 → 1 - exp(-0.75) ≈ 0.528
    assert score["93"] > 0.5
    assert "m16 rifle" in hits["93"]


def test_signal_only_chapter_score() -> None:
    rows = [
        ("battery", 0.5, "85", "signal", []),
    ]
    score, _ = _per_chapter_score("alkaline battery", rows, retyped_chapters=set())
    assert "85" in score
    assert 0.0 < score["85"] < 1.0


def test_suppressor_reduces_chapter_score() -> None:
    rows_without = [
        ("battery", 0.6, "95", "signal",     []),
    ]
    rows_with = [
        ("battery",        0.6, "95", "signal",     []),
        ("military-grade", 1.0, "95", "suppressor", []),
    ]
    text = "battery, military-grade"
    score_a, _ = _per_chapter_score(text, rows_without, retyped_chapters=set())
    score_b, _ = _per_chapter_score(text, rows_with, retyped_chapters=set())
    assert score_a["95"] > score_b.get("95", 0.0)


def test_modifier_redirects_signal_to_target_chapter() -> None:
    """`toy gun`: the `gun` signal in ch93 (defense) is retyped to ch95 (toys)
    when the `toy` modifier fires. ch93 should see no contribution from
    `gun`; ch95 should get the redirected weight."""
    rows = [
        ("toy", 1.0, "95", "modifier", ["95"]),
        ("gun", 0.7, "93", "signal",   []),
    ]
    text = "toy gun"
    targets = _modifier_retype_targets(text, rows)
    assert targets == {"95"}

    score, hits = _per_chapter_score(text, rows, retyped_chapters=targets)
    # ch93 should NOT have accumulated weight from the gun signal — it was redirected.
    assert score.get("93", 0.0) == 0.0
    # ch95 should carry the redirected weight.
    assert score.get("95", 0.0) > 0.0
    assert any("gun->ch95" in h for h in hits.get("95", []))


def test_signal_in_target_chapter_keeps_native_when_modifier_endorses() -> None:
    """When a signal's native chapter IS in the modifier's target set, it
    contributes to its native chapter normally (no double-counting)."""
    rows = [
        ("toy",     1.0, "95", "modifier", ["95"]),
        ("battery", 0.5, "95", "signal",   []),    # already in ch95
    ]
    text = "toy battery"
    targets = _modifier_retype_targets(text, rows)
    assert targets == {"95"}

    score, hits = _per_chapter_score(text, rows, retyped_chapters=targets)
    assert score.get("95", 0.0) > 0.0
    assert any(h == "battery" for h in hits.get("95", []))   # not a redirected tag


# ── End-to-end: _score_one_typed ─────────────────────────────────────────────

def test_score_one_typed_returns_full_shape() -> None:
    """The typed scorer produces the same dict shape the rest of the pipeline
    expects (`_apply_bands`, the API response shape, the audit log)."""
    centroids = {
        "electronics": [("85", _vec(1))],
        "toys":        [("95", _vec(2))],
        "defense":     [("93", _vec(3))],
    }
    keywords_typed = {
        "electronics": [("battery", 0.5, "85", "signal", [])],
        "toys":        [
            ("toy",     1.0, "95", "modifier", ["95"]),
            ("battery", 0.4, "95", "signal",   []),
        ],
        "defense":     [("gun", 0.7, "93", "signal", [])],
    }
    cfg = {
        "electronics": {"semantic_weight": 0.8, "keyword_weight": 0.2},
        "toys":        {"semantic_weight": 0.8, "keyword_weight": 0.2},
        "defense":     {"semantic_weight": 0.8, "keyword_weight": 0.2},
    }
    embedding = _vec(99)

    scores, max_score = _score_one_typed(
        text_norm="toy battery",
        embedding=embedding,
        centroids=centroids,
        keywords_typed=keywords_typed,
        category_config=cfg,
    )

    for cat in ("electronics", "toys", "defense"):
        assert cat in scores
        for k in ("semantic_score", "keyword_score", "final_score",
                  "best_chapter", "keywords_hit", "modifiers_active"):
            assert k in scores[cat], f"missing {k} on {cat}"

    # toys had the modifier fire; its modifiers_active list reflects that.
    assert "95" in scores["toys"]["modifiers_active"]
    # max_score is the largest final across categories.
    assert max_score == max(s["final_score"] for s in scores.values())


def test_score_one_typed_no_keywords_falls_through_cleanly() -> None:
    """An empty keywords_typed for a category should not error — semantic
    score alone determines the final score."""
    centroids = {"electronics": [("85", _vec(1))]}
    scores, _ = _score_one_typed(
        text_norm="some text",
        embedding=_vec(2),
        centroids=centroids,
        keywords_typed={"electronics": []},
        category_config={},
    )
    assert "electronics" in scores
    assert scores["electronics"]["keyword_score"] == 0.0


# ── collision_resolver.py unit checks ────────────────────────────────────────

def test_collision_resolver_default_when_no_token_in_text() -> None:
    from collision_resolver import CollisionRow, ResolverContext, resolve

    rows = [CollisionRow(
        token="gun",
        home_chapters=["93", "95"],
        risk_tier="high",
        resolution=[
            {"if": "modifier_hit:toy", "then": "force_chapter:95"},
            {"if": "default",          "then": "use_per_chapter_weights"},
        ],
    )]
    ctx = ResolverContext(
        text_lower="alkaline battery pack",
        hs_codes=[], hs_implied_chapters=[],
        top1_category="electronics", top1_chapter="85",
        top1_score=0.7, top2_score=0.5,
        active_modifiers=[], anchor_chapters_hit=[],
    )
    outcome, token = resolve(rows, ctx)
    assert outcome == "use_per_chapter_weights"
    assert token is None


def test_collision_resolver_modifier_path_fires() -> None:
    from collision_resolver import CollisionRow, ResolverContext, resolve

    rows = [CollisionRow(
        token="gun",
        home_chapters=["93", "95"],
        risk_tier="high",
        resolution=[
            {"if": "modifier_hit:toy", "then": "force_chapter:95"},
            {"if": "default",          "then": "use_per_chapter_weights"},
        ],
    )]
    ctx = ResolverContext(
        text_lower="toy gun for kids",
        hs_codes=[], hs_implied_chapters=[],
        top1_category="defense", top1_chapter="93",
        top1_score=0.7, top2_score=0.65,
        active_modifiers=["toy"], anchor_chapters_hit=[],
    )
    outcome, token = resolve(rows, ctx)
    assert outcome == "force_chapter:95"
    assert token == "gun"


def test_collision_resolver_high_tier_defers_on_thin_margin() -> None:
    from collision_resolver import CollisionRow, ResolverContext, resolve

    rows = [CollisionRow(
        token="helicopter",
        home_chapters=["88", "95"],
        risk_tier="high",
        resolution=[
            {"if": "hs_code_present", "then": "use_hs_chapter"},
            {"if": "modifier_hit",    "then": "use_modifier_chapter"},
            {"if": "margin < 0.05",   "then": "defer_ambiguous_high_risk"},
            {"if": "default",         "then": "use_per_chapter_weights"},
        ],
    )]
    ctx = ResolverContext(
        text_lower="helicopter, 1 unit",
        hs_codes=[], hs_implied_chapters=[],
        top1_category="aircraft", top1_chapter="88",
        top1_score=0.50, top2_score=0.48,    # thin margin
        active_modifiers=[], anchor_chapters_hit=[],
    )
    outcome, token = resolve(rows, ctx)
    assert outcome == "defer_ambiguous_high_risk"
    assert token == "helicopter"


def test_collision_resolver_unknown_predicate_falls_through() -> None:
    """Unknown 'if' predicates must fail closed so a typo in the registry
    doesn't silently match every request."""
    from collision_resolver import CollisionRow, ResolverContext, resolve

    rows = [CollisionRow(
        token="gun",
        home_chapters=["93", "95"],
        risk_tier="high",
        resolution=[
            {"if": "completely_made_up_predicate", "then": "force_chapter:95"},
            {"if": "default",                      "then": "use_per_chapter_weights"},
        ],
    )]
    ctx = ResolverContext(
        text_lower="toy gun",
        hs_codes=[], hs_implied_chapters=[],
        top1_category="defense", top1_chapter="93",
        top1_score=0.7, top2_score=0.65,
        active_modifiers=["toy"], anchor_chapters_hit=[],
    )
    outcome, _ = resolve(rows, ctx)
    assert outcome == "use_per_chapter_weights"
