"""
Pinned tests for every CCTR anchor/suppressor/modifier seeded in
seed/seed_anchors.sql.

Each test below is referenced by `notes` of the seed row it covers
(CCTR-001..CCTR-015). When `scripts/audit_collisions.py` runs in CI it
shells out to pytest with `-k <CCTR-id>` to verify the pinned behaviour
still holds against the live centroids + keywords.

These tests do NOT touch the database. They exercise `_score_one_typed`
end-to-end with hand-built `keywords_typed` and `centroids` dicts that
mirror the shape of the loader's output. That keeps the test fast, makes
the seed→behavior mapping verifiable in pure Python, and isolates the
test from DB seed churn (the same logic re-runs against the DB inside
`audit_collisions.py`).

Centroids are deterministic but arbitrary unit vectors — the cosine path
is essentially constant noise. The keyword path is what's under test.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from classifier import _score_one_typed, _per_chapter_score, _modifier_retype_targets


def _vec(seed: int, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


# A small stable centroid map covering the chapters touched by the seed.
# We use a fixed embedding for the request so cosine contributions are
# constant across categories and the keyword score is what flips outcomes.
_CENTROIDS = {
    "electronics":     [("85", _vec(85))],
    "toys":            [("95", _vec(95))],
    "defense":         [("93", _vec(93))],
    "automotive":      [("87", _vec(87))],
    "machinery":       [("88", _vec(88))],
    "chemicals":       [("28", _vec(28)), ("29", _vec(29))],
    "pharmaceuticals": [("30", _vec(30))],
    "food_beverages":  [("09", _vec(9))],
}

# These tests isolate the keyword pipeline — the cosine path is exercised
# separately by test_keyword_scoring.py and the live 99-row regression. We
# weight semantic to 0 here so a randomly-aligned centroid can't flip a
# pinned outcome (which would make this file a flaky test of cosine, not
# of the seed rows we actually want to verify).
_CFG = {
    cat: {"semantic_weight": 0.0, "keyword_weight": 1.0} for cat in _CENTROIDS
}

_REQUEST_EMB = _vec(0)


# ── seed mirror ──────────────────────────────────────────────────────────────
# This dict mirrors what load_keywords_typed would return after the seed file
# loads. Keep it in sync with seed_anchors.sql when adding rows. Each test
# below cites the CCTR-id it covers.

_TYPED: dict[str, list[tuple[str, float, str, str, list[str]]]] = {
    "toys": [
        # anchors
        ("plush toy",         1.0, "95", "anchor",     []),
        ("doll",              1.0, "95", "anchor",     []),
        ("board game",        1.0, "95", "anchor",     []),
        # suppressors
        ("military-grade",    1.0, "95", "suppressor", []),
        ("mil-spec",          1.0, "95", "suppressor", []),
        ("live-fire",         1.0, "95", "suppressor", []),
        # modifiers
        ("toy",               1.0, "95", "modifier",   ["95"]),
        ("nerf",              1.0, "95", "modifier",   ["95"]),
        ("lego",              1.0, "95", "modifier",   ["95"]),
        ("rc",                0.8, "95", "modifier",   ["95"]),
        # polysemous signals (mirror of what seed_keywords_v2.sql supplies)
        ("gun",               0.7, "95", "signal",     []),
        ("tank",              0.7, "95", "signal",     []),
        ("helicopter",        0.7, "95", "signal",     []),
        ("aircraft",          0.6, "95", "signal",     []),
        ("car",               0.5, "95", "signal",     []),
        ("battery",           0.5, "95", "signal",     []),
        ("blaster",           0.7, "95", "signal",     []),
    ],
    "defense": [
        # anchors
        ("m16 rifle",         1.0, "93", "anchor",     []),
        ("ak-47",             1.0, "93", "anchor",     []),
        ("5.56mm",            1.0, "93", "anchor",     []),
        ("live ammunition",   1.0, "93", "anchor",     []),
        ("firearm",           1.0, "93", "anchor",     []),
        ("rocket launcher",   1.0, "93", "anchor",     []),
        # suppressors
        ("toy",               1.0, "93", "suppressor", []),
        ("nerf",              1.0, "93", "suppressor", []),
        ("lego",              1.0, "93", "suppressor", []),
        # modifiers
        ("military",          1.0, "93", "modifier",   ["93"]),
        ("military-grade",    1.0, "93", "modifier",   ["93"]),
        ("tactical",          0.9, "93", "modifier",   ["93"]),
        # polysemous signals
        ("gun",               0.8, "93", "signal",     []),
        ("tank",              0.8, "93", "signal",     []),
    ],
    "automotive": [
        ("passenger car",     1.0, "87", "anchor",     []),
        ("brake pad",         0.9, "87", "anchor",     []),
        ("transmission",      0.85,"87", "anchor",     []),
        ("toy",               0.9, "87", "suppressor", []),
        ("scale model",       0.9, "87", "suppressor", []),
        # polysemous signal
        ("car",               0.7, "87", "signal",     []),
    ],
    "machinery": [
        ("commercial aircraft", 1.0, "88", "anchor",     []),
        ("turbofan",            1.0, "88", "anchor",     []),
        ("rotorcraft",          1.0, "88", "anchor",     []),
        ("toy",                 0.9, "88", "suppressor", []),
        ("scale model",         0.9, "88", "suppressor", []),
        # polysemous signals
        ("helicopter",          0.8, "88", "signal",     []),
        ("aircraft",            0.8, "88", "signal",     []),
    ],
    "chemicals": [
        ("sulphuric acid",    1.0, "28", "anchor", []),
        ("hydrochloric acid", 1.0, "28", "anchor", []),
        ("methanol",          1.0, "29", "anchor", []),
        ("toluene",           1.0, "29", "anchor", []),
    ],
    "pharmaceuticals": [
        ("vaccine",           1.0, "30", "anchor",     []),
        ("insulin",           1.0, "30", "anchor",     []),
        ("antibiotic",        1.0, "30", "anchor",     []),
        ("cosmetic",          0.7, "30", "suppressor", []),
    ],
    "food_beverages": [
        ("tea",               1.0, "09", "anchor", []),
        ("tea bags",          1.0, "09", "anchor", []),
        ("coffee",            1.0, "09", "anchor", []),
        ("turmeric",          1.0, "09", "anchor", []),
    ],
    "electronics": [
        ("lithium-ion battery",1.0, "85", "anchor",     []),
        ("lithium ion battery",1.0, "85", "anchor",     []),
        ("li-ion cell",        1.0, "85", "anchor",     []),
        ("pcb assembly",       1.0, "85", "anchor",     []),
        ("toy",                0.7, "85", "suppressor", []),
    ],
}


def _winner(text: str) -> tuple[str, dict]:
    """Run the typed scorer and return (best_category, scores_dict)."""
    scores, _ = _score_one_typed(
        text_norm=text,
        embedding=_REQUEST_EMB,
        centroids=_CENTROIDS,
        keywords_typed=_TYPED,
        category_config=_CFG,
    )
    best = max(scores, key=lambda c: scores[c]["final_score"])
    return best, scores


# ── CCTR-001 / CCTR-002: ch95 anchors + suppressors ──────────────────────────

def test_ccts001_plush_toy_resolves_to_toys() -> None:
    best, _ = _winner("plush toy panda for children")
    assert best == "toys"


def test_cctr001_board_game_resolves_to_toys() -> None:
    best, _ = _winner("board game with dice")
    assert best == "toys"


def test_cctr002_military_grade_suppresses_toy_chapter() -> None:
    """A 'military-grade lithium-ion battery' is electronics, not toys —
    the ch95 suppressor for 'military-grade' must beat any toy-side noise."""
    _best, scores = _winner("military-grade lithium-ion battery cells")
    # ch85 anchor (lithium-ion battery) should dominate.
    assert scores["electronics"]["final_score"] > scores["toys"]["final_score"]


# ── CCTR-003 / CCTR-004: ch93 real arms anchors + toy suppressor ─────────────

def test_cctr003_m16_rifle_resolves_to_defense() -> None:
    """SID equivalent: a real M16 rifle is ch93, regardless of toy-side noise."""
    best, _ = _winner("M16 rifle 5.56mm cartridge box")
    assert best == "defense"


def test_cctr004_toy_gun_does_not_resolve_to_defense() -> None:
    """The "toy" modifier in ch95 retypes signals; the ch93 suppressor on
    "toy" stops defense from ever winning a toy-context shipment."""
    best, scores = _winner("toy gun for kids, plastic")
    assert best != "defense"
    # toys should rank above defense.
    assert scores["toys"]["final_score"] > scores["defense"]["final_score"]


def test_cctr004_nerf_blaster_is_toys_not_defense() -> None:
    best, _ = _winner("nerf blaster, kids")
    assert best == "toys"


# ── CCTR-005 / CCTR-006: ch87 vehicle anchors + toy-car suppressor ───────────

def test_cctr005_brake_pad_resolves_to_automotive() -> None:
    best, _ = _winner("brake pad oem replacement")
    assert best == "automotive"


def test_cctr006_toy_car_is_toys_not_automotive() -> None:
    best, scores = _winner("toy car for kids, scale model")
    assert best != "automotive"
    assert scores["toys"]["final_score"] > scores["automotive"]["final_score"]


# ── CCTR-007 / CCTR-008: ch88 aircraft anchors + toy-helicopter suppressor ───

def test_cctr007_turbofan_resolves_to_machinery() -> None:
    best, _ = _winner("turbofan engine assembly")
    assert best == "machinery"


def test_cctr008_toy_helicopter_is_toys_not_machinery() -> None:
    best, scores = _winner("toy helicopter, RC, kids")
    assert best != "machinery"
    assert scores["toys"]["final_score"] > scores["machinery"]["final_score"]


# ── CCTR-009 / CCTR-010: ch28 / ch29 chemistry anchors ───────────────────────

def test_cctr009_sulphuric_acid_resolves_to_chemicals() -> None:
    best, _ = _winner("sulphuric acid, 98%")
    assert best == "chemicals"


def test_cctr010_methanol_resolves_to_chemicals() -> None:
    best, _ = _winner("methanol industrial grade")
    assert best == "chemicals"


# ── CCTR-011 / CCTR-012: ch30 pharma anchors + cosmetic suppressor ───────────

def test_cctr011_vaccine_resolves_to_pharmaceuticals() -> None:
    best, _ = _winner("vaccine doses, cold-chain shipment")
    assert best == "pharmaceuticals"


def test_cctr012_cosmetic_suppresses_pharma() -> None:
    """A cosmetic claim should weaken ch30 — the test asserts pharma score
    drops when the suppressor token is present, vs the same shipment without
    it."""
    _, scores_with = _winner("antibiotic cream, cosmetic")
    _, scores_without = _winner("antibiotic cream")
    assert scores_with["pharmaceuticals"]["final_score"] <= scores_without["pharmaceuticals"]["final_score"]


# ── CCTR-013: ch09 tea/coffee/spice anchors ──────────────────────────────────

def test_cctr013_tea_bags_resolves_to_food_beverages() -> None:
    """The headline regression: "Tea, bags" was hitting electronics through
    'bag' overlap. With 'tea' as ch09 anchor it should never lose."""
    best, _ = _winner("Tea, bags")
    assert best == "food_beverages"


def test_cctr013_coffee_beans_resolves_to_food_beverages() -> None:
    best, _ = _winner("coffee beans, arabica")
    assert best == "food_beverages"


# ── CCTR-014 / CCTR-015: ch85 lithium-ion anchor + toy suppressor ────────────

def test_cctr014_lithium_ion_battery_resolves_to_electronics() -> None:
    best, _ = _winner("lithium-ion battery pack 18650 cells")
    assert best == "electronics"


def test_cctr015_toy_battery_does_not_resolve_to_electronics() -> None:
    """Toys with batteries (the SID-6 regression). The 'toy' modifier on ch95
    plus the 'toy' suppressor on ch85 means electronics doesn't win."""
    best, scores = _winner("plush toy with battery for children")
    # toys must rank at least as high as electronics — the toy modifier
    # plus the plush-toy anchor are doing their job.
    assert scores["toys"]["final_score"] >= scores["electronics"]["final_score"]
    assert best != "electronics"


# ── Modifier composition smoke test (cross-cutting) ──────────────────────────

def test_modifier_retype_collected_for_toys_when_toy_present() -> None:
    """The /classify?debug=1 audit field `modifiers_active` should reflect
    the active modifier targets for the toys row."""
    _, scores = _winner("toy gun for kids")
    assert "95" in scores["toys"]["modifiers_active"]


def test_modifier_retype_empty_when_no_modifier_in_text() -> None:
    _, scores = _winner("M16 rifle 5.56mm cartridge")
    # No toys-side modifier ('toy', 'nerf', 'lego', 'rc') is in the text.
    assert scores["toys"]["modifiers_active"] == []


# ── Pinned tests for token_collisions registry rows (seed_collisions.sql) ────
#
# Each test below is referenced by exactly one `test_case` value in
# seed_collisions.sql. scripts/audit_collisions.py parses that column and
# runs `pytest -k <test_case>` against this file as a CI gate.
#
# These tests exercise the resolver directly with hand-built rules + a
# ResolverContext, instead of going through the full /classify pipeline.
# That keeps the assertions tight: each test pins one decision arm in one
# rule, not the whole inference engine.

from collision_resolver import CollisionRow, ResolverContext, resolve  # noqa: E402


def _row_gun() -> CollisionRow:
    return CollisionRow(
        token="gun",
        home_chapters=["93", "95"],
        risk_tier="high",
        resolution=[
            {"if": "hs_code_present",                                         "then": "use_hs_chapter"},
            {"if": "modifier_hit:toy|plush|nerf|lego|rc|scale model",         "then": "force_chapter:95"},
            {"if": "modifier_hit:military|tactical|military-grade|live-fire", "then": "force_chapter:93"},
            {"if": "anchor_hit_in_home",                                      "then": "use_anchor_chapter"},
            {"if": "margin < 0.05",                                           "then": "defer_ambiguous_high_risk"},
            {"if": "default",                                                 "then": "use_per_chapter_weights"},
        ],
    )


def _row_tank() -> CollisionRow:
    return CollisionRow(
        token="tank",
        home_chapters=["93", "95", "73", "87", "84"],
        risk_tier="high",
        resolution=[
            {"if": "hs_code_present",                                  "then": "use_hs_chapter"},
            {"if": "modifier_hit:toy|plush|nerf|lego|rc|scale model",  "then": "force_chapter:95"},
            {"if": "modifier_hit:military|tactical|military-grade",    "then": "force_chapter:93"},
            {"if": "anchor_hit_in_home",                               "then": "use_anchor_chapter"},
            {"if": "margin < 0.05",                                    "then": "defer_ambiguous_high_risk"},
            {"if": "default",                                          "then": "use_per_chapter_weights"},
        ],
    )


def _ctx(text: str, **kw) -> ResolverContext:
    """Build a ResolverContext with sensible defaults; override fields per-test."""
    return ResolverContext(
        text_lower=text.lower(),
        hs_codes=kw.get("hs_codes", []),
        hs_implied_chapters=kw.get("hs_implied_chapters", []),
        top1_category=kw.get("top1_category"),
        top1_chapter=kw.get("top1_chapter"),
        top1_score=kw.get("top1_score", 0.6),
        top2_score=kw.get("top2_score", 0.4),
        active_modifiers=kw.get("active_modifiers", []),
        anchor_chapters_hit=kw.get("anchor_chapters_hit", []),
    )


# ── tank: high-tier defer when ambiguous ─────────────────────────────────────

def test_cctr_tank_high_tier_defers_when_ambiguous() -> None:
    """Pinned for token_collisions.token='tank'. 'tank, 1 unit' with no HS,
    no modifier, and a tight top-2 margin must defer rather than guess."""
    outcome, token = resolve([_row_tank()], _ctx(
        "tank, 1 unit",
        top1_score=0.50, top2_score=0.48,
    ))
    assert outcome == "defer_ambiguous_high_risk"
    assert token == "tank"


def test_cctr_tank_toy_modifier_forces_ch95() -> None:
    outcome, _ = resolve([_row_tank()], _ctx(
        "toy tank, plastic",
        active_modifiers=["toy"],
    ))
    assert outcome == "force_chapter:95"


def test_cctr_tank_military_modifier_forces_ch93() -> None:
    outcome, _ = resolve([_row_tank()], _ctx(
        "military tank, m1 abrams",
        active_modifiers=["military"],
    ))
    assert outcome == "force_chapter:93"


# ── chemical products: medium tier, low_confidence on thin margin ─────────────

def _row_chemical_products() -> CollisionRow:
    return CollisionRow(
        token="chemical products",
        home_chapters=["28", "29", "32", "38"],
        risk_tier="medium",
        resolution=[
            {"if": "hs_code_present",     "then": "use_hs_chapter"},
            {"if": "anchor_hit_in_home",  "then": "use_anchor_chapter"},
            {"if": "margin < 0.05",       "then": "low_confidence"},
            {"if": "default",             "then": "use_per_chapter_weights"},
        ],
    )


def test_cctr_chemical_products_nos_low_confidence() -> None:
    """The 99-row regression: "Chemical products, nos" with no HS, no anchor
    hit, thin margin → mark low_confidence rather than picking blindly."""
    outcome, token = resolve([_row_chemical_products()], _ctx(
        "chemical products, nos",
        top1_score=0.45, top2_score=0.42,    # thin
    ))
    assert outcome == "low_confidence"
    assert token == "chemical products"


def test_cctr_chemical_products_with_hs_uses_hs_chapter() -> None:
    outcome, _ = resolve([_row_chemical_products()], _ctx(
        "chemical products, nos, HS 2806.10",
        hs_codes=["2806.10"], hs_implied_chapters=["28"],
    ))
    assert outcome == "use_hs_chapter"


# ── bag / cell / core: low-tier polysemes that fall through ──────────────────

def _row_bag() -> CollisionRow:
    return CollisionRow(
        token="bag",
        home_chapters=["42", "63", "48", "39", "73"],
        risk_tier="low",
        resolution=[
            {"if": "hs_code_present",     "then": "use_hs_chapter"},
            {"if": "anchor_hit_in_home",  "then": "use_anchor_chapter"},
            {"if": "default",             "then": "use_per_chapter_weights"},
        ],
    )


def test_cctr_bag_falls_through_to_per_chapter_weights() -> None:
    outcome, _ = resolve([_row_bag()], _ctx("textile bags, polypropylene"))
    assert outcome == "use_per_chapter_weights"


def _row_cell() -> CollisionRow:
    return CollisionRow(
        token="cell",
        home_chapters=["85", "30"],
        risk_tier="low",
        resolution=[
            {"if": "hs_code_present",                              "then": "use_hs_chapter"},
            {"if": "modifier_hit:toy|plush|nerf|lego|rc",          "then": "use_modifier_chapter"},
            {"if": "anchor_hit_in_home",                           "then": "use_anchor_chapter"},
            {"if": "default",                                      "then": "use_per_chapter_weights"},
        ],
    )


def test_cctr_cell_falls_through_to_per_chapter_weights() -> None:
    outcome, _ = resolve([_row_cell()], _ctx("battery cell, 18650"))
    assert outcome == "use_per_chapter_weights"


def _row_core() -> CollisionRow:
    return CollisionRow(
        token="core",
        home_chapters=["84", "85", "48", "73"],
        risk_tier="low",
        resolution=[
            {"if": "hs_code_present",     "then": "use_hs_chapter"},
            {"if": "anchor_hit_in_home",  "then": "use_anchor_chapter"},
            {"if": "default",             "then": "use_per_chapter_weights"},
        ],
    )


def test_cctr_core_falls_through_to_per_chapter_weights() -> None:
    outcome, _ = resolve([_row_core()], _ctx("transformer core, laminated"))
    assert outcome == "use_per_chapter_weights"


# ── gun: anchor-in-home wins over default ────────────────────────────────────

def test_cctr_gun_anchor_in_home_uses_anchor_chapter() -> None:
    outcome, _ = resolve([_row_gun()], _ctx(
        "gun, m16 rifle, ammunition",
        anchor_chapters_hit=["93"],
    ))
    assert outcome == "use_anchor_chapter"


def test_cctr_gun_hs_code_takes_priority() -> None:
    """HS code is the deterministic prior: even if a toy modifier fires,
    the registered rule order means hs_code_present wins."""
    outcome, _ = resolve([_row_gun()], _ctx(
        "toy gun, HS 9304.00",
        hs_codes=["9304.00"], hs_implied_chapters=["93"],
        active_modifiers=["toy"],
    ))
    assert outcome == "use_hs_chapter"
