"""
Tunable constants in one place.

Every knob that used to be scattered across classifier.py / main.py /
diagnostics/prune_keywords.py is declared here, with an env override so
operators can retune without a code change (restart required — nothing
here is a per-request parameter).

To tune in a principled way instead of guessing:
    python calibrate.py          # grid-search threshold x margin_delta, report F1

Import from here; do not re-declare these values anywhere else. If you need
to add a new tunable, add it here and import it — that keeps the audit
surface small.
"""

from __future__ import annotations

import os


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


# ── Confidence bands ──────────────────────────────────────────────────────────
# Score below UNCLASSIFIED_THRESHOLD → `unclassified`
# UNCLASSIFIED_THRESHOLD ≤ score < THRESHOLD → `low_confidence` (top candidate only)
# Score ≥ THRESHOLD → `classified` (all labels within MARGIN_DELTA of top)
THRESHOLD_DEFAULT              = _float_env("CLASSIFY_THRESHOLD",       0.45)
UNCLASSIFIED_THRESHOLD_DEFAULT = _float_env("UNCLASSIFIED_THRESHOLD",   0.35)

# ── Top-margin rule (Phase 1) ─────────────────────────────────────────────────
# Secondary labels fire only if their final_score is within this delta of the
# top score. Suppresses weak tail labels produced by single-token keyword
# pollution. Lower = more multi-label permissive; higher = one-label-dominant.
MARGIN_DELTA_DEFAULT           = _float_env("MARGIN_DELTA",             0.06)

# ── Keyword scoring ───────────────────────────────────────────────────────────
# Saturating curve: score = 1 - exp(-alpha * sum_weights).
# alpha=0.5 → 1 hit@1.0 = 0.39, 2 = 0.63, 3 = 0.78.
# Raising alpha makes keyword hits count more; lowering damps them.
KEYWORD_SATURATION_ALPHA       = _float_env("KEYWORD_SATURATION_ALPHA", 0.5)

# ── Keyword prune (Phase 2, offline) ──────────────────────────────────────────
# Cosine threshold below which a single-token keyword is pruned from its
# category. Multi-token keywords are always kept (rarely polluted).
MIN_KEYWORD_COSINE_DEFAULT     = _float_env("MIN_KEYWORD_COSINE",       0.30)

# ── Cross-encoder reranker (Phase 3, runtime, opt-in) ─────────────────────────
# When enabled, the reranker runs on `classified` rows with ≥2 candidates.
# The margin is applied in sigmoid-wrapped space (0, 1) — NOT on raw logits —
# so the same value is meaningful across different CE models.
RERANKER_ENABLED               = _bool_env("RERANKER_ENABLED",          False)
RERANKER_MODEL                 = os.environ.get("RERANKER_MODEL",
                                                "cross-encoder/ms-marco-MiniLM-L-6-v2")
CROSS_ENCODER_MARGIN_DEFAULT   = _float_env("CROSS_ENCODER_MARGIN",     0.15)

# ── CCTR (Context-Conditional Token Resolution) knobs ────────────────────────
# Whether _score_one prefers the typed (per-chapter, signal-class-aware)
# scoring path when a keywords_typed dict is supplied. False forces the legacy
# untyped path even when typed data is available — lets ops A/B-compare.
CCTR_ENABLED                   = _bool_env("CCTR_ENABLED",              True)

# Margin in final_score below which a high-tier collision causes the system to
# defer rather than guess. See ml-service/collision_resolver.py.
CCTR_DANGER_PAIR_MARGIN        = _float_env("CCTR_DANGER_PAIR_MARGIN",  0.05)

# Bounds the rebalancer respects on `signal_class IN ('signal',)` rows.
# Anchors / suppressors / modifiers are immune to the rebalancer regardless.
CCTR_WEIGHT_FLOOR              = _float_env("CCTR_WEIGHT_FLOOR",        0.10)
CCTR_WEIGHT_CAP                = _float_env("CCTR_WEIGHT_CAP",          1.00)
