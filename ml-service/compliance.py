"""
Semantic compliance decision layer.

Uses the same sentence-transformer embeddings as the classifier to match
shipments against hard-negative risk phrases. Instead of brittle keyword regex,
each risk phrase (+ aliases) is embedded at startup. At inference, the
*already-computed* shipment embedding is compared against risk vectors via
cosine similarity.

This means:
  - "depleted uranium fuel rods" catches "spent nuclear fuel", "DU penetrator"
  - "armored" / "armoured" / "armor-plated" are naturally similar vectors
  - No duplicated entries for spelling variants

Schema (risk_profile.json):
  - default_thresholds: {block: float, review: float}
  - global_blocked: list of {phrase, aliases?, risky: true, reason}
        Every global entry is unconditionally treated as a block on match.
        The `risky: true` flag is required to make the intent explicit.
  - categories: dict of {risk_level, hard_negatives: list of {phrase, aliases?, action, reason}}
        Category entries use action ∈ {"block", "review"}.

Decision cascade (evaluated top-to-bottom, first match wins):
    1. Global blocked phrase hit (cosine >= block_threshold) -> block
    2. Category hard-negative hit (action=block, cosine >= review_threshold) -> block
    3. Category hard-negative hit (action=review, cosine >= review_threshold) -> review
    4. risk_level=high (any confidence) -> review
    5. confidence_state=unclassified or low_confidence -> review
    6. risk_level=medium or low + classified -> allow

Usage::

    from compliance import load_risk_profile, prepare_risk_vectors, apply_compliance

    profile = load_risk_profile()
    risk_vectors = prepare_risk_vectors(profile, embed_fn)
    decision = apply_compliance(classifier_result, risk_vectors)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_PROFILE_PATH = Path(__file__).parent / "risk_profile.json"

# Single source of truth for cascade thresholds. Overridable via
# default_thresholds in risk_profile.json.
_DEFAULT_THRESHOLDS = {"block": 0.62, "review": 0.55}

# Sentinel category name used for global_blocked hits.
_GLOBAL_SCOPE = "_global"

# Schema validation
_REQUIRED_GLOBAL_KEYS = {"phrase", "reason", "risky"}
_REQUIRED_CATEGORY_KEYS = {"phrase", "action", "reason"}
_VALID_CATEGORY_ACTIONS = {"block", "review"}


# ── Profile loading ───────────────────────────────────────────────────────────

def load_risk_profile(path: str | Path | None = None) -> dict:
    """Load the JSON risk profile. Returns the parsed dict."""
    p = Path(path) if path else _DEFAULT_PROFILE_PATH
    if not p.exists():
        raise FileNotFoundError(f"Risk profile not found: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_global(entries: list[dict]) -> None:
    """Fail fast on malformed global_blocked entries."""
    for entry in entries:
        missing = _REQUIRED_GLOBAL_KEYS - entry.keys()
        if missing:
            raise ValueError(
                f"Global risk entry missing keys {sorted(missing)}: "
                f"{entry.get('phrase', '<no phrase>')}"
            )
        if entry["risky"] is not True:
            raise ValueError(
                f"Global risk entry must have risky=true: {entry['phrase']}"
            )


def _validate_category(category: str, entries: list[dict]) -> None:
    """Fail fast on malformed category hard_negative entries."""
    for entry in entries:
        missing = _REQUIRED_CATEGORY_KEYS - entry.keys()
        if missing:
            raise ValueError(
                f"Category '{category}' risk entry missing keys "
                f"{sorted(missing)}: {entry.get('phrase', '<no phrase>')}"
            )
        if entry["action"] not in _VALID_CATEGORY_ACTIONS:
            raise ValueError(
                f"Category '{category}' entry '{entry['phrase']}' has "
                f"invalid action '{entry['action']}'; expected one of "
                f"{sorted(_VALID_CATEGORY_ACTIONS)}"
            )


# ── Vector preparation (called once at startup / reload) ─────────────────────

def _embed_entries(
    entries: list[dict],
    embed_fn: Callable[[list[str]], np.ndarray],
) -> tuple[list[dict], int]:
    """
    Embed each entry's phrase + aliases into vectors.

    Returns ``(enriched_entries, total_vector_count)``. Phrase + aliases for
    every entry are batched into a single embed call.
    """
    all_texts: list[str] = []
    entry_spans: list[tuple[int, int]] = []  # (start_idx, count) per entry

    for entry in entries:
        texts = [entry["phrase"]] + entry.get("aliases", [])
        start = len(all_texts)
        all_texts.extend(texts)
        entry_spans.append((start, len(texts)))

    if not all_texts:
        return entries, 0

    all_vectors = embed_fn(all_texts)  # (N, dim), L2-normalized

    enriched: list[dict] = []
    n_vectors = 0
    for i, entry in enumerate(entries):
        start, count = entry_spans[i]
        vectors = all_vectors[start : start + count]  # (count, dim)
        enriched.append({**entry, "_vectors": vectors})
        n_vectors += vectors.shape[0]

    return enriched, n_vectors


def prepare_risk_vectors(
    profile: dict,
    embed_fn: Callable[[list[str]], np.ndarray],
) -> dict:
    """
    Embed all risk phrases and return an enriched profile with numpy vectors.

    Validates the schema before embedding so typos fail at startup, not at
    request time.
    """
    global_raw = profile.get("global_blocked", [])
    _validate_global(global_raw)
    global_entries, n_global_vectors = _embed_entries(global_raw, embed_fn)

    categories: dict[str, dict] = {}
    n_cat_vectors = 0
    for cat_name, cat_cfg in profile.get("categories", {}).items():
        negs = cat_cfg.get("hard_negatives", [])
        _validate_category(cat_name, negs)
        enriched, n_v = _embed_entries(negs, embed_fn)
        categories[cat_name] = {
            "risk_level": cat_cfg.get("risk_level", "medium"),
            "hard_negatives": enriched,
        }
        n_cat_vectors += n_v

    return {
        "default_thresholds": profile.get("default_thresholds", _DEFAULT_THRESHOLDS),
        "global_blocked": global_entries,
        "categories": categories,
        "_stats": {
            "global_entries":   len(global_entries),
            "category_entries": sum(len(c["hard_negatives"]) for c in categories.values()),
            "total_vectors":    n_global_vectors + n_cat_vectors,
        },
    }


# ── Semantic matching ─────────────────────────────────────────────────────────

def _match_entries(
    embedding: np.ndarray,
    entries: list[dict],
    threshold: float,
    category: str,
    force_action: str | None = None,
) -> list[dict]:
    """
    Compare a shipment embedding against a list of risk entries.

    For each entry, compute max cosine across its phrase + alias vectors.
    Returns hits that exceed the threshold. If ``force_action`` is provided
    (e.g., "block" for global entries), it overrides the entry's action.
    """
    hits = []
    for entry in entries:
        vectors = entry.get("_vectors")
        if vectors is None or len(vectors) == 0:
            continue

        # Cosine similarities (vectors and embedding are L2-normalized)
        sims = vectors @ embedding  # (k,)
        max_sim = float(sims.max())
        best_idx = int(sims.argmax())

        if max_sim >= threshold:
            all_texts = [entry["phrase"]] + entry.get("aliases", [])
            hits.append({
                "phrase":       entry["phrase"],
                "matched_text": all_texts[best_idx],
                "similarity":   round(max_sim, 4),
                "category":     category,
                "action":       force_action or entry["action"],
                "reason":       entry["reason"],
            })

    return hits


def _sample_risk_vector(risk_vectors: dict) -> np.ndarray | None:
    """Return the first available risk vector matrix, or None if empty."""
    for entry in risk_vectors.get("global_blocked", []):
        v = entry.get("_vectors")
        if v is not None and len(v) > 0:
            return v
    for cat in risk_vectors.get("categories", {}).values():
        for entry in cat.get("hard_negatives", []):
            v = entry.get("_vectors")
            if v is not None and len(v) > 0:
                return v
    return None


# ── Decision engine ───────────────────────────────────────────────────────────

def apply_compliance(classifier_result: dict, risk_vectors: dict) -> dict:
    """
    Produce a compliance decision from classifier output + pre-computed risk vectors.

    Parameters
    ----------
    classifier_result : dict
        Output of ``classifier.predict()`` — must have ``categories``,
        ``confidence_state``, ``scores``, and (when available) ``embedding``.
    risk_vectors : dict
        Output of ``prepare_risk_vectors()`` — profile enriched with numpy vectors.

    Returns
    -------
    dict with keys:
        compliance_decision : str   -- "allow", "review", or "block"
        decision_reasons    : list[str]
        hard_negative_hits  : list[dict]
        risk_levels         : dict[str, str]
    """
    embedding    = classifier_result.get("embedding")
    categories   = classifier_result.get("categories", [])
    conf_state   = classifier_result.get("confidence_state", "unclassified")
    cat_profiles = risk_vectors.get("categories", {})
    thresholds   = risk_vectors.get("default_thresholds", _DEFAULT_THRESHOLDS)

    reasons: list[str] = []
    all_hits: list[dict] = []
    risk_levels: dict[str, str] = {
        c: cat_profiles.get(c, {}).get("risk_level", "medium")
        for c in categories
    }

    # ── Defensive: dim mismatch (model swapped without /reload) ───────────
    if embedding is not None:
        sample = _sample_risk_vector(risk_vectors)
        if sample is not None:
            emb_dim = np.asarray(embedding).shape[-1]
            if sample.shape[1] != emb_dim:
                return {
                    "compliance_decision": "review",
                    "decision_reasons": [
                        f"embedding dim {emb_dim} != risk vector dim "
                        f"{sample.shape[1]}; semantic check skipped"
                    ],
                    "hard_negative_hits": [],
                    "risk_levels": risk_levels,
                }

    # ── No embedding (quality gate rejected) ──────────────────────────────
    if embedding is None:
        if conf_state in ("low_confidence", "unclassified"):
            reasons.append(f"confidence_state={conf_state}")
            return {
                "compliance_decision": "review",
                "decision_reasons": reasons,
                "hard_negative_hits": [],
                "risk_levels": risk_levels,
            }
        high_risk = [c for c, rl in risk_levels.items() if rl == "high"]
        if high_risk:
            reasons.append(f"high-risk category: {', '.join(sorted(high_risk))}")
            return {
                "compliance_decision": "review",
                "decision_reasons": reasons,
                "hard_negative_hits": [],
                "risk_levels": risk_levels,
            }
        reasons.append("classified (no embedding for semantic check)")
        return {
            "compliance_decision": "allow",
            "decision_reasons": reasons,
            "hard_negative_hits": [],
            "risk_levels": risk_levels,
        }

    # ── Step 1: Global blocked phrases (always block on match) ────────────
    block_threshold  = thresholds.get("block",  _DEFAULT_THRESHOLDS["block"])
    review_threshold = thresholds.get("review", _DEFAULT_THRESHOLDS["review"])

    global_hits = _match_entries(
        embedding,
        risk_vectors.get("global_blocked", []),
        threshold=block_threshold,
        category=_GLOBAL_SCOPE,
        force_action="block",
    )
    all_hits.extend(global_hits)

    # ── Step 2: Category-scoped hard negatives ────────────────────────────
    # If the classifier produced no categories, fall back to the top-scorer
    # so we can still apply category-scoped checks and risk-level routing.
    # NOTE: this mutates `risk_levels` to include the implied category.
    check_cats = list(categories)
    if not check_cats and classifier_result.get("scores"):
        scores = classifier_result["scores"]
        if scores:
            top = max(scores, key=lambda c: scores[c].get("final_score", 0))
            check_cats = [top]
            risk_levels[top] = cat_profiles.get(top, {}).get("risk_level", "medium")

    for cat in check_cats:
        cat_cfg = cat_profiles.get(cat, {})
        negatives = cat_cfg.get("hard_negatives", [])
        if not negatives:
            continue
        cat_hits = _match_entries(
            embedding,
            negatives,
            threshold=review_threshold,
            category=cat,
        )
        all_hits.extend(cat_hits)

    # ── Decision cascade ──────────────────────────────────────────────────

    # 1. Any block-action hit -> BLOCK
    block_hits = [h for h in all_hits if h["action"] == "block"]
    if block_hits:
        for h in block_hits:
            scope = "global" if h["category"] == _GLOBAL_SCOPE else f"category '{h['category']}'"
            reasons.append(
                f"semantic match '{h['phrase']}' ({scope}, "
                f"sim={h['similarity']:.3f}, matched='{h['matched_text']}'): "
                f"{h['reason']}"
            )
        return {
            "compliance_decision": "block",
            "decision_reasons": reasons,
            "hard_negative_hits": all_hits,
            "risk_levels": risk_levels,
        }

    # 2. Any review-action hit -> REVIEW
    review_hits = [h for h in all_hits if h["action"] == "review"]
    if review_hits:
        for h in review_hits:
            reasons.append(
                f"semantic match '{h['phrase']}' in category '{h['category']}' "
                f"(sim={h['similarity']:.3f}): {h['reason']}"
            )
        return {
            "compliance_decision": "review",
            "decision_reasons": reasons,
            "hard_negative_hits": all_hits,
            "risk_levels": risk_levels,
        }

    # 3. High-risk category -> REVIEW
    high_risk_cats = [c for c, rl in risk_levels.items() if rl == "high"]
    if high_risk_cats:
        reasons.append(f"high-risk category: {', '.join(sorted(high_risk_cats))}")
        return {
            "compliance_decision": "review",
            "decision_reasons": reasons,
            "hard_negative_hits": all_hits,
            "risk_levels": risk_levels,
        }

    # 4. Low confidence or unclassified -> REVIEW
    if conf_state in ("low_confidence", "unclassified"):
        reasons.append(f"confidence_state={conf_state}")
        return {
            "compliance_decision": "review",
            "decision_reasons": reasons,
            "hard_negative_hits": all_hits,
            "risk_levels": risk_levels,
        }

    # 5. Classified + medium/low risk -> ALLOW
    reasons.append(
        f"classified with confidence, risk_level(s): "
        f"{', '.join(f'{c}={rl}' for c, rl in sorted(risk_levels.items()))}"
    )
    return {
        "compliance_decision": "allow",
        "decision_reasons": reasons,
        "hard_negative_hits": all_hits,
        "risk_levels": risk_levels,
    }
