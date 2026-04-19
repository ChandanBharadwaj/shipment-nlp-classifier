"""
Semantic compliance decision layer.

Uses the same sentence-transformer embeddings as the classifier to match
shipments against risk phrases. Instead of brittle keyword regex, each risk
phrase (+ aliases) is embedded at startup. At inference, the *already-computed*
shipment embedding is compared against risk vectors via cosine similarity.

This means:
  - "depleted uranium fuel rods" catches "spent nuclear fuel", "DU penetrator"
  - "armored" / "armoured" / "armor-plated" are naturally similar vectors
  - No duplicated entries for spelling variants

Both shipment text and risk phrases pass through ``preprocess.normalize`` before
embedding so the two distributions stay aligned. The original phrase text is
preserved for audit output.

Schema (risk_profile.json):
  - default_thresholds: {risky: float}
  - global_blocked: list of {phrase, aliases?, risky: true, reason}
  - categories: dict of {hard_negatives, safe_exceptions?}
        hard_negatives   — list of {phrase, aliases?, reason}
        safe_exceptions  — list of {phrase, aliases?, deflects: [phrase,...], reason}
          When a safe_exception hits with higher cosine similarity than the
          hard_negatives listed in ``deflects``, those hits are suppressed.

Decision rule — one bit, phrase matches only:
    Any global_blocked or category hard_negative hit (cosine ≥ risky_threshold)
    that is not deflected by a safe_exception  ->  is_risky = True
    Otherwise                                   ->  is_risky = False
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from preprocess import normalize as _normalize_text

_DEFAULT_PROFILE_PATH = Path(__file__).parent / "risk_profile.json"

# Single cosine threshold for "is this shipment risky?". Overridable via
# default_thresholds.risky in risk_profile.json.
_DEFAULT_THRESHOLDS = {"risky": 0.55}

_GLOBAL_SCOPE = "_global"

_REQUIRED_GLOBAL_KEYS         = {"phrase", "reason", "risky"}
_REQUIRED_CATEGORY_KEYS       = {"phrase", "reason"}
_REQUIRED_SAFE_EXCEPTION_KEYS = {"phrase", "deflects", "reason"}


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
    for entry in entries:
        missing = _REQUIRED_CATEGORY_KEYS - entry.keys()
        if missing:
            raise ValueError(
                f"Category '{category}' hard_negative missing keys "
                f"{sorted(missing)}: {entry.get('phrase', '<no phrase>')}"
            )


def _validate_safe_exceptions(category: str, entries: list[dict]) -> None:
    for entry in entries:
        missing = _REQUIRED_SAFE_EXCEPTION_KEYS - entry.keys()
        if missing:
            raise ValueError(
                f"Category '{category}' safe_exception missing keys "
                f"{sorted(missing)}: {entry.get('phrase', '<no phrase>')}"
            )
        deflects = entry["deflects"]
        if not isinstance(deflects, list) or not deflects:
            raise ValueError(
                f"Category '{category}' safe_exception '{entry['phrase']}' "
                f"must have a non-empty 'deflects' list"
            )


# ── Vector preparation (called once at startup / reload) ─────────────────────

def _embed_entries(
    entries: list[dict],
    embed_fn: Callable[[list[str]], np.ndarray],
) -> tuple[list[dict], int]:
    all_texts: list[str] = []
    entry_spans: list[tuple[int, int]] = []
    entry_displays: list[list[str]] = []

    for entry in entries:
        originals = [entry["phrase"]] + entry.get("aliases", [])
        kept_orig: list[str] = []
        kept_norm: list[str] = []
        for orig in originals:
            norm = _normalize_text(orig)
            if not norm:
                continue
            kept_orig.append(orig)
            kept_norm.append(norm)
        start = len(all_texts)
        all_texts.extend(kept_norm)
        entry_spans.append((start, len(kept_norm)))
        entry_displays.append(kept_orig)

    if not all_texts:
        return entries, 0

    all_vectors = embed_fn(all_texts)

    enriched: list[dict] = []
    n_vectors = 0
    for i, entry in enumerate(entries):
        start, count = entry_spans[i]
        vectors = all_vectors[start : start + count]
        enriched.append({
            **entry,
            "_vectors": vectors,
            "_display_texts": entry_displays[i],
        })
        n_vectors += vectors.shape[0]

    return enriched, n_vectors


def prepare_risk_vectors(
    profile: dict,
    embed_fn: Callable[[list[str]], np.ndarray],
) -> dict:
    """Embed all risk phrases and return an enriched profile with numpy vectors."""
    global_raw = profile.get("global_blocked", [])
    _validate_global(global_raw)
    global_entries, n_global_vectors = _embed_entries(global_raw, embed_fn)

    categories: dict[str, dict] = {}
    n_cat_vectors = 0
    n_safe_vectors = 0
    for cat_name, cat_cfg in profile.get("categories", {}).items():
        negs = cat_cfg.get("hard_negatives", [])
        _validate_category(cat_name, negs)
        enriched_negs, n_v = _embed_entries(negs, embed_fn)
        n_cat_vectors += n_v

        safe_raw = cat_cfg.get("safe_exceptions", [])
        _validate_safe_exceptions(cat_name, safe_raw)
        enriched_safe, n_s = _embed_entries(safe_raw, embed_fn)
        n_safe_vectors += n_s

        categories[cat_name] = {
            "hard_negatives":  enriched_negs,
            "safe_exceptions": enriched_safe,
        }

    thresholds = dict(_DEFAULT_THRESHOLDS)
    thresholds.update(profile.get("default_thresholds", {}))

    return {
        "default_thresholds": thresholds,
        "global_blocked":     global_entries,
        "categories":         categories,
        "_stats": {
            "global_entries":         len(global_entries),
            "category_entries":       sum(len(c["hard_negatives"]) for c in categories.values()),
            "safe_exception_entries": sum(len(c["safe_exceptions"]) for c in categories.values()),
            "total_vectors":          n_global_vectors + n_cat_vectors + n_safe_vectors,
        },
    }


# ── Semantic matching ─────────────────────────────────────────────────────────

def _match_entries(
    embedding: np.ndarray,
    entries: list[dict],
    threshold: float,
    category: str,
) -> list[dict]:
    """Return all entries with max cosine similarity >= threshold."""
    M = np.atleast_2d(embedding)  # (n_chunks, dim)

    hits = []
    for entry in entries:
        vectors = entry.get("_vectors")
        if vectors is None or len(vectors) == 0:
            continue

        sims = vectors @ M.T
        max_sim = float(sims.max())

        if max_sim >= threshold:
            ph_idx, ch_idx = np.unravel_index(int(sims.argmax()), sims.shape)
            display_texts = entry.get("_display_texts") or (
                [entry["phrase"]] + entry.get("aliases", [])
            )
            matched = (
                display_texts[ph_idx] if ph_idx < len(display_texts)
                else entry["phrase"]
            )
            hit = {
                "phrase":       entry["phrase"],
                "matched_text": matched,
                "similarity":   round(max_sim, 4),
                "category":     category,
                "reason":       entry["reason"],
                "chunk_idx":    int(ch_idx),
            }
            if "deflects" in entry:
                hit["deflects"] = list(entry["deflects"])
            hits.append(hit)

    return hits


def _apply_safe_exceptions(
    hard_hits: list[dict],
    safe_hits: list[dict],
) -> tuple[list[dict], list[dict], list[str]]:
    """Drop hard_negative hits that are deflected by a higher-similarity safe_exception."""
    if not safe_hits or not hard_hits:
        return hard_hits, [], []

    best_deflect: dict[str, tuple[float, dict]] = {}
    for s in safe_hits:
        s_sim = s["similarity"]
        for phrase in s.get("deflects", []):
            prev = best_deflect.get(phrase)
            if prev is None or s_sim > prev[0]:
                best_deflect[phrase] = (s_sim, s)

    kept: list[dict] = []
    dropped: list[dict] = []
    reasons: list[str] = []
    for h in hard_hits:
        deflect = best_deflect.get(h["phrase"])
        if deflect is not None and deflect[0] >= h["similarity"]:
            s_sim, s = deflect
            drop = dict(h)
            drop["deflected_by"] = s["phrase"]
            drop["deflected_by_similarity"] = s_sim
            dropped.append(drop)
            reasons.append(
                f"deflected '{h['phrase']}' (sim={h['similarity']:.3f}) "
                f"by safe_exception '{s['phrase']}' (sim={s_sim:.3f})"
            )
        else:
            kept.append(h)

    return kept, dropped, reasons


def _sample_risk_vector(risk_vectors: dict) -> np.ndarray | None:
    for entry in risk_vectors.get("global_blocked", []):
        v = entry.get("_vectors")
        if v is not None and len(v) > 0:
            return v
    for cat in risk_vectors.get("categories", {}).values():
        for entry in cat.get("hard_negatives", []):
            v = entry.get("_vectors")
            if v is not None and len(v) > 0:
                return v
        for entry in cat.get("safe_exceptions", []):
            v = entry.get("_vectors")
            if v is not None and len(v) > 0:
                return v
    return None


# ── Decision engine ───────────────────────────────────────────────────────────

def apply_compliance(classifier_result: dict, risk_vectors: dict) -> dict:
    """
    Produce a compliance decision from classifier output + pre-computed risk vectors.

    Returns
    -------
    dict with keys:
        is_risky            : bool  — True if any non-deflected phrase hit
        hard_negative_hits  : list[dict]  — surviving hits that drove is_risky
        deflected_hits      : list[dict]  — safe-exception-suppressed hits (audit)
        decision_reasons    : list[str]
    """
    chunk_matrix = classifier_result.get("chunk_embeddings")
    embedding    = chunk_matrix if chunk_matrix is not None else classifier_result.get("embedding")
    categories   = classifier_result.get("categories", [])
    cat_profiles = risk_vectors.get("categories", {})
    thresholds   = risk_vectors.get("default_thresholds", _DEFAULT_THRESHOLDS)

    reasons: list[str] = []
    all_hits: list[dict] = []
    deflected_hits: list[dict] = []

    risky_threshold = thresholds.get("risky", _DEFAULT_THRESHOLDS["risky"])

    def _build() -> dict:
        return {
            "is_risky":           bool(all_hits),
            "hard_negative_hits": all_hits,
            "deflected_hits":     deflected_hits,
            "decision_reasons":   reasons,
        }

    # Defensive: dim mismatch (model swapped without /reload)
    if embedding is not None:
        sample = _sample_risk_vector(risk_vectors)
        if sample is not None:
            emb_dim = np.asarray(embedding).shape[-1]
            if sample.shape[1] != emb_dim:
                reasons.append(
                    f"embedding dim {emb_dim} != risk vector dim "
                    f"{sample.shape[1]}; semantic check skipped"
                )
                return _build()

    if embedding is None:
        reasons.append("no embedding available for semantic check")
        return _build()

    # Global blocked phrases
    global_hits = _match_entries(
        embedding,
        risk_vectors.get("global_blocked", []),
        threshold=risky_threshold,
        category=_GLOBAL_SCOPE,
    )
    all_hits.extend(global_hits)

    # Category hard_negatives — fall back to top-scoring category when classifier
    # returned no labels so we can still apply category-scoped checks.
    check_cats = list(categories)
    if not check_cats and classifier_result.get("scores"):
        scores = classifier_result["scores"]
        if scores:
            top = max(scores, key=lambda c: scores[c].get("final_score", 0))
            check_cats = [top]

    for cat in check_cats:
        cat_cfg = cat_profiles.get(cat, {})
        negatives = cat_cfg.get("hard_negatives", [])
        safe_ex   = cat_cfg.get("safe_exceptions", [])
        if not negatives:
            continue

        cat_hits = _match_entries(
            embedding, negatives, threshold=risky_threshold, category=cat,
        )
        if not cat_hits:
            continue

        safe_hits = _match_entries(
            embedding, safe_ex, threshold=risky_threshold, category=cat,
        ) if safe_ex else []

        kept, dropped, deflect_reasons = _apply_safe_exceptions(cat_hits, safe_hits)
        all_hits.extend(kept)
        deflected_hits.extend(dropped)
        reasons.extend(deflect_reasons)

    if all_hits:
        for h in all_hits:
            scope = "global" if h["category"] == _GLOBAL_SCOPE else f"category '{h['category']}'"
            reasons.append(
                f"semantic match '{h['phrase']}' ({scope}, "
                f"sim={h['similarity']:.3f}, matched='{h['matched_text']}'): "
                f"{h['reason']}"
            )
    else:
        reasons.append("no risk phrase matches")

    return _build()
