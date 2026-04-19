"""
Core classification logic.

No FastAPI imports here — this module is used by main.py, centroid_builder.py,
evaluate_threshold.py, fit_calibration.py, and discover_unknowns.py alike.

Scoring formula (per category):
    semantic_score = max_k cosine_similarity(query_embedding, sub_centroid_k)
                   = max_k dot(query_emb, sub_centroid_k)   [L2-normalized]
    keyword_score  = 1 - exp(-alpha * sum_of_weights_of_matched_keywords)
                     (word-boundary matching, naive plural stripping)
    final_score    = semantic_weight[cat] * semantic_score
                   + keyword_weight[cat]  * keyword_score
                     (defaults: 0.8 / 0.2; per-category override in DB)
    probability    = sigmoid(platt_a[cat] * final_score + platt_b[cat])
                     (NULL calibration → probability = final_score passthrough)

Confidence bands (evaluated on `final_score`, not probability):
    final_score (max across categories) < unclassified_threshold → unclassified
    unclassified_threshold ≤ max_score < threshold               → low_confidence
    max_score ≥ threshold                                        → classified

unclassified_threshold defaults to min(UNCLASSIFIED_THRESHOLD_DEFAULT, threshold)
so callers passing a low threshold (e.g. 0.25) don't get their results
clobbered by a higher floor.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize as sk_normalize

from chunking import chunk_text
from config import (
    CROSS_ENCODER_MARGIN_DEFAULT,
    KEYWORD_SATURATION_ALPHA,
    MARGIN_DELTA_DEFAULT,
    UNCLASSIFIED_THRESHOLD_DEFAULT,
)
from preprocess import build_query_text, extract_signals, normalize as preprocess_normalize

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Runtime metrics counters (Phase 1/3 observability) ────────────────────────
# Simple in-process counters, read by main.py's /metrics endpoint. Single-worker
# by design — if you scale to multiple uvicorn workers you'll get per-worker
# counts. That's fine for local tuning; switch to statsd/prom if ops cares.
_metrics: dict[str, int] = {
    "predictions_total":        0,
    "margin_suppressions_total": 0,  # labels dropped by the top-margin rule
    "rerank_fires_total":       0,   # rows where the reranker actually ran
    "rerank_drops_total":       0,   # labels dropped by the reranker
}


def get_metrics() -> dict[str, int]:
    """Snapshot of in-process counters for /metrics."""
    return dict(_metrics)


def reset_metrics() -> None:
    """Zero the counters — called by tests and, optionally, on /reload."""
    for k in _metrics:
        _metrics[k] = 0

# Generic tokens — if the normalized text consists entirely of these, it's
# flagged as "generic" (insufficient information to classify).
GENERIC_TOKENS: frozenset[str] = frozenset({
    "cargo", "goods", "items", "various", "misc", "miscellaneous",
    "freight", "shipment", "commodity", "general", "merchandise",
    "package", "parcel", "stuff", "thing", "things", "product", "products",
    "as", "per", "invoice", "the", "a", "an",
})

_TOKEN_RE = re.compile(r"\w+")

# Model is loaded once at module import. For BGE-small-en-v1.5 this is ~130MB
# and takes ~3-5s on first run, near-instant afterwards (HF cache).
# Fallback quietly to MiniLM if the env-specified model can't be loaded — keeps
# local dev frictionless.
try:
    model = SentenceTransformer(MODEL_NAME)
except Exception as exc:  # pragma: no cover - defensive
    print(f"WARNING: could not load EMBEDDING_MODEL={MODEL_NAME} ({exc}); falling back to all-MiniLM-L6-v2")
    MODEL_NAME = "all-MiniLM-L6-v2"
    model      = SentenceTransformer(MODEL_NAME)

# Cross-encoder reranker (Phase 3, optional). None = disabled. Callers opt in
# by invoking `load_cross_encoder()` once at startup (typically gated behind
# the RERANKER_ENABLED env var in main.py).
_cross_encoder = None


def load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Load (or reload) the cross-encoder reranker. Returns the handle."""
    global _cross_encoder
    from sentence_transformers import CrossEncoder  # lazy import — skip when disabled
    _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


def _sigmoid(x: float) -> float:
    """Stable sigmoid over raw CE logits. CE scores are unbounded, so we
    squash to (0, 1) before applying the margin — same threshold then means
    the same thing across different cross-encoder models."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _rerank_candidates(
    text: str,
    matched: list[str],
    scores: dict,
    category_descriptions: dict[str, str] | None,
    margin: float = CROSS_ENCODER_MARGIN_DEFAULT,
) -> list[str]:
    """Rerank `matched` via the cross-encoder; drop candidates more than
    `margin` below the top, measured in sigmoid-wrapped probability space.

    Short-circuits (returns `matched` unchanged) when:
      - the cross-encoder isn't loaded (disabled), OR
      - fewer than 2 candidates survived Phase 1 (nothing to rerank), OR
      - no category_descriptions were provided.

    Side effect: annotates each scored candidate with both the raw logit
    (``cross_encoder_score``) and the squashed probability (``cross_encoder_prob``)
    so the audit trail shows what the CE thought AND the comparable number
    the margin rule actually used.
    """
    if _cross_encoder is None or len(matched) < 2 or not category_descriptions:
        return matched
    pairs: list[tuple[str, str]] = []
    cats: list[str] = []
    for c in matched:
        desc = category_descriptions.get(c)
        if desc:
            pairs.append((text, desc))
            cats.append(c)
    if len(cats) < 2:
        return matched
    ce_logits = _cross_encoder.predict(pairs)
    ce_probs = [_sigmoid(float(s)) for s in ce_logits]
    for cat, logit, prob in zip(cats, ce_logits, ce_probs):
        scores[cat]["cross_encoder_score"] = round(float(logit), 4)
        scores[cat]["cross_encoder_prob"]  = round(prob, 4)
    ranked = sorted(zip(cats, ce_probs), key=lambda t: t[1], reverse=True)
    top_prob = ranked[0][1]
    kept = {c for c, p in ranked if p >= top_prob - margin}
    return [c for c in matched if c in kept]


# ── DB loaders ─────────────────────────────────────────────────────────────────

def load_centroids(conn) -> dict[str, list[tuple[str, np.ndarray]]]:
    """
    Load L2-normalized centroid vectors for all active categories, tagged with
    their HS 2-digit chapter.

    Returns:
        {category_name: [(hs_chapter, np.ndarray(dim,)), ...]}
        One list entry per child chapter. Categories with no centroids
        (seeded but not yet built) are omitted. The ``hs_chapter`` tag drives
        the ``best_chapter`` field in scoring output.

    Legacy rows written by older centroid_builder (hs_chapter NULL) are still
    loaded — their tag is the empty string, so they behave as a generic
    fallback centroid for the category.
    """
    result: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT cc.name, COALESCE(cen.hs_chapter, ''), cen.centroid
            FROM   category_centroids cen
            JOIN   classification_categories cc ON cc.id = cen.category_id
            WHERE  cc.is_active = true
            ORDER  BY cc.name, cen.hs_chapter
        """)
        for name, hs_chapter, centroid in cur.fetchall():
            result[name].append((hs_chapter, np.array(centroid)))
    return dict(result)


def load_chapter_titles(conn) -> dict[str, str]:
    """Return {hs_chapter: chapter_title} for all 96 active HS chapters."""
    titles: dict[str, str] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT hs_chapter, chapter_title FROM category_hs_chapters")
        for hs_chapter, title in cur.fetchall():
            titles[hs_chapter] = title
    return titles


def load_chapter_to_category(conn) -> dict[str, str]:
    """Return {hs_chapter: category_name} for the HS-code tiebreaker.

    Schema guarantees ``UNIQUE(hs_chapter)`` — each chapter maps to exactly one
    active category — so no ambiguity here.
    """
    mapping: dict[str, str] = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chc.hs_chapter, cc.name
            FROM   category_hs_chapters chc
            JOIN   classification_categories cc ON cc.id = chc.category_id
            WHERE  cc.is_active = true
        """)
        for hs_chapter, name in cur.fetchall():
            mapping[hs_chapter] = name
    return mapping


def _hs_implied_categories(
    hs_codes: list[str],
    chapter_to_category: dict[str, str] | None,
) -> list[str]:
    """Map extracted HS codes to the set of implied categories via their
    2-digit chapter prefix. Returns a deduplicated list preserving first-seen
    order for stable audit output.
    """
    if not hs_codes or not chapter_to_category:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for code in hs_codes:
        if len(code) < 2:
            continue
        chapter = code[:2]
        cat = chapter_to_category.get(chapter)
        if cat and cat not in seen:
            seen.add(cat)
            out.append(cat)
    return out


def _hs_text_mismatch(hs_implied: list[str], matched_categories: list[str]) -> bool:
    """True when HS chapter implies a category the text classifier did not commit to.

    Returns False when either side has no opinion (no HS code extracted, or text
    classifier returned nothing) — a missing opinion isn't a disagreement. Only
    returns True when both sides have concrete categories AND they don't overlap.
    """
    if not hs_implied or not matched_categories:
        return False
    return not (set(hs_implied) & set(matched_categories))


def load_keywords(conn) -> dict[str, list[tuple[str, float]]]:
    """
    Load keywords (lowercased) and their weights for all active categories.

    Returns:
        {category_name: [(keyword_lowercase, weight), ...]}
    """
    keywords: dict[str, list] = defaultdict(list)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT cc.name, ck.keyword, ck.weight
            FROM   category_keywords ck
            JOIN   classification_categories cc ON cc.id = ck.category_id
            WHERE  cc.is_active = true
            ORDER  BY cc.name, ck.keyword
        """)
        for name, keyword, weight in cur.fetchall():
            keywords[name].append((keyword.lower(), float(weight)))
    return dict(keywords)


def load_category_config(conn) -> dict[str, dict]:
    """
    Load per-category tuning parameters:
        semantic_weight, keyword_weight, platt_a, platt_b, threshold.

    Returns:
        {category_name: {"semantic_weight": float, "keyword_weight": float,
                         "platt_a": float|None, "platt_b": float|None,
                         "threshold": float|None}}
    """
    config: dict[str, dict] = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT name, semantic_weight, keyword_weight,
                   platt_a, platt_b, threshold
            FROM   classification_categories
            WHERE  is_active = true
        """)
        for row in cur.fetchall():
            name, sw, kw, pa, pb, thr = row
            config[name] = {
                "semantic_weight": float(sw) if sw is not None else 0.8,
                "keyword_weight":  float(kw) if kw is not None else 0.2,
                "platt_a":         float(pa) if pa is not None else None,
                "platt_b":         float(pb) if pb is not None else None,
                "threshold":       float(thr) if thr is not None else None,
            }
    return config


# ── Text helpers ───────────────────────────────────────────────────────────────

def text_quality_check(cargo: str, commodity: str) -> str:
    """
    Quality gate on the raw (pre-normalization) cargo + commodity pair.

    Returns:
        'too_short' — combined length < 10 chars after stripping
        'generic'   — every token is in GENERIC_TOKENS (e.g. "cargo goods")
        'ok'        — worth embedding
    """
    combined = f"{cargo or ''} {commodity or ''}".strip().lower()
    if len(combined) < 10:
        return "too_short"

    # Token-level check: strip punctuation, split on non-word, ignore 1-char tokens
    tokens = [t for t in _TOKEN_RE.findall(combined) if len(t) > 1]
    if not tokens:
        return "too_short"
    if all(t in GENERIC_TOKENS for t in tokens):
        return "generic"
    return "ok"


def _singularize(word: str) -> str:
    """Very lightweight plural→singular: 'batteries'→'batterie', 'boxes'→'box', 'cells'→'cell'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def keyword_score(
    text: str,
    keywords: list[tuple[str, float]],
) -> tuple[float, list[str]]:
    """
    Saturating keyword match score with word-boundary matching and plural
    tolerance.

    For each keyword, we match if either the keyword OR its naive plural/
    singular variant appears as a whole word in `text`. Punctuation and case
    are already normalized by the caller (preprocess.normalize).

    Returns:
        (score: float in [0, 1], matched_keywords: list[str])
    """
    if not keywords:
        return 0.0, []

    # Text was already lowercased upstream (normalize), but be defensive.
    text_lower = text.lower()

    # Pre-tokenize so word-boundary matching is cheap for multi-word keywords.
    # For single-word keywords we can use a regex set; for multi-word keywords
    # we fall back to substring with word-boundary anchors.

    hits: list[str] = []
    hit_weight = 0.0
    for kw, weight in keywords:
        kw_l = kw.lower()
        kw_singular = _singularize(kw_l)

        pattern_kw       = rf"\b{re.escape(kw_l)}s?\b"           # match kw and trailing 's'
        pattern_singular = rf"\b{re.escape(kw_singular)}s?\b"

        if re.search(pattern_kw, text_lower) or (
            kw_singular != kw_l and re.search(pattern_singular, text_lower)
        ):
            hits.append(kw)
            hit_weight += weight

    # Saturating: 1 - exp(-alpha * hit_weight_sum)
    # With alpha=1.2: 1 hit@1.0 → 0.70, 2 hits → 0.91, 3 hits → 0.97.
    # Very different from the old "hit_weight/total_weight" which gave ~0.05
    # per hit and made keyword_score contribute almost nothing.
    score = round(1.0 - math.exp(-KEYWORD_SATURATION_ALPHA * hit_weight), 4)
    return score, hits


# ── Embedding helper ───────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Encode and L2-normalize a list of preprocessed texts.
    Returns an (n, dim) array.
    """
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()))
    return sk_normalize(model.encode(texts, batch_size=512, show_progress_bar=False))


# ── Scoring core (shared by predict and predict_batch) ────────────────────────

def _score_one(
    text_norm: str,
    embedding: np.ndarray,
    centroids: dict[str, list[tuple[str, np.ndarray]]],
    keywords: dict[str, list[tuple[str, float]]],
    category_config: dict[str, dict],
    chapter_titles: dict[str, str] | None = None,
) -> tuple[dict, float]:
    """
    Compute per-category scores for one already-embedded, already-normalized text.
    Returns (scores_dict, max_final_score).

    `centroids[category]` is a list of (hs_chapter, centroid_vec) tuples — one
    entry per HS chapter assigned to that category. The winning chapter (by
    max cosine) is surfaced in ``scores[category]["best_chapter"]``.
    """
    if chapter_titles is None:
        chapter_titles = {}

    scores: dict[str, dict] = {}
    max_score = -1.0

    for category, children in centroids.items():
        # Max cosine across child-chapter centroids. Supervised per-chapter
        # clusters replace the old unsupervised k-means split.
        chapters = [c[0] for c in children]
        stacked  = np.stack([c[1] for c in children])     # (k, dim)
        sims     = stacked @ embedding                    # (k,)
        sem      = round(float(sims.max()), 4)
        best_k   = int(sims.argmax())
        best_chapter = chapters[best_k]

        kw_s, hits = keyword_score(text_norm, keywords.get(category, []))

        cfg = category_config.get(category, {})
        sw  = cfg.get("semantic_weight", 0.8)
        kw_w = cfg.get("keyword_weight",  0.2)
        # Guard against weights that don't sum near 1 (misconfiguration).
        if sw + kw_w <= 0:
            sw, kw_w = 0.8, 0.2

        final = round(sw * sem + kw_w * kw_s, 4)

        # Platt calibration if fit, else passthrough.
        pa = cfg.get("platt_a")
        pb = cfg.get("platt_b")
        if pa is not None and pb is not None:
            probability = round(1.0 / (1.0 + math.exp(-(pa * final + pb))), 4)
        else:
            probability = final

        scores[category] = {
            "semantic_score":     sem,
            "keyword_score":      kw_s,
            "final_score":        final,
            "probability":        probability,
            "matched":            False,
            "keywords_hit":       hits,
            "best_chapter":       best_chapter,
            "best_chapter_title": chapter_titles.get(best_chapter, ""),
            "best_cluster":       best_k,  # retained for one release; prefer best_chapter
        }
        if final > max_score:
            max_score = final

    return scores, max_score


def _apply_bands(
    scores: dict,
    max_score: float,
    threshold: float,
    unclassified_threshold: float,
    margin_delta: float = MARGIN_DELTA_DEFAULT,
) -> tuple[list[str], str, str | None]:
    """
    Apply the confidence bands to already-computed per-category scores.
    Returns (matched_categories, confidence_state, reason).

    Decision is on `final_score`. Platt `probability` is reported alongside
    for UI/confidence display but is not used for thresholding — with a
    severe class imbalance (8:152 per category in the training set) Platt
    sigmoids compress everything below ~0.1, which would make any reasonable
    threshold exclude all positives.

    Top-margin rule: in the `classified` band, secondary labels must be within
    `margin_delta` of the top score to fire. Prevents weak tail labels from
    leaking through on multi-label output when single-token keyword hits nudge
    an unrelated category just above `threshold`.
    """
    if max_score < unclassified_threshold:
        return [], "unclassified", "low_similarity"

    if max_score < threshold:
        top_cat = max(scores, key=lambda c: scores[c]["final_score"])
        scores[top_cat]["matched"] = True
        return [top_cat], "low_confidence", None

    cutoff = max_score - margin_delta
    above_threshold = [c for c, s in scores.items() if s["final_score"] >= threshold]
    matched = [c for c in above_threshold if scores[c]["final_score"] >= cutoff]
    for c in matched:
        scores[c]["matched"] = True
    _metrics["margin_suppressions_total"] += len(above_threshold) - len(matched)
    return matched, "classified", None


def _maybe_rerank(
    text: str,
    matched: list[str],
    state: str,
    scores: dict,
    category_descriptions: dict[str, str] | None,
) -> tuple[list[str], bool]:
    """Call the cross-encoder reranker when appropriate; update `scores[cat]["matched"]`
    flags for any categories the reranker drops.

    Returns ``(matched_list, reranked_flag)``. `reranked_flag` is True iff the
    reranker actually ran (regardless of whether it shortened the list).
    """
    if state != "classified" or len(matched) < 2 or _cross_encoder is None:
        return matched, False
    new_matched = _rerank_candidates(text, matched, scores, category_descriptions)
    _metrics["rerank_fires_total"] += 1
    if len(new_matched) != len(matched):
        _metrics["rerank_drops_total"] += len(matched) - len(new_matched)
        kept = set(new_matched)
        for c in matched:
            if c not in kept:
                scores[c]["matched"] = False
    return new_matched, True


# ── Chunk aggregation ─────────────────────────────────────────────────────────

def _aggregate_chunk_scores(
    per_chunk: list[dict],
) -> tuple[dict, float, int]:
    """
    Combine per-chunk score dicts into one shipment-level score dict.

    For each category, take the row from the chunk with the highest
    ``final_score`` for that category — preserves the invariant
    ``final_score == sw*sem + kw*kw`` (i.e. all sub-scores come from the
    same chunk and aren't Frankenstein-merged across chunks). The only
    union field is ``keywords_hit``: gathered from every chunk for audit
    completeness.

    Returns
    -------
    (aggregated_scores, max_final_score, overall_winner_chunk_idx)
        ``overall_winner_chunk_idx`` is the index of the chunk that produced
        the highest ``final_score`` across all categories. The caller uses it
        to choose which chunk's embedding to retain on the result.
    """
    if not per_chunk:
        return {}, -1.0, 0
    if len(per_chunk) == 1:
        scores = per_chunk[0]
        max_score = max((s["final_score"] for s in scores.values()), default=-1.0)
        return scores, max_score, 0

    # Categories are identical across chunks (same centroids, same config).
    categories = list(per_chunk[0].keys())
    aggregated: dict[str, dict] = {}
    overall_max = -1.0
    overall_idx = 0

    for cat in categories:
        # Per-category: pick the chunk with the highest final_score.
        best_idx = max(
            range(len(per_chunk)),
            key=lambda i: per_chunk[i][cat]["final_score"],
        )
        winning_row = dict(per_chunk[best_idx][cat])  # shallow copy

        # Union keywords across chunks (audit completeness).
        union_hits: list[str] = []
        seen: set[str] = set()
        for row_idx in range(len(per_chunk)):
            for kw in per_chunk[row_idx][cat].get("keywords_hit", []):
                if kw not in seen:
                    seen.add(kw)
                    union_hits.append(kw)
        winning_row["keywords_hit"] = union_hits

        aggregated[cat] = winning_row

        if winning_row["final_score"] > overall_max:
            overall_max = winning_row["final_score"]
            overall_idx = best_idx

    return aggregated, overall_max, overall_idx


# ── Public API ─────────────────────────────────────────────────────────────────

def predict(
    cargo_description: str,
    commodity_description: str,
    centroids: dict[str, list[tuple[str, np.ndarray]]],
    keywords: dict[str, list[tuple[str, float]]],
    category_config: dict[str, dict] | None = None,
    threshold: float = 0.45,
    unclassified_threshold: float | None = None,
    chapter_titles: dict[str, str] | None = None,
    chapter_to_category: dict[str, str] | None = None,
    category_descriptions: dict[str, str] | None = None,
    margin_delta: float = MARGIN_DELTA_DEFAULT,
) -> dict:
    """
    Classify a single shipment.

    Backwards-compatible with the old signature (category_config,
    chapter_to_category optional). Callers that haven't been updated yet can
    still call: ``predict(cargo, commodity, centroids, keywords, threshold=0.45)``.
    """
    if category_config is None:
        category_config = {}

    if unclassified_threshold is None:
        unclassified_threshold = min(UNCLASSIFIED_THRESHOLD_DEFAULT, threshold)
    else:
        unclassified_threshold = min(unclassified_threshold, threshold)

    _metrics["predictions_total"] += 1

    # Extract structured HS/HTS codes BEFORE normalization strips them.
    raw_combined = f"{cargo_description or ''} {commodity_description or ''}"
    signals = extract_signals(raw_combined)
    hs_implied = _hs_implied_categories(signals.hs_codes, chapter_to_category)

    quality = text_quality_check(cargo_description, commodity_description)
    if quality != "ok":
        return {
            "categories":             [],
            "confidence_state":       "unclassified",
            "reason":                 "insufficient_input",
            "scores":                 {},
            "embedding":              None,
            "text_quality":           quality,
            "chunks_processed":       0,
            "chunk_embeddings":       None,
            "hs_codes_extracted":     signals.hs_codes,
            "hs_implied_categories":  hs_implied,
            "hs_text_mismatch":       False,
            "reranked":               False,
        }

    text_for_embedding = build_query_text(cargo_description, commodity_description)
    text_for_keywords  = preprocess_normalize(f"{cargo_description} {commodity_description}")

    chunks = chunk_text(text_for_embedding, model)

    if len(chunks) == 1:
        # Fast path — byte-identical behavior to the pre-chunking version.
        embedding = embed_texts([text_for_embedding])[0]
        scores, max_score = _score_one(
            text_for_keywords, embedding, centroids, keywords, category_config,
            chapter_titles=chapter_titles,
        )
        matched, state, reason = _apply_bands(scores, max_score, threshold, unclassified_threshold, margin_delta)
        matched, reranked = _maybe_rerank(
            text_for_embedding, matched, state, scores, category_descriptions
        )
        return {
            "categories":            matched,
            "confidence_state":      state,
            "reason":                reason,
            "scores":                scores,
            "embedding":             embedding,
            "text_quality":          quality,
            "chunks_processed":      1,
            "chunk_embeddings":      None,
            "hs_codes_extracted":    signals.hs_codes,
            "hs_implied_categories": hs_implied,
            "hs_text_mismatch":      _hs_text_mismatch(hs_implied, matched),
            "reranked":              reranked,
        }

    # ── Multi-chunk path ─────────────────────────────────────────────────
    chunk_embeddings = embed_texts(chunks)  # (n_chunks, dim)
    per_chunk_scores: list[dict] = []
    for i in range(len(chunks)):
        s, _ = _score_one(
            text_for_keywords, chunk_embeddings[i],
            centroids, keywords, category_config,
            chapter_titles=chapter_titles,
        )
        per_chunk_scores.append(s)

    scores, max_score, winner_idx = _aggregate_chunk_scores(per_chunk_scores)
    matched, state, reason = _apply_bands(scores, max_score, threshold, unclassified_threshold)
    matched, reranked = _maybe_rerank(
        text_for_embedding, matched, state, scores, category_descriptions
    )
    embedding = chunk_embeddings[winner_idx]

    return {
        "categories":            matched,
        "confidence_state":      state,
        "reason":                reason,
        "scores":                scores,
        "embedding":             embedding,
        "text_quality":          quality,
        "chunks_processed":      len(chunks),
        "chunk_embeddings":      chunk_embeddings,
        "hs_codes_extracted":    signals.hs_codes,
        "hs_implied_categories": hs_implied,
        "hs_text_mismatch":      _hs_text_mismatch(hs_implied, matched),
        "reranked":              reranked,
    }


def predict_batch(
    inputs: Iterable[tuple[str, str]],
    centroids: dict[str, list[tuple[str, np.ndarray]]],
    keywords: dict[str, list[tuple[str, float]]],
    category_config: dict[str, dict] | None = None,
    thresholds: list[float] | float = 0.45,
    unclassified_thresholds: list[float | None] | float | None = None,
    chapter_titles: dict[str, str] | None = None,
    chapter_to_category: dict[str, str] | None = None,
    category_descriptions: dict[str, str] | None = None,
    margin_delta: float = MARGIN_DELTA_DEFAULT,
) -> list[dict]:
    """
    Batched inference — single model.encode() call for all valid rows.

    `inputs` is an iterable of (cargo, commodity) pairs. `thresholds` may be a
    single float or a per-row list. Returns a list of per-row result dicts in
    the same order.

    This is the code path used by both /classify and /classify/batch; the old
    duplicated inline-scoring logic in main.py is gone.
    """
    if category_config is None:
        category_config = {}

    inputs_list = list(inputs)
    n = len(inputs_list)
    _metrics["predictions_total"] += n

    if isinstance(thresholds, (int, float)):
        thresholds = [float(thresholds)] * n
    if not isinstance(unclassified_thresholds, list):
        unclassified_thresholds = [unclassified_thresholds] * n  # type: ignore[list-item]

    results: list[dict | None] = [None] * n
    valid_indices: list[int] = []
    keyword_texts_list: list[str] = []
    # Raw query text per row — fed to the reranker when Phase 3 is enabled.
    query_texts_list: list[str] = []
    # HS signals extracted per-row (aligned with valid_indices).
    row_hs_codes: list[list[str]] = []
    row_hs_implied: list[list[str]] = []
    # all_chunks is the flat list passed to ONE embed_texts call. row_chunk_spans
    # lets us slice the resulting matrix back into per-row chunk groups.
    all_chunks: list[str] = []
    row_chunk_spans: list[tuple[int, int]] = []  # (start_offset, count) per valid row

    # Quality gate + normalization + chunking
    for i, (cargo, commodity) in enumerate(inputs_list):
        signals = extract_signals(f"{cargo or ''} {commodity or ''}")
        hs_implied = _hs_implied_categories(signals.hs_codes, chapter_to_category)

        quality = text_quality_check(cargo, commodity)
        if quality != "ok":
            results[i] = {
                "categories":            [],
                "confidence_state":      "unclassified",
                "reason":                "insufficient_input",
                "scores":                {},
                "embedding":             None,
                "text_quality":          quality,
                "chunks_processed":      0,
                "chunk_embeddings":      None,
                "hs_codes_extracted":    signals.hs_codes,
                "hs_implied_categories": hs_implied,
                "hs_text_mismatch":      False,
                "reranked":              False,
            }
            continue
        valid_indices.append(i)
        text_for_embedding = build_query_text(cargo, commodity)
        keyword_texts_list.append(preprocess_normalize(f"{cargo} {commodity}"))
        query_texts_list.append(text_for_embedding)
        row_hs_codes.append(signals.hs_codes)
        row_hs_implied.append(hs_implied)
        chunks = chunk_text(text_for_embedding, model)
        row_chunk_spans.append((len(all_chunks), len(chunks)))
        all_chunks.extend(chunks)

    # Single batched embed call for ALL chunks across the entire batch.
    if all_chunks:
        all_embeddings = embed_texts(all_chunks)
        for offset, idx in enumerate(valid_indices):
            start, count = row_chunk_spans[offset]
            row_embeddings = all_embeddings[start : start + count]  # (count, dim)
            text_norm  = keyword_texts_list[offset]
            hs_codes   = row_hs_codes[offset]
            hs_implied = row_hs_implied[offset]
            thr = thresholds[idx]
            unc = unclassified_thresholds[idx]
            if unc is None:
                unc = min(UNCLASSIFIED_THRESHOLD_DEFAULT, thr)
            else:
                unc = min(unc, thr)

            query_text = query_texts_list[offset]

            if count == 1:
                # Fast path — same shape as today.
                embedding = row_embeddings[0]
                scores, max_score = _score_one(
                    text_norm, embedding, centroids, keywords, category_config,
                    chapter_titles=chapter_titles,
                )
                matched, state, reason = _apply_bands(scores, max_score, thr, unc, margin_delta)
                matched, reranked = _maybe_rerank(
                    query_text, matched, state, scores, category_descriptions
                )
                results[idx] = {
                    "categories":            matched,
                    "confidence_state":      state,
                    "reason":                reason,
                    "scores":                scores,
                    "embedding":             embedding,
                    "text_quality":          "ok",
                    "chunks_processed":      1,
                    "chunk_embeddings":      None,
                    "hs_codes_extracted":    hs_codes,
                    "hs_implied_categories": hs_implied,
                    "hs_text_mismatch":      _hs_text_mismatch(hs_implied, matched),
                    "reranked":              reranked,
                }
            else:
                per_chunk_scores: list[dict] = []
                for ci in range(count):
                    s, _ = _score_one(
                        text_norm, row_embeddings[ci],
                        centroids, keywords, category_config,
                        chapter_titles=chapter_titles,
                    )
                    per_chunk_scores.append(s)
                scores, max_score, winner_idx = _aggregate_chunk_scores(per_chunk_scores)
                matched, state, reason = _apply_bands(scores, max_score, thr, unc, margin_delta)
                matched, reranked = _maybe_rerank(
                    query_text, matched, state, scores, category_descriptions
                )
                results[idx] = {
                    "categories":            matched,
                    "confidence_state":      state,
                    "reason":                reason,
                    "scores":                scores,
                    "embedding":             row_embeddings[winner_idx],
                    "text_quality":          "ok",
                    "chunks_processed":      count,
                    "chunk_embeddings":      row_embeddings,
                    "hs_codes_extracted":    hs_codes,
                    "hs_implied_categories": hs_implied,
                    "hs_text_mismatch":      _hs_text_mismatch(hs_implied, matched),
                    "reranked":              reranked,
                }

    return results  # type: ignore[return-value]
