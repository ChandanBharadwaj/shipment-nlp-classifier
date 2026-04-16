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
from preprocess import build_query_text, normalize as preprocess_normalize

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_NAME                    = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
UNCLASSIFIED_THRESHOLD_DEFAULT = 0.35
KEYWORD_SATURATION_ALPHA       = 0.5   # 1 hit@1.0 → 0.39, 2 hits → 0.63, 3 hits → 0.78

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
) -> tuple[list[str], str, str | None]:
    """
    Apply the confidence bands to already-computed per-category scores.
    Returns (matched_categories, confidence_state, reason).

    Decision is on `final_score`. Platt `probability` is reported alongside
    for UI/confidence display but is not used for thresholding — with a
    severe class imbalance (8:152 per category in the training set) Platt
    sigmoids compress everything below ~0.1, which would make any reasonable
    threshold exclude all positives.
    """
    if max_score < unclassified_threshold:
        return [], "unclassified", "low_similarity"

    if max_score < threshold:
        top_cat = max(scores, key=lambda c: scores[c]["final_score"])
        scores[top_cat]["matched"] = True
        return [top_cat], "low_confidence", None

    matched = [c for c, s in scores.items() if s["final_score"] >= threshold]
    for c in matched:
        scores[c]["matched"] = True
    return matched, "classified", None


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
) -> dict:
    """
    Classify a single shipment.

    Backwards-compatible with the old signature (category_config optional).
    Callers that haven't been updated yet can still call:
        predict(cargo, commodity, centroids, keywords, threshold=0.45)
    """
    if category_config is None:
        category_config = {}

    if unclassified_threshold is None:
        unclassified_threshold = min(UNCLASSIFIED_THRESHOLD_DEFAULT, threshold)
    else:
        unclassified_threshold = min(unclassified_threshold, threshold)

    quality = text_quality_check(cargo_description, commodity_description)
    if quality != "ok":
        return {
            "categories":       [],
            "confidence_state": "unclassified",
            "reason":           "insufficient_input",
            "scores":           {},
            "embedding":        None,
            "quality":          quality,
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
        matched, state, reason = _apply_bands(scores, max_score, threshold, unclassified_threshold)
        return {
            "categories":        matched,
            "confidence_state":  state,
            "reason":            reason,
            "scores":            scores,
            "embedding":         embedding,
            "chunks_processed":  1,
            "chunk_embeddings":  None,
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
    embedding = chunk_embeddings[winner_idx]

    return {
        "categories":        matched,
        "confidence_state":  state,
        "reason":            reason,
        "scores":            scores,
        "embedding":         embedding,
        "chunks_processed":  len(chunks),
        "chunk_embeddings":  chunk_embeddings,
    }


def predict_batch(
    inputs: Iterable[tuple[str, str]],
    centroids: dict[str, list[tuple[str, np.ndarray]]],
    keywords: dict[str, list[tuple[str, float]]],
    category_config: dict[str, dict] | None = None,
    thresholds: list[float] | float = 0.45,
    unclassified_thresholds: list[float | None] | float | None = None,
    chapter_titles: dict[str, str] | None = None,
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

    if isinstance(thresholds, (int, float)):
        thresholds = [float(thresholds)] * n
    if not isinstance(unclassified_thresholds, list):
        unclassified_thresholds = [unclassified_thresholds] * n  # type: ignore[list-item]

    results: list[dict | None] = [None] * n
    valid_indices: list[int] = []
    keyword_texts_list: list[str] = []
    # all_chunks is the flat list passed to ONE embed_texts call. row_chunk_spans
    # lets us slice the resulting matrix back into per-row chunk groups.
    all_chunks: list[str] = []
    row_chunk_spans: list[tuple[int, int]] = []  # (start_offset, count) per valid row

    # Quality gate + normalization + chunking
    for i, (cargo, commodity) in enumerate(inputs_list):
        quality = text_quality_check(cargo, commodity)
        if quality != "ok":
            results[i] = {
                "categories":       [],
                "confidence_state": "unclassified",
                "reason":           "insufficient_input",
                "scores":           {},
                "embedding":        None,
                "quality":          quality,
                "chunks_processed": 0,
                "chunk_embeddings": None,
            }
            continue
        valid_indices.append(i)
        text_for_embedding = build_query_text(cargo, commodity)
        keyword_texts_list.append(preprocess_normalize(f"{cargo} {commodity}"))
        chunks = chunk_text(text_for_embedding, model)
        row_chunk_spans.append((len(all_chunks), len(chunks)))
        all_chunks.extend(chunks)

    # Single batched embed call for ALL chunks across the entire batch.
    if all_chunks:
        all_embeddings = embed_texts(all_chunks)
        for offset, idx in enumerate(valid_indices):
            start, count = row_chunk_spans[offset]
            row_embeddings = all_embeddings[start : start + count]  # (count, dim)
            text_norm = keyword_texts_list[offset]
            thr = thresholds[idx]
            unc = unclassified_thresholds[idx]
            if unc is None:
                unc = min(UNCLASSIFIED_THRESHOLD_DEFAULT, thr)
            else:
                unc = min(unc, thr)

            if count == 1:
                # Fast path — same shape as today.
                embedding = row_embeddings[0]
                scores, max_score = _score_one(
                    text_norm, embedding, centroids, keywords, category_config,
                    chapter_titles=chapter_titles,
                )
                matched, state, reason = _apply_bands(scores, max_score, thr, unc)
                results[idx] = {
                    "categories":        matched,
                    "confidence_state":  state,
                    "reason":            reason,
                    "scores":            scores,
                    "embedding":         embedding,
                    "chunks_processed":  1,
                    "chunk_embeddings":  None,
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
                matched, state, reason = _apply_bands(scores, max_score, thr, unc)
                results[idx] = {
                    "categories":        matched,
                    "confidence_state":  state,
                    "reason":            reason,
                    "scores":            scores,
                    "embedding":         row_embeddings[winner_idx],
                    "chunks_processed":  count,
                    "chunk_embeddings":  row_embeddings,
                }

    return results  # type: ignore[return-value]
