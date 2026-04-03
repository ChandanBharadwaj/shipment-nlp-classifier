"""
Core classification logic.

No FastAPI imports here — this module is used by main.py, centroid_builder.py,
evaluate_threshold.py, and discover_unknowns.py alike.

Scoring formula (per category):
    semantic_score = cosine_similarity(query_embedding, centroid)
                   = dot(query_emb, centroid)   [both are L2-normalized]
    keyword_score  = weighted fraction of known keywords found in text
    final_score    = 0.8 * semantic_score + 0.2 * keyword_score

Confidence bands:
    final_score (max across categories) < 0.35   → unclassified  (low_similarity)
    0.35 ≤ max_score < threshold (default 0.45)  → low_confidence (top-1 only)
    max_score ≥ threshold                        → classified     (all above threshold)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_NAME             = "all-MiniLM-L6-v2"
KEYWORD_WEIGHT         = 0.2        # 80% semantic, 20% keyword boost
UNCLASSIFIED_THRESHOLD = 0.35       # below this → unclassified (low_similarity)

GENERIC_PHRASES: frozenset[str] = frozenset({
    "cargo",
    "goods",
    "items",
    "as per invoice",
    "general merchandise",
    "various",
    "misc",
    "miscellaneous",
    "freight",
    "shipment",
    "commodity",
    "general cargo",
})

# Model is loaded once at module import (~2–5 s, ~90 MB).
# For serverless/cold-start environments, move this inside predict() with caching.
model = SentenceTransformer(MODEL_NAME)


# ── DB loaders ─────────────────────────────────────────────────────────────────

def load_centroids(conn) -> dict[str, np.ndarray]:
    """
    Load L2-normalized centroid vectors for all active categories.

    Returns:
        {category_name: np.ndarray of shape (384,)}
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT cc.name, cen.centroid
            FROM   category_centroids cen
            JOIN   classification_categories cc ON cc.id = cen.category_id
            WHERE  cc.is_active = true
        """)
        return {row[0]: np.array(row[1]) for row in cur.fetchall()}


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


# ── Text helpers ───────────────────────────────────────────────────────────────

def text_quality_check(text: str) -> str:
    """
    Quick pre-embedding quality gate.

    Returns one of:
        'too_short' — fewer than 10 characters after stripping
        'generic'   — matches a known meaningless phrase exactly
        'ok'        — text is worth embedding
    """
    stripped = text.strip().lower()
    if len(stripped) < 10:
        return "too_short"
    if stripped in GENERIC_PHRASES:
        return "generic"
    return "ok"


def keyword_score(
    text: str,
    keywords: list[tuple[str, float]],
) -> tuple[float, list[str]]:
    """
    Weighted keyword match score.

    Score = sum of weights for matched keywords / sum of all keyword weights.
    Returns (score: float, matched_keywords: list[str]).
    """
    if not keywords:
        return 0.0, []

    text_lower = text.lower()
    total_weight = sum(w for _, w in keywords)
    hits: list[str] = []
    hit_weight = 0.0

    for kw, weight in keywords:
        if kw in text_lower:
            hits.append(kw)
            hit_weight += weight

    score = round(hit_weight / total_weight, 3) if total_weight > 0 else 0.0
    return score, hits


# ── Main prediction ────────────────────────────────────────────────────────────

def predict(
    cargo_description: str,
    commodity_description: str,
    centroids: dict[str, np.ndarray],
    keywords: dict[str, list[tuple[str, float]]],
    threshold: float = 0.45,
) -> dict:
    """
    Classify a single shipment.

    Args:
        cargo_description:     Value of cargo_document_description field.
        commodity_description: Value of commodity_description field.
        centroids:             Dict returned by load_centroids().
        keywords:              Dict returned by load_keywords().
        threshold:             Final-score cutoff for 'classified' state.

    Returns a dict with keys:
        categories        list[str]   — assigned category names
        confidence_state  str         — 'classified' | 'low_confidence' | 'unclassified'
        reason            str | None  — 'insufficient_input' | 'low_similarity' | None
        scores            dict        — full per-category breakdown
        embedding         np.ndarray  — L2-normalized query embedding (384,)
    """
    text    = f"{cargo_description} {commodity_description}".strip()
    quality = text_quality_check(text)

    if quality != "ok":
        return {
            "categories":       [],
            "confidence_state": "unclassified",
            "reason":           "insufficient_input",
            "scores":           {},
            "embedding":        None,
        }

    # Embed and L2-normalize so dot product == cosine similarity
    embedding = normalize(model.encode([text]))[0]

    scores: dict[str, dict] = {}
    max_score = 0.0

    for category, centroid in centroids.items():
        sem        = round(float(np.dot(embedding, centroid)), 4)
        kw_s, hits = keyword_score(text, keywords.get(category, []))
        final      = round((1 - KEYWORD_WEIGHT) * sem + KEYWORD_WEIGHT * kw_s, 4)

        scores[category] = {
            "semantic_score": sem,
            "keyword_score":  kw_s,
            "final_score":    final,
            "matched":        False,
            "keywords_hit":   hits,
        }
        max_score = max(max_score, final)

    # Apply confidence bands
    if max_score < UNCLASSIFIED_THRESHOLD:
        confidence_state = "unclassified"
        reason           = "low_similarity"
        matched_cats: list[str] = []

    elif max_score < threshold:
        confidence_state = "low_confidence"
        reason           = None
        top_cat          = max(scores, key=lambda c: scores[c]["final_score"])
        matched_cats     = [top_cat]
        scores[top_cat]["matched"] = True

    else:
        confidence_state = "classified"
        reason           = None
        matched_cats     = [c for c, s in scores.items() if s["final_score"] >= threshold]
        for c in matched_cats:
            scores[c]["matched"] = True

    return {
        "categories":       matched_cats,
        "confidence_state": confidence_state,
        "reason":           reason,
        "scores":           scores,
        "embedding":        embedding,
    }
