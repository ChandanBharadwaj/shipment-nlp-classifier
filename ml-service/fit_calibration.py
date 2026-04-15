"""
Per-category Platt-scaling calibration.

For each category, fit a 2-parameter sigmoid that maps the raw final_score
produced by classifier._score_one into a calibrated probability:

    probability = sigmoid(platt_a * final_score + platt_b)

Training data: every row in shipment_labels where split='validation'. For each
row we compute final_scores for all 20 categories. For category C, we collect:
    positives: final_score from rows where C is in the expected set
    negatives: final_score from rows where C is NOT in the expected set
then fit a scikit-learn LogisticRegression on a single feature (final_score).

Writes (platt_a, platt_b) back to classification_categories so classifier.predict
can apply the calibration at inference time.

Run AFTER centroid_builder.py, BEFORE evaluate_threshold.py:
    python fit_calibration.py

Why Platt and not isotonic: 8-16 positives per category is enough for a 2-param
sigmoid but not enough to fit a step function stably. Platt is the correct
choice at this data scale.
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

from classifier import (
    _score_one,
    embed_texts,
    load_category_config,
    load_centroids,
    load_keywords,
    text_quality_check,
)
from db import get_connection
from preprocess import build_query_text, normalize as preprocess_normalize


def _collect_scores(conn, centroids, keywords, category_config) -> tuple[list[dict], set[str]]:
    """Load validation samples and compute per-category final_score for each."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT shipment_id,
                   MAX(cargo_text)                                 AS cargo,
                   MAX(commodity_text)                             AS commodity,
                   array_agg(category_name ORDER BY category_name) AS expected_labels
            FROM   shipment_labels
            WHERE  split = 'validation'
            GROUP  BY shipment_id
        """)
        rows = cur.fetchall()

    # Batch-embed all texts in one pass
    pairs = [(r[1], r[2]) for r in rows]
    valid_indices = []
    embed_inputs = []
    keyword_texts = []
    for i, (c, m) in enumerate(pairs):
        if text_quality_check(c, m) != "ok":
            continue
        valid_indices.append(i)
        embed_inputs.append(build_query_text(c, m, role="query"))
        keyword_texts.append(preprocess_normalize(f"{c} {m}"))

    if not embed_inputs:
        raise RuntimeError("No valid validation samples to calibrate on.")

    embeddings = embed_texts(embed_inputs)

    samples = []
    for offset, idx in enumerate(valid_indices):
        scores, _max = _score_one(
            keyword_texts[offset],
            embeddings[offset],
            centroids,
            keywords,
            category_config,
        )
        samples.append({
            "expected":     set(rows[idx][3]),
            "final_scores": {cat: s["final_score"] for cat, s in scores.items()},
        })

    all_categories = set(centroids.keys())
    return samples, all_categories


def _fit_platt_per_category(samples, all_categories):
    """Fit a LogisticRegression on final_score for each category.

    Returns {category: (a, b)} where probability = sigmoid(a * x + b).

    Falls back to (a=None, b=None) if a category has fewer than 2 positives OR
    fewer than 2 negatives — we can't fit a sigmoid on degenerate data.
    """
    fits: dict[str, tuple[float | None, float | None]] = {}
    for cat in sorted(all_categories):
        X: list[list[float]] = []
        y: list[int]         = []
        for s in samples:
            score = s["final_scores"].get(cat)
            if score is None:
                continue
            X.append([score])
            y.append(1 if cat in s["expected"] else 0)

        n_pos = sum(y)
        n_neg = len(y) - n_pos
        if n_pos < 2 or n_neg < 2:
            print(f"  [{cat:<18}] skip (pos={n_pos}, neg={n_neg})")
            fits[cat] = (None, None)
            continue

        # Single-feature logistic regression. C=1.0 default, no class weighting —
        # calibration wants to honor the raw class balance, not rebalance it.
        clf = LogisticRegression(solver="lbfgs", max_iter=1000)
        clf.fit(X, y)
        a = float(clf.coef_[0][0])
        b = float(clf.intercept_[0])
        fits[cat] = (a, b)
        print(f"  [{cat:<18}] pos={n_pos:>3} neg={n_neg:>4}  a={a:+.3f}  b={b:+.3f}")
    return fits


def _persist_fits(conn, fits: dict) -> None:
    with conn.cursor() as cur:
        for cat, (a, b) in fits.items():
            cur.execute(
                "UPDATE classification_categories SET platt_a = %s, platt_b = %s WHERE name = %s",
                (a, b, cat),
            )
    conn.commit()


def run(conn) -> None:
    # IMPORTANT: pass an empty category_config — we want raw (uncalibrated)
    # final_scores here. The already-persisted Platt params would otherwise
    # get applied recursively.
    uncalibrated_config = {
        name: {**cfg, "platt_a": None, "platt_b": None}
        for name, cfg in load_category_config(conn).items()
    }
    centroids = load_centroids(conn)
    keywords  = load_keywords(conn)
    if not centroids:
        raise RuntimeError("No centroids found. Run centroid_builder.py first.")

    print(f"Collecting validation scores across {len(centroids)} categories...")
    samples, all_categories = _collect_scores(conn, centroids, keywords, uncalibrated_config)
    print(f"Validation samples: {len(samples)}")

    print("\nFitting per-category Platt scaling:")
    fits = _fit_platt_per_category(samples, all_categories)

    fitted = [c for c, (a, b) in fits.items() if a is not None]
    skipped = [c for c, (a, b) in fits.items() if a is None]
    print(f"\nFit {len(fitted)} categories; skipped {len(skipped)} (insufficient data).")

    _persist_fits(conn, fits)
    print("Saved platt_a, platt_b to classification_categories.")
    print("Next step: python evaluate_threshold.py --split validation")


if __name__ == "__main__":
    conn = None
    try:
        conn = get_connection()
        run(conn)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
