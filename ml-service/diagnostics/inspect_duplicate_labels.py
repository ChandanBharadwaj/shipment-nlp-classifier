"""
Diagnostic: do identical commodity strings in ``shipment_labels`` carry multiple
``hs_chapter`` labels?

The 99-row test CSV surfaced "same commodity → different categories" on strings
like ``"Chemical products, nos"`` and ``"Furniture, nos"``. The classifier is
deterministic given a fixed centroid set, so variance has to come from one of:

    1. Training-label ambiguity — the same commodity text appears under more
       than one chapter in ``shipment_labels``, so the centroids themselves
       straddle categories and every near-tie breaks arbitrarily.
    2. Input-text variance — the CSV rows have trivially different
       punctuation/whitespace so embeddings drift even though the strings
       look identical to humans.

This script answers (1) directly with one SQL query and then — for each
ambiguous string — runs the current classifier against it to show the top-3
scores + ``hs_codes_extracted`` + ``hs_implied_categories``. If the SQL finds
no dupes, the answer is (2) and the next step is to inspect the input CSV.

READ-ONLY. No writes. Does not touch centroids, keywords, or categories.

Usage:
    python -m diagnostics.inspect_duplicate_labels
    python -m diagnostics.inspect_duplicate_labels --limit 100
    python -m diagnostics.inspect_duplicate_labels --commodity "Chemical products, nos"
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from classifier import (                         # noqa: E402
    load_category_config,
    load_centroids,
    load_chapter_titles,
    load_chapter_to_category,
    load_keywords,
    predict,
)
from db import pooled_connection                 # noqa: E402


_DUPE_SQL = """
    SELECT commodity_text,
           COUNT(DISTINCT hs_chapter)                         AS distinct_chapters,
           ARRAY_AGG(DISTINCT hs_chapter ORDER BY hs_chapter) AS chapters,
           COUNT(*)                                           AS row_count
    FROM   shipment_labels
    WHERE  commodity_text IS NOT NULL
      AND  hs_chapter IS NOT NULL
    GROUP  BY commodity_text
    HAVING COUNT(DISTINCT hs_chapter) > 1
    ORDER  BY row_count DESC
    LIMIT  %s
"""

_CHAPTER_BREAKDOWN_SQL = """
    SELECT hs_chapter, COUNT(*) AS n
    FROM   shipment_labels
    WHERE  commodity_text = %s
      AND  hs_chapter IS NOT NULL
    GROUP  BY hs_chapter
    ORDER  BY n DESC, hs_chapter
"""


def _fetch_ambiguous(conn, limit: int) -> list[tuple[str, int, list[str], int]]:
    with conn.cursor() as cur:
        cur.execute(_DUPE_SQL, (limit,))
        return cur.fetchall()


def _fetch_one(conn, commodity: str) -> list[tuple[str, int, list[str], int]]:
    """Return a single-row result shaped like ``_fetch_ambiguous`` for one commodity."""
    with conn.cursor() as cur:
        cur.execute(_CHAPTER_BREAKDOWN_SQL, (commodity,))
        rows = cur.fetchall()
    if not rows:
        return []
    chapters   = [ch for ch, _ in rows]
    row_count  = sum(n for _, n in rows)
    return [(commodity, len(chapters), chapters, row_count)]


def _top_k_scores(scores: dict[str, float], k: int = 3) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]


def run(limit: int, commodity_filter: str | None) -> None:
    with pooled_connection() as conn:
        ambiguous = (
            _fetch_one(conn, commodity_filter)
            if commodity_filter
            else _fetch_ambiguous(conn, limit)
        )
        if not ambiguous:
            if commodity_filter:
                print(f'No rows in shipment_labels for commodity "{commodity_filter}".')
            else:
                print("No ambiguous commodity_texts found in shipment_labels.")
                print("Same-text-different-category variance is NOT explained by training labels.")
                print("Next step: diff the 99-row CSV inputs for trailing whitespace / case / punctuation.")
            return

        centroids           = load_centroids(conn)
        keywords            = load_keywords(conn)
        category_config     = load_category_config(conn)
        chapter_titles      = load_chapter_titles(conn)
        chapter_to_category = load_chapter_to_category(conn)

        # Chapter breakdowns for the top matches.
        breakdowns: dict[str, list[tuple[str, int]]] = {}
        with conn.cursor() as cur:
            for commodity, *_ in ambiguous:
                cur.execute(_CHAPTER_BREAKDOWN_SQL, (commodity,))
                breakdowns[commodity] = list(cur.fetchall())

    # ── Section 1: ambiguous training rows ───────────────────────────────────
    print("=" * 78)
    print(f"Ambiguous training rows   (top {len(ambiguous)} by row count)")
    print("=" * 78)
    print(f"{'commodity_text':50s} {'# chapters':>11} {'# rows':>8}")
    print("-" * 78)
    for commodity, distinct_chapters, chapters, row_count in ambiguous:
        print(f"{commodity[:50]:50s} {distinct_chapters:>11} {row_count:>8}")
        per_chapter = breakdowns.get(commodity, [])
        dist = ", ".join(f"ch{ch}={n}" for ch, n in per_chapter)
        print(f"    chapters: [{', '.join(chapters)}]   distribution: {dist}")
    print()

    # ── Section 2: what the current classifier says about each ────────────────
    print("=" * 78)
    print("Current classifier behaviour on the ambiguous set")
    print("=" * 78)
    for commodity, *_ in ambiguous:
        result = predict(
            cargo_description     = "",
            commodity_text = commodity,
            centroids             = centroids,
            keywords              = keywords,
            category_config       = category_config,
            chapter_titles        = chapter_titles,
            chapter_to_category   = chapter_to_category,
        )
        top3 = _top_k_scores(result.get("scores") or {}, k=3)
        matched = result.get("categories") or []
        state   = result.get("confidence_state")
        hs      = result.get("hs_codes_extracted") or []
        implied = result.get("hs_implied_categories") or []

        print(f'"{commodity}"')
        print(f"    predicted: {matched or '<none>'}   state: {state}")
        print(f"    top3:      {[(c, round(s, 3)) for c, s in top3]}")
        if hs:
            print(f"    hs_codes:  {hs}")
        if implied:
            print(f"    hs_implied:{implied}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=50,
                    help="Max ambiguous commodity_texts to inspect (default: 50)")
    ap.add_argument("--commodity", type=str, default=None,
                    help="Focus on one commodity string; prints its chapter distribution + classifier behaviour.")
    args = ap.parse_args()
    run(limit=args.limit, commodity_filter=args.commodity)


if __name__ == "__main__":
    main()
