"""
Offline centroid builder — supervised per-chapter centroids.

Reads labeled shipments from `shipment_labels` (split='train'), embeds their
text using the configured sentence-transformer, and produces ONE centroid per
(category, hs_chapter) pair — roughly 96 centroids across 20 categories.

Why per-chapter instead of k-means:
    The prior version ran k-means inside each category to split heterogeneous
    ones (machinery spanning pumps / compressors / CNC tools). Clusters were
    unsupervised and drifted into neighboring categories' topic space.

    Now that every v2 training row carries its source `hs_chapter` (2-digit
    HS code), we have labeled clusters for free. Chapter 84 (machinery) still
    gets sub-clusters — one each for heading-groups like pumps (8413),
    compressors (8414), CNC (8456-8466) — but each is supervised by the HS
    hierarchy rather than k-means geometry.

    Single-chapter categories (cosmetics=33, automotive=87, defense=93,
    energy=27, minerals=26) naturally produce one centroid.

Legacy rows:
    Legacy rows (shipment_id matching the xx_### pattern) have hs_chapter=NULL.
    They're assigned to the PRIMARY chapter of their coarse category — looked
    up from `category_hs_chapters` where `is_primary = true`. This keeps
    legacy samples contributing signal without a special code path at
    inference time.

Configuration:
    EMBEDDING_MODEL env var — defaults to all-MiniLM-L6-v2 (384-dim). Must
    match what the API service uses at inference time.

Run:
    python centroid_builder.py

Schedule:
    After any data change: init_db.py → centroid_builder.py → fit_calibration.py.
    Then POST /reload on the API so the new centroids load without a restart.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

from classifier import embed_texts, model  # reuses configured model
from db import get_connection
from preprocess import build_query_text

# ── Config ─────────────────────────────────────────────────────────────────────

MIN_SAMPLES_PER_CHAPTER = 3   # below this we warn but still build a centroid


# ── Queries ────────────────────────────────────────────────────────────────────

# Pull every train-split row for a category plus its hs_chapter.
# Legacy rows (hs_chapter IS NULL) are re-labeled downstream to the category's
# primary chapter so they feed into the main centroid.
TRAIN_ROWS_QUERY = """
    SELECT sl.shipment_id,
           sl.cargo_text,
           sl.commodity_text,
           sl.hs_chapter
    FROM   shipment_labels sl
    JOIN   classification_categories cc ON cc.name = sl.category_name
    WHERE  cc.is_active = true
      AND  sl.category_name = %s
      AND  sl.split = 'train'
"""

PRIMARY_CHAPTER_QUERY = """
    SELECT cc.name, chc.hs_chapter
    FROM   category_hs_chapters chc
    JOIN   classification_categories cc ON cc.id = chc.category_id
    WHERE  chc.is_primary = true
"""

ALL_CHAPTERS_QUERY = """
    SELECT cc.name, chc.hs_chapter
    FROM   category_hs_chapters chc
    JOIN   classification_categories cc ON cc.id = chc.category_id
    WHERE  cc.is_active = true
    ORDER  BY cc.name, chc.hs_chapter
"""


def _load_primary_chapters(conn) -> dict[str, str]:
    """{category_name: primary_hs_chapter} — one row per category."""
    out: dict[str, str] = {}
    with conn.cursor() as cur:
        cur.execute(PRIMARY_CHAPTER_QUERY)
        for name, hs_chapter in cur.fetchall():
            out[name] = hs_chapter
    return out


def _load_category_chapters(conn) -> dict[str, list[str]]:
    """{category_name: [hs_chapter, ...]} — every edge in the taxonomy."""
    out: dict[str, list[str]] = defaultdict(list)
    with conn.cursor() as cur:
        cur.execute(ALL_CHAPTERS_QUERY)
        for name, hs_chapter in cur.fetchall():
            out[name].append(hs_chapter)
    return dict(out)


def build_and_persist(conn) -> None:
    """
    Build one centroid per (category, hs_chapter) and UPSERT into
    category_centroids. Wipe-and-rebuild semantics so stale chapters from a
    prior taxonomy version don't linger.
    """
    primary_by_cat = _load_primary_chapters(conn)
    chapters_by_cat = _load_category_chapters(conn)

    if not chapters_by_cat:
        print("No category_hs_chapters rows found. Run init_db.py first.")
        return

    print(f"Building supervised per-chapter centroids for {len(chapters_by_cat)} categories...")
    model_name = (
        model._first_module().auto_model.config.name_or_path
        if hasattr(model, "_first_module") else "unknown"
    )
    print(f"Model: {model_name}")

    # [(category_name, hs_chapter, centroid_vec, sample_count), ...]
    to_upsert: list[tuple[str, str, list, int]] = []

    for category in sorted(chapters_by_cat.keys()):
        with conn.cursor() as cur:
            cur.execute(TRAIN_ROWS_QUERY, (category,))
            rows = cur.fetchall()

        if not rows:
            print(f"  [{category}] WARNING: no train rows — skipping.")
            continue

        # Bucket rows by chapter. Legacy rows (hs_chapter NULL) fall into the
        # category's primary chapter.
        primary = primary_by_cat.get(category)
        if primary is None:
            # Should not happen if seed_categories.sql marked one chapter primary.
            primary = chapters_by_cat[category][0]

        buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for _sid, cargo, commodity, hs_chapter in rows:
            bucket = hs_chapter if hs_chapter else primary
            buckets[bucket].append((cargo, commodity))

        # Batch-embed all rows for this category once (fewer model calls).
        all_texts: list[str] = []
        flat_buckets: list[str] = []
        for chap in sorted(buckets):
            for cargo, commodity in buckets[chap]:
                all_texts.append(build_query_text(cargo, commodity))
                flat_buckets.append(chap)

        embeddings = embed_texts(all_texts)  # L2-normalized, shape (n, dim)

        # One centroid per bucket: mean then re-normalize.
        bucket_sizes: list[tuple[str, int]] = []
        for chap in sorted(buckets):
            idxs = [i for i, b in enumerate(flat_buckets) if b == chap]
            sub = embeddings[idxs]
            n_in = len(idxs)
            centroid = sub.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            to_upsert.append((category, chap, centroid.tolist(), n_in))
            bucket_sizes.append((chap, n_in))

        # Warn on any assigned chapter that ended up with no training data.
        assigned = set(chapters_by_cat[category])
        seen = set(buckets.keys())
        missing = sorted(assigned - seen)

        sizes_str = " ".join(f"{c}={n}" for c, n in bucket_sizes)
        print(f"  [{category}] {len(rows)} samples → {len(buckets)} chapters  {sizes_str}")
        if missing:
            print(f"    MISSING (assigned, no train data): {missing}")

        thin = [c for c, n in bucket_sizes if n < MIN_SAMPLES_PER_CHAPTER]
        if thin:
            print(f"    thin chapters (<{MIN_SAMPLES_PER_CHAPTER} samples): {thin}")

    if not to_upsert:
        print("No centroids built. Check that shipment_labels has train data.")
        return

    # Wipe-and-rebuild: centroids are derived. A chapter being reassigned or
    # removed from a category must not leave stale rows behind.
    with conn.cursor() as cur:
        cur.execute("DELETE FROM category_centroids")
        for category, hs_chapter, centroid, sample_count in to_upsert:
            cur.execute("""
                INSERT INTO category_centroids
                    (category_id, hs_chapter, cluster_id, centroid, sample_count, updated_at)
                SELECT id, %s, 0, %s, %s, now()
                FROM   classification_categories
                WHERE  name = %s
            """, (hs_chapter, centroid, sample_count, category))
    conn.commit()

    n_categories = len({row[0] for row in to_upsert})
    n_centroids = len(to_upsert)
    print(f"\nDone. Built {n_centroids} centroids across {n_categories} categories.")
    print("Next steps:")
    print("  1. Fit calibration:  python fit_calibration.py")
    print("  2. Reload API:       curl -X POST http://localhost:8001/reload")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    conn = None
    try:
        conn = get_connection()
        build_and_persist(conn)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
