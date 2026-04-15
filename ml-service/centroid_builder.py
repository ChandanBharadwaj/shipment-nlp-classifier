"""
Offline centroid builder — multi-centroid k-means per category.

Reads labeled shipments from `shipment_labels` (split='train'), embeds their
text using the configured sentence-transformer, runs k-means inside each
category to produce 1–4 sub-centroids, and UPSERTs them into
`category_centroids` keyed by (category_id, cluster_id).

Why multi-centroid:
    Heterogeneous categories (e.g. `machinery` spanning pumps, compressors,
    tractors, CNC tools) are poorly represented by a single mean. Sub-clusters
    give each sub-topic its own centroid and recall improves substantially —
    at inference time the score for a category is the max cosine across its
    sub-centroids.

Adaptive k:
    k = clip(n_samples // 20, 1, 4)
    So a 35-sample category uses k=1 (no benefit to splitting), a 65-sample
    category uses k=3. Purely data-driven; no per-category tuning.

Configuration:
    EMBEDDING_MODEL env var — defaults to BAAI/bge-small-en-v1.5 (384-dim).
    Must match what the API service uses at inference time.

Run:
    python centroid_builder.py

Schedule:
    Weekly via cron or task scheduler. After each run, call POST /reload on the
    API service so the new centroids are loaded into memory without a restart.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from classifier import embed_texts, model  # reuses configured model
from db import get_connection
from preprocess import build_query_text

# ── Config ─────────────────────────────────────────────────────────────────────
BATCH_SIZE          = 512
SAMPLE_PER_CATEGORY = 2000


def _choose_k(n_samples: int) -> int:
    """Adaptive cluster count per category.

    Multi-centroid helps heterogeneous categories (e.g. machinery covering
    pumps, compressors, CNC tools) but hurts precision for homogeneous ones
    by letting sub-centroids drift toward neighboring categories' topic
    space. Require a higher per-cluster sample count (50 vs 20) so we only
    split categories that are genuinely multi-modal.
    """
    if n_samples <= 0:
        return 0
    return int(max(1, min(4, n_samples // 50)))


# Only train-split rows feed centroid building — validation/test held out for
# tuning and final evaluation.
SAMPLING_QUERY = """
    SELECT sl.shipment_id,
           sl.cargo_text,
           sl.commodity_text
    FROM   shipment_labels sl
    JOIN   classification_categories cc ON cc.name = sl.category_name
    WHERE  cc.is_active = true
      AND  sl.category_name = %s
      AND  sl.split = 'train'
    ORDER  BY RANDOM()
    LIMIT  %s
"""


def build_and_persist(conn) -> None:
    """
    Build 1-4 sub-centroids per active category (k-means on the train split)
    and UPSERT them into category_centroids.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT name FROM classification_categories WHERE is_active = true ORDER BY name")
        active_categories = [row[0] for row in cur.fetchall()]

    if not active_categories:
        print("No active categories found. Seed classification_categories first.")
        return

    print(f"Building centroids for {len(active_categories)} active categories...")
    print(f"Model: {model._first_module().auto_model.config.name_or_path if hasattr(model, '_first_module') else 'unknown'}")

    # Collect all sub-centroid rows to write in one transaction.
    # Format: [(category_name, cluster_id, centroid_vec, sample_count), ...]
    to_upsert: list[tuple[str, int, list, int]] = []

    for category in active_categories:
        with conn.cursor() as cur:
            cur.execute(SAMPLING_QUERY, (category, SAMPLE_PER_CATEGORY))
            rows = cur.fetchall()

        if not rows:
            print(f"  [{category}] WARNING: no labeled rows — skipping.")
            continue

        texts = [build_query_text(cargo, commodity) for _sid, cargo, commodity in rows]
        embeddings = embed_texts(texts)   # already L2-normalized, shape (n, dim)

        k = _choose_k(len(rows))
        if k == 1:
            centroid = embeddings.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            to_upsert.append((category, 0, centroid.tolist(), len(rows)))
            print(f"  [{category}] {len(rows)} samples → k=1")
        else:
            # n_init='auto' is the modern sklearn default and suppresses the
            # deprecation warning on sklearn >= 1.4.
            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            labels = km.fit_predict(embeddings)
            for cluster_id in range(k):
                mask = labels == cluster_id
                n_in_cluster = int(mask.sum())
                if n_in_cluster == 0:
                    continue
                centroid = embeddings[mask].mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                to_upsert.append((category, cluster_id, centroid.tolist(), n_in_cluster))
            sizes = [int((labels == c).sum()) for c in range(k)]
            print(f"  [{category}] {len(rows)} samples → k={k} cluster_sizes={sizes}")

    if not to_upsert:
        print("No centroids built. Check that shipment_labels has data.")
        return

    # Wipe-and-rebuild: centroids are a derived artifact. Delete all existing
    # rows then insert the new set so a reduction in k for a category doesn't
    # leave stale cluster_ids behind.
    with conn.cursor() as cur:
        cur.execute("DELETE FROM category_centroids")
        for category, cluster_id, centroid, sample_count in to_upsert:
            cur.execute("""
                INSERT INTO category_centroids (category_id, cluster_id, centroid, sample_count, updated_at)
                SELECT id, %s, %s, %s, now()
                FROM   classification_categories
                WHERE  name = %s
            """, (cluster_id, centroid, sample_count, category))
    conn.commit()

    n_categories = len({row[0] for row in to_upsert})
    n_centroids  = len(to_upsert)
    print(f"\nDone. Built {n_centroids} sub-centroids across {n_categories} categories.")
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
