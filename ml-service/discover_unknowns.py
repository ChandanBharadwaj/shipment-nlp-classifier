"""
Unknown category discovery.

Periodically run this script to detect emerging shipment types that don't fit
any existing category. It fetches the embeddings of all `unclassified` shipments,
runs HDBSCAN to find clusters, and prints candidate new categories for analyst review.

Usage:
    python discover_unknowns.py [--min-cluster-size 50]

When an analyst confirms a new category:
    1. INSERT into classification_categories (name, display_name, is_active=true)
    2. INSERT representative keywords into category_keywords
    3. Label those shipments in shipment_labels (or your real labels table)
    4. Re-run centroid_builder.py
    5. Call POST /reload on the API service

HDBSCAN note (Windows):
    pip install hdbscan --prefer-binary
    Or: conda install -c conda-forge hdbscan
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import hdbscan
except ImportError:
    print(
        "ERROR: hdbscan is not installed.\n"
        "  Windows: pip install hdbscan --prefer-binary\n"
        "  Conda:   conda install -c conda-forge hdbscan",
        file=sys.stderr,
    )
    sys.exit(1)

from classifier import load_centroids
from db import get_connection

MIN_CLUSTER_SIZE_DEFAULT = 50
MIN_SAMPLES_DEFAULT      = 5


def find_unknown_clusters(
    conn,
    min_cluster_size: int = MIN_CLUSTER_SIZE_DEFAULT,
    min_samples: int      = MIN_SAMPLES_DEFAULT,
) -> None:
    """
    Fetch unclassified shipment embeddings, cluster with HDBSCAN, and report
    candidate new categories to stdout.

    For each cluster found, also reports the nearest existing category centroid
    as a hint to the analyst.
    """
    # Load existing centroids for nearest-neighbour hint
    centroids = load_centroids(conn)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT shipment_id, embedding
            FROM   shipment_classifications
            WHERE  confidence_state = 'unclassified'
              AND  embedding IS NOT NULL
        """)
        rows = cur.fetchall()

    n_unclassified = len(rows)
    print(f"Unclassified shipments with embeddings: {n_unclassified}")

    if n_unclassified < min_cluster_size:
        print(
            f"Not enough unclassified shipments to cluster "
            f"(need ≥ {min_cluster_size}, found {n_unclassified})."
        )
        return

    ids        = [row[0] for row in rows]
    embeddings = np.array([row[1] for row in rows])

    print(f"Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)

    unique_labels = sorted(set(labels))
    cluster_labels = [l for l in unique_labels if l != -1]
    noise_count    = int((labels == -1).sum())

    print(f"\nResults: {len(cluster_labels)} cluster(s) found, {noise_count} noise points.")

    if not cluster_labels:
        print("No clusters found — all unclassified shipments appear to be one-offs or noise.")
        return

    # Centroid matrix for nearest-neighbour computation. Each category now
    # holds a list of (hs_chapter, centroid_vec) tuples; collapse to a single
    # mean-vector per category for the nearest-neighbour hint.
    if centroids:
        cat_names = list(centroids.keys())
        cat_vecs  = []
        for c in cat_names:
            stacked = np.stack([vec for _chap, vec in centroids[c]])
            mean_v  = stacked.mean(axis=0)
            norm    = np.linalg.norm(mean_v)
            cat_vecs.append(mean_v / norm if norm > 0 else mean_v)
        cat_matrix = np.stack(cat_vecs)
    else:
        cat_names  = []
        cat_matrix = None

    print("\n" + "=" * 60)
    for label in cluster_labels:
        mask         = labels == label
        cluster_ids  = [ids[i] for i, m in enumerate(mask) if m]
        cluster_embs = embeddings[mask]
        cluster_centroid = cluster_embs.mean(axis=0)
        cluster_centroid /= np.linalg.norm(cluster_centroid)

        nearest_cat = None
        nearest_sim = None
        if cat_matrix is not None:
            sims        = cat_matrix @ cluster_centroid
            best_idx    = int(np.argmax(sims))
            nearest_cat = cat_names[best_idx]
            nearest_sim = round(float(sims[best_idx]), 3)

        print(f"\nCANDIDATE CLUSTER {label}")
        print(f"  Size:           {len(cluster_ids)} shipments")
        if nearest_cat:
            print(f"  Nearest category: {nearest_cat} (cosine sim={nearest_sim})")
            if nearest_sim and nearest_sim > 0.80:
                print(f"  NOTE: High similarity to '{nearest_cat}' — may be a sub-category or variant.")
        print(f"  Sample IDs:     {cluster_ids[:10]}")

    print("\n" + "=" * 60)
    print("\nNext steps for each confirmed cluster:")
    print("  1. Review sample shipments and decide on a category name")
    print("  2. INSERT into classification_categories")
    print("  3. INSERT keywords into category_keywords")
    print("  4. Label those shipment IDs in shipment_labels")
    print("  5. Re-run centroid_builder.py")
    print("  6. Call POST /reload")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover unknown shipment categories.")
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=MIN_CLUSTER_SIZE_DEFAULT,
        help=f"Minimum cluster size for HDBSCAN (default: {MIN_CLUSTER_SIZE_DEFAULT})",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=MIN_SAMPLES_DEFAULT,
        help=f"min_samples for HDBSCAN (default: {MIN_SAMPLES_DEFAULT})",
    )
    args = parser.parse_args()

    conn = None
    try:
        conn = get_connection()
        find_unknown_clusters(conn, args.min_cluster_size, args.min_samples)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
