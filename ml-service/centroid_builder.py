"""
Offline centroid builder.

Reads labeled shipments from the `shipment_labels` table, embeds their text
using all-MiniLM-L6-v2, computes an L2-normalized incremental mean per category,
and UPSERTs the result into `category_centroids`.

Run:
    python centroid_builder.py

Schedule:
    Weekly via cron or task scheduler. After each run, call POST /reload on the
    API service so the new centroids are loaded into memory without a restart.

# CONFIGURE THIS (production):
    The SAMPLING_QUERY below targets the `shipment_labels` PoC table. In
    production, replace it with a JOIN against your real shipments table, e.g.:
        SELECT cc.name, s.shipment_id,
               s.cargo_document_description || ' ' || s.commodity_description
        FROM   your_labels_table sl
        JOIN   your_shipments_table s ON s.id = sl.shipment_id
        JOIN   classification_categories cc ON cc.name = sl.category_name
        WHERE  cc.is_active = true AND cc.name = %s
        LIMIT  %s
"""

from __future__ import annotations

import sys

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from db import get_connection

# ── Config ─────────────────────────────────────────────────────────────────────
BATCH_SIZE          = 512   # rows to embed in one model.encode() call
SAMPLE_PER_CATEGORY = 2000  # max labeled rows sampled per category
MODEL_NAME          = "all-MiniLM-L6-v2"


# ── Incremental centroid accumulator ───────────────────────────────────────────

class IncrementalCentroid:
    """Accumulates embeddings batch-by-batch and returns the L2-normalized mean."""

    def __init__(self):
        self.sum   = None
        self.count = 0

    def update(self, batch_embeddings: np.ndarray) -> None:
        """Add a batch of shape (n, 384) to the running sum."""
        if self.sum is None:
            self.sum = batch_embeddings.sum(axis=0)
        else:
            self.sum = self.sum + batch_embeddings.sum(axis=0)
        self.count += len(batch_embeddings)

    def centroid(self) -> np.ndarray:
        """Return the L2-normalized mean vector."""
        if self.count == 0:
            raise ValueError("No embeddings have been added to this accumulator.")
        mean = self.sum / self.count
        norm = np.linalg.norm(mean)
        if norm == 0:
            raise ValueError("Centroid has zero norm — all embeddings may be zero vectors.")
        return mean / norm


# ── Core builder ───────────────────────────────────────────────────────────────

# CONFIGURE THIS (production): replace with your real labeled data query.
# Only 'train' split rows are used — validation and test rows are held out
# for threshold tuning and final evaluation respectively.
SAMPLING_QUERY = """
    SELECT sl.category_name,
           sl.shipment_id,
           sl.cargo_text || ' ' || sl.commodity_text AS combined_text
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
    Build one centroid per active category and UPSERT into category_centroids.
    Uses a separate per-category query (up to SAMPLE_PER_CATEGORY rows each).
    """
    model = SentenceTransformer(MODEL_NAME)

    # Fetch active category names
    with conn.cursor() as cur:
        cur.execute("SELECT name FROM classification_categories WHERE is_active = true ORDER BY name")
        active_categories = [row[0] for row in cur.fetchall()]

    if not active_categories:
        print("No active categories found. Seed classification_categories first.")
        return

    print(f"Building centroids for {len(active_categories)} active categories...")

    accumulators: dict[str, IncrementalCentroid] = {}

    for category in active_categories:
        # Server-side cursor streams rows without loading all into memory
        cursor_name = f"centroid_cursor_{category}"
        with conn.cursor(name=cursor_name) as cur:
            cur.execute(SAMPLING_QUERY, (category, SAMPLE_PER_CATEGORY))

            rows_processed = 0
            acc = IncrementalCentroid()

            while True:
                rows = cur.fetchmany(BATCH_SIZE)
                if not rows:
                    break

                texts      = [row[2] for row in rows]
                embeddings = normalize(model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False))
                acc.update(embeddings)
                rows_processed += len(rows)

        if acc.count == 0:
            print(f"  [{category}] WARNING: no labeled rows found — skipping.")
            continue

        accumulators[category] = acc
        print(f"  [{category}] {acc.count} samples processed.")

    if not accumulators:
        print("No centroids built. Check that shipment_labels has data.")
        return

    # UPSERT all centroids in a single transaction
    with conn.cursor() as wc:
        for category, acc in accumulators.items():
            centroid_vec = acc.centroid()
            wc.execute("""
                INSERT INTO category_centroids (category_id, centroid, sample_count, updated_at)
                SELECT id, %s, %s, now()
                FROM   classification_categories
                WHERE  name = %s
                ON CONFLICT (category_id) DO UPDATE
                    SET centroid     = EXCLUDED.centroid,
                        sample_count = EXCLUDED.sample_count,
                        updated_at   = now()
            """, (centroid_vec.tolist(), acc.count, category))

    conn.commit()
    print(f"\nDone. Built centroids for {len(accumulators)} categories.")
    print("Next step: call POST /reload on the API service to pick up the new centroids.")


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
