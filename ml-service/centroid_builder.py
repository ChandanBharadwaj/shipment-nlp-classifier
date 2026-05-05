"""
Offline centroid builder — v3 source-driven centroids.

Reads `public_source_descriptions` (USITC + UK Trade Tariff), groups by the
LLM-derived (category, hs_chapter) map, embeds with the configured
SentenceTransformer, and writes ONE L2-normalized centroid per
(category_slug, hs_chapter) into `category_centroids`.

shipment_labels is NOT used here — it stays as the F1 evaluation set only.

Configuration:
    EMBEDDING_MODEL env var (defaults to all-MiniLM-L6-v2, 384-dim). Must
    match what the API service uses at inference time.

Run:
    python ml-service/centroid_builder.py

Schedule:
    After any pipeline data change:
        scripts/load_public_sources.py
        scripts/load_llm_categories.py
        scripts/build_public_keywords.py
        scripts/validate_chat_cctr.py
        ml-service/centroid_builder.py
        curl -X POST http://localhost:8000/reload
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
# Categories come from `categories`/`chapter_categories` (LLM-derived).
# Reference text comes from `public_source_descriptions` (USITC + UK).
# shipment_labels is NOT used here — it stays as the F1 evaluation set only.
PUBLIC_TEXT_QUERY = """
    SELECT cm.category_slug,
           p.hs_chapter,
           p.description
    FROM   public_source_descriptions p
    JOIN   chapter_categories cm USING (hs_chapter)
    ORDER  BY cm.category_slug, p.hs_chapter
"""

ALL_CHAPTERS_QUERY = """
    SELECT cc.category_slug, cc.hs_chapter
    FROM   chapter_categories cc
    ORDER  BY cc.category_slug, cc.hs_chapter
"""


def _load_category_chapters(conn) -> dict[str, list[str]]:
    """{category_slug: [hs_chapter, ...]} v3 taxonomy edges."""
    out: dict[str, list[str]] = defaultdict(list)
    with conn.cursor() as cur:
        cur.execute(ALL_CHAPTERS_QUERY)
        for slug, hs_chapter in cur.fetchall():
            out[slug].append(hs_chapter)
    return dict(out)


def _load_public_text(conn) -> dict[str, dict[str, list[str]]]:
    """{category_slug: {hs_chapter: [description, ...]}} from
    public_source_descriptions, deduped per chapter to avoid parent-context
    rollup inflating the average (a 4-digit heading's text appears verbatim
    inside many child rows)."""
    out: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    with conn.cursor() as cur:
        cur.execute(PUBLIC_TEXT_QUERY)
        for slug, ch, descr in cur.fetchall():
            out[slug][ch].add(descr)
    return {slug: {ch: sorted(s) for ch, s in by_ch.items()}
            for slug, by_ch in out.items()}


def build_and_persist(conn) -> None:
    """
    Build one centroid per (category, hs_chapter) from public reference
    text and UPSERT into category_centroids. Wipe-and-rebuild semantics.
    """
    chapters_by_cat = _load_category_chapters(conn)
    text_by_cat = _load_public_text(conn)

    if not chapters_by_cat:
        print("No chapter_categories rows. Run scripts/load_llm_categories.py first.")
        return
    if not text_by_cat:
        print("No public_source_descriptions. Run scripts/load_public_sources.py first.")
        return

    print(f"Building per-chapter centroids from public source text "
          f"({len(chapters_by_cat)} categories)...")
    model_name = (
        model._first_module().auto_model.config.name_or_path
        if hasattr(model, "_first_module") else "unknown"
    )
    print(f"Model: {model_name}")

    # [(category_slug, hs_chapter, centroid_vec, sample_count), ...]
    to_upsert: list[tuple[str, str, list, int]] = []

    for category in sorted(chapters_by_cat.keys()):
        buckets_dict = text_by_cat.get(category, {})
        if not buckets_dict:
            print(f"  [{category}] WARNING: no public text -- skipping.")
            continue

        # Convert into the same shape the rest of the function expects:
        # buckets[hs_chapter] = [(cargo, commodity), ...]. We treat each
        # description as cargo_text with empty commodity_text.
        buckets: dict[str, list[tuple[str, str]]] = {}
        for chap, descriptions in buckets_dict.items():
            buckets[chap] = [(d, "") for d in descriptions]

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
        total_samples = sum(n for _, n in bucket_sizes)
        print(f"  [{category}] {total_samples} samples -> {len(buckets)} chapters  {sizes_str}")
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
        for category_slug, hs_chapter, centroid, sample_count in to_upsert:
            cur.execute("""
                INSERT INTO category_centroids
                    (category_slug, hs_chapter, centroid, sample_count, updated_at)
                VALUES (%s, %s, %s, %s, now())
            """, (category_slug, hs_chapter, centroid, sample_count))
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
