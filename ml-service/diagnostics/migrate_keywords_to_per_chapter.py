"""
One-shot migration: populate hs_chapter + signal_class on legacy
``category_keywords`` rows.

Why
---
Pre-CCTR rows in ``category_keywords`` carry only ``(category_id, keyword,
weight)``. The CCTR schema (see schema.sql) added ``hs_chapter``,
``signal_class``, ``source``, ``notes`` so the same word can route to different
chapters with different weights.

This script does NOT split a single legacy row into multiple per-chapter rows.
It picks the single best-matching chapter for each existing row, on the
principle that the legacy row already represents one decision and we don't want
the migration to invent new ones. Per-chapter splits happen later, by hand or
via the ``rebalance_keyword_weights`` job, with a clear paper trail in
``keyword_audit_log``.

Algorithm
---------
For each legacy row ``(category_id, keyword, weight)`` with ``hs_chapter IS NULL``::

    kw_vec     = embed(keyword)               # L2-normalized
    candidates = category_centroids[(category, ch)] for ch in this category
    best_ch    = argmax_ch cosine(kw_vec, candidates[ch])
    UPDATE category_keywords SET hs_chapter = best_ch,
                                 signal_class = 'signal',
                                 source = COALESCE(source, 'merged')
                  WHERE id = row.id

Fallback: if a category has zero built centroids (newly seeded, never run
``centroid_builder``), fall back to its primary chapter from
``category_hs_chapters`` where ``is_primary = true`` (or the lexicographically
smallest chapter if none is flagged primary).

Idempotency
-----------
Running this script twice is a no-op — the second run finds zero rows with
``hs_chapter IS NULL``. Safe to wire into ``init_db.py`` as a post-seed step.

Usage
-----
    python -m diagnostics.migrate_keywords_to_per_chapter
    python -m diagnostics.migrate_keywords_to_per_chapter --dry-run
    python -m diagnostics.migrate_keywords_to_per_chapter --category electronics
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from classifier import embed_texts, load_centroids  # noqa: E402
from db import get_connection                        # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def _fetch_legacy_rows(conn, category_filter: str | None) -> list[tuple[int, int, str, str, float]]:
    """
    Returns (id, category_id, category_name, keyword, weight) for every row
    where ``hs_chapter IS NULL`` (i.e. not yet migrated).
    """
    sql = """
        SELECT ck.id, ck.category_id, cc.name, ck.keyword, ck.weight
        FROM   category_keywords ck
        JOIN   classification_categories cc ON cc.id = ck.category_id
        WHERE  ck.hs_chapter IS NULL
    """
    args: tuple = ()
    if category_filter:
        sql += " AND cc.name = %s"
        args = (category_filter,)
    sql += " ORDER BY cc.name, ck.keyword"
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchall()


def _fetch_primary_chapters(conn) -> dict[str, str]:
    """Return {category_name: primary_chapter_or_first_chapter}. Used as
    a centroid-less fallback so the migration can still pick a defensible
    chapter for newly-seeded categories."""
    fallbacks: dict[str, str] = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT cc.name, chc.hs_chapter, chc.is_primary
            FROM   category_hs_chapters chc
            JOIN   classification_categories cc ON cc.id = chc.category_id
            ORDER  BY cc.name, chc.is_primary DESC, chc.hs_chapter
        """)
        for name, ch, _is_primary in cur.fetchall():
            fallbacks.setdefault(name, ch)        # first row per category wins (primary first)
    return fallbacks


def _best_chapter(
    kw_vec: np.ndarray,
    chapter_centroids: list[tuple[str, np.ndarray]],
) -> tuple[str, float]:
    """argmax cosine. Both kw_vec and centroids are pre-normalized → dot product."""
    best_ch, best_cos = "", -1.0
    for ch, vec in chapter_centroids:
        cos = float(np.dot(kw_vec, vec))
        if cos > best_cos:
            best_cos, best_ch = cos, ch
    return best_ch, best_cos


def _backup_csv(rows: list[tuple]) -> str:
    """Write a pre-migration snapshot for rollback. Returns the path."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_HERE, f"migrate_keywords_backup_{ts}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "category_id", "category_name", "keyword", "weight"])
        for row in rows:
            w.writerow(row)
    return path


# ── main ──────────────────────────────────────────────────────────────────────

def run(category_filter: str | None, dry_run: bool) -> None:
    conn = get_connection()
    try:
        rows = _fetch_legacy_rows(conn, category_filter)
        if not rows:
            print("No legacy rows found (all category_keywords already have hs_chapter). Nothing to do.")
            return

        centroids = load_centroids(conn)              # {cat_name: [(ch, vec), ...]}
        fallbacks = _fetch_primary_chapters(conn)     # {cat_name: ch}
        print(f"Loaded centroids for {len(centroids)} categories; "
              f"fallback chapters for {len(fallbacks)} categories.")
        print(f"Migrating {len(rows)} legacy keyword rows"
              f"{f' (filter: category={category_filter})' if category_filter else ''}.")

        # Embed all keywords in one batch — much faster than one-by-one.
        keywords = [kw for _id, _cid, _cn, kw, _w in rows]
        kw_vecs  = embed_texts([k.lower() for k in keywords])

        updates: list[tuple[int, str, str, float]] = []      # (id, chapter, source, cosine)
        per_category_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        no_centroid_categories: set[str] = set()

        for (row_id, _cid, cat_name, keyword, _w), kw_vec in zip(rows, kw_vecs):
            ch_centroids = centroids.get(cat_name, [])
            ch_centroids = [(c, v) for c, v in ch_centroids if c]   # drop legacy '' tag
            if ch_centroids:
                best_ch, cos = _best_chapter(kw_vec, ch_centroids)
                source = "merged"
            else:
                best_ch = fallbacks.get(cat_name, "")
                cos     = float("nan")
                source  = "fallback_primary"
                no_centroid_categories.add(cat_name)

            if not best_ch:
                print(f"  ⚠ skipping {cat_name}/{keyword!r} — no centroid and no fallback chapter.")
                continue

            updates.append((row_id, best_ch, source, cos))
            per_category_stats[cat_name][best_ch] += 1

        # ── Reporting ────────────────────────────────────────────────────────
        print()
        print("Migration plan (per category, chapter assignment counts):")
        for cat in sorted(per_category_stats):
            assigns = per_category_stats[cat]
            total   = sum(assigns.values())
            note    = "  [fallback used]" if cat in no_centroid_categories else ""
            print(f"  {cat:20s} ({total:>3} keywords){note}")
            for ch in sorted(assigns):
                print(f"      ch{ch}  →  {assigns[ch]:>3}")
        print()

        if dry_run:
            print(f"DRY RUN — would update {len(updates)} rows. Pass --apply to commit.")
            return

        # ── Backup before writing ────────────────────────────────────────────
        backup_path = _backup_csv(rows)
        print(f"Backup written: {backup_path}")

        with conn.cursor() as cur:
            for row_id, best_ch, source, _cos in updates:
                cur.execute(
                    """
                    UPDATE category_keywords
                       SET hs_chapter   = %s,
                           signal_class = COALESCE(signal_class, 'signal'),
                           source       = COALESCE(source, %s)
                     WHERE id = %s
                       AND hs_chapter IS NULL
                    """,
                    (best_ch, source, row_id),
                )
        conn.commit()
        print(f"Applied {len(updates)} updates.")

    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply",    action="store_true",
                    help="Commit the migration. Without this flag the script is a dry run.")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Explicit dry run (default).")
    ap.add_argument("--category", type=str, default=None,
                    help="Limit migration to one category by name.")
    args = ap.parse_args()

    dry = args.dry_run or not args.apply
    run(category_filter=args.category, dry_run=dry)


if __name__ == "__main__":
    main()
