"""
Inspect chapter centroids — geometric sanity checks.

Reports:
  1. Inter-centroid cosine matrix, sorted by pair similarity.
  2. Cross-category pairs > 0.85 (highest-risk confusables).
  3. Intra-category pairs > 0.90 (sibling chapters that are too close to
     help resolution — they'll effectively behave as one centroid).
  4. Per-chapter sample count (as recorded at build time).

Read-only. Run after centroid_builder.py. Intended output lands in
`results/diag_centroids_v2.txt`.

Usage:
    python -m diagnostics.inspect_centroids
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

# Allow running from ml-service/ or from repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from db import get_connection  # noqa: E402


CROSS_CATEGORY_FLAG = 0.85
INTRA_CATEGORY_FLAG = 0.90


def _load_all_centroids(conn):
    """Returns list of dicts: [{category, chapter, vec, n}, ...] in DB order."""
    rows = []
    with conn.cursor() as cur:
        cur.execute("""
            SELECT cc.name, cen.hs_chapter, cen.centroid, cen.sample_count
            FROM   category_centroids cen
            JOIN   classification_categories cc ON cc.id = cen.category_id
            WHERE  cc.is_active = true
            ORDER  BY cc.name, cen.hs_chapter
        """)
        for name, hs_chapter, centroid, sample_count in cur.fetchall():
            rows.append({
                "category":  name,
                "chapter":   hs_chapter or "",
                "vec":       np.array(centroid),
                "n_samples": int(sample_count) if sample_count is not None else 0,
            })
    return rows


def run(conn) -> None:
    rows = _load_all_centroids(conn)
    if not rows:
        print("No centroids found. Run centroid_builder.py first.")
        return

    n = len(rows)
    vecs = np.stack([r["vec"] for r in rows])
    # All vectors are already L2-normalized at build time, but defensively
    # normalize again so cosine == dot.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    sims = vecs @ vecs.T  # (n, n) cosine matrix

    print(f"Loaded {n} centroids across "
          f"{len({r['category'] for r in rows})} categories.\n")

    # ── Per-category chapter count ───────────────────────────────────────────
    by_cat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    print("Per-category chapter centroids (count, sample totals):")
    print(f"  {'category':<20}  {'chapters':>8}  {'samples':>8}")
    print("  " + "-" * 45)
    for cat in sorted(by_cat):
        children = by_cat[cat]
        total_n = sum(c["n_samples"] for c in children)
        print(f"  {cat:<20}  {len(children):>8}  {total_n:>8}")

    # ── Flagged pairs ────────────────────────────────────────────────────────
    cross_hits: list[tuple[float, int, int]] = []
    intra_hits: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sims[i, j])
            same_cat = rows[i]["category"] == rows[j]["category"]
            if same_cat and s > INTRA_CATEGORY_FLAG:
                intra_hits.append((s, i, j))
            elif (not same_cat) and s > CROSS_CATEGORY_FLAG:
                cross_hits.append((s, i, j))

    cross_hits.sort(reverse=True)
    intra_hits.sort(reverse=True)

    print(f"\nCROSS-category pairs with cosine > {CROSS_CATEGORY_FLAG}:")
    if not cross_hits:
        print("  (none — good separation between categories)")
    else:
        print(f"  {'cos':>6}   {'left':<28}  {'right':<28}")
        print("  " + "-" * 68)
        for s, i, j in cross_hits[:40]:
            left  = f"{rows[i]['category']}/{rows[i]['chapter']}"
            right = f"{rows[j]['category']}/{rows[j]['chapter']}"
            print(f"  {s:>6.3f}   {left:<28}  {right:<28}")
        if len(cross_hits) > 40:
            print(f"  ... ({len(cross_hits) - 40} more)")

    print(f"\nINTRA-category pairs with cosine > {INTRA_CATEGORY_FLAG} "
          f"(siblings that may be redundant):")
    if not intra_hits:
        print("  (none)")
    else:
        print(f"  {'cos':>6}   {'category':<18}  {'chap_a':<6}  {'chap_b':<6}")
        print("  " + "-" * 48)
        for s, i, j in intra_hits[:40]:
            print(f"  {s:>6.3f}   {rows[i]['category']:<18}  "
                  f"{rows[i]['chapter']:<6}  {rows[j]['chapter']:<6}")
        if len(intra_hits) > 40:
            print(f"  ... ({len(intra_hits) - 40} more)")

    # ── Top-of-matrix headline (most confusable overall) ─────────────────────
    off_diag: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append((float(sims[i, j]), i, j))
    off_diag.sort(reverse=True)
    print("\nTop 10 most similar centroid pairs (any category):")
    for s, i, j in off_diag[:10]:
        left  = f"{rows[i]['category']}/{rows[i]['chapter']}"
        right = f"{rows[j]['category']}/{rows[j]['chapter']}"
        same = "same-cat" if rows[i]["category"] == rows[j]["category"] else "CROSS"
        print(f"  {s:>6.3f}  [{same:<8}]  {left:<28}  {right:<28}")


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
