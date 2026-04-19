"""
Prune non-discriminative single-token keywords.

For each (category, keyword) pair in ``category_keywords``:

    kw_vec  = embed(keyword)
    cat_vec = mean of that category's chapter centroids (L2-normalized)
    cosine  = dot(kw_vec, cat_vec)

If ``cosine < MIN_KEYWORD_COSINE`` AND the keyword is a single token, mark it
for removal. Multi-token keywords are always kept — short phrases
(``"action figure"``, ``"lithium ion battery"``) are already semantically
grounded and rarely pollute.

Dry-run by default; prints a per-category summary. Pass ``--apply`` to commit
the prune to the database, after a CSV snapshot is written to
``diagnostics/prune_backup_YYYYMMDD_HHMMSS.csv``.

Rollback: the CSV is a plain ``(category_id, keyword, weight)`` dump. Replay
with ``COPY category_keywords(category_id, keyword, weight) FROM '...' CSV
HEADER`` to restore.

Usage:
    python -m diagnostics.prune_keywords                       # dry run, all cats
    python -m diagnostics.prune_keywords --category toys       # preview one category
    python -m diagnostics.prune_keywords --min-cosine 0.25     # adjust threshold
    python -m diagnostics.prune_keywords --apply               # commit

Ordering with Phase 3 (cross-encoder reranker):
    Always prune BEFORE enabling the reranker, not after. The reranker needs
    per-category descriptions, and ``main.py::_build_category_descriptions``
    builds them from the top-10 highest-weighted keywords in the DB. If you
    prune after the service is already running with the reranker on, the
    descriptions will still reflect the pre-prune keyword set until you
    reload them.

    Correct sequence:
        1. python -m diagnostics.prune_keywords             # dry run, review
        2. python -m diagnostics.prune_keywords --apply     # commit
        3. POST /reload on the service                      # rebuilds keywords
                                                            # AND category_descriptions
        4. (optional) set RERANKER_ENABLED=true and restart.

    /reload already calls _build_category_descriptions, so descriptions stay
    in lockstep with the keyword table — no manual step needed there.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from classifier import embed_texts, load_centroids, load_keywords  # noqa: E402
from db import pooled_connection  # noqa: E402

MIN_KEYWORD_COSINE_DEFAULT = 0.30
SAMPLE_PER_CATEGORY        = 8  # how many removed/kept examples to print


def _mean_centroid(children: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """Collapse per-chapter centroids into one L2-normalized vector per category."""
    mean = np.mean(np.stack([c[1] for c in children]), axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


def _load_category_ids(conn) -> dict[str, int]:
    ids: dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT id, name FROM classification_categories WHERE is_active = true")
        for cid, name in cur.fetchall():
            ids[name] = cid
    return ids


def _snapshot(rows_to_delete: list[tuple[int, str, float]]) -> str:
    """Dump (category_id, keyword, weight) to a timestamped CSV. Returns path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_HERE, f"prune_backup_{ts}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category_id", "keyword", "weight"])
        w.writerows(rows_to_delete)
    return path


def run(
    min_cosine: float = MIN_KEYWORD_COSINE_DEFAULT,
    category_filter: str | None = None,
    apply: bool = False,
) -> None:
    with pooled_connection() as conn:
        keywords    = load_keywords(conn)
        centroids   = load_centroids(conn)
        category_id = _load_category_ids(conn)

    if category_filter:
        if category_filter not in keywords:
            print(f"Category '{category_filter}' not found (or has no keywords).")
            return
        keywords = {category_filter: keywords[category_filter]}

    # Per-category mean centroid (L2-normalized so dot == cosine).
    cat_vecs: dict[str, np.ndarray] = {}
    for cat, children in centroids.items():
        if children:
            cat_vecs[cat] = _mean_centroid(children)

    # Collect every unique keyword across the categories we're scanning. We
    # embed the *union* in one batched call, then look up per pair.
    keyword_set: set[str] = set()
    for kws in keywords.values():
        for kw, _ in kws:
            keyword_set.add(kw)
    keyword_list = sorted(keyword_set)
    print(f"Embedding {len(keyword_list)} unique keywords…")
    kw_vecs = embed_texts(keyword_list)     # (n, dim), L2-normalized
    kw_index = {kw: i for i, kw in enumerate(keyword_list)}

    total_before = 0
    total_pruned = 0
    pruned_rows: list[tuple[int, str, float]] = []  # (category_id, keyword, weight)
    per_cat_report: list[tuple[str, int, int, list[str], list[str]]] = []

    for cat, kws in sorted(keywords.items()):
        total_before += len(kws)
        if cat not in cat_vecs:
            print(f"{cat:30s}  no centroid — skipped ({len(kws)} keywords unchanged)")
            continue
        cat_vec = cat_vecs[cat]
        removed: list[tuple[str, float]] = []
        kept:    list[tuple[str, float]] = []
        for kw, weight in kws:
            # Always keep multi-token keywords (conservative first-pass policy).
            if " " in kw or "-" in kw:
                kept.append((kw, float("nan")))
                continue
            vec = kw_vecs[kw_index[kw]]
            cos = float(np.dot(vec, cat_vec))
            if cos < min_cosine:
                removed.append((kw, cos))
            else:
                kept.append((kw, cos))
        per_cat_report.append((
            cat, len(kws), len(removed),
            [f"{kw} ({cos:.2f})" for kw, cos in removed[:SAMPLE_PER_CATEGORY]],
            [f"{kw}" for kw, _ in kept[:SAMPLE_PER_CATEGORY]],
        ))
        if cat in category_id:
            for kw, _cos in removed:
                # Look up the original weight from `kws`.
                for k, w in kws:
                    if k == kw:
                        pruned_rows.append((category_id[cat], kw, float(w)))
                        break
        total_pruned += len(removed)

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\nThreshold: cosine < {min_cosine:.2f}  (single-token only)\n")
    print(f"{'category':30s} {'total':>6} {'pruned':>7} {'%':>6}")
    print("-" * 80)
    for cat, total, pruned, rem_sample, kept_sample in per_cat_report:
        pct = (100.0 * pruned / total) if total else 0.0
        print(f"{cat:30s} {total:>6} {pruned:>7} {pct:>5.1f}%")
        if rem_sample:
            print(f"    removed: {', '.join(rem_sample)}")
        if kept_sample:
            print(f"    kept:    {', '.join(kept_sample)}")
    print("-" * 80)
    pct_total = (100.0 * total_pruned / total_before) if total_before else 0.0
    print(f"Total: {total_before} → {total_before - total_pruned} "
          f"(pruned {total_pruned}, {pct_total:.1f}%)")

    # ── Apply ────────────────────────────────────────────────────────────────
    if not apply:
        print("\nDry run. Re-run with --apply to commit. "
              "Snapshot is written only on --apply.")
        return

    if not pruned_rows:
        print("\nNothing to prune.")
        return

    snapshot_path = _snapshot(pruned_rows)
    print(f"\nSnapshot written: {snapshot_path}")

    with pooled_connection() as conn:
        with conn.cursor() as cur:
            # Parameterized batch delete — one statement per row keeps things simple
            # and safe; with ~500 rows this is well under a second.
            cur.executemany(
                "DELETE FROM category_keywords WHERE category_id = %s AND keyword = %s",
                [(cid, kw) for cid, kw, _ in pruned_rows],
            )
        conn.commit()
    print(f"Deleted {len(pruned_rows)} rows. "
          f"Hit POST /reload on the running service to pick up the change.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-cosine", type=float, default=MIN_KEYWORD_COSINE_DEFAULT,
                    help=f"Cosine threshold below which a keyword is pruned (default: {MIN_KEYWORD_COSINE_DEFAULT})")
    ap.add_argument("--category", type=str, default=None,
                    help="Limit the run to one category (dry run only).")
    ap.add_argument("--apply", action="store_true",
                    help="Commit the prune to the DB after writing a CSV snapshot.")
    args = ap.parse_args()

    if args.apply and args.category:
        print("Refusing to --apply while --category is set. Drop --category for a full apply.")
        sys.exit(2)

    run(min_cosine=args.min_cosine, category_filter=args.category, apply=args.apply)


if __name__ == "__main__":
    main()
