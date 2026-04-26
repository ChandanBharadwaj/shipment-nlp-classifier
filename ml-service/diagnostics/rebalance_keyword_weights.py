"""
Rebalance per-(category, hs_chapter, keyword) weights — CCTR replacement
for prune_keywords.py.

Property P1 (CCTR plan): no keyword is removed; every keyword stays
attached to its right HS chapter. This script enforces P1 by *adjusting
weights*, not by deleting rows. Default mode is non-destructive: weights
get pulled toward the bounds [WEIGHT_FLOOR, WEIGHT_CAP] based on the
keyword's cosine to the chapter centroid the row lives in. Anchors and
modifiers are sticky — never adjusted regardless of cosine — because
their value is structural, not statistical.

For each row in ``category_keywords`` with hs_chapter set:

    kw_vec       = embed(keyword)
    chapter_vec  = the row's (category, hs_chapter) centroid
    cosine       = dot(kw_vec, chapter_vec)

Rebalancing rule (signal_class='signal'):
    new_weight = clip(WEIGHT_FLOOR, old_weight * (1 + (cosine - 0.5) * 0.5), WEIGHT_CAP)

That's a soft pull toward the bounds: very-aligned keywords nudge up,
very-unaligned ones nudge down to the floor (still queryable). The
formula intentionally damps the swing — one rebalance pass should not
drastically restructure the table.

Anchors and modifiers are skipped:
    - anchor   weight is structural; rebalancer never lowers it below
               0.9. This is the "anchors stay anchors" guarantee.
    - modifier weight is the modifier's *match strength*, not its
               importance to a chapter — cosine is meaningless for it.
    - suppressor rows ARE rebalanced because their weight is a real
               numeric subtractor that should track empirical signal
               strength.

Modes:
    (default)  Dry-run summary only. No DB writes.
    --apply    Persist new weights via UPDATE. Snapshot CSV written to
               diagnostics/rebalance_backup_YYYYMMDD_HHMMSS.csv first.
    --strict   EMERGENCY ROLLBACK: behaves like the legacy prune_keywords —
               deletes single-token rows whose cosine < min_cosine.
               This violates P1 and exists only so an operator can match
               legacy behaviour during a rollback. Not used in CI.

Comparison to the legacy pruner:

    | Aspect              | prune_keywords (legacy) | rebalance (default) | rebalance --strict |
    |---------------------|-------------------------|---------------------|--------------------|
    | Deletes rows        | yes                     | NO                  | yes (legacy parity)|
    | Per-chapter aware   | no                      | yes                 | no                 |
    | Respects anchors    | no                      | yes                 | no                 |
    | Respects modifiers  | no                      | yes                 | no                 |
    | P1 compliant        | no                      | YES                 | no                 |

Usage:
    python -m diagnostics.rebalance_keyword_weights                 # dry run
    python -m diagnostics.rebalance_keyword_weights --category toys # one category
    python -m diagnostics.rebalance_keyword_weights --apply         # commit
    python -m diagnostics.rebalance_keyword_weights --strict --apply # legacy parity
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

from classifier import embed_texts, load_centroids  # noqa: E402
from config import (  # noqa: E402
    CCTR_WEIGHT_CAP,
    CCTR_WEIGHT_FLOOR,
    MIN_KEYWORD_COSINE_DEFAULT,
)
from db import pooled_connection  # noqa: E402


SAMPLE_PER_CATEGORY = 8


def _load_keyword_rows(conn, category_filter: str | None = None
                       ) -> list[tuple[int, str, str, str, float, str]]:
    """Return [(id, category_name, hs_chapter, keyword, weight, signal_class)]
    for every category_keywords row with hs_chapter NOT NULL. Legacy NULL-
    chapter rows are out of scope for the rebalancer (the migration is
    expected to have populated them by now)."""
    out: list[tuple[int, str, str, str, float, str]] = []
    sql = """
        SELECT ck.id, cc.name, ck.hs_chapter, ck.keyword, ck.weight, ck.signal_class
        FROM   category_keywords ck
        JOIN   classification_categories cc ON cc.id = ck.category_id
        WHERE  cc.is_active = true
          AND  ck.hs_chapter IS NOT NULL
    """
    params: tuple = ()
    if category_filter:
        sql += " AND cc.name = %s"
        params = (category_filter,)
    sql += " ORDER BY cc.name, ck.hs_chapter, ck.keyword"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        for row_id, name, hs, kw, weight, sclass in cur.fetchall():
            out.append((row_id, name, hs, kw, float(weight), sclass))
    return out


def _chapter_vec_index(centroids) -> dict[tuple[str, str], np.ndarray]:
    """Flatten {category: [(hs, vec), ...]} to {(category, hs): vec}."""
    out: dict[tuple[str, str], np.ndarray] = {}
    for cat, children in centroids.items():
        for hs, vec in children:
            out[(cat, hs)] = vec
    return out


def _new_weight(old: float, cosine: float, floor: float, cap: float) -> float:
    """Soft pull toward the bounds. Cosine 0.5 is neutral; >0.5 nudges up,
    <0.5 nudges down. The 0.5 attenuator dampens the swing so one pass
    can't crater a row."""
    proposed = old * (1.0 + (cosine - 0.5) * 0.5)
    return max(floor, min(cap, proposed))


def _snapshot(rows: list[tuple[int, str, str, str, float, float, str]]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_HERE, f"rebalance_backup_{ts}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "hs_chapter", "keyword", "old_weight", "new_weight", "signal_class"])
        w.writerows(rows)
    return path


def run(
    min_cosine: float = MIN_KEYWORD_COSINE_DEFAULT,
    category_filter: str | None = None,
    apply: bool = False,
    strict: bool = False,
    floor: float = CCTR_WEIGHT_FLOOR,
    cap: float = CCTR_WEIGHT_CAP,
) -> int:
    """Returns the number of rows that would change (or were changed when --apply)."""
    with pooled_connection() as conn:
        rows = _load_keyword_rows(conn, category_filter)
        centroids = load_centroids(conn)

    chapter_vec = _chapter_vec_index(centroids)

    # Embed every unique keyword once.
    unique_kws = sorted({kw for _id, _cat, _hs, kw, _w, _sc in rows})
    print(f"Embedding {len(unique_kws)} unique keywords…")
    kw_vecs = embed_texts(unique_kws)
    kw_index = {kw: i for i, kw in enumerate(unique_kws)}

    # Per-row cosine + new weight.
    changes_apply: list[tuple[int, str, str, str, float, float, str]] = []
    deletes_strict: list[tuple[int, str, str, str, float, float, str]] = []
    skipped_anchor_modifier = 0
    no_centroid = 0

    per_cat_report: dict[str, dict[str, int]] = {}

    for row_id, cat, hs, kw, old_w, sclass in rows:
        if cat not in per_cat_report:
            per_cat_report[cat] = {"total": 0, "rebalanced": 0, "deleted": 0, "sticky": 0}
        per_cat_report[cat]["total"] += 1

        # P1 stickiness: anchors and modifiers never get touched.
        if sclass in ("anchor", "modifier"):
            skipped_anchor_modifier += 1
            per_cat_report[cat]["sticky"] += 1
            continue

        vec = chapter_vec.get((cat, hs))
        if vec is None:
            no_centroid += 1
            continue

        cos = float(np.dot(kw_vecs[kw_index[kw]], vec))

        if strict:
            # Legacy parity: single-token rows with cosine < threshold are
            # deleted. Multi-token rows are kept (matches prune_keywords).
            if " " not in kw and "-" not in kw and cos < min_cosine:
                deletes_strict.append((row_id, cat, hs, kw, old_w, 0.0, sclass))
                per_cat_report[cat]["deleted"] += 1
            continue

        # Default mode: rebalance weight; never delete.
        new_w = _new_weight(old_w, cos, floor, cap)
        if abs(new_w - old_w) > 1e-4:
            changes_apply.append((row_id, cat, hs, kw, old_w, new_w, sclass))
            per_cat_report[cat]["rebalanced"] += 1

    # ── Report ───────────────────────────────────────────────────────────────
    mode = "STRICT (delete)" if strict else "default (rebalance only)"
    print(f"\nMode: {mode}")
    print(f"Bounds: [{floor:.2f}, {cap:.2f}]   strict-cosine: {min_cosine:.2f}")
    print(f"Sticky (anchor/modifier): {skipped_anchor_modifier}   no-centroid: {no_centroid}\n")
    print(f"{'category':30s} {'total':>6} {'rebal':>6} {'del':>4} {'sticky':>7}")
    print("-" * 70)
    for cat in sorted(per_cat_report):
        c = per_cat_report[cat]
        print(f"{cat:30s} {c['total']:>6} {c['rebalanced']:>6} {c['deleted']:>4} {c['sticky']:>7}")
    print("-" * 70)
    total = sum(c["total"] for c in per_cat_report.values())
    n_rebal = sum(c["rebalanced"] for c in per_cat_report.values())
    n_del = sum(c["deleted"] for c in per_cat_report.values())
    print(f"TOTAL  {total} rows  |  rebalanced={n_rebal}  deleted={n_del}\n")

    if not apply:
        print("Dry run. Re-run with --apply to commit.")
        return n_rebal + n_del

    # ── Apply ────────────────────────────────────────────────────────────────
    if strict and deletes_strict:
        snap = _snapshot(deletes_strict)
        print(f"Snapshot (strict deletes) written: {snap}")
        with pooled_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "DELETE FROM category_keywords WHERE id = %s",
                    [(rid,) for rid, *_ in deletes_strict],
                )
            conn.commit()
        print(f"Deleted {len(deletes_strict)} rows (--strict).")
        return len(deletes_strict)

    if not strict and changes_apply:
        snap = _snapshot(changes_apply)
        print(f"Snapshot (rebalanced weights) written: {snap}")
        with pooled_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE category_keywords SET weight = %s WHERE id = %s",
                    [(round(new_w, 2), rid) for rid, _c, _h, _k, _o, new_w, _s in changes_apply],
                )
            conn.commit()
        print(f"Rebalanced {len(changes_apply)} rows. "
              f"POST /reload on the service to pick up the change.")
        return len(changes_apply)

    print("No changes to apply.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-cosine", type=float, default=MIN_KEYWORD_COSINE_DEFAULT,
                    help="Cosine threshold (only used by --strict). Default %(default)s.")
    ap.add_argument("--category", type=str, default=None,
                    help="Limit to one category (dry run only).")
    ap.add_argument("--apply", action="store_true",
                    help="Commit changes to the DB (snapshot CSV written first).")
    ap.add_argument("--strict", action="store_true",
                    help="EMERGENCY ROLLBACK: delete single-token rows below "
                         "--min-cosine (legacy prune_keywords parity). Violates P1.")
    ap.add_argument("--floor", type=float, default=CCTR_WEIGHT_FLOOR,
                    help="Minimum weight after rebalance. Default %(default)s.")
    ap.add_argument("--cap", type=float, default=CCTR_WEIGHT_CAP,
                    help="Maximum weight after rebalance. Default %(default)s.")
    args = ap.parse_args()

    if args.apply and args.category:
        print("Refusing to --apply while --category is set. Drop --category for a full apply.")
        sys.exit(2)

    run(
        min_cosine=args.min_cosine,
        category_filter=args.category,
        apply=args.apply,
        strict=args.strict,
        floor=args.floor,
        cap=args.cap,
    )


if __name__ == "__main__":
    main()
