"""
Training-data coverage diagnostics.

Reports:
  1. Row counts per split, per category, per chapter.
  2. Per-category mean token count and Jaccard similarity across train rows
     (ballpark diversity metric — low Jaccard = diverse vocabulary).
  3. HS-chapter coverage: every assigned chapter should have >= 20 train
     rows drawn from its 6-digit vocabulary.
  4. Categories whose train rows are dominated by a single chapter (>90%)
     — flags unbalanced categories that will lean on one centroid.

Read-only. Run after init_db.py. Intended output lands in
`results/diag_coverage_v2.txt`.

Usage:
    python -m diagnostics.data_coverage
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter, defaultdict

# Allow running from ml-service/ or from repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from db import get_connection  # noqa: E402


MIN_CHAPTER_TRAIN_ROWS   = 20
SINGLE_CHAPTER_DOMINANCE = 0.90

_TOKEN_RE = re.compile(r"[a-z]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 2}


def _fetch_rows(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT category_name, split, hs_chapter,
                   cargo_text, commodity_text
            FROM   shipment_labels
        """)
        return cur.fetchall()


def run(conn) -> None:
    rows = _fetch_rows(conn)
    if not rows:
        print("No rows in shipment_labels.")
        return

    # Bucket: by_cat_split[(cat, split)] -> list[(chap, cargo, commodity)]
    by_cat_split: dict[tuple[str, str], list] = defaultdict(list)
    for cat, split, chap, cargo, commodity in rows:
        by_cat_split[(cat, split)].append((chap, cargo or "", commodity or ""))

    # ── Global split counts ──────────────────────────────────────────────────
    split_counts = Counter((r[1] for r in rows))
    print("Rows by split:")
    for s in ("train", "validation", "test"):
        print(f"  {s:<12}  {split_counts.get(s, 0)}")
    print(f"  {'total':<12}  {sum(split_counts.values())}")

    categories = sorted({r[0] for r in rows})

    # ── Per-category totals ──────────────────────────────────────────────────
    print("\nPer-category row counts:")
    print(f"  {'category':<20}  {'train':>6}  {'val':>6}  {'test':>6}  {'total':>6}")
    print("  " + "-" * 52)
    for cat in categories:
        tr = len(by_cat_split.get((cat, "train"), []))
        va = len(by_cat_split.get((cat, "validation"), []))
        te = len(by_cat_split.get((cat, "test"), []))
        print(f"  {cat:<20}  {tr:>6}  {va:>6}  {te:>6}  {tr+va+te:>6}")

    # ── Per-chapter training coverage ────────────────────────────────────────
    train_chap_rows: dict[tuple[str, str], int] = defaultdict(int)
    for cat, split, chap, _c, _m in rows:
        if split == "train" and chap:
            train_chap_rows[(cat, chap)] += 1

    assigned_chapters: dict[str, list[str]] = defaultdict(list)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT cc.name, chc.hs_chapter
            FROM   category_hs_chapters chc
            JOIN   classification_categories cc ON cc.id = chc.category_id
            ORDER  BY cc.name, chc.hs_chapter
        """)
        for cat, chap in cur.fetchall():
            assigned_chapters[cat].append(chap)

    print("\nHS-chapter training coverage (target: >= "
          f"{MIN_CHAPTER_TRAIN_ROWS} rows per assigned chapter):")
    print(f"  {'category':<20}  {'chapter':>7}  {'rows':>5}  status")
    print("  " + "-" * 50)
    thin_chapters: list[tuple[str, str, int]] = []
    for cat in categories:
        for chap in sorted(assigned_chapters.get(cat, [])):
            n_train = train_chap_rows.get((cat, chap), 0)
            status = "OK" if n_train >= MIN_CHAPTER_TRAIN_ROWS else "THIN"
            if n_train < MIN_CHAPTER_TRAIN_ROWS:
                thin_chapters.append((cat, chap, n_train))
            print(f"  {cat:<20}  {chap:>7}  {n_train:>5}  {status}")

    if thin_chapters:
        print(f"\nThin chapters ({len(thin_chapters)}):")
        for cat, chap, n in thin_chapters:
            print(f"  {cat:<20}  {chap:>3}  {n}")

    # ── Single-chapter dominance ─────────────────────────────────────────────
    print("\nChapter distribution within each category (train split):")
    dominated: list[tuple[str, str, float]] = []
    for cat in categories:
        train_rows = by_cat_split.get((cat, "train"), [])
        if not train_rows:
            continue
        chap_counts = Counter(chap for chap, _c, _m in train_rows if chap)
        if not chap_counts:
            continue
        total = sum(chap_counts.values())
        top_chap, top_n = chap_counts.most_common(1)[0]
        share = top_n / total if total else 0.0
        if share >= SINGLE_CHAPTER_DOMINANCE:
            dominated.append((cat, top_chap, share))
    if dominated:
        print("  Categories dominated by a single chapter "
              f"(>= {int(SINGLE_CHAPTER_DOMINANCE*100)}% of train rows):")
        for cat, chap, share in dominated:
            print(f"    {cat:<20}  chapter {chap}  {share*100:.1f}%")
    else:
        print("  (no category is single-chapter-dominated — good balance)")

    # ── Jaccard + token counts ───────────────────────────────────────────────
    print("\nPer-category train vocabulary:")
    print(f"  {'category':<20}  {'rows':>5}  {'avg tokens':>10}  {'median Jaccard':>14}")
    print("  " + "-" * 58)
    for cat in categories:
        train_rows = by_cat_split.get((cat, "train"), [])
        if len(train_rows) < 2:
            continue
        token_sets = [_tokens(f"{c} {m}") for _ch, c, m in train_rows]
        avg_toks = sum(len(s) for s in token_sets) / len(token_sets)
        # Sample-based Jaccard (N^2 over 1000+ rows is slow — subsample)
        import random
        random.seed(42)
        subs = random.sample(token_sets, min(80, len(token_sets)))
        jaccards = []
        for i in range(len(subs)):
            for j in range(i + 1, len(subs)):
                a, b = subs[i], subs[j]
                if not a and not b: continue
                u = len(a | b)
                if u == 0: continue
                jaccards.append(len(a & b) / u)
        if jaccards:
            jaccards.sort()
            median = jaccards[len(jaccards) // 2]
        else:
            median = 0.0
        print(f"  {cat:<20}  {len(train_rows):>5}  {avg_toks:>10.1f}  {median:>14.3f}")


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
