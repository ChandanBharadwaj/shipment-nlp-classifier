"""
Validate chat-generated CCTR rows against every data source we have.

For each chat-generated keyword row, run multiple checks and emit a
"flagged for review" CSV with status + reason for the suspicious ones.

Then auto-promote unused high-TF-IDF tokens to anchors so the gap between
"what TF-IDF found in the source text" and "what chat anchors covered" closes.

Validation checks:
    A. Anchor not in source AND ALSO not in any other chapter's source
       → likely hallucination
    B. Anchor in source for a DIFFERENT chapter (more text matches there)
       → possible mis-assignment
    C. Single-word anchor that's a top-frequency word across all chapters
       → too generic
    D. Suppressor's suppress_category doesn't exist in `categories`
       → broken FK
    E. Modifier's target_chapter is the same as its hs_chapter
       → circular routing (likely bug from chat generation)
    F. Modifier's target_chapter not in chapter_categories
       → routes nowhere
    G. Anchor IS in source AND TF-IDF also picked it
       → double coverage (informational, not a problem)

Promotion (writes to DB):
    - For each chapter, take TF-IDF tokens at weight 1.5 (highest)
      that are NOT already in chat anchors. Insert as anchors at
      weight 0.9, source='tfidf_promoted:YYYY-MM-DD'. This captures
      data-driven trade vocabulary the chat generation missed.

Usage:
    python scripts/validate_chat_cctr.py
    python scripts/validate_chat_cctr.py --no-promote   # validation only
    python scripts/validate_chat_cctr.py --review-csv path/out.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVIEW_CSV = ROOT / "data" / "llm_generated" / "anchor_review.csv"

PROMOTED_TAG = f"tfidf_promoted:{date.today().isoformat()}"
PROMOTED_WEIGHT = 0.9


# ── Validators ────────────────────────────────────────────────────────────────

def _build_source_index(conn) -> dict[str, str]:
    """Return {hs_chapter: lowercased concatenated description text}."""
    out: dict[str, list[str]] = defaultdict(list)
    with conn.cursor() as cur:
        cur.execute("SELECT hs_chapter, description FROM public_source_descriptions")
        for ch, descr in cur.fetchall():
            out[ch].append((descr or "").lower())
    return {ch: " | ".join(texts) for ch, texts in out.items()}


def _validate_anchors(conn, source_text: dict[str, str]) -> list[dict]:
    """Return a list of validation records for every anchor row."""
    flags: list[dict] = []
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, keyword, hs_chapter
            FROM   keywords
            WHERE  source LIKE 'chat:%' AND signal_class = 'anchor'
            ORDER  BY hs_chapter, keyword
        """)
        anchors = list(cur.fetchall())

    # Pre-compute: in which chapters does each keyword's substring appear?
    keyword_chapters: dict[str, list[str]] = defaultdict(list)
    for a in anchors:
        kw_lc = a["keyword"].lower()
        for ch, text in source_text.items():
            if kw_lc in text:
                keyword_chapters[a["keyword"]].append(ch)

    # Generic-word detection: if a keyword appears in >40 of 96 chapters,
    # it's too generic to be a useful anchor.
    GENERIC_THRESHOLD = 40

    for a in anchors:
        kw = a["keyword"]
        ch = a["hs_chapter"]
        chapters_with_kw = keyword_chapters.get(kw, [])
        in_own = ch in chapters_with_kw
        elsewhere = [c for c in chapters_with_kw if c != ch]

        status = "ok"
        reasons = []

        # Check A: not in source AND not elsewhere → likely hallucination
        if not in_own and not elsewhere:
            status = "flagged"
            reasons.append("not_in_any_source_text")

        # Check B: appears in OTHER chapters but not its own
        elif not in_own and elsewhere:
            status = "review"
            reasons.append(f"in_other_chapters:{','.join(sorted(elsewhere)[:5])}")

        # Check C: too generic — appears in many chapters
        if len(chapters_with_kw) > GENERIC_THRESHOLD:
            status = "flagged"
            reasons.append(f"appears_in_{len(chapters_with_kw)}_chapters")

        flags.append({
            "id": a["id"],
            "keyword": kw,
            "hs_chapter": ch,
            "signal_class": "anchor",
            "status": status,
            "in_own_chapter_source": in_own,
            "appears_in_n_chapters": len(chapters_with_kw),
            "reason": "; ".join(reasons) if reasons else "",
        })
    return flags


def _validate_suppressors(conn) -> list[dict]:
    flags: list[dict] = []
    with conn.cursor() as cur:
        cur.execute("SELECT slug FROM categories")
        valid_slugs = {r[0] for r in cur.fetchall()}
        cur.execute("""
            SELECT id, keyword, hs_chapter, suppress_category
            FROM   keywords
            WHERE  source LIKE 'chat:%' AND signal_class = 'suppressor'
            ORDER  BY hs_chapter, keyword
        """)
        for sid, kw, ch, sup in cur.fetchall():
            status = "ok"
            reasons = []
            if not sup:
                status = "flagged"
                reasons.append("missing_suppress_category")
            elif sup not in valid_slugs:
                status = "flagged"
                reasons.append(f"unknown_category:{sup}")
            flags.append({
                "id": sid,
                "keyword": kw,
                "hs_chapter": ch,
                "signal_class": "suppressor",
                "status": status,
                "in_own_chapter_source": "",
                "appears_in_n_chapters": "",
                "reason": "; ".join(reasons) if reasons else "",
            })
    return flags


def _validate_modifiers(conn) -> list[dict]:
    flags: list[dict] = []
    with conn.cursor() as cur:
        cur.execute("SELECT hs_chapter FROM chapter_categories")
        valid_chapters = {r[0] for r in cur.fetchall()}
        cur.execute("""
            SELECT id, keyword, hs_chapter, target_chapter, head_noun
            FROM   keywords
            WHERE  source LIKE 'chat:%' AND signal_class = 'modifier'
            ORDER  BY hs_chapter, keyword
        """)
        for mid, kw, ch, target, head in cur.fetchall():
            status = "ok"
            reasons = []
            if not target:
                status = "flagged"
                reasons.append("missing_target_chapter")
            elif target not in valid_chapters:
                status = "flagged"
                reasons.append(f"unknown_target:{target}")
            elif target == ch:
                status = "review"
                reasons.append("circular_routing_to_own_chapter")
            if not head:
                status = "flagged"
                reasons.append("missing_head_noun")
            flags.append({
                "id": mid,
                "keyword": kw,
                "hs_chapter": ch,
                "signal_class": "modifier",
                "status": status,
                "in_own_chapter_source": "",
                "appears_in_n_chapters": "",
                "reason": "; ".join(reasons) if reasons else "",
            })
    return flags


# ── TF-IDF promotion ──────────────────────────────────────────────────────────

PROMOTE_SQL = """
    INSERT INTO keywords
        (keyword, hs_chapter, signal_class, weight, source)
    VALUES %s
    ON CONFLICT (keyword, hs_chapter, signal_class, source) DO UPDATE
        SET weight = EXCLUDED.weight,
            generated_at = now()
"""


def _promote_tfidf_to_anchors(conn) -> tuple[int, int]:
    """For each chapter, find TF-IDF tokens at weight 1.5 that aren't already
    a chat anchor for that chapter, and insert them as anchors. Returns
    (n_candidates, n_promoted)."""
    with conn.cursor() as cur:
        cur.execute("""
            WITH top_tfidf AS (
                SELECT hs_chapter, keyword
                FROM   keywords
                WHERE  source = 'public_tfidf' AND weight = 1.5
            ),
            chat_anchors AS (
                SELECT DISTINCT hs_chapter, keyword
                FROM   keywords
                WHERE  source LIKE 'chat:%' AND signal_class = 'anchor'
            )
            SELECT t.hs_chapter, t.keyword
            FROM   top_tfidf t
            LEFT   JOIN chat_anchors a USING (hs_chapter, keyword)
            WHERE  a.keyword IS NULL
            ORDER  BY t.hs_chapter, t.keyword
        """)
        candidates = list(cur.fetchall())

    n_candidates = len(candidates)
    if n_candidates == 0:
        return 0, 0

    rows = [(kw, ch, "anchor", PROMOTED_WEIGHT, PROMOTED_TAG) for (ch, kw) in candidates]

    with conn.cursor() as cur:
        # Wipe prior promotion runs first so re-running is clean.
        cur.execute("DELETE FROM keywords WHERE source LIKE 'tfidf_promoted:%'")
        psycopg2.extras.execute_values(cur, PROMOTE_SQL, rows, page_size=500)
    conn.commit()
    return n_candidates, len(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--url", default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--review-csv", default=str(DEFAULT_REVIEW_CSV))
    ap.add_argument("--no-promote", action="store_true",
                    help="Skip TF-IDF promotion; validation only.")
    args = ap.parse_args()

    if not args.url:
        print("ERROR: --url not set and DATABASE_URL not in env.", file=sys.stderr)
        return 2

    conn = psycopg2.connect(args.url)
    print("Loading source-text index...")
    source_text = _build_source_index(conn)
    print(f"  -> {len(source_text)} chapters of source text")

    print("Validating anchors...")
    anchor_flags = _validate_anchors(conn, source_text)
    print("Validating suppressors...")
    suppressor_flags = _validate_suppressors(conn)
    print("Validating modifiers...")
    modifier_flags = _validate_modifiers(conn)

    all_flags = anchor_flags + suppressor_flags + modifier_flags

    # Summary by status
    by_status: Counter = Counter(f["status"] for f in all_flags)
    print(f"\n--- Validation summary ({len(all_flags)} rows checked) ---")
    for status, n in sorted(by_status.items()):
        print(f"  {status:<10} {n}")

    by_reason: Counter = Counter()
    for f in all_flags:
        if f["reason"]:
            for r in f["reason"].split(";"):
                tag = r.strip().split(":")[0]
                by_reason[tag] += 1
    if by_reason:
        print("\n--- Flag reasons ---")
        for reason, n in by_reason.most_common():
            print(f"  {reason:<35} {n}")

    # Write the review CSV (only flagged + review rows; ok ones are noise).
    review_rows = [f for f in all_flags if f["status"] in ("flagged", "review")]
    review_path = Path(args.review_csv)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "keyword", "hs_chapter", "signal_class",
            "status", "in_own_chapter_source",
            "appears_in_n_chapters", "reason",
        ])
        writer.writeheader()
        for r in review_rows:
            writer.writerow(r)
    print(f"\nFlagged/review rows written to {review_path}")
    print(f"  ({len(review_rows)} rows; review and edit data/llm_generated/cctr_rows.csv accordingly)")

    if not args.no_promote:
        print("\n--- Promoting unused high-TF-IDF tokens to anchors ---")
        n_cands, n_promoted = _promote_tfidf_to_anchors(conn)
        print(f"  candidates: {n_cands}")
        print(f"  promoted as anchors (weight {PROMOTED_WEIGHT}, source={PROMOTED_TAG}): {n_promoted}")

        # Final keyword distribution
        with conn.cursor() as cur:
            cur.execute("""
                SELECT signal_class, source, COUNT(*) FROM keywords
                GROUP BY signal_class, source ORDER BY signal_class, source
            """)
            print("\n--- Final keyword distribution ---")
            for cls, src, n in cur.fetchall():
                print(f"  {cls:<11} {src:<35} {n}")

    conn.close()
    return 0


if __name__ == "__main__":
    for env_path in [ROOT / "ml-service" / ".env", ROOT / ".env"]:
        if env_path.exists():
            load_dotenv(env_path)
            break
    else:
        load_dotenv()
    sys.exit(main())
