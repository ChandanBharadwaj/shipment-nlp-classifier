"""
Load anchor/suppressor/modifier rows from a CSV into `keywords`.

Substitute for `llm_generate_cctr.py` when no Anthropic API key is available
— the user (or Claude in chat) generates the CSV manually and this script
applies it to the DB.

CSV format (data/llm_generated/cctr_rows.csv):
    hs_chapter,signal_class,keyword,target_chapter,suppress_category_slug,head_noun

Empty cells for fields irrelevant to a signal_class are fine — anchors leave
the type-specific fields blank, suppressors fill suppress_category_slug,
modifiers fill target_chapter + head_noun.

Idempotent: ON CONFLICT DO UPDATE.

Weights are fixed defaults applied at insert time:
    anchor=1.0, suppressor=0.7, modifier=1.0

Usage:
    python scripts/load_llm_cctr.py
    python scripts/load_llm_cctr.py --csv path/to/other.csv
    python scripts/load_llm_cctr.py --source-tag chat:2026-05-04
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "llm_generated" / "cctr_rows.csv"

WEIGHTS = {"anchor": 1.0, "suppressor": 0.7, "modifier": 1.0}
VALID_CLASSES = set(WEIGHTS.keys())

INSERT_SQL = """
    INSERT INTO keywords
        (keyword, hs_chapter, signal_class, weight,
         target_chapter, suppress_category, head_noun, source)
    VALUES %s
    ON CONFLICT (keyword, hs_chapter, signal_class, source) DO UPDATE
        SET weight = EXCLUDED.weight,
            target_chapter = EXCLUDED.target_chapter,
            suppress_category = EXCLUDED.suppress_category,
            head_noun = EXCLUDED.head_noun,
            generated_at = now()
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--url", default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--source-tag",
                    default=f"chat:claude:{date.today().isoformat()}",
                    help="Tagged into the keywords.source column.")
    args = ap.parse_args()

    if not args.url:
        print("ERROR: --url not set and DATABASE_URL not in env.", file=sys.stderr)
        return 2

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 2

    print(f"Reading {csv_path}...")
    rows: list[tuple] = []
    skipped = 0
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader, 2):
            ch = (r.get("hs_chapter") or "").strip()
            cls = (r.get("signal_class") or "").strip()
            kw = (r.get("keyword") or "").strip().lower()
            target = (r.get("target_chapter") or "").strip() or None
            suppress = (r.get("suppress_category_slug") or "").strip() or None
            head = (r.get("head_noun") or "").strip() or None

            if not ch or not cls or not kw:
                print(f"  WARN line {i}: missing required field, skipped: {r}",
                      file=sys.stderr)
                skipped += 1
                continue
            if cls not in VALID_CLASSES:
                print(f"  WARN line {i}: invalid signal_class {cls!r}, skipped",
                      file=sys.stderr)
                skipped += 1
                continue

            rows.append((kw, ch, cls, WEIGHTS[cls], target, suppress, head, args.source_tag))

    print(f"  -> {len(rows)} rows ready ({skipped} skipped)")

    conn = psycopg2.connect(args.url)
    print(f"Inserting with source='{args.source_tag}'...")
    with conn.cursor() as cur:
        # Wipe prior chat-loaded rows so re-running this script with an updated
        # CSV doesn't leave orphaned rows from a previous load.
        cur.execute("DELETE FROM keywords WHERE source = %s", (args.source_tag,))
        psycopg2.extras.execute_values(cur, INSERT_SQL, rows, page_size=200)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("""
            SELECT signal_class, COUNT(*)
            FROM   keywords
            WHERE  source = %s
            GROUP  BY signal_class
            ORDER  BY 1
        """, (args.source_tag,))
        breakdown = cur.fetchall()
        cur.execute("""
            SELECT hs_chapter, COUNT(*)
            FROM   keywords
            WHERE  source = %s
            GROUP  BY hs_chapter
            ORDER  BY 2 DESC
            LIMIT  5
        """, (args.source_tag,))
        top_chapters = cur.fetchall()

    print("\n--- breakdown by signal_class ---")
    for cls, n in breakdown:
        print(f"  {cls:<11} {n}")
    print("\n--- top 5 chapters by row count ---")
    for ch, n in top_chapters:
        print(f"  ch {ch}: {n}")

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
