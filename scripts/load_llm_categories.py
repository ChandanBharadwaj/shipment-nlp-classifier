"""
Load chapter→category mapping from a CSV into `categories` + `chapter_categories`.

Used as a substitute for `llm_classify_chapters.py` when no Anthropic API key
is available — the user (or Claude in chat) generates the CSV manually and
this script applies it to the DB.

CSV format (data/llm_generated/chapter_categories.csv):
    hs_chapter,category_slug,category_display_name,chapter_title

Idempotent: re-running with the same CSV is a no-op (ON CONFLICT DO UPDATE).

Usage:
    python scripts/load_llm_categories.py
    python scripts/load_llm_categories.py --csv path/to/other.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "llm_generated" / "chapter_categories.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--url", default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    args = ap.parse_args()

    if not args.url:
        print("ERROR: --url not set and DATABASE_URL not in env.", file=sys.stderr)
        return 2

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 2

    print(f"Reading {csv_path}...")
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  -> {len(rows)} rows")

    # Validate: every chapter exactly once, slugs are URL-safe.
    seen: dict[str, str] = {}
    categories: dict[str, str] = {}   # slug -> display_name
    for r in rows:
        ch = r["hs_chapter"]
        if ch in seen:
            print(f"ERROR: duplicate hs_chapter {ch}", file=sys.stderr)
            return 2
        seen[ch] = r["category_slug"]
        slug = r["category_slug"]
        display = r["category_display_name"]
        if slug in categories and categories[slug] != display:
            print(f"ERROR: slug {slug!r} maps to multiple display names "
                  f"({categories[slug]!r} vs {display!r})", file=sys.stderr)
            return 2
        categories[slug] = display

    print(f"  {len(categories)} distinct categories")

    conn = psycopg2.connect(args.url)

    print("Inserting categories...")
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO categories (slug, display_name) VALUES %s "
            "ON CONFLICT (slug) DO UPDATE SET display_name = EXCLUDED.display_name",
            [(slug, name) for slug, name in categories.items()],
        )

    print("Inserting chapter_categories...")
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO chapter_categories (hs_chapter, category_slug, chapter_title) VALUES %s "
            "ON CONFLICT (hs_chapter) DO UPDATE SET "
            "  category_slug = EXCLUDED.category_slug, "
            "  chapter_title = EXCLUDED.chapter_title",
            [(r["hs_chapter"], r["category_slug"], r["chapter_title"]) for r in rows],
        )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM categories")
        n_cats = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chapter_categories")
        n_map = cur.fetchone()[0]
        cur.execute("""
            SELECT c.display_name, COUNT(cm.hs_chapter) AS chapters
            FROM   categories c
            LEFT   JOIN chapter_categories cm ON cm.category_slug = c.slug
            GROUP  BY c.display_name
            ORDER  BY chapters DESC, c.display_name
        """)
        breakdown = cur.fetchall()

    print(f"\nResult: {n_cats} categories, {n_map} chapters mapped\n")
    for name, count in breakdown:
        print(f"  {count:>3}  {name}")

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
