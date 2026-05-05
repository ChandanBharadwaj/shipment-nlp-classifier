"""
LLM-derive a business-category taxonomy + chapter mapping.

Two-pass call to Claude Sonnet 4.6:
    1. Generate: propose N categories with chapter assignments.
    2. Validate: refine — drop too-narrow, split too-broad, fix mis-assignments.

Inputs:
    - public_source_descriptions table (read; uses chapter_title from UK rows
      where present, else falls back to a default per-chapter title from a
      static map).
    - prompts/chapter_classification.md (prompt template, two passes).

Outputs (committed to git for reproducibility):
    - seed/seed_categories_v3.sql        INSERT INTO categories ...
    - seed/seed_chapter_categories_v3.sql INSERT INTO chapter_categories ...

The script also INSERTs directly into the live DB so downstream pipeline
steps can run immediately without manual seed-load.

REQUIRES ANTHROPIC_API_KEY in env. Refuses to run without it. Cost: ~2 LLM
calls × ~10K tokens each ≈ ~$0.30-1.00 per run.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/llm_classify_chapters.py
    python scripts/llm_classify_chapters.py --url postgresql://...
    python scripts/llm_classify_chapters.py --dry-run    # print prompt, don't call LLM
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import httpx
import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT / "prompts" / "chapter_classification.md"
OUT_CATEGORIES_SQL = ROOT / "seed" / "seed_categories_v3.sql"
OUT_CHAPTER_MAP_SQL = ROOT / "seed" / "seed_chapter_categories_v3.sql"

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 8192


# ── Prompt loading ────────────────────────────────────────────────────────────

def _extract_prompt(name: str) -> str:
    """Extract a fenced code block from chapter_classification.md by section
    title ('Pass 1 — Generate' or 'Pass 2 — Validate / refine')."""
    text = PROMPT_FILE.read_text(encoding="utf-8")
    # Find the section header, then the next fenced ``` ... ``` block.
    pattern = rf"## {re.escape(name)}.*?```\n(.*?)```"
    m = re.search(pattern, text, flags=re.DOTALL)
    if not m:
        raise RuntimeError(f"Could not find prompt section {name!r} in {PROMPT_FILE}")
    return m.group(1).strip()


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_chapter_titles(conn) -> dict[str, str]:
    """Pull the best chapter title we have per HS chapter. UK rows tagged
    hs_code='{ch}00' carry the chapter description; fall back to the
    first USITC item for that chapter if UK is missing."""
    titles: dict[str, str] = {}
    with conn.cursor() as cur:
        # UK chapter-titles first (preferred — clean human label).
        cur.execute("""
            SELECT hs_chapter, description
            FROM   public_source_descriptions
            WHERE  source = 'uk'
              AND  hs_code LIKE '%00'
              AND  LENGTH(hs_code) = 4
        """)
        for ch, descr in cur.fetchall():
            titles[ch] = descr.strip()

        # Fill gaps with the first USITC root row (indent 0, shortest
        # description for that chapter).
        cur.execute("""
            SELECT DISTINCT ON (hs_chapter) hs_chapter, description
            FROM   public_source_descriptions
            WHERE  source = 'usitc'
            ORDER  BY hs_chapter, LENGTH(description) ASC
        """)
        for ch, descr in cur.fetchall():
            if ch not in titles:
                titles[ch] = descr.strip()

    return titles


def _slugify(name: str) -> str:
    """Derive a URL-safe slug from a natural-language category name.
    Examples:
      'Live Animals & Meat Products' -> 'live_animals_meat_products'
      'Vegetables, Grains & Cereals'  -> 'vegetables_grains_cereals'
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm(api_key: str, prompt: str) -> str:
    """One round-trip to Claude. Returns the assistant's text content."""
    response = httpx.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120.0,
    )
    response.raise_for_status()
    body = response.json()
    # Sonnet returns content as a list of blocks; concatenate text blocks.
    parts = [b.get("text", "") for b in body.get("content", []) if b.get("type") == "text"]
    return "".join(parts).strip()


def _parse_json(text: str) -> dict:
    """Extract the JSON object from an LLM response. Tolerates extra
    surrounding text by finding the first '{' and the matching last '}'."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    return json.loads(text[start:end + 1])


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_grouping(grouping: dict, all_chapters: set[str]) -> None:
    """Sanity check: every chapter present exactly once, no unknown chapters,
    no empty category names."""
    if "categories" not in grouping:
        raise ValueError("Missing 'categories' key in grouping")

    seen: dict[str, str] = {}
    for cat in grouping["categories"]:
        if not cat.get("name"):
            raise ValueError(f"Category missing name: {cat!r}")
        for ch in cat.get("hs_chapters", []):
            if ch not in all_chapters:
                raise ValueError(f"Unknown chapter {ch!r} in category {cat['name']!r}")
            if ch in seen:
                raise ValueError(
                    f"Chapter {ch!r} appears in both {seen[ch]!r} and {cat['name']!r}"
                )
            seen[ch] = cat["name"]

    missing = all_chapters - set(seen.keys())
    if missing:
        raise ValueError(f"Chapters not assigned to any category: {sorted(missing)}")


# ── Output emission ───────────────────────────────────────────────────────────

def _emit_sql(grouping: dict, chapter_titles: dict[str, str]) -> tuple[str, str]:
    """Build the two SQL files: categories + chapter_categories."""
    cat_lines = ["-- Generated by scripts/llm_classify_chapters.py. Do not edit by hand.",
                 "TRUNCATE TABLE chapter_categories CASCADE;",
                 "TRUNCATE TABLE categories CASCADE;",
                 ""]
    map_lines = ["-- Generated by scripts/llm_classify_chapters.py. Do not edit by hand.",
                 ""]

    for cat in grouping["categories"]:
        slug = _slugify(cat["name"])
        display = cat["name"].replace("'", "''")
        cat_lines.append(
            f"INSERT INTO categories (slug, display_name) VALUES "
            f"('{slug}', '{display}') ON CONFLICT (slug) DO UPDATE SET display_name = EXCLUDED.display_name;"
        )
        for ch in cat["hs_chapters"]:
            title = chapter_titles.get(ch, f"Chapter {ch}").replace("'", "''")
            map_lines.append(
                f"INSERT INTO chapter_categories (hs_chapter, category_slug, chapter_title) "
                f"VALUES ('{ch}', '{slug}', '{title}') "
                f"ON CONFLICT (hs_chapter) DO UPDATE SET "
                f"category_slug = EXCLUDED.category_slug, "
                f"chapter_title = EXCLUDED.chapter_title;"
            )

    return "\n".join(cat_lines) + "\n", "\n".join(map_lines) + "\n"


def _apply_to_db(conn, categories_sql: str, chapter_map_sql: str) -> None:
    """Execute the generated SQL atomically against the live DB."""
    with conn.cursor() as cur:
        cur.execute(categories_sql)
        cur.execute(chapter_map_sql)
    conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--url", default=os.environ.get("DATABASE_URL"),
                    help="Postgres URL. Default: $DATABASE_URL.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the prompts that would be sent; don't call LLM or DB.")
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set in env.", file=sys.stderr)
        print("  → export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        print("  → or use --dry-run to see prompts without calling LLM", file=sys.stderr)
        return 2

    if not args.url and not args.dry_run:
        print("ERROR: --url not set and DATABASE_URL not in env.", file=sys.stderr)
        return 2

    if args.dry_run:
        # Load chapter titles from disk-only sources for dry-run preview.
        # Build a synthetic chapter list from a hardcoded HS title map if no DB.
        if args.url:
            conn = psycopg2.connect(args.url)
            chapter_titles = _load_chapter_titles(conn)
            conn.close()
        else:
            # Minimal fallback for true dry-run with no DB.
            chapter_titles = {f"{n:02d}": f"Chapter {n:02d}"
                              for n in range(1, 98) if n != 77}
    else:
        conn = psycopg2.connect(args.url)
        chapter_titles = _load_chapter_titles(conn)

    if not chapter_titles:
        print("ERROR: no chapter titles found in DB. Run load_public_sources.py first.",
              file=sys.stderr)
        return 2

    chapters_list = "\n".join(
        f"{ch}: {chapter_titles[ch]}" for ch in sorted(chapter_titles)
    )
    all_chapters = set(chapter_titles.keys())

    # ── Pass 1 — generate ────────────────────────────────────────────────
    gen_template = _extract_prompt("Pass 1 — Generate")
    gen_prompt = gen_template.replace("{CHAPTERS_LIST}", chapters_list)

    if args.dry_run:
        print("=== Pass 1 prompt ===")
        print(gen_prompt[:2000])
        print(f"... ({len(gen_prompt):,} chars total)")
        print()
        print("--- (dry-run; not calling LLM) ---")
        return 0

    print("Calling Claude (pass 1: generate)...")
    gen_response = _call_llm(api_key, gen_prompt)
    proposed = _parse_json(gen_response)
    print(f"  -> {len(proposed.get('categories', []))} categories proposed")
    _validate_grouping(proposed, all_chapters)

    # ── Pass 2 — validate / refine ───────────────────────────────────────
    val_template = _extract_prompt("Pass 2 — Validate / refine")
    val_prompt = (val_template
                  .replace("{CHAPTERS_LIST}", chapters_list)
                  .replace("{PROPOSED_GROUPING}", json.dumps(proposed, indent=2)))

    print("Calling Claude (pass 2: validate)...")
    val_response = _call_llm(api_key, val_prompt)
    refined = _parse_json(val_response)
    print(f"  -> {len(refined.get('categories', []))} categories after refinement")
    _validate_grouping(refined, all_chapters)

    # ── Emit SQL + apply ──────────────────────────────────────────────────
    categories_sql, chapter_map_sql = _emit_sql(refined, chapter_titles)
    OUT_CATEGORIES_SQL.write_text(categories_sql, encoding="utf-8")
    OUT_CHAPTER_MAP_SQL.write_text(chapter_map_sql, encoding="utf-8")
    print(f"Wrote {OUT_CATEGORIES_SQL}")
    print(f"Wrote {OUT_CHAPTER_MAP_SQL}")

    print("Applying to live DB...")
    _apply_to_db(conn, categories_sql, chapter_map_sql)

    # Summary
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM categories")
        n_cats = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chapter_categories")
        n_map = cur.fetchone()[0]
        cur.execute("""
            SELECT c.display_name, COUNT(cm.hs_chapter)
            FROM   categories c
            LEFT   JOIN chapter_categories cm ON cm.category_slug = c.slug
            GROUP  BY c.display_name
            ORDER  BY 2 DESC, 1
        """)
        rows = cur.fetchall()

    print(f"\n--- Result: {n_cats} categories, {n_map} chapters mapped ---")
    for name, count in rows:
        print(f"  {count:>3} chapters  {name}")

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
