"""
LLM-generate anchor/suppressor/modifier rows for the keywords table.

For each chapter (96 active), make a 2-pass LLM call:
    Pass 1 — generate: propose anchors, suppressors, modifiers.
    Pass 2 — validate: drop hallucinations, keep only confident rows.

Inserts the surviving rows into the live `keywords` table with
source='llm:claude-sonnet-4-6:YYYY-MM-DD'. Also dumps to
seed/seed_cctr_v3.sql for git-committed reproducibility.

REQUIRES ANTHROPIC_API_KEY in env. Refuses to run without it.
REQUIRES `categories` and `chapter_categories` populated — run
scripts/llm_classify_chapters.py first.

Cost: ~96 chapters × 2 calls × ~5K tokens ≈ ~$10-15 per full run.
Use --resume-from-chapter NN to continue after an interruption.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/llm_generate_cctr.py
    python scripts/llm_generate_cctr.py --only-chapter 09     # one chapter for testing
    python scripts/llm_generate_cctr.py --resume-from-chapter 30
    python scripts/llm_generate_cctr.py --dry-run --only-chapter 09  # see prompt without calling
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import httpx
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT / "prompts" / "cctr_generation.md"
OUT_SQL = ROOT / "seed" / "seed_cctr_v3.sql"

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

# Fixed weights per signal class — LLM never sets these.
WEIGHT_ANCHOR = 1.0
WEIGHT_SUPPRESSOR = 0.7
WEIGHT_MODIFIER = 1.0

SOURCE_TAG = f"llm:{MODEL}:{date.today().isoformat()}"


# ── Prompt loading ────────────────────────────────────────────────────────────

def _extract_prompt(name: str) -> str:
    text = PROMPT_FILE.read_text(encoding="utf-8")
    pattern = rf"## {re.escape(name)}.*?```\n(.*?)```"
    m = re.search(pattern, text, flags=re.DOTALL)
    if not m:
        raise RuntimeError(f"Could not find prompt section {name!r} in {PROMPT_FILE}")
    return m.group(1).strip()


# ── DB readers ────────────────────────────────────────────────────────────────

def _load_chapter_context(conn) -> dict[str, dict]:
    """Return per-chapter context for prompting:
    {hs_chapter: {title, category_slug, category_display, sample_descriptions, adjacent_chapters}}.
    """
    out: dict[str, dict] = {}

    # Chapter title + category mapping
    with conn.cursor() as cur:
        cur.execute("""
            SELECT cm.hs_chapter, cm.chapter_title, c.slug, c.display_name
            FROM   chapter_categories cm
            JOIN   categories c ON c.slug = cm.category_slug
            ORDER  BY cm.hs_chapter
        """)
        for ch, title, slug, display in cur.fetchall():
            out[ch] = {
                "title": title,
                "category_slug": slug,
                "category_display": display,
                "sample_descriptions": [],
                "adjacent_chapters": [],
            }

        # Top USITC descriptions per chapter (longest 10 — proxy for richness).
        cur.execute("""
            SELECT hs_chapter, description
            FROM (
                SELECT hs_chapter, description,
                       ROW_NUMBER() OVER (PARTITION BY hs_chapter ORDER BY LENGTH(description) DESC) AS rn
                FROM   public_source_descriptions
                WHERE  source = 'usitc'
            ) ranked
            WHERE  rn <= 10
            ORDER  BY hs_chapter, rn
        """)
        for ch, descr in cur.fetchall():
            if ch in out:
                out[ch]["sample_descriptions"].append(descr)

    # Adjacent chapters in the same category — rough heuristic for "commonly confused"
    by_cat: dict[str, list[tuple[str, str]]] = {}
    for ch, ctx in out.items():
        by_cat.setdefault(ctx["category_slug"], []).append((ch, ctx["title"]))
    for ch, ctx in out.items():
        siblings = [(c, t) for (c, t) in by_cat.get(ctx["category_slug"], []) if c != ch]
        ctx["adjacent_chapters"] = siblings[:8]   # cap at 8 for prompt size

    return out


def _load_all_categories(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT display_name FROM categories ORDER BY display_name")
        return [r[0] for r in cur.fetchall()]


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm(api_key: str, prompt: str) -> str:
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
    parts = [b.get("text", "") for b in body.get("content", []) if b.get("type") == "text"]
    return "".join(parts).strip()


def _parse_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    return json.loads(text[start:end + 1])


# ── Schema validation of LLM output ───────────────────────────────────────────

def _validate_cctr(data: dict, all_category_displays: set[str], chapter: str) -> dict:
    """Schema-validate the LLM's CCTR output. Returns the validated dict
    with all three arrays present (defaulting to []). Raises ValueError on
    structural problems; silently drops individual rows that fail
    type/value checks."""
    out = {"anchors": [], "suppressors": [], "modifiers": []}

    for row in data.get("anchors", []):
        kw = row.get("keyword")
        if isinstance(kw, str) and kw.strip():
            out["anchors"].append({"keyword": kw.strip().lower()})

    for row in data.get("suppressors", []):
        kw = row.get("keyword")
        sup = row.get("suppress_category")
        if (isinstance(kw, str) and kw.strip()
                and isinstance(sup, str) and sup in all_category_displays):
            out["suppressors"].append({
                "keyword": kw.strip().lower(),
                "suppress_category": sup,
            })

    for row in data.get("modifiers", []):
        kw = row.get("keyword")
        head = row.get("head_noun")
        target = row.get("target_chapter")
        if (isinstance(kw, str) and kw.strip()
                and isinstance(head, str) and head.strip()
                and isinstance(target, str) and re.fullmatch(r"\d{2}", target)):
            out["modifiers"].append({
                "keyword": kw.strip().lower(),
                "head_noun": head.strip().lower(),
                "target_chapter": target,
            })

    return out


# ── DB writes ─────────────────────────────────────────────────────────────────

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


def _row_to_tuple(chapter: str, signal_class: str, row: dict,
                  category_slug_by_display: dict[str, str]) -> tuple:
    """Convert a validated LLM row → INSERT tuple matching INSERT_SQL.
    suppress_category is stored as a slug (FK-friendly)."""
    if signal_class == "anchor":
        return (row["keyword"], chapter, "anchor", WEIGHT_ANCHOR,
                None, None, None, SOURCE_TAG)
    if signal_class == "suppressor":
        sup_slug = category_slug_by_display.get(row["suppress_category"])
        return (row["keyword"], chapter, "suppressor", WEIGHT_SUPPRESSOR,
                None, sup_slug, None, SOURCE_TAG)
    if signal_class == "modifier":
        return (row["keyword"], chapter, "modifier", WEIGHT_MODIFIER,
                row["target_chapter"], None, row["head_noun"], SOURCE_TAG)
    raise ValueError(f"unknown signal_class: {signal_class}")


def _persist(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, INSERT_SQL, rows, page_size=200)
    conn.commit()
    return len(rows)


# ── Per-chapter generation ────────────────────────────────────────────────────

def _process_chapter(api_key: str, conn, chapter: str, ctx: dict,
                     all_category_displays: set[str],
                     category_slug_by_display: dict[str, str],
                     gen_template: str, val_template: str,
                     dry_run: bool) -> tuple[int, dict]:
    """Returns (n_rows_inserted, validated_dict) for one chapter."""
    samples = "\n".join(f"  - {s[:160]}" for s in ctx["sample_descriptions"][:10])
    adjacent = ", ".join(f"{c}: {t}" for (c, t) in ctx["adjacent_chapters"]) or "(none in same category)"
    cats = ", ".join(sorted(all_category_displays))

    gen_prompt = (gen_template
                  .replace("{HS_CHAPTER}", chapter)
                  .replace("{CHAPTER_TITLE}", ctx["title"])
                  .replace("{CATEGORY_DISPLAY_NAME}", ctx["category_display"])
                  .replace("{TOP_USITC_DESCRIPTIONS}", samples)
                  .replace("{ADJACENT_CHAPTERS}", adjacent)
                  .replace("{ALL_CATEGORY_DISPLAY_NAMES}", cats))

    if dry_run:
        print(f"=== ch {chapter} pass-1 prompt ({len(gen_prompt)} chars) ===")
        print(gen_prompt[:1500])
        print(f"... ({len(gen_prompt)} total chars)")
        return 0, {"anchors": [], "suppressors": [], "modifiers": []}

    gen_text = _call_llm(api_key, gen_prompt)
    proposed_raw = _parse_json(gen_text)
    proposed = _validate_cctr(proposed_raw, all_category_displays, chapter)

    val_prompt = (val_template
                  .replace("{HS_CHAPTER}", chapter)
                  .replace("{CHAPTER_TITLE}", ctx["title"])
                  .replace("{CATEGORY_DISPLAY_NAME}", ctx["category_display"])
                  .replace("{PROPOSED_ROWS}", json.dumps(proposed, indent=2)))

    val_text = _call_llm(api_key, val_prompt)
    refined_raw = _parse_json(val_text)
    refined = _validate_cctr(refined_raw, all_category_displays, chapter)

    rows: list[tuple] = []
    for r in refined["anchors"]:
        rows.append(_row_to_tuple(chapter, "anchor", r, category_slug_by_display))
    for r in refined["suppressors"]:
        rows.append(_row_to_tuple(chapter, "suppressor", r, category_slug_by_display))
    for r in refined["modifiers"]:
        rows.append(_row_to_tuple(chapter, "modifier", r, category_slug_by_display))

    n = _persist(conn, rows)
    return n, refined


# ── SQL emission ──────────────────────────────────────────────────────────────

def _emit_sql(all_refined: dict[str, dict],
              category_slug_by_display: dict[str, str]) -> str:
    """Build the deterministic seed file from accumulated per-chapter outputs."""
    lines = ["-- Generated by scripts/llm_generate_cctr.py. Do not edit by hand.",
             f"-- Source tag: {SOURCE_TAG}",
             "",
             "DELETE FROM keywords WHERE source LIKE 'llm:%';",
             ""]

    def _esc(s: str) -> str:
        return s.replace("'", "''")

    for chapter in sorted(all_refined):
        refined = all_refined[chapter]
        for r in refined["anchors"]:
            lines.append(
                f"INSERT INTO keywords (keyword, hs_chapter, signal_class, weight, source) "
                f"VALUES ('{_esc(r['keyword'])}', '{chapter}', 'anchor', {WEIGHT_ANCHOR:.2f}, '{SOURCE_TAG}') "
                f"ON CONFLICT (keyword, hs_chapter, signal_class, source) DO UPDATE SET weight = EXCLUDED.weight;"
            )
        for r in refined["suppressors"]:
            sup_slug = category_slug_by_display.get(r["suppress_category"], "")
            lines.append(
                f"INSERT INTO keywords (keyword, hs_chapter, signal_class, weight, suppress_category, source) "
                f"VALUES ('{_esc(r['keyword'])}', '{chapter}', 'suppressor', {WEIGHT_SUPPRESSOR:.2f}, "
                f"'{_esc(sup_slug)}', '{SOURCE_TAG}') "
                f"ON CONFLICT (keyword, hs_chapter, signal_class, source) DO UPDATE SET "
                f"weight = EXCLUDED.weight, suppress_category = EXCLUDED.suppress_category;"
            )
        for r in refined["modifiers"]:
            lines.append(
                f"INSERT INTO keywords (keyword, hs_chapter, signal_class, weight, target_chapter, head_noun, source) "
                f"VALUES ('{_esc(r['keyword'])}', '{chapter}', 'modifier', {WEIGHT_MODIFIER:.2f}, "
                f"'{r['target_chapter']}', '{_esc(r['head_noun'])}', '{SOURCE_TAG}') "
                f"ON CONFLICT (keyword, hs_chapter, signal_class, source) DO UPDATE SET "
                f"weight = EXCLUDED.weight, target_chapter = EXCLUDED.target_chapter, head_noun = EXCLUDED.head_noun;"
            )

    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--url", default=os.environ.get("DATABASE_URL"),
                    help="Postgres URL. Default: $DATABASE_URL.")
    ap.add_argument("--only-chapter", help="Process just this 2-digit chapter.")
    ap.add_argument("--resume-from-chapter",
                    help="Skip chapters before this 2-digit chapter (for resume after interrupt).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the prompts; don't call LLM or DB.")
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set in env.", file=sys.stderr)
        print("  → export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        print("  → or use --dry-run to see prompts without calling LLM", file=sys.stderr)
        return 2

    if not args.url:
        print("ERROR: --url not set and DATABASE_URL not in env.", file=sys.stderr)
        return 2

    conn = psycopg2.connect(args.url)

    print("Loading chapter context...")
    chapter_ctx = _load_chapter_context(conn)
    if not chapter_ctx:
        print("ERROR: chapter_categories or categories table is empty.\n"
              "  → run: python scripts/llm_classify_chapters.py", file=sys.stderr)
        return 2

    all_category_displays_list = _load_all_categories(conn)
    all_category_displays = set(all_category_displays_list)
    category_slug_by_display: dict[str, str] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT slug, display_name FROM categories")
        for slug, display in cur.fetchall():
            category_slug_by_display[display] = slug

    gen_template = _extract_prompt("Pass 1 — Generate")
    val_template = _extract_prompt("Pass 2 — Validate / refine")

    chapters = sorted(chapter_ctx.keys())
    if args.only_chapter:
        chapters = [args.only_chapter] if args.only_chapter in chapter_ctx else []
    elif args.resume_from_chapter:
        chapters = [c for c in chapters if c >= args.resume_from_chapter]

    if not chapters:
        print(f"No chapters to process (filter: only={args.only_chapter}, "
              f"resume={args.resume_from_chapter})", file=sys.stderr)
        return 2

    print(f"Processing {len(chapters)} chapter(s)...")
    all_refined: dict[str, dict] = {}
    total_rows = 0
    for i, ch in enumerate(chapters, 1):
        ctx = chapter_ctx[ch]
        try:
            n, refined = _process_chapter(
                api_key, conn, ch, ctx, all_category_displays,
                category_slug_by_display, gen_template, val_template,
                args.dry_run,
            )
            all_refined[ch] = refined
            total_rows += n
            if not args.dry_run:
                print(f"  [{i:>3}/{len(chapters)}] ch {ch} ({ctx['title'][:40]}): "
                      f"{len(refined['anchors'])} anchors, "
                      f"{len(refined['suppressors'])} suppressors, "
                      f"{len(refined['modifiers'])} modifiers "
                      f"= {n} rows")
        except Exception as e:
            print(f"  [{i:>3}/{len(chapters)}] ch {ch} FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)
            print(f"     → re-run with --resume-from-chapter {ch} to retry from here",
                  file=sys.stderr)
            # Persist what we have so far before exiting.
            if all_refined:
                OUT_SQL.write_text(_emit_sql(all_refined, category_slug_by_display),
                                   encoding="utf-8")
                print(f"     → partial seed written to {OUT_SQL}", file=sys.stderr)
            return 3

    if not args.dry_run:
        OUT_SQL.write_text(_emit_sql(all_refined, category_slug_by_display),
                           encoding="utf-8")
        print(f"\nWrote {OUT_SQL}")
        print(f"Total rows inserted: {total_rows:,}")

        with conn.cursor() as cur:
            cur.execute("""
                SELECT signal_class, COUNT(*) FROM keywords
                WHERE source LIKE 'llm:%' GROUP BY signal_class ORDER BY 1
            """)
            for cls, n in cur.fetchall():
                print(f"  {cls:<11} {n}")

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
