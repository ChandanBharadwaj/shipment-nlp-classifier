"""
Load USITC + UK Trade Tariff cached data into public_source_descriptions.

Walks the cached files written by scripts/fetch_public_data.py and inserts one
row per (source, hs_code) into the public_source_descriptions table.

Unlike fetch_public_data._clean_usitc_to_6digit (which dedupes to one row per
6-digit code), this script emits ALL depths (4, 6, 8, 10) so per-chapter TF-IDF
downstream sees the full descriptive vocabulary at every level. The
parent-context rollup logic is the same — child rows like 'Males' are useless
without the parent path 'Live horses, asses, mules and hinnies, Horses,
Purebred breeding animals, Males'.

Idempotent: re-running with the same fetched data is a no-op (or refreshes the
description if upstream changed).

Usage:
    python scripts/load_public_sources.py
    python scripts/load_public_sources.py --url postgresql://...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
USITC_JSON = ROOT / "data" / "hs" / "usitc_hts_2024.json"
UK_RAW_DIR = ROOT / "data" / "hs" / "uk_chapters_raw"

# Active 2-digit HS chapters: 01-97 minus 77 (WCO-reserved). Same set as
# fetch_public_data.py's TARGET_CHAPTERS — kept in sync intentionally.
TARGET_CHAPTERS = {f"{n:02d}" for n in range(1, 98) if n != 77}

# Match the leading "{4digit}.{2digit}" of an htsno (e.g. "0101.21.00.10")
# so we can extract the 6-digit prefix for chapter detection.
HTSNO_HEAD = re.compile(r"^(\d{4})\.(\d{2})")


# ── USITC walk ────────────────────────────────────────────────────────────────

def _walk_usitc(rows: list[dict]) -> list[tuple[str, str, str, str]]:
    """Walk the USITC `indent` tree, emit one record per row that has an
    htsno (4/6/8/10-digit) with parent context rolled into the description.

    Returns a list of (source, hs_code, hs_chapter, description) tuples ready
    for INSERT. Rows without htsno (parent grouping rows like "Horses:") are
    NOT emitted on their own — their text appears in children via the rollup.
    """
    stack: dict[int, str] = {}             # indent level -> description
    out: list[tuple[str, str, str, str]] = []

    for row in rows:
        # Indent may arrive as int, str, or missing. Default 0.
        raw = row.get("indent")
        try:
            indent = int(raw) if raw not in (None, "") else 0
        except (TypeError, ValueError):
            indent = 0

        descr = (row.get("description") or "").strip().rstrip(":").strip()
        if not descr:
            continue

        # Pop deeper stack levels — we're at this depth now.
        for k in list(stack.keys()):
            if k >= indent:
                stack.pop(k)
        stack[indent] = descr

        htsno = (row.get("htsno") or "").strip()
        if not htsno:
            continue

        # Chapter from htsno's first 2 digits. The HTSNO_HEAD pattern catches
        # the "NNNN.NN" prefix and we take its first two characters.
        m = HTSNO_HEAD.match(htsno)
        if not m:
            # Some short htsno entries (just "0101") may not have the dot;
            # take first 2 digits if they exist.
            if len(htsno) >= 2 and htsno[:2].isdigit():
                chapter = htsno[:2]
            else:
                continue
        else:
            chapter = m.group(1)[:2]

        if chapter not in TARGET_CHAPTERS:
            continue

        # Build full descriptive path from indent 0 upward.
        full = ", ".join(stack[k] for k in sorted(stack.keys()) if k in stack)
        out.append(("usitc", htsno, chapter, full))

    return out


# ── UK walk ───────────────────────────────────────────────────────────────────

def _walk_uk(uk_dir: Path) -> list[tuple[str, str, str, str]]:
    """Walk per-chapter cached UK Trade Tariff JSON, emit one record per
    heading (4-digit HS code level). The `chapter` description is also
    captured, indexed under hs_code = chapter+"00" for sorting consistency.

    Returns the same shape as _walk_usitc.
    """
    out: list[tuple[str, str, str, str]] = []

    for chapter in sorted(TARGET_CHAPTERS):
        path = uk_dir / f"{chapter}.json"
        if not path.exists():
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        # Chapter title — index under {chapter}00 so it sorts first.
        chapter_descr = (
            data.get("data", {}).get("attributes", {}).get("description") or ""
        ).strip()
        if chapter_descr:
            out.append(("uk", f"{chapter}00", chapter, chapter_descr))

        # Headings (4-digit children).
        for inc in data.get("included", []):
            if inc.get("type") != "heading":
                continue
            attrs = inc.get("attributes", {}) or {}
            descr = (attrs.get("description") or "").strip()
            if not descr:
                continue

            # UK API returns short_code like "0101" or sometimes the full
            # goods_nomenclature_item_id ("0101000000"). Normalize to
            # 4-digit by taking first 4 chars of whichever is populated.
            code_raw = (
                attrs.get("short_code")
                or attrs.get("goods_nomenclature_item_id")
                or ""
            ).strip()
            if not code_raw or len(code_raw) < 4:
                continue
            heading_code = code_raw[:4]
            if heading_code[:2] != chapter:
                continue
            out.append(("uk", heading_code, chapter, descr))

    return out


# ── DB load ───────────────────────────────────────────────────────────────────

INSERT_SQL = """
    INSERT INTO public_source_descriptions
        (source, hs_code, hs_chapter, description, fetched_at)
    VALUES %s
    ON CONFLICT (source, hs_code) DO UPDATE
        SET description = EXCLUDED.description,
            fetched_at  = EXCLUDED.fetched_at
"""


def _load(conn: psycopg2.extensions.connection,
          rows: list[tuple[str, str, str, str]]) -> int:
    """Bulk-insert rows. Returns count inserted/updated.

    Dedupes by (source, hs_code) within the input list — UK chapter pages
    can emit the same heading code more than once (chapter title under
    "{chapter}00" vs. a heading at the same code, or duplicate `included`
    entries). ON CONFLICT DO UPDATE can't update the same row twice in
    one INSERT, so we collapse to the LAST description seen per key.
    """
    if not rows:
        return 0
    deduped: dict[tuple[str, str], tuple[str, str, str, str]] = {}
    for s, c, ch, d in rows:
        deduped[(s, c)] = (s, c, ch, d)
    rows = list(deduped.values())

    now = time.strftime("%Y-%m-%d %H:%M:%S+00", time.gmtime())
    values = [(s, c, ch, d, now) for (s, c, ch, d) in rows]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, INSERT_SQL, values, page_size=500)
    conn.commit()
    return len(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--url", default=os.environ.get("DATABASE_URL"),
                    help="Postgres URL. Default: $DATABASE_URL.")
    args = ap.parse_args()

    if not args.url:
        print("ERROR: --url not set and DATABASE_URL not in env.", file=sys.stderr)
        return 2

    if not USITC_JSON.exists():
        print(f"ERROR: missing {USITC_JSON}\n"
              f"  -> run: python scripts/fetch_public_data.py", file=sys.stderr)
        return 2
    if not UK_RAW_DIR.exists():
        print(f"ERROR: missing {UK_RAW_DIR}\n"
              f"  -> run: python scripts/fetch_public_data.py", file=sys.stderr)
        return 2

    print(f"Reading USITC: {USITC_JSON}")
    usitc_rows = json.loads(USITC_JSON.read_text(encoding="utf-8"))
    usitc_records = _walk_usitc(usitc_rows)
    print(f"  -> {len(usitc_records):,} USITC rows to load (all depths)")

    print(f"Reading UK chapters: {UK_RAW_DIR}")
    uk_records = _walk_uk(UK_RAW_DIR)
    print(f"  -> {len(uk_records):,} UK rows to load (chapter + heading)")

    print(f"Connecting: {args.url}")
    conn = psycopg2.connect(args.url)

    print("Loading USITC...")
    n_usitc = _load(conn, usitc_records)
    print("Loading UK...")
    n_uk = _load(conn, uk_records)

    # Summary against DB to confirm idempotent behaviour.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT source, COUNT(*), COUNT(DISTINCT hs_chapter)
            FROM   public_source_descriptions
            GROUP  BY source
            ORDER  BY source
        """)
        rows = cur.fetchall()

    print("\n--- public_source_descriptions ---")
    print(f"{'source':<10} {'rows':>10} {'chapters':>10}")
    for source, count, chapters in rows:
        print(f"{source:<10} {count:>10,} {chapters:>10}")

    conn.close()
    print(f"\nDone. USITC inserted/updated: {n_usitc:,}, UK: {n_uk:,}.")
    return 0


if __name__ == "__main__":
    for env_path in [ROOT / "ml-service" / ".env", ROOT / ".env"]:
        if env_path.exists():
            load_dotenv(env_path)
            break
    else:
        load_dotenv()
    sys.exit(main())
