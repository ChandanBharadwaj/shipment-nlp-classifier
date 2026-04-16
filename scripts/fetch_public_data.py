"""
Fetch public HS reference data once and commit cleaned artifacts under data/.

Idempotent: each output is skipped if already present and non-empty, unless
--force is passed. Resumable: the UK tariff chapter loop writes per-chapter
files and only re-pulls missing ones.

Outputs
-------
    data/hs/usitc_hts_2024.json       raw USITC HTS dump (single JSON array)
    data/hs/hs_6digit.csv             cleaned: hs_code, chapter, description
    data/hs/uk_tariff_descriptions.csv cleaned: chapter, source, description
    data/hs/uk_chapters_raw/NN.json   per-chapter UK API responses (cache)

WHO INN and UNSPSC augmentations are optional and are NOT fetched here — the
keyword generator falls back to HS-derived vocabulary when they are absent.
If curated lists exist at data/augmentation/who_inn.txt or unspsc_tech.csv,
they will be picked up at keyword-generation time.

Usage:
    python scripts/fetch_public_data.py
    python scripts/fetch_public_data.py --force
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_HS = ROOT / "data" / "hs"
UK_RAW = DATA_HS / "uk_chapters_raw"

USITC_URL = (
    "https://hts.usitc.gov/reststop/exportList"
    "?from=0100&to=9999&format=JSON&styles=false"
)
UK_CHAPTER_URL = "https://www.trade-tariff.service.gov.uk/api/v2/chapters/{nn:02d}"

# HS 2-digit chapters we care about (01-97, excluding 77 which is WCO-reserved).
# USITC returns 98/99 for US-only special chapters; drop them.
TARGET_CHAPTERS = [f"{n:02d}" for n in range(1, 98) if n != 77]

HTSNO_6DIGIT = re.compile(r"^(\d{4})\.(\d{2})")


# ── USITC ─────────────────────────────────────────────────────────────────────

def _fetch_usitc(path: Path, force: bool) -> list[dict]:
    if path.exists() and path.stat().st_size > 0 and not force:
        print(f"  [usitc] cache hit → {path} ({path.stat().st_size:,} bytes)")
        return json.loads(path.read_text(encoding="utf-8"))

    print(f"  [usitc] GET {USITC_URL}")
    t0 = time.time()
    r = requests.get(USITC_URL, timeout=120)
    r.raise_for_status()
    data = json.loads(r.text)
    print(f"  [usitc] {len(data):,} rows, {len(r.content):,} bytes, {time.time()-t0:.1f}s")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def _clean_usitc_to_6digit(rows: list[dict], out_path: Path) -> int:
    """Walk the HTS tree, collapse parent context into each 6-digit row.

    The USITC JSON is a flat list with an `indent` column encoding tree depth.
    A child's true description is "<parent>, <parent>, ... <child>" — without
    that roll-up, rows like "Purebred breeding animals" are useless.

    We emit one row per distinct 6-digit code, using the FIRST occurrence's
    full descriptive path (later subheadings under the same 6-digit are
    10-digit tariff lines that share the same HS6 definition).
    """
    stack: dict[int, str] = {}           # indent -> description at that level
    emitted: dict[str, tuple[str, str]]  = {}   # hs6 -> (chapter, description)

    for row in rows:
        indent_raw = row.get("indent")
        try:
            indent = int(indent_raw) if indent_raw not in (None, "") else 0
        except (TypeError, ValueError):
            indent = 0
        descr = (row.get("description") or "").strip().rstrip(":").strip()
        if not descr:
            continue

        # Reset deeper stack levels
        for k in list(stack.keys()):
            if k >= indent:
                stack.pop(k)
        stack[indent] = descr

        htsno = (row.get("htsno") or "").strip()
        m = HTSNO_6DIGIT.match(htsno)
        if not m:
            continue
        chapter = m.group(1)[:2]
        hs6 = m.group(1) + m.group(2)          # e.g. "010121"
        if hs6 in emitted:
            continue
        if chapter not in TARGET_CHAPTERS:
            continue

        # Build descriptive path from indent 0 upward, keeping current stack.
        path_parts = [stack[k] for k in sorted(stack.keys()) if k in stack]
        full_descr = ", ".join(p for p in path_parts if p)
        emitted[hs6] = (chapter, full_descr)

    # Write CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hs_code", "chapter", "description"])
        for hs6 in sorted(emitted.keys()):
            chapter, descr = emitted[hs6]
            w.writerow([hs6, chapter, descr])
    return len(emitted)


# ── UK Trade Tariff ──────────────────────────────────────────────────────────

def _fetch_uk_chapter(nn: int, cache_path: Path, force: bool) -> dict | None:
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force:
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    url = UK_CHAPTER_URL.format(nn=nn)
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [uk] chapter {nn:02d}: {exc}")
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(r.text, encoding="utf-8")
    # Polite pacing — UK API is unauthenticated and public.
    time.sleep(0.25)
    return r.json()


def _clean_uk_to_csv(out_path: Path, force: bool) -> int:
    """Pull 96 UK chapter pages; emit chapter title + heading titles as
    alternate phrasings.
    """
    UK_RAW.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, str]] = []   # (chapter, source, description)
    for chapter in TARGET_CHAPTERS:
        nn = int(chapter)
        cache = UK_RAW / f"{chapter}.json"
        data = _fetch_uk_chapter(nn, cache, force)
        if not data:
            continue
        chapter_descr = (
            data.get("data", {}).get("attributes", {}).get("description") or ""
        ).strip()
        if chapter_descr:
            rows.append((chapter, "chapter", chapter_descr))
        for inc in data.get("included", []):
            if inc.get("type") != "heading":
                continue
            descr = (inc.get("attributes", {}).get("description") or "").strip()
            if descr:
                rows.append((chapter, "heading", descr))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chapter", "source", "description"])
        for r in rows:
            w.writerow(r)
    return len(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore caches")
    args = ap.parse_args()

    print("Fetching USITC HTS (primary)...")
    raw = _fetch_usitc(DATA_HS / "usitc_hts_2024.json", args.force)

    print("Cleaning to 6-digit CSV...")
    n6 = _clean_usitc_to_6digit(raw, DATA_HS / "hs_6digit.csv")
    print(f"  wrote {n6:,} distinct 6-digit codes")

    print("Fetching UK Trade Tariff chapters (cross-check phrasing)...")
    nuk = _clean_uk_to_csv(DATA_HS / "uk_tariff_descriptions.csv", args.force)
    print(f"  wrote {nuk:,} UK chapter/heading descriptions")

    print("\nDone.")
    print(f"  {DATA_HS/'usitc_hts_2024.json'}")
    print(f"  {DATA_HS/'hs_6digit.csv'}")
    print(f"  {DATA_HS/'uk_tariff_descriptions.csv'}")
    print("\nOptional augmentations (not fetched here):")
    print("  data/augmentation/who_inn.txt   — if present, pharma vocab boost")
    print("  data/augmentation/unspsc_tech.csv — if present, consumer-product vocab")
    return 0


if __name__ == "__main__":
    sys.exit(main())
