"""
Generate shipment_labels seed rows from public HS descriptions.

Inputs:
    data/hs/hs_6digit.csv              5,612 HS 6-digit codes + descriptions
    data/hs/uk_tariff_descriptions.csv  UK phrasing variants (chapter/heading)
    data/chapter_to_category.csv       96-row chapter → coarse category map
    data/multilabel_overrides.csv      HS prefix → extra categories

Output:
    seed/seed_shipment_labels_v2.sql   ~10,000 rows with hs_chapter + split

Design
------
For each HS 6-digit code we emit N variants (N sized per-category so weak
categories with few chapters still get enough training signal). Each variant
cycles through one of 5 templates that mimic the real distribution of
shipment-record phrasings:

    T0 — clean USITC descriptive (default)
    T1 — quantity-prefixed ("1200 cartons of X")
    T2 — BOL uppercase with inline HS code ("STC 1200 CTNS X HS 850110")
    T3 — abbreviation-noisy (reverse of preprocess.py's abbreviation map)
    T4 — UK phrasing (chapter heading) when available

Commodity field is varied too: parent 4-digit descr, sibling 6-digit descr,
or plain chapter title. The combination exercises the preprocessing gate
and the semantic-vs-keyword blend in the classifier.

Split assignment is deterministic: hash(shipment_id) % 100 mapped to
train (<72), validation (<86), else test — yielding a ~72/14/14 split.

Multi-label: if a 4-digit HS prefix appears in multilabel_overrides.csv
we emit one row per category (same shipment_id, different category_name)
so the generated set includes realistic multi-label shipments.

Seeded RNG (42) → byte-deterministic SQL output.
"""

from __future__ import annotations

import csv
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent
HS_CSV          = ROOT / "data" / "hs" / "hs_6digit.csv"
UK_CSV          = ROOT / "data" / "hs" / "uk_tariff_descriptions.csv"
MAP_CSV         = ROOT / "data" / "chapter_to_category.csv"
OVERRIDES_CSV   = ROOT / "data" / "multilabel_overrides.csv"
OUT_SQL         = ROOT / "seed" / "seed_shipment_labels_v2.sql"

# Target train rows per coarse category. Weak categories (low HS coverage or
# historically worst recall) get a higher target so the per-chapter centroids
# see enough signal.
TARGET_ROWS = defaultdict(lambda: 400, {
    "electronics":     600,
    "pharmaceuticals": 600,
    "machinery":       600,
    "construction":    500,
    "toys":            500,
})

SPLIT_TRAIN_PCT = 72
SPLIT_VAL_PCT   = 14   # → val covers [72, 86)
# → test covers [86, 100)

# Abbreviation-noisy template: reverse of preprocess.py abbreviations.
# (Only the reversible "word → abbrev" direction is useful for noise.)
NOISE_SUBS = [
    ("refrigerated", "refrig"),
    ("frozen",       "frz"),
    ("electronic",   "elec"),
    ("electronics",  "elec"),
    ("pharmaceutical", "pharma"),
    ("chocolate",    "choco"),
    ("manufactured", "mfg"),
    ("manufacturer", "mfr"),
    ("assembly",     "assy"),
    ("component",    "comp"),
    ("equipment",    "equip"),
    ("machinery",    "mach"),
    ("material",     "mat"),
    ("chemical",     "chem"),
    ("chemicals",    "chems"),
    ("industrial",   "ind"),
    ("automotive",   "auto"),
    ("vehicle",      "veh"),
    ("agriculture",  "agri"),
    ("stainless steel", "ss"),
    ("mild steel",   "ms"),
]

BOL_UNITS  = ["CTNS", "CARTONS", "PKGS", "BOXES", "BAGS", "PALLETS", "DRUMS"]
QTY_UNITS  = ["cartons", "pallets", "boxes", "bags", "drums", "crates"]


# ── Load reference data ──────────────────────────────────────────────────────

def load_chapter_map() -> dict[str, tuple[str, str]]:
    """hs_chapter → (category_name, chapter_title)"""
    out: dict[str, tuple[str, str]] = {}
    with MAP_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["hs_chapter"]] = (row["category"], row["chapter_title"])
    return out


def load_hs_codes() -> list[dict]:
    """Load hs_6digit.csv as list of {hs_code, chapter, description, heading, parent4}"""
    rows = []
    with HS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "hs_code":     row["hs_code"],
                "chapter":     row["chapter"],
                "description": row["description"],
                "heading":     row["hs_code"][:4],
            })
    # Group by 4-digit heading for sibling lookup / commodity field
    by_heading: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_heading[r["heading"]].append(r)
    for r in rows:
        r["siblings"] = [s for s in by_heading[r["heading"]] if s["hs_code"] != r["hs_code"]]
    return rows


def load_uk_headings() -> dict[str, list[str]]:
    """chapter → list of heading descriptions (title-cased)"""
    out: dict[str, list[str]] = defaultdict(list)
    if not UK_CSV.exists():
        return out
    with UK_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["source"] == "heading":
                # Normalize the UK uppercase → sentence-case
                desc = row["description"].strip()
                if desc:
                    out[row["chapter"]].append(desc.lower())
    return out


def load_overrides() -> dict[str, list[str]]:
    """4/6-digit hs prefix → extra categories list"""
    out: dict[str, list[str]] = {}
    if not OVERRIDES_CSV.exists():
        return out
    with OVERRIDES_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prefix = row["hs_prefix"].strip()
            cats = [c.strip() for c in row["categories"].split(",") if c.strip()]
            out[prefix] = cats
    return out


# ── Templates ────────────────────────────────────────────────────────────────

def _clean_descr(descr: str) -> str:
    """Title-case single-clause summary suitable for cargo_text."""
    parts = [p.strip() for p in descr.split(",") if p.strip()]
    # Use the most specific (last) leaf + one ancestor for context.
    if len(parts) >= 2:
        return f"{parts[-1]} ({parts[-2].lower()})".strip()
    return parts[0] if parts else descr


def _quantity_prefix(base: str, rng: random.Random) -> str:
    qty = rng.choice([50, 100, 120, 200, 240, 300, 500, 800, 1000, 1200, 2400])
    unit = rng.choice(QTY_UNITS)
    return f"{qty} {unit} of {base.lower()}"


def _bol_uppercase(base: str, hs6: str, rng: random.Random) -> str:
    qty = rng.choice([100, 200, 480, 1000, 1200, 2400, 4800])
    unit = rng.choice(BOL_UNITS)
    return f"STC {qty} {unit} {base.upper()} HS {hs6}"


def _abbrev_noisy(base: str, rng: random.Random) -> str:
    s = base.lower()
    # Apply 1-3 random subs if any match
    picks = rng.sample(NOISE_SUBS, k=min(3, len(NOISE_SUBS)))
    for full, abbr in picks:
        if full in s:
            s = s.replace(full, abbr, 1)
    return s


def _uk_phrased(base: str, chapter: str, uk_headings: dict[str, list[str]],
                rng: random.Random) -> str | None:
    headings = uk_headings.get(chapter, [])
    if not headings:
        return None
    # Pick a heading as an alternate phrasing, optionally combined with base.
    alt = rng.choice(headings)
    if rng.random() < 0.5:
        return alt
    return f"{base.lower()} — {alt}"


def render_variant(
    code: dict,
    variant_idx: int,
    uk_headings: dict[str, list[str]],
    chapter_titles: dict[str, str],
    rng: random.Random,
) -> tuple[str, str]:
    """Returns (cargo_text, commodity_text)."""
    base = _clean_descr(code["description"])
    hs6 = code["hs_code"]
    chapter = code["chapter"]

    templates = [0, 1, 2, 3, 4]
    # Deterministic template choice: seeded by hs6+variant_idx
    h = int(hashlib.md5(f"{hs6}:{variant_idx}".encode()).hexdigest(), 16)
    t = templates[h % len(templates)]

    if t == 0:
        cargo = base
    elif t == 1:
        cargo = _quantity_prefix(base, rng)
    elif t == 2:
        cargo = _bol_uppercase(base, hs6, rng)
    elif t == 3:
        cargo = _abbrev_noisy(base, rng)
    else:  # t == 4 — UK phrasing if available, else fall back to clean
        uk = _uk_phrased(base, chapter, uk_headings, rng)
        cargo = uk if uk else base

    # Commodity field: mix parent-4-digit description + 1-2 siblings + chapter
    siblings = code.get("siblings", [])
    parts = [code["description"].split(",")[0].strip()]          # heading context
    if siblings and rng.random() < 0.7:
        sibs = rng.sample(siblings, k=min(2, len(siblings)))
        parts.extend(_clean_descr(s["description"]).lower() for s in sibs)
    if rng.random() < 0.35:
        parts.append(chapter_titles.get(chapter, "").lower())
    commodity = "; ".join(p for p in parts if p)[:240]
    return cargo, commodity


# ── Split assignment ─────────────────────────────────────────────────────────

def split_of(shipment_id: str) -> str:
    h = int(hashlib.md5(shipment_id.encode()).hexdigest(), 16) % 100
    if h < SPLIT_TRAIN_PCT:
        return "train"
    if h < SPLIT_TRAIN_PCT + SPLIT_VAL_PCT:
        return "validation"
    return "test"


# ── Main generation loop ─────────────────────────────────────────────────────

def compute_variants_per_code(
    hs_rows: list[dict],
    chapter_map: dict[str, tuple[str, str]],
) -> dict[str, int]:
    """Compute variant count honoring both chapter and category floors.

    Chapter floor: ≥ 20 train rows per chapter. Train share is ~72%, so we
    aim for ≥ 30 total rows per chapter — variants = ceil(30 / n_codes_in_chap).

    Category floor: ≥ TARGET_ROWS[cat] train rows per coarse category.
    variants = ceil(target / (n_codes_in_cat * train_pct)).

    Actual variants = max(chapter_floor, category_floor).
    """
    import math

    codes_by_cat: dict[str, int]       = defaultdict(int)
    codes_by_chapter: dict[str, int]   = defaultdict(int)
    for r in hs_rows:
        if r["chapter"] in chapter_map:
            codes_by_cat[chapter_map[r["chapter"]][0]] += 1
            codes_by_chapter[r["chapter"]] += 1

    train_pct = SPLIT_TRAIN_PCT / 100.0
    CHAPTER_TRAIN_FLOOR = 28          # headroom for hash-based split variance

    per_code: dict[str, int] = {}
    for r in hs_rows:
        cat_info = chapter_map.get(r["chapter"])
        if not cat_info:
            continue
        cat = cat_info[0]
        chap = r["chapter"]

        chapter_floor = max(1, math.ceil(CHAPTER_TRAIN_FLOOR / max(1, codes_by_chapter[chap] * train_pct)))
        category_floor = max(1, math.ceil(TARGET_ROWS[cat] / max(1, codes_by_cat[cat] * train_pct)))
        per_code[r["hs_code"]] = max(chapter_floor, category_floor)
    return per_code


def iter_rows(
    hs_rows: list[dict],
    chapter_map: dict[str, tuple[str, str]],
    overrides: dict[str, list[str]],
    uk_headings: dict[str, list[str]],
) -> Iterator[tuple[str, str, str, str, str, str]]:
    """Yield (shipment_id, category_name, cargo, commodity, hs_chapter, split)."""
    chapter_titles = {ch: info[1] for ch, info in chapter_map.items()}
    variants_per_code = compute_variants_per_code(hs_rows, chapter_map)
    rng_master = random.Random(42)

    ship_counter = 0
    for code in hs_rows:
        chapter = code["chapter"]
        primary = chapter_map.get(chapter)
        if not primary:
            continue
        primary_cat = primary[0]

        # Multi-label resolution: check both 4-digit and 6-digit prefixes.
        extra_cats: list[str] = []
        for prefix in (code["hs_code"], code["hs_code"][:4]):
            if prefix in overrides:
                for c in overrides[prefix]:
                    if c != primary_cat and c not in extra_cats:
                        extra_cats.append(c)

        variants = variants_per_code[code["hs_code"]]
        for v in range(variants):
            ship_counter += 1
            sid = f"v2_{ship_counter:06d}"
            # Per-variant RNG stream — deterministic per (hs_code, v)
            rng = random.Random(int(hashlib.md5(f"{code['hs_code']}:{v}:{rng_master.random()}".encode()).hexdigest(), 16) & 0xFFFFFFFF)
            # Actually, seed strictly from (hs_code, v) so output is byte-stable:
            rng = random.Random(f"{code['hs_code']}:{v}")
            cargo, commodity = render_variant(code, v, uk_headings, chapter_titles, rng)
            split = split_of(sid)
            yield (sid, primary_cat, cargo, commodity, chapter, split)
            for extra in extra_cats:
                yield (sid, extra, cargo, commodity, chapter, split)


# ── SQL emission ─────────────────────────────────────────────────────────────

def _sql_esc(s: str) -> str:
    return s.replace("'", "''")


def write_sql(rows: list[tuple[str, str, str, str, str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    BATCH = 200

    header = (
        "-- Generated by scripts/generate_labels.py. Do not edit by hand.\n"
        "-- Regenerate: python scripts/generate_labels.py\n"
        f"-- Rows: {len(rows):,}\n"
        "-- Each row carries hs_chapter and split inline.\n"
        "-- shipment_id prefix 'v2_' distinguishes generated from legacy rows.\n"
        "-- Multi-label rows share shipment_id across categories.\n"
        "\n"
    )

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            f.write(
                "INSERT INTO shipment_labels "
                "(shipment_id, category_name, cargo_text, commodity_text, hs_chapter, split) VALUES\n"
            )
            lines = []
            for sid, cat, cargo, com, chap, split in batch:
                lines.append(
                    f"    ('{sid}', '{cat}', '{_sql_esc(cargo)}', "
                    f"'{_sql_esc(com)}', '{chap}', '{split}')"
                )
            f.write(",\n".join(lines))
            f.write("\nON CONFLICT (shipment_id, category_name) DO NOTHING;\n\n")


# ── Entry ────────────────────────────────────────────────────────────────────

def main() -> int:
    print("Loading reference data...")
    chapter_map = load_chapter_map()
    hs_rows     = load_hs_codes()
    uk_headings = load_uk_headings()
    overrides   = load_overrides()
    print(f"  {len(hs_rows):,} HS codes, {len(chapter_map)} chapters, "
          f"{sum(len(v) for v in uk_headings.values())} UK heading phrasings, "
          f"{len(overrides)} multi-label overrides")

    print("Generating rows...")
    rows = list(iter_rows(hs_rows, chapter_map, overrides, uk_headings))
    print(f"  emitted {len(rows):,} rows")

    # Summary counts
    by_split: dict[str, int] = defaultdict(int)
    by_cat: dict[str, int]   = defaultdict(int)
    by_chap: dict[str, int]  = defaultdict(int)
    for sid, cat, cargo, com, chap, split in rows:
        by_split[split] += 1
        if split == "train":
            by_cat[cat] += 1
            by_chap[chap] += 1

    print(f"\nSplit counts: "
          f"train={by_split['train']:,}  "
          f"validation={by_split['validation']:,}  "
          f"test={by_split['test']:,}")

    print("\nTrain rows per category (target floor 300; weak-5 floor 500):")
    for cat in sorted(by_cat):
        print(f"  {cat:<18} {by_cat[cat]:>5}")

    below = [ch for ch, n in by_chap.items() if n < 20]
    print(f"\nChapters with <20 train rows: {len(below)} "
          f"(target: 0)")
    if below:
        print(f"  underfilled: {sorted(below)}")

    print(f"\nWriting SQL → {OUT_SQL}")
    write_sql(rows, OUT_SQL)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
