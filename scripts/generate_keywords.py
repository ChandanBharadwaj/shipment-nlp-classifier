"""
Generate per-category keywords from HS subheading descriptions via TF-IDF.

Input:
    data/hs/hs_6digit.csv         HS descriptions per 6-digit code
    data/chapter_to_category.csv  Chapter → category map
    seed/seed_keywords.sql        Hand-curated keywords (preserved; hand weights win)

Output:
    seed/seed_keywords_v2.sql     ~1,200 keywords (≥ 40 / category, median ~60)

Method
------
1. Group HS descriptions by coarse category (via chapter mapping).
2. Build a TF-IDF vectorizer across the 20 category "documents" (each doc =
   concatenated descriptions of that category's HS codes).
3. Rank distinctive tokens per category by tf-idf score.
4. Keep the top N per category, assigning weight:
      1.5 if ≤ 2 categories also contain the term (highly distinctive)
      1.0 if 3-5 categories contain it (standard)
      0.8 if 6+ categories contain it (common but relevant)
5. Merge with existing hand-curated keywords from seed_keywords.sql —
   hand weights win on conflict.

Phrase handling: ngram_range=(1, 2) captures "iron ore", "fresh tomatoes",
"mineral wool", etc., which single tokens miss. Stopwords use sklearn's
English list plus a custom list of HS boilerplate ("thereof", "other",
"nesoi", "whether", "used", "similar", etc.).
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

ROOT           = Path(__file__).resolve().parent.parent
HS_CSV         = ROOT / "data" / "hs" / "hs_6digit.csv"
MAP_CSV        = ROOT / "data" / "chapter_to_category.csv"
EXISTING_SQL   = ROOT / "seed" / "seed_keywords.sql"
OUT_SQL        = ROOT / "seed" / "seed_keywords_v2.sql"

TOP_N_PER_CAT  = 80       # then filter by min-length and presence in other cats

# HS boilerplate / non-discriminative tokens beyond sklearn's english stopwords.
EXTRA_STOPWORDS = {
    "thereof", "other", "nesoi", "otherwise", "specified", "included",
    "whether", "containing", "preparation", "preparations", "similar",
    "used", "use", "parts", "accessories", "based", "within", "weight",
    "non", "nes", "per", "kind", "kinds", "including", "composed",
    "having", "made", "form", "forms", "types", "certain", "mixed",
    "excluding", "entirely", "wholly", "following", "articles", "article",
    "contains", "containing", "material", "materials", "content", "not",
    "exceeding", "less", "more", "except", "either", "without", "with",
    "kg", "cm", "mm", "new", "used", "raw", "mixed", "extracted", "obtained",
    "different", "similar", "products", "product", "various", "all",
    "also", "under", "over", "above", "below", "being", "use", "uses",
    # Bigram-tail noise observed in HS text
    "heading", "example", "percent", "weighing", "mainly",
    "consisting", "retail", "textile", "namely",
    # Generic descriptors that hit too many categories
    "boys", "girls", "men", "women", "children",
}

# Single-character or all-digit tokens get filtered.
TOKEN_OK = re.compile(r"^[a-z][a-z\-]{1,}(?: [a-z][a-z\-]{1,})?$")


# ── Hand-curated keyword parse ───────────────────────────────────────────────

HAND_RE = re.compile(
    r"\(\s*'([^']+)'\s*,\s*([\d.]+)\s*\).*?WHERE name = '(\w+)'",
    re.DOTALL,
)


def parse_existing_keywords() -> dict[str, dict[str, float]]:
    """Return {category: {keyword_lower: weight}} from seed_keywords.sql."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    if not EXISTING_SQL.exists():
        return out

    txt = EXISTING_SQL.read_text(encoding="utf-8")
    # Split by INSERT blocks to keep category scoping clean
    blocks = re.split(r"-- ── ([a-z_]+)", txt)
    # blocks[0] = preamble, then alternating category-name and body
    for i in range(1, len(blocks) - 1, 2):
        cat = blocks[i].strip()
        body = blocks[i + 1]
        for m in re.finditer(r"\(\s*'([^']+)'\s*,\s*([\d.]+)\s*\)", body):
            kw, wt = m.group(1).strip().lower(), float(m.group(2))
            if kw and kw not in EXTRA_STOPWORDS:
                out[cat][kw] = wt
    return out


# ── HS doc corpus ────────────────────────────────────────────────────────────

def load_category_corpus() -> dict[str, str]:
    """Return {category_name: concatenated HS descriptions}."""
    chap_to_cat: dict[str, str] = {}
    with MAP_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            chap_to_cat[r["hs_chapter"]] = r["category"]

    docs: dict[str, list[str]] = defaultdict(list)
    with HS_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cat = chap_to_cat.get(r["chapter"])
            if cat:
                docs[cat].append(r["description"])
    return {cat: " ".join(texts).lower() for cat, texts in docs.items()}


# ── TF-IDF ───────────────────────────────────────────────────────────────────

def extract_keywords(corpus: dict[str, str]) -> dict[str, list[tuple[str, float, int]]]:
    """
    Returns {category: [(term, tfidf_score, df_count), ...]} sorted by score desc.
    df_count is the number of categories the term appears in (used for weighting).
    """
    cats = sorted(corpus.keys())
    docs = [corpus[c] for c in cats]

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stopwords = list(ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS))

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
        stop_words=stopwords,
        token_pattern=r"(?u)\b[a-z][a-z\-]{1,}\b",
        sublinear_tf=True,
    )
    X = vec.fit_transform(docs)
    terms = vec.get_feature_names_out()

    # df across categories: for each term, count cats where tf > 0
    df_count = {}
    import numpy as np
    X_bin = (X > 0).toarray()
    for j, term in enumerate(terms):
        df_count[term] = int(X_bin[:, j].sum())

    out: dict[str, list[tuple[str, float, int]]] = {}
    for i, cat in enumerate(cats):
        row = X[i].toarray().flatten()
        scored = [
            (terms[j], float(row[j]), df_count[terms[j]])
            for j in range(len(terms))
            if row[j] > 0
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        out[cat] = scored
    return out


def weight_for(df: int) -> float:
    if df <= 2:
        return 1.5
    if df <= 5:
        return 1.0
    return 0.8


# ── Emit SQL ─────────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    return s.replace("'", "''")


def emit_sql(
    hand: dict[str, dict[str, float]],
    scored: dict[str, list[tuple[str, float, int]]],
    path: Path,
) -> dict[str, int]:
    """Emit one INSERT block per category, combining hand + TF-IDF keywords."""
    counts: dict[str, int] = {}
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(
            "-- Generated by scripts/generate_keywords.py. Do not edit by hand.\n"
            "-- TF-IDF keywords derived from HS 6-digit descriptions, merged with\n"
            "-- hand-curated keywords from seed_keywords.sql (hand weights win).\n\n"
        )

        for cat in sorted(scored.keys()):
            # Start with hand-curated entries (preserved verbatim)
            merged: dict[str, float] = dict(hand.get(cat, {}))

            # Layer TF-IDF picks; skip if already present (hand wins) or filtered.
            for term, score, df in scored[cat][:TOP_N_PER_CAT]:
                if term in merged:
                    continue
                if not TOKEN_OK.match(term):
                    continue
                if term in EXTRA_STOPWORDS:
                    continue
                # Filter bigrams: both parts must be ≥ 4 chars and non-stopword.
                parts = term.split()
                if len(parts) == 2:
                    if any(p in EXTRA_STOPWORDS or len(p) < 4 for p in parts):
                        continue
                else:
                    if any(p in EXTRA_STOPWORDS or len(p) < 3 for p in parts):
                        continue
                merged[term] = weight_for(df)

            # Enforce ≥ 40 per category (pad from next-tier tf-idf if short)
            if len(merged) < 40:
                i = TOP_N_PER_CAT
                while len(merged) < 40 and i < len(scored[cat]):
                    term, _, df = scored[cat][i]
                    i += 1
                    if term in merged: continue
                    if not TOKEN_OK.match(term): continue
                    if term in EXTRA_STOPWORDS: continue
                    parts = term.split()
                    if len(parts) == 2:
                        if any(p in EXTRA_STOPWORDS or len(p) < 4 for p in parts): continue
                    else:
                        if any(p in EXTRA_STOPWORDS or len(p) < 3 for p in parts): continue
                    merged[term] = weight_for(df)

            counts[cat] = len(merged)

            # Write category INSERT block
            f.write(f"-- ── {cat} ─────────────────────────────────────────────\n")
            f.write(
                "INSERT INTO category_keywords (category_id, keyword, weight)\n"
                "SELECT id, kw, wt FROM classification_categories,\n(VALUES\n"
            )
            entries = sorted(merged.items())
            lines = [
                f"    ('{_esc(kw)}', {wt:.2f})" for kw, wt in entries
            ]
            f.write(",\n".join(lines))
            f.write(
                f"\n) AS t(kw, wt)\nWHERE name = '{cat}'\n"
                "ON CONFLICT (category_id, keyword) DO NOTHING;\n\n"
            )
    return counts


# ── Entry ────────────────────────────────────────────────────────────────────

def main() -> int:
    print("Loading corpus...")
    corpus = load_category_corpus()
    print(f"  {len(corpus)} category documents, "
          f"avg {sum(len(d.split()) for d in corpus.values()) // len(corpus)} tokens each")

    print("Parsing hand-curated keywords...")
    hand = parse_existing_keywords()
    print(f"  loaded hand keywords for {len(hand)} categories "
          f"({sum(len(v) for v in hand.values())} total)")

    print("Fitting TF-IDF...")
    scored = extract_keywords(corpus)

    print(f"Writing SQL → {OUT_SQL}")
    counts = emit_sql(hand, scored, OUT_SQL)

    print("\nKeywords per category:")
    for cat in sorted(counts):
        print(f"  {cat:<18} {counts[cat]:>4}")
    print(f"Total: {sum(counts.values())}")
    print(f"Median: {sorted(counts.values())[len(counts)//2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
