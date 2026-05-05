"""
Per-chapter TF-IDF over public_source_descriptions → keywords table.

For each of the 96 active HS chapters, gather all USITC + UK descriptions,
DEDUPE the text per chapter (parent rollup means a 4-digit heading's text
appears verbatim inside many child rows; without dedup, parent vocabulary is
counted N times), then run TF-IDF across the chapters as documents. Top N
distinctive terms per chapter become signal-class keywords pinned to that
chapter.

The chapter→category mapping is read from the `chapter_categories` table —
LLM-derived, from scripts/llm_classify_chapters.py.

Output:
    seed/seed_keywords_v3.sql  — INSERTs into the new `keywords` table

The script also INSERTs directly into the live DB so downstream pipeline steps
can run immediately without manual seed-load.

Usage:
    python scripts/build_public_keywords.py
    python scripts/build_public_keywords.py --top-n 60 --min-df 4
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
OUT_SQL = ROOT / "seed" / "seed_keywords_v3.sql"

# HS-domain + connective + structural-word stopword list. Inlined here so
# build_public_keywords.py doesn't depend on legacy scripts/generate_keywords.py.
EXTRA_STOPWORDS: set[str] = {
    # HS-dictionary boilerplate
    "thereof", "other", "nesoi", "otherwise", "specified", "included",
    "whether", "containing", "preparation", "preparations", "similar",
    "used", "use", "parts", "accessories", "based", "within", "weight",
    "non", "nes", "per", "kind", "kinds", "including", "composed",
    "having", "made", "form", "forms", "types", "certain", "mixed",
    "excluding", "entirely", "wholly", "following", "articles", "article",
    "contains", "containing", "material", "materials", "content", "not",
    "exceeding", "less", "more", "except", "either", "without", "with",
    "kg", "cm", "mm", "new", "raw", "extracted", "obtained",
    "different", "products", "product", "various", "all",
    "also", "under", "over", "above", "below", "being", "uses",
    # Bigram-tail noise observed in HS text
    "heading", "example", "percent", "weighing", "mainly",
    "consisting", "retail", "textile", "namely",
    # Generic descriptors that hit too many categories
    "boys", "girls", "men", "women", "children",
    # NEW v3 — connectives & structural HS phrases that polluted polysemy view
    "but", "any", "like", "well", "neither", "etc", "includes",
    "chapter", "chapters", "note", "notes", "elsewhere", "subheading",
    "headings", "subheadings", "up", "from", "into", "through",
    "their", "these", "those", "this", "that", "such", "only",
    "general", "general note", "section",
}

# Single-character / all-digit tokens get filtered. Bigrams allowed (one space).
TOKEN_OK = re.compile(r"^[a-z][a-z\-]{1,}(?: [a-z][a-z\-]{1,})?$")

SOURCE_TAG = "public_tfidf"


# ── DB readers ────────────────────────────────────────────────────────────────

def _pull_chapter_text(conn) -> dict[str, list[str]]:
    """Return {hs_chapter: [unique_description_strings]} from public_source_descriptions.

    Dedup happens at SQL level per (hs_chapter, description) — parent rollup
    means many child USITC rows for chapter NN share long prefix substrings,
    but distinct hs_codes still emit distinct descriptions, so we dedup on
    description text directly.
    """
    by_chapter: dict[str, list[str]] = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT hs_chapter, description
            FROM (
                SELECT hs_chapter, description,
                       ROW_NUMBER() OVER (PARTITION BY hs_chapter, description) AS rn
                FROM   public_source_descriptions
            ) dedup
            WHERE  rn = 1
            ORDER  BY hs_chapter
        """)
        for ch, descr in cur.fetchall():
            by_chapter.setdefault(ch, []).append(descr)
    return by_chapter


def _pull_chapter_categories(conn) -> dict[str, str]:
    """Return {hs_chapter: category_slug}. Empty if llm_classify_chapters.py
    has not been run yet."""
    out: dict[str, str] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT hs_chapter, category_slug FROM chapter_categories")
        for ch, slug in cur.fetchall():
            out[ch] = slug
    return out


# ── TF-IDF + filtering ────────────────────────────────────────────────────────

def _weight_for_rank(rank: int, kept_count: int) -> float:
    """Rank-based weight assignment within a chapter. Top 1/4 of kept tokens
    get 1.5 (highly distinctive), middle half get 1.0, bottom 1/4 get 0.8.

    Rank-based (not score-based) because TF-IDF score magnitudes shift
    drastically with corpus size — the legacy 20-doc pipeline produced scores
    in [0, 3]; the 96-chapter pipeline produces [0, 0.6]. Same distinctiveness
    rank, very different absolute number. Bucketing by rank keeps behaviour
    stable as the source grows."""
    fraction = rank / max(1, kept_count)
    if fraction < 0.25:
        return 1.5
    if fraction < 0.75:
        return 1.0
    return 0.8


def _build_tfidf(per_chapter_text: dict[str, list[str]],
                 min_df: int, top_n: int) -> dict[str, list[tuple[str, float]]]:
    """Return {hs_chapter: [(token, tfidf_score), ...]} top_n per chapter."""
    chapters = sorted(per_chapter_text.keys())
    documents = ["\n".join(per_chapter_text[ch]) for ch in chapters]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=list(set(["the", "and", "or", "of", "in", "to", "for",
                              "on", "at", "by", "an", "a", "is", "are",
                              "be", "as", "from", "this", "that", "these",
                              "those"]) | EXTRA_STOPWORDS),
        min_df=min_df,
        max_df=0.7,           # token in >70% of chapters is too common
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{1,}\b",
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(documents)
    vocab = vectorizer.get_feature_names_out()

    out: dict[str, list[tuple[str, float]]] = {}
    for i, ch in enumerate(chapters):
        row = matrix[i]
        # Sort tokens by descending TF-IDF score.
        coo = row.tocoo()
        ranked = sorted(
            zip(coo.col, coo.data),
            key=lambda p: p[1],
            reverse=True,
        )
        kept: list[tuple[str, float]] = []
        for idx, score in ranked:
            token = vocab[idx]
            if not TOKEN_OK.match(token):
                continue
            kept.append((token, float(score)))
            if len(kept) >= top_n:
                break
        out[ch] = kept
    return out


# ── Output ────────────────────────────────────────────────────────────────────

INSERT_SQL = """
    INSERT INTO keywords
        (keyword, hs_chapter, signal_class, weight, source)
    VALUES %s
    ON CONFLICT (keyword, hs_chapter, signal_class, source) DO UPDATE
        SET weight = EXCLUDED.weight,
            generated_at = now()
"""


def _emit_sql(rows: list[tuple[str, str, float]]) -> str:
    """Build a deterministic SQL file for git commit. Matches the INSERT_SQL
    semantics so re-running the script doesn't change behaviour vs the file."""
    lines = ["-- Generated by scripts/build_public_keywords.py. Do not edit by hand.",
             "-- Per-chapter TF-IDF over public_source_descriptions.",
             "",
             "DELETE FROM keywords WHERE source = 'public_tfidf';",
             ""]
    for kw, ch, w in rows:
        kw_esc = kw.replace("'", "''")
        lines.append(
            f"INSERT INTO keywords (keyword, hs_chapter, signal_class, weight, source) "
            f"VALUES ('{kw_esc}', '{ch}', 'signal', {w:.2f}, '{SOURCE_TAG}') "
            f"ON CONFLICT (keyword, hs_chapter, signal_class, source) DO UPDATE "
            f"SET weight = EXCLUDED.weight, generated_at = now();"
        )
    return "\n".join(lines) + "\n"


def _load_to_db(conn, rows: list[tuple[str, str, float]]) -> None:
    """Bulk-insert keyword rows."""
    if not rows:
        return
    values = [(kw, ch, "signal", w, SOURCE_TAG) for (kw, ch, w) in rows]
    with conn.cursor() as cur:
        cur.execute("DELETE FROM keywords WHERE source = %s", (SOURCE_TAG,))
        psycopg2.extras.execute_values(cur, INSERT_SQL, values, page_size=500)
    conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--url", default=os.environ.get("DATABASE_URL"),
                    help="Postgres URL. Default: $DATABASE_URL.")
    ap.add_argument("--top-n", type=int, default=80,
                    help="Top tokens to keep per chapter. Default 80.")
    ap.add_argument("--min-df", type=int, default=3,
                    help="Min document frequency for TF-IDF. Default 3.")
    args = ap.parse_args()

    if not args.url:
        print("ERROR: --url not set and DATABASE_URL not in env.", file=sys.stderr)
        return 2

    conn = psycopg2.connect(args.url)

    print("Reading public_source_descriptions...")
    per_chapter = _pull_chapter_text(conn)
    print(f"  -> {len(per_chapter)} chapters with text "
          f"(median {sorted(len(v) for v in per_chapter.values())[len(per_chapter)//2]} unique descriptions)")

    if not per_chapter:
        print("ERROR: public_source_descriptions is empty. Run load_public_sources.py first.",
              file=sys.stderr)
        return 2

    # Chapter→category mapping is OPTIONAL here — the keywords table doesn't
    # store a category_id (looked up via JOIN at query time), so we don't need
    # the mapping to write rows. We DO warn if it's missing so the user knows
    # downstream lookups will fail until llm_classify_chapters.py has run.
    chapter_cats = _pull_chapter_categories(conn)
    if not chapter_cats:
        print("WARN: chapter_categories table is empty. "
              "Run llm_classify_chapters.py before relying on category JOINs.")
    elif len(chapter_cats) < len(per_chapter):
        missing = set(per_chapter) - set(chapter_cats)
        print(f"WARN: {len(missing)} chapters have public-source text but no "
              f"category mapping yet: {sorted(missing)[:5]}...")

    print(f"Building TF-IDF (top_n={args.top_n}, min_df={args.min_df})...")
    per_chapter_tokens = _build_tfidf(per_chapter, min_df=args.min_df, top_n=args.top_n)

    rows: list[tuple[str, str, float]] = []
    for ch in sorted(per_chapter_tokens):
        kept = per_chapter_tokens[ch]
        for rank, (token, _score) in enumerate(kept):
            weight = _weight_for_rank(rank, len(kept))
            rows.append((token, ch, weight))

    print(f"  -> {len(rows):,} keyword rows (rank-based weight assignment)")

    OUT_SQL.write_text(_emit_sql(rows), encoding="utf-8")
    print(f"Wrote {OUT_SQL}")

    print("Applying to live DB...")
    _load_to_db(conn, rows)

    # Summary against DB.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT hs_chapter, COUNT(*)
            FROM   keywords
            WHERE  source = %s
            GROUP  BY hs_chapter
            ORDER  BY 2 DESC
            LIMIT  5
        """, (SOURCE_TAG,))
        top_chapters = cur.fetchall()
        cur.execute("""
            SELECT hs_chapter, COUNT(*)
            FROM   keywords
            WHERE  source = %s
            GROUP  BY hs_chapter
            ORDER  BY 2 ASC
            LIMIT  5
        """, (SOURCE_TAG,))
        thin_chapters = cur.fetchall()

    print("\n--- richest chapters by keyword count ---")
    for ch, n in top_chapters:
        print(f"  ch {ch}: {n} keywords")
    print("\n--- thinnest chapters (likely <10 source descriptions) ---")
    for ch, n in thin_chapters:
        print(f"  ch {ch}: {n} keywords")

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
