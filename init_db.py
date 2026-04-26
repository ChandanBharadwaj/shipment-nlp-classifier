"""
Database initializer.

Applies the schema (schema.sql) and optionally seeds reference data.
Safe to re-run — every statement uses IF NOT EXISTS / ON CONFLICT DO NOTHING.

Usage:
    # From the repo root (with .env or DATABASE_URL set):
    python init_db.py

    # Or pass a connection string directly:
    python init_db.py --url "postgresql://shipment:shipment@localhost:5555/shipment_db"

    # Schema only (skip seed data):
    python init_db.py --no-seed

    # Dry run (print files that would be executed, don't run them):
    python init_db.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent

SCHEMA: Path = ROOT / "schema.sql"

SEEDS: list[Path] = [
    ROOT / "seed" / "seed_categories.sql",       # 20 categories + 96 HS chapter edges
    ROOT / "seed" / "seed_keywords.sql",         # legacy hand-curated keywords
    ROOT / "seed" / "seed_keywords_v2.sql",      # TF-IDF + hand merge (generated)
    ROOT / "seed" / "seed_anchors.sql",          # CCTR anchors / suppressors / modifiers (per-chapter, signal-class-typed)
    ROOT / "seed" / "seed_collisions.sql",       # CCTR token-collision registry (polysemy resolution rules)
    ROOT / "seed" / "seed_shipment_labels.sql",  # legacy 1000 labeled rows
    ROOT / "seed" / "seed_splits.sql",           # splits legacy rows only
    ROOT / "seed" / "seed_shipment_labels_v2.sql",  # generated ~15k rows w/ hs_chapter + split
    ROOT / "seed" / "seed_synthetic_labels.sql", # CCTR hand-curated chapter-locked labels (Commit 6)
    ROOT / "seed" / "seed_confusables.sql",      # hand-curated confusables (train-only)
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _connect(url: str) -> psycopg2.extensions.connection:
    """Connect to PostgreSQL with pgvector registered. Retries up to 15s for Docker startup."""
    last_exc = None
    for attempt in range(1, 7):
        try:
            conn = psycopg2.connect(url)
            register_vector(conn)
            return conn
        except psycopg2.OperationalError as exc:
            last_exc = exc
            wait = attempt * 2
            print(f"  DB not ready (attempt {attempt}/6) — retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(
        f"Could not connect to the database after 6 attempts.\n"
        f"Last error: {last_exc}\n\n"
        f"Is Docker running?  →  docker compose up -d\n"
        f"Check DATABASE_URL  →  {url}"
    )


def _run_sql_file(conn: psycopg2.extensions.connection, path: Path) -> None:
    """Execute a SQL file inside a single transaction."""
    sql = path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def _row_count(conn: psycopg2.extensions.connection, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        return cur.fetchone()[0]


# ── Main ───────────────────────────────────────────────────────────────────────

def run(url: str, include_seed: bool = True, dry_run: bool = False) -> None:
    files = [SCHEMA] + (SEEDS if include_seed else [])

    if dry_run:
        print("DRY RUN — the following files would be executed:\n")
        for f in files:
            print(f"  {f.relative_to(ROOT)}")
        print("\nNo changes made.")
        return

    print(f"Connecting to: {url}")
    conn = _connect(url)
    print("Connected.\n")

    # ── Schema ─────────────────────────────────────────────────────────────────
    print("── Applying schema ──────────────────────────────────────────")
    try:
        _run_sql_file(conn, SCHEMA)
        print(f"  ✓  {SCHEMA.name}")
    except Exception as exc:
        print(f"  ✗  {SCHEMA.name}")
        print(f"     ERROR: {exc}")
        conn.rollback()
        conn.close()
        sys.exit(1)

    # ── Seeds ──────────────────────────────────────────────────────────────────
    if include_seed:
        print("\n── Seeding data ─────────────────────────────────────────────")
        for path in SEEDS:
            label = path.name
            try:
                _run_sql_file(conn, path)
                print(f"  ✓  {label}")
            except Exception as exc:
                print(f"  ✗  {label}")
                print(f"     ERROR: {exc}")
                conn.rollback()
                conn.close()
                sys.exit(1)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────")
    counts = {
        "classification_categories": "categories",
        "category_hs_chapters":      "HS-chapter → category edges (target: 96)",
        "category_keywords":         "keywords",
        "shipment_labels":           "labeled rows (legacy + v2-generated)",
    }
    for table, label in counts.items():
        try:
            n = _row_count(conn, table)
            print(f"  {table:<40} {n:>5} {label}")
        except Exception:
            print(f"  {table:<40}  (table not found — migration may have failed)")

    conn.close()
    print("\nDone. Next step: build centroids.")
    print("  cd ml-service && python centroid_builder.py")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load .env from ml-service/.env if present, then fall back to root .env
    for env_path in [ROOT / "ml-service" / ".env", ROOT / ".env"]:
        if env_path.exists():
            load_dotenv(env_path)
            break
    else:
        load_dotenv()  # default search

    parser = argparse.ArgumentParser(description="Initialize shipment classification database.")
    parser.add_argument(
        "--url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection string. Defaults to DATABASE_URL env var.",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Apply schema only; skip seed data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be executed without making any changes.",
    )
    args = parser.parse_args()

    if not args.url:
        print(
            "ERROR: No database URL provided.\n"
            "  Set DATABASE_URL in ml-service/.env  or pass --url.",
            file=sys.stderr,
        )
        sys.exit(1)

    run(url=args.url, include_seed=not args.no_seed, dry_run=args.dry_run)
