"""
discover_collisions.py — weekly cron job that surfaces candidate collisions.

A "candidate collision" is a token that *looks like* a polysemous word —
it appears as `signal_class='signal'` in three or more chapters with
weight >= 0.30 — but is NOT yet listed in token_collisions. Surfacing
these proactively catches new polysemy before it shows up as a wrong
classification in production.

This script doesn't auto-add anything. It writes a CSV the team
reviews in the weekly polysemy triage; if a row deserves to become a
registry entry, the curator runs scripts/apply_collision_change.py to
land it (which records the action in keyword_audit_log).

Usage:
    python -m scripts.discover_collisions [--output discovered_collisions.csv]
                                          [--min-chapters 3] [--min-weight 0.30]

Exit codes:
    0  candidates written (or none found — script always succeeds)
    1  database unreachable

CCTR plan §5a — discovery feeds the registry; governance prevents rot.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _connect(url: str):
    import psycopg2
    return psycopg2.connect(url)


def _registered_tokens(conn) -> set[str]:
    """Tokens already in token_collisions, lowercased for stable comparison."""
    out: set[str] = set()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM information_schema.tables WHERE table_name='token_collisions'
        """)
        if cur.fetchone() is None:
            return out
        cur.execute("SELECT token FROM token_collisions")
        for (tok,) in cur.fetchall():
            out.add(tok.lower())
    return out


def _candidate_signals(conn, min_weight: float) -> list[tuple[str, str, str, float]]:
    """Pull signal-class rows with weight >= min_weight, return
    [(keyword, category, hs_chapter, weight)]."""
    out: list[tuple[str, str, str, float]] = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ck.keyword, cc.name, ck.hs_chapter, ck.weight
            FROM   category_keywords ck
            JOIN   classification_categories cc ON cc.id = ck.category_id
            WHERE  ck.signal_class = 'signal'
              AND  ck.hs_chapter IS NOT NULL
              AND  ck.weight >= %s
              AND  cc.is_active = true
            """,
            (min_weight,),
        )
        for keyword, category, hs, weight in cur.fetchall():
            out.append((keyword.lower(), category, hs, float(weight)))
    return out


def compute_candidates(
    conn,
    min_chapters: int = 3,
    min_weight: float = 0.30,
) -> list[dict]:
    """Pure DB-read step — no I/O. Returns the candidate list ordered by
    breadth (most chapters first) so the highest-fanout polysemy is
    reviewed first.

    The HTTP admin route calls this directly; the CLI wraps it with the
    CSV writer below. Keeping this seam means /admin/api/discover and
    the cron job stay in lock-step on the candidate definition.
    """
    registered = _registered_tokens(conn)
    rows = _candidate_signals(conn, min_weight)

    # Group by keyword → set of chapters (with categories so the
    # operator can see the home chapters at a glance).
    by_keyword: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for kw, cat, hs, w in rows:
        by_keyword[kw].append((cat, hs, w))

    candidates: list[dict] = []
    for kw, hits in by_keyword.items():
        chapters = sorted({hs for _c, hs, _w in hits})
        if len(chapters) < min_chapters:
            continue
        if kw in registered:
            continue
        max_w = max(w for _c, _h, w in hits)
        candidates.append({
            "keyword":         kw,
            "n_chapters":      len(chapters),
            "home_chapters":   ",".join(chapters),
            "categories":      ",".join(sorted({c for c, _h, _w in hits})),
            "max_weight":      f"{max_w:.2f}",
            "n_rows":          len(hits),
            "suggested_tier":  _suggested_tier(chapters),
        })

    # Sort by descending breadth — broader collisions reviewed first.
    candidates.sort(key=lambda r: (-r["n_chapters"], r["keyword"]))
    return candidates


def discover(
    db_url: str,
    output: Path,
    min_chapters: int = 3,
    min_weight: float = 0.30,
) -> int:
    """CLI wrapper — connect, compute candidates, write CSV. Returns
    the number of candidate rows written, or -1 on connection failure."""
    try:
        conn = _connect(db_url)
    except Exception as exc:
        print(f"[error] could not connect to DB: {exc}", file=sys.stderr)
        return -1

    try:
        candidates = compute_candidates(conn, min_chapters=min_chapters, min_weight=min_weight)
    finally:
        conn.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["keyword", "n_chapters", "home_chapters", "categories",
                        "max_weight", "n_rows", "suggested_tier"],
        )
        writer.writeheader()
        writer.writerows(candidates)

    print(f"[discover_collisions] {len(candidates)} candidate(s) -> {output}")
    if not candidates:
        print("[discover_collisions] nothing new this week.")
    return len(candidates)


# Heuristic: any candidate touching ch93 (defense) or ch88 (aircraft) gets
# tier='high' suggested. The operator can override; this is just a starting
# point for the CSV review.
_HIGH_RISK_CHAPTERS = {"93", "88"}
_MEDIUM_RISK_CHAPTERS = {"28", "29", "30", "38"}


def _suggested_tier(chapters: list[str]) -> str:
    chset = set(chapters)
    if chset & _HIGH_RISK_CHAPTERS:
        return "high"
    if chset & _MEDIUM_RISK_CHAPTERS:
        return "medium"
    return "low"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=os.environ.get("DATABASE_URL"),
                    help="Postgres URL (or DATABASE_URL env var).")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results" / "discovered_collisions.csv",
                    help="Output CSV path. Default: results/discovered_collisions.csv")
    ap.add_argument("--min-chapters", type=int, default=3,
                    help="Token must appear in this many chapters as a signal. Default 3.")
    ap.add_argument("--min-weight", type=float, default=0.30,
                    help="Per-row weight threshold for inclusion. Default 0.30.")
    args = ap.parse_args()

    if not args.db:
        print("[error] DATABASE_URL not set and --db not given", file=sys.stderr)
        return 1

    n = discover(args.db, args.output, args.min_chapters, args.min_weight)
    return 1 if n < 0 else 0


if __name__ == "__main__":
    sys.exit(main())
