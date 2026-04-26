"""
admin_routes.py — FastAPI router backing the /admin/* SPA.

Mounted by main.py under prefix ``/admin/api``. All endpoints are
JSON in / JSON out. Reads check connections out of ``db.pooled_connection``
and use ``RealDictCursor`` so rows round-trip naturally to JSON.

Surface (Commit 2 — read-only + discovery):

    GET  /admin/api/overview              — counts + top polysemous tokens
    GET  /admin/api/categories            — every active category + its chapters
    GET  /admin/api/keywords              — paginated/filterable keyword list
    GET  /admin/api/keywords/by-token/{t} — every chapter row for a token
    GET  /admin/api/collisions            — full collision registry
    GET  /admin/api/collisions/{token}    — one collision row
    GET  /admin/api/audit-log             — paginated audit entries
    POST /admin/api/discover              — run the discovery job in-process

Write endpoints land in Commit 6 once the UI's <ChangeModal> is in place.

Each DB-touching helper is a module-level function so tests can
``monkeypatch`` it without standing up a real Postgres — no SQL is
embedded inside the route bodies.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from psycopg2.extras import RealDictCursor

from db import pooled_connection

router = APIRouter(prefix="/admin/api", tags=["admin"])


# ── Read-shape helpers ────────────────────────────────────────────────────────
#
# Each helper takes a psycopg2 connection and returns plain Python (dicts /
# lists / scalars). Tests monkeypatch the helpers, not the route handlers, so
# the FastAPI surface stays exercised end-to-end without a DB.


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _serialize_audit(row: dict) -> dict:
    """Normalize a keyword_audit_log row for JSON output (timestamps as
    ISO-8601 strings; before/after JSONB arrives as a dict already)."""
    return {
        **row,
        "occurred_at": _isoformat(row.get("occurred_at")),
    }


def fetch_overview(conn) -> dict:
    """Counts + top-10 polysemous tokens. One round-trip per query so each
    fragment is independently testable."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM category_keywords)         AS total_keywords,
                (SELECT COUNT(*) FROM token_collisions)          AS total_collisions,
                (SELECT MAX(occurred_at) FROM keyword_audit_log) AS last_audit_ts
            """
        )
        totals = cur.fetchone() or {}

        cur.execute(
            """
            SELECT signal_class, COUNT(*) AS n
            FROM   category_keywords
            GROUP  BY signal_class
            ORDER  BY signal_class
            """
        )
        by_class = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT cc.name AS category, COUNT(*) AS n
            FROM   category_keywords ck
            JOIN   classification_categories cc ON cc.id = ck.category_id
            WHERE  cc.is_active = true
            GROUP  BY cc.name
            ORDER  BY n DESC, cc.name
            """
        )
        by_category = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT hs_chapter, COUNT(*) AS n
            FROM   category_keywords
            WHERE  hs_chapter IS NOT NULL
            GROUP  BY hs_chapter
            ORDER  BY hs_chapter
            """
        )
        by_chapter = [dict(r) for r in cur.fetchall()]

        # Top-10 polysemous: signals appearing in ≥2 distinct chapters,
        # ordered by breadth.
        cur.execute(
            """
            SELECT keyword,
                   COUNT(DISTINCT hs_chapter)                   AS n_chapters,
                   array_agg(DISTINCT hs_chapter)               AS chapters
            FROM   category_keywords
            WHERE  signal_class = 'signal' AND hs_chapter IS NOT NULL
            GROUP  BY keyword
            HAVING COUNT(DISTINCT hs_chapter) >= 2
            ORDER  BY n_chapters DESC, keyword
            LIMIT  10
            """
        )
        top_polysemous = [
            {**dict(r), "chapters": sorted(r["chapters"] or [])}
            for r in cur.fetchall()
        ]

    return {
        "total_keywords":   int(totals.get("total_keywords") or 0),
        "total_collisions": int(totals.get("total_collisions") or 0),
        "last_audit_ts":    _isoformat(totals.get("last_audit_ts")),
        "counts": {
            "by_class":    by_class,
            "by_category": by_category,
            "by_chapter":  by_chapter,
        },
        "top_polysemous": top_polysemous,
    }


def fetch_categories(conn) -> list[dict]:
    """Active categories with their chapters and per-chapter keyword
    counts. The chapter sub-list lets the Browse view render the second
    pane without a follow-up call."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, name, display_name, semantic_weight, keyword_weight, threshold
            FROM   classification_categories
            WHERE  is_active = true
            ORDER  BY name
            """
        )
        cats = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT chc.category_id, chc.hs_chapter, chc.chapter_title, chc.is_primary,
                   COALESCE(kc.n, 0) AS keyword_count
            FROM   category_hs_chapters chc
            LEFT   JOIN (
                       SELECT category_id, hs_chapter, COUNT(*) AS n
                       FROM   category_keywords
                       WHERE  hs_chapter IS NOT NULL
                       GROUP  BY category_id, hs_chapter
                   ) kc ON kc.category_id = chc.category_id AND kc.hs_chapter = chc.hs_chapter
            ORDER  BY chc.category_id, chc.hs_chapter
            """
        )
        by_cat: dict[int, list[dict]] = {}
        for r in cur.fetchall():
            by_cat.setdefault(r["category_id"], []).append({
                "hs_chapter":    r["hs_chapter"],
                "chapter_title": r["chapter_title"],
                "is_primary":    r["is_primary"],
                "keyword_count": int(r["keyword_count"] or 0),
            })

    for c in cats:
        c["chapters"] = by_cat.get(c["id"], [])
    return cats


def fetch_keywords(
    conn,
    category: str | None,
    chapter: str | None,
    signal_class: str | None,
    search: str | None,
    limit: int,
    offset: int,
) -> dict:
    """Filterable keyword list. Returns ``{rows, total}`` so the Browse
    table can render row-count without a second query."""
    where: list[str] = ["cc.is_active = true"]
    params: list[Any] = []
    if category:
        where.append("cc.name = %s")
        params.append(category)
    if chapter:
        where.append("ck.hs_chapter = %s")
        params.append(chapter)
    if signal_class:
        where.append("ck.signal_class = %s")
        params.append(signal_class)
    if search:
        where.append("ck.keyword ILIKE %s")
        params.append(f"%{search}%")

    where_sql = " AND ".join(where)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM   category_keywords ck
            JOIN   classification_categories cc ON cc.id = ck.category_id
            WHERE  {where_sql}
            """,
            params,
        )
        total = int((cur.fetchone() or {}).get("n") or 0)

        cur.execute(
            f"""
            SELECT ck.id, ck.category_id, cc.name AS category_name,
                   ck.hs_chapter, ck.keyword, ck.weight, ck.signal_class,
                   ck.source, ck.notes, ck.created_at
            FROM   category_keywords ck
            JOIN   classification_categories cc ON cc.id = ck.category_id
            WHERE  {where_sql}
            ORDER  BY cc.name, ck.hs_chapter NULLS LAST,
                      ck.signal_class, ck.keyword
            LIMIT  %s OFFSET %s
            """,
            [*params, limit, offset],
        )
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            d["weight"]     = float(d["weight"]) if d["weight"] is not None else None
            d["created_at"] = _isoformat(d["created_at"])
            rows.append(d)

    return {"rows": rows, "total": total, "limit": limit, "offset": offset}


def fetch_keyword_by_token(conn, token: str) -> dict:
    """Every chapter row for a token — the polysemy detail view. Match
    is case-insensitive on the keyword column to mirror how the
    inference path normalises tokens."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT ck.id, ck.category_id, cc.name AS category_name,
                   ck.hs_chapter, ck.keyword, ck.weight, ck.signal_class,
                   ck.source, ck.notes, ck.created_at
            FROM   category_keywords ck
            JOIN   classification_categories cc ON cc.id = ck.category_id
            WHERE  LOWER(ck.keyword) = LOWER(%s)
              AND  cc.is_active = true
            ORDER  BY cc.name, ck.hs_chapter NULLS LAST, ck.signal_class
            """,
            (token,),
        )
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            d["weight"]     = float(d["weight"]) if d["weight"] is not None else None
            d["created_at"] = _isoformat(d["created_at"])
            rows.append(d)
    return {"token": token, "rows": rows}


def fetch_collisions(conn) -> list[dict]:
    """The full collision registry, ordered by risk tier (high first)."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, token, home_chapters, risk_tier, resolution,
                   owner, test_case, notes, created_at
            FROM   token_collisions
            ORDER  BY CASE risk_tier
                          WHEN 'high'   THEN 0
                          WHEN 'medium' THEN 1
                          WHEN 'low'    THEN 2
                          ELSE 3
                      END,
                     token
            """
        )
        out = []
        for r in cur.fetchall():
            d = dict(r)
            d["created_at"] = _isoformat(d["created_at"])
            out.append(d)
        return out


def fetch_collision_by_token(conn, token: str) -> dict | None:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, token, home_chapters, risk_tier, resolution,
                   owner, test_case, notes, created_at
            FROM   token_collisions
            WHERE  LOWER(token) = LOWER(%s)
            """,
            (token,),
        )
        r = cur.fetchone()
        if not r:
            return None
        d = dict(r)
        d["created_at"] = _isoformat(d["created_at"])
        return d


def fetch_audit_log(
    conn,
    table: str | None,
    actor: str | None,
    operation: str | None,
    limit: int,
    offset: int,
) -> dict:
    where: list[str] = []
    params: list[Any] = []
    if table:
        where.append("table_name = %s")
        params.append(table)
    if actor:
        where.append("actor = %s")
        params.append(actor)
    if operation:
        where.append("operation = %s")
        params.append(operation)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"SELECT COUNT(*) AS n FROM keyword_audit_log {where_sql}", params)
        total = int((cur.fetchone() or {}).get("n") or 0)

        cur.execute(
            f"""
            SELECT id, table_name, operation, row_pk, before, after,
                   actor, ticket, reason, occurred_at
            FROM   keyword_audit_log
            {where_sql}
            ORDER  BY occurred_at DESC, id DESC
            LIMIT  %s OFFSET %s
            """,
            [*params, limit, offset],
        )
        rows = [_serialize_audit(dict(r)) for r in cur.fetchall()]

    return {"rows": rows, "total": total, "limit": limit, "offset": offset}


def run_discover(conn, min_chapters: int, min_weight: float) -> dict:
    """Run the discovery candidate computation in-process. Imports lazily
    so a missing scripts/ package (e.g. in stripped-down deployments)
    surfaces as a clean 500 instead of a router-load failure."""
    # Add repo root to sys.path so we can `from scripts.discover_collisions ...`
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.discover_collisions import compute_candidates, _registered_tokens

    candidates = compute_candidates(
        conn,
        min_chapters=min_chapters,
        min_weight=min_weight,
    )
    n_already_registered = len(_registered_tokens(conn))
    return {
        "candidates":           candidates,
        "n_already_registered": n_already_registered,
        "min_chapters":         min_chapters,
        "min_weight":           min_weight,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/overview")
def get_overview() -> dict:
    with pooled_connection() as conn:
        return fetch_overview(conn)


@router.get("/categories")
def get_categories() -> list[dict]:
    with pooled_connection() as conn:
        return fetch_categories(conn)


@router.get("/keywords")
def get_keywords(
    category:     str | None = Query(None,  description="Filter by category name."),
    chapter:      str | None = Query(None,  description="Filter by HS 2-digit chapter."),
    signal_class: str | None = Query(None,  description="anchor | signal | suppressor | modifier"),
    search:       str | None = Query(None,  description="ILIKE substring match on keyword."),
    limit:        int        = Query(100, ge=1, le=1000),
    offset:       int        = Query(0,   ge=0),
) -> dict:
    if signal_class and signal_class not in ("anchor", "signal", "suppressor", "modifier"):
        raise HTTPException(422, f"invalid signal_class={signal_class!r}")
    with pooled_connection() as conn:
        return fetch_keywords(conn, category, chapter, signal_class, search, limit, offset)


@router.get("/keywords/by-token/{token}")
def get_keyword_by_token(token: str) -> dict:
    if not token.strip():
        raise HTTPException(422, "token cannot be empty")
    with pooled_connection() as conn:
        return fetch_keyword_by_token(conn, token)


@router.get("/collisions")
def get_collisions() -> list[dict]:
    with pooled_connection() as conn:
        return fetch_collisions(conn)


@router.get("/collisions/{token}")
def get_collision(token: str) -> dict:
    with pooled_connection() as conn:
        row = fetch_collision_by_token(conn, token)
    if row is None:
        raise HTTPException(404, f"no collision registered for token {token!r}")
    return row


@router.get("/audit-log")
def get_audit_log(
    table:     str | None = Query(None, description="category_keywords | token_collisions"),
    actor:     str | None = Query(None),
    operation: str | None = Query(None, description="insert | update | delete"),
    limit:     int        = Query(50, ge=1, le=500),
    offset:    int        = Query(0,  ge=0),
) -> dict:
    if table and table not in ("category_keywords", "token_collisions"):
        raise HTTPException(422, f"invalid table={table!r}")
    if operation and operation not in ("insert", "update", "delete"):
        raise HTTPException(422, f"invalid operation={operation!r}")
    with pooled_connection() as conn:
        return fetch_audit_log(conn, table, actor, operation, limit, offset)


@router.post("/discover")
def post_discover(
    min_chapters: int   = Query(3,    ge=2, le=10),
    min_weight:   float = Query(0.30, ge=0.0, le=1.0),
) -> dict:
    with pooled_connection() as conn:
        return run_discover(conn, min_chapters=min_chapters, min_weight=min_weight)
