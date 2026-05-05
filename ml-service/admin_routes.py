"""
admin_routes.py — FastAPI router backing the /admin/* SPA.

Mounted by main.py under prefix ``/admin/api``. All endpoints are
JSON in / JSON out. Reads check connections out of ``db.pooled_connection``
and use ``RealDictCursor`` so rows round-trip naturally to JSON.

Surface (v3 source-driven pipeline — read-only):

    GET  /admin/api/overview              — counts of v3 keywords, categories, sources
    GET  /admin/api/categories            — LLM-derived categories + chapters
    GET  /admin/api/chapters              — flat HS chapter list with rollups
    GET  /admin/api/keywords              — paginated/filterable keyword list
    GET  /admin/api/keywords/by-token/{t} — every chapter row for a token
    GET  /admin/api/labels                — paginated shipment labels with predictions
    GET  /admin/api/centroids             — UMAP/PCA scatter of centroids
    GET  /admin/api/keywords/{t}/centroid-affinity — per-token cosine to every centroid

Removed in v3 (no longer applicable to source-driven pipeline):
    /admin/api/collisions, /admin/api/audit-log, /admin/api/discover

Each DB-touching helper is a module-level function so tests can
``monkeypatch`` it without standing up a real Postgres — no SQL is
embedded inside the route bodies.
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from psycopg2.extras import RealDictCursor

from db import pooled_connection
from centroid_projection import fetch_centroid_projection, transform_point

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


def fetch_overview(conn) -> dict:
    """v3 overview: counts of v3 keywords, broken down by signal_class,
    category and chapter, plus top-10 polysemous tokens.

    Polysemy is now found at the token level: signal-class tokens that
    show up in ≥2 distinct chapters are interesting because they suggest
    cross-category vocabulary.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM keywords)                   AS total_keywords,
                (SELECT COUNT(*) FROM categories)                 AS total_categories,
                (SELECT COUNT(*) FROM public_source_descriptions) AS total_source_rows
            """
        )
        totals = cur.fetchone() or {}

        cur.execute(
            """
            SELECT signal_class, COUNT(*) AS n
            FROM   keywords
            GROUP  BY signal_class
            ORDER  BY signal_class
            """
        )
        by_class = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT c.display_name AS category, COUNT(*) AS n
            FROM   keywords k
            JOIN   chapter_categories cm USING (hs_chapter)
            JOIN   categories c ON c.slug = cm.category_slug
            GROUP  BY c.display_name
            ORDER  BY n DESC, c.display_name
            """
        )
        by_category = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT hs_chapter, COUNT(*) AS n
            FROM   keywords
            GROUP  BY hs_chapter
            ORDER  BY hs_chapter
            """
        )
        by_chapter = [dict(r) for r in cur.fetchall()]

        # Top-10 polysemous: signal tokens spanning ≥2 chapters.
        cur.execute(
            """
            SELECT keyword,
                   COUNT(DISTINCT hs_chapter) AS n_chapters,
                   array_agg(DISTINCT hs_chapter) AS chapters
            FROM   keywords
            WHERE  signal_class = 'signal'
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
        "total_keywords":    int(totals.get("total_keywords") or 0),
        "total_categories":  int(totals.get("total_categories") or 0),
        "total_source_rows": int(totals.get("total_source_rows") or 0),
        # Kept for API contract compatibility; v3 doesn't use these.
        "total_collisions":  0,
        "last_audit_ts":     None,
        "counts": {
            "by_class":    by_class,
            "by_category": by_category,
            "by_chapter":  by_chapter,
        },
        "top_polysemous": top_polysemous,
    }


def fetch_categories(conn) -> list[dict]:
    """v3 categories with their chapters + per-chapter keyword counts.
    The 'id' field is synthesised from the row order so the front end
    keeps working — v3 keys on slug, not numeric id.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT slug, display_name, generated_at
            FROM   categories
            ORDER  BY display_name
            """
        )
        cats: list[dict] = []
        for i, r in enumerate(cur.fetchall(), 1):
            cats.append({
                "id":              i,                  # synthetic — for SPA compat
                "name":            r["slug"],
                "display_name":    r["display_name"],
                "semantic_weight": 0.80,               # v3 uses defaults; calibration dropped
                "keyword_weight":  0.20,
                "threshold":       None,
                "chapters":        [],
            })

        cur.execute(
            """
            SELECT cm.category_slug, cm.hs_chapter, cm.chapter_title,
                   COALESCE(kc.n, 0) AS keyword_count
            FROM   chapter_categories cm
            LEFT   JOIN (
                       SELECT hs_chapter, COUNT(*) AS n
                       FROM   keywords
                       GROUP  BY hs_chapter
                   ) kc USING (hs_chapter)
            ORDER  BY cm.category_slug, cm.hs_chapter
            """
        )
        by_slug: dict[str, list[dict]] = {}
        for r in cur.fetchall():
            by_slug.setdefault(r["category_slug"], []).append({
                "hs_chapter":    r["hs_chapter"],
                "chapter_title": r["chapter_title"],
                "is_primary":    False,                # v3 has no primary concept
                "keyword_count": int(r["keyword_count"] or 0),
            })

    for c in cats:
        c["chapters"] = by_slug.get(c["name"], [])
    return cats


def fetch_chapters(conn) -> list[dict]:
    """Flat HS-chapter list with per-chapter rollups (keyword count, label
    count, centroid presence) for SPA views. Single JOIN-heavy query
    avoids N+1 from the front end."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Synthetic category_id (ROW_NUMBER over slug) — stable across queries
        # since fetch_categories uses the same ordering. SPA only uses
        # category_id for client-side keying, never for back-fetch.
        cur.execute(
            """
            WITH cat_id AS (
                SELECT slug, ROW_NUMBER() OVER (ORDER BY display_name) AS id
                FROM   categories
            )
            SELECT cm.hs_chapter,
                   cm.chapter_title,
                   false                          AS is_primary,
                   ci.id                          AS category_id,
                   cm.category_slug               AS category_name,
                   COALESCE(kw.n, 0)              AS keyword_count,
                   COALESCE(lb.n, 0)              AS label_count,
                   (cn.hs_chapter IS NOT NULL)    AS centroid_present,
                   cn.sample_count                AS sample_count
            FROM   chapter_categories cm
            JOIN   cat_id ci ON ci.slug = cm.category_slug
            LEFT   JOIN (
                       SELECT hs_chapter, COUNT(*) AS n
                       FROM   keywords
                       GROUP  BY hs_chapter
                   ) kw USING (hs_chapter)
            LEFT   JOIN (
                       SELECT hs_chapter, COUNT(*) AS n
                       FROM   shipment_labels
                       WHERE  hs_chapter IS NOT NULL
                       GROUP  BY hs_chapter
                   ) lb USING (hs_chapter)
            LEFT   JOIN category_centroids cn
                       ON cn.category_slug = cm.category_slug
                      AND cn.hs_chapter    = cm.hs_chapter
            ORDER  BY cm.hs_chapter
            """
        )
        out: list[dict] = []
        for r in cur.fetchall():
            d = dict(r)
            d["keyword_count"] = int(d["keyword_count"] or 0)
            d["label_count"]   = int(d["label_count"] or 0)
            d["sample_count"]  = int(d["sample_count"]) if d["sample_count"] is not None else None
            out.append(d)
        return out


def _predict_for_label_rows(rows: list[dict]) -> list[str | None]:
    """Run the classifier on label rows and return one predicted_category
    name per row (None if the classifier returned no categories — e.g.
    the row failed the input-quality gate).

    Lazy imports: ``main`` imports this module at startup, so a top-level
    ``import main`` would be circular. We only need its module globals
    (centroids, keywords, etc.) at call time."""
    if not rows:
        return []
    import main as _main
    from classifier import predict_batch

    inputs = [(r["cargo_text"], r["commodity_text"]) for r in rows]
    results = predict_batch(
        inputs,
        _main.centroids,
        _main.keywords,
        _main.category_config,
        chapter_titles=_main.chapter_titles,
        chapter_to_category=_main.chapter_to_category,
        category_descriptions=_main.category_descriptions if _main.RERANKER_ENABLED else None,
        keywords_typed=_main.keywords_typed,
    )
    # predict_batch returns each row as {"categories": list[str], ...} —
    # the categories list is already category name strings (sorted by score),
    # so the top match is just cats[0].
    out: list[str | None] = []
    for res in results:
        cats = res.get("categories") or []
        out.append(cats[0] if cats else None)
    return out


def fetch_labels(
    conn,
    split: str | None,
    category: str | None,
    chapter: str | None,
    search: str | None,
    limit: int,
    offset: int,
    prediction_status: str | None = None,
) -> dict:
    """Paginated shipment_labels browser. Mirrors fetch_keywords' shape so
    the front-end DataTable component is reusable.

    Predictions are always-on: each returned row carries a
    ``predicted_category`` (str | None). When ``prediction_status`` is
    ``"correct"`` or ``"mispredicted"`` we must classify *every* row that
    matches the SQL filters before pagination — that can take a few
    seconds on large filtered sets. Caching is intentionally out of
    scope for this round."""
    where: list[str] = []
    params: list[Any] = []
    if split:
        where.append("split = %s")
        params.append(split)
    if category:
        where.append("category_name = %s")
        params.append(category)
    if chapter:
        where.append("hs_chapter = %s")
        params.append(chapter)
    if search:
        where.append("(cargo_text ILIKE %s OR commodity_text ILIKE %s OR shipment_id ILIKE %s)")
        like = f"%{search}%"
        params.extend([like, like, like])
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if prediction_status in ("correct", "mispredicted"):
            # Pull every matching row, classify, filter, then paginate in Python.
            cur.execute(
                f"""
                SELECT id, shipment_id, category_name, cargo_text, commodity_text,
                       hs_chapter, split
                FROM   shipment_labels
                {where_sql}
                ORDER  BY id
                """,
                params,
            )
            all_rows = [dict(r) for r in cur.fetchall()]
            preds = _predict_for_label_rows(all_rows)
            for r, p in zip(all_rows, preds):
                r["predicted_category"] = p

            if prediction_status == "correct":
                filtered = [r for r in all_rows
                            if r["predicted_category"] is not None
                            and r["predicted_category"] == r["category_name"]]
            else:  # mispredicted
                filtered = [r for r in all_rows
                            if r["predicted_category"] is not None
                            and r["predicted_category"] != r["category_name"]]

            total = len(filtered)
            rows  = filtered[offset:offset + limit]
        else:
            # Default path: SQL pagination, then classify just the page.
            cur.execute(f"SELECT COUNT(*) AS n FROM shipment_labels {where_sql}", params)
            total = int((cur.fetchone() or {}).get("n") or 0)

            cur.execute(
                f"""
                SELECT id, shipment_id, category_name, cargo_text, commodity_text,
                       hs_chapter, split
                FROM   shipment_labels
                {where_sql}
                ORDER  BY id
                LIMIT  %s OFFSET %s
                """,
                [*params, limit, offset],
            )
            rows = [dict(r) for r in cur.fetchall()]
            preds = _predict_for_label_rows(rows)
            for r, p in zip(rows, preds):
                r["predicted_category"] = p

    return {"rows": rows, "total": total, "limit": limit, "offset": offset}


def fetch_keywords(
    conn,
    category: str | None,
    chapter: str | None,
    signal_class: str | None,
    search: str | None,
    limit: int,
    offset: int,
) -> dict:
    """Filterable v3 keyword list. Returns ``{rows, total}``.
    Category filter accepts the slug (URL-safe key)."""
    where: list[str] = []
    params: list[Any] = []
    if category:
        where.append("cm.category_slug = %s")
        params.append(category)
    if chapter:
        where.append("k.hs_chapter = %s")
        params.append(chapter)
    if signal_class:
        where.append("k.signal_class = %s")
        params.append(signal_class)
    if search:
        where.append("k.keyword ILIKE %s")
        params.append(f"%{search}%")

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM   keywords k
            JOIN   chapter_categories cm USING (hs_chapter)
            {where_sql}
            """,
            params,
        )
        total = int((cur.fetchone() or {}).get("n") or 0)

        cur.execute(
            f"""
            SELECT k.id,
                   ci.id AS category_id,
                   cm.category_slug AS category_name,
                   k.hs_chapter, k.keyword, k.weight, k.signal_class,
                   k.source,
                   COALESCE(NULLIF(CONCAT_WS(' ',
                       CASE WHEN k.target_chapter    IS NOT NULL THEN 'targets:' || k.target_chapter END,
                       CASE WHEN k.suppress_category IS NOT NULL THEN 'suppresses:' || k.suppress_category END,
                       CASE WHEN k.head_noun         IS NOT NULL THEN 'on:' || k.head_noun END
                   ), ''), '') AS notes,
                   k.generated_at AS created_at
            FROM   keywords k
            JOIN   chapter_categories cm USING (hs_chapter)
            JOIN   (SELECT slug, ROW_NUMBER() OVER (ORDER BY display_name) AS id
                    FROM categories) ci ON ci.slug = cm.category_slug
            {where_sql}
            ORDER  BY cm.category_slug, k.hs_chapter, k.signal_class, k.keyword
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
    """Every chapter row for a token. Case-insensitive match."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT k.id,
                   ci.id AS category_id,
                   cm.category_slug AS category_name,
                   k.hs_chapter, k.keyword, k.weight, k.signal_class,
                   k.source,
                   COALESCE(NULLIF(CONCAT_WS(' ',
                       CASE WHEN k.target_chapter    IS NOT NULL THEN 'targets:' || k.target_chapter END,
                       CASE WHEN k.suppress_category IS NOT NULL THEN 'suppresses:' || k.suppress_category END,
                       CASE WHEN k.head_noun         IS NOT NULL THEN 'on:' || k.head_noun END
                   ), ''), '') AS notes,
                   k.generated_at AS created_at
            FROM   keywords k
            JOIN   chapter_categories cm USING (hs_chapter)
            JOIN   (SELECT slug, ROW_NUMBER() OVER (ORDER BY display_name) AS id
                    FROM categories) ci ON ci.slug = cm.category_slug
            WHERE  LOWER(k.keyword) = LOWER(%s)
            ORDER  BY cm.category_slug, k.hs_chapter, k.signal_class
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


# v3 removes the collision registry, manual audit log, and discovery
# scan — those existed to support hand-curation flows that the source-
# driven pipeline replaces. Endpoints below return empty/410 for
# compatibility while the SPA migrates away from those views.


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/overview")
def get_overview() -> dict:
    with pooled_connection() as conn:
        return fetch_overview(conn)


@router.get("/categories")
def get_categories() -> list[dict]:
    with pooled_connection() as conn:
        return fetch_categories(conn)


@router.get("/chapters")
def get_chapters() -> list[dict]:
    with pooled_connection() as conn:
        return fetch_chapters(conn)


@router.get("/labels")
def get_labels(
    split:             str | None = Query(None, description="train | validation | test"),
    category:          str | None = Query(None, description="Filter by category name."),
    chapter:           str | None = Query(None, description="Filter by HS 2-digit chapter."),
    search:            str | None = Query(None, description="ILIKE on cargo_text / commodity_text / shipment_id."),
    prediction_status: str | None = Query(None, description="correct | mispredicted — when set, classifies every matching row before pagination."),
    limit:             int        = Query(100, ge=1, le=1000),
    offset:            int        = Query(0,   ge=0),
) -> dict:
    if split and split not in ("train", "validation", "test"):
        raise HTTPException(422, f"invalid split={split!r}")
    if prediction_status and prediction_status not in ("correct", "mispredicted"):
        raise HTTPException(422, f"invalid prediction_status={prediction_status!r}")
    with pooled_connection() as conn:
        return fetch_labels(conn, split, category, chapter, search, limit, offset,
                            prediction_status=prediction_status)


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


# v3: collisions / audit-log / discover endpoints removed. The features
# they backed (manual collision registry, governed audit log, discovery
# scan for polysemous tokens to register manually) don't apply to the
# source-driven pipeline — categories, anchors, suppressors, modifiers
# are now generated, not curated. The SPA's Collisions / Audit / Discover
# views have been removed from the router.


@router.get("/centroids")
def get_centroids() -> dict:
    """2D UMAP/PCA projection of every (category, hs_chapter) centroid.

    Memoized in-process; first call after a centroid rebuild pays the
    UMAP fit cost (~1s for ~96 points), subsequent calls return cached.
    """
    with pooled_connection() as conn:
        return fetch_centroid_projection(conn)


# ── Per-keyword centroid affinity ────────────────────────────────────────────


def fetch_centroid_affinity(conn, token: str) -> dict[str, Any]:
    """Embed ``token``, score it against every (category, chapter) centroid
    by cosine similarity, and project it into the same 2D UMAP space the
    /centroids endpoint uses.

    Token need not be in the registry — this works for any free-text input,
    which is what makes it useful for discovery (e.g. "where would
    'graphene' land?").

    Output shape:
        {token, embedding_dim, reducer, point: {x, y},
         similarities: [{category_id, category_name, hs_chapter, chapter_title,
                          sample_count, cosine, has_registry_row, registry_weight,
                          signal_class}]}

    The same token can have several rows for one (category, chapter) — e.g.
    one ``anchor`` row and one ``signal`` row. We collapse to the highest-
    weight row per (category, chapter) and surface its ``signal_class`` so
    the UI can show which class won.

    The ``embed_texts`` import is local so admin_routes can be imported
    without the SentenceTransformer model loading (model is heavy; tests
    that monkeypatch this helper bypass it).
    """
    import numpy as np
    from classifier import embed_texts

    # 1. Embed the token. Output is L2-normalized, so dot = cosine.
    vec = embed_texts([token])
    if vec.size == 0:
        raise HTTPException(500, "embedding model returned empty vector")
    embedding = np.asarray(vec[0], dtype=np.float32)

    # 2. Pull every centroid + its display fields. v3: category derived
    # from category_centroids.category_slug; synthetic category_id ordinal
    # matches fetch_categories' ROW_NUMBER ordering by display_name.
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            WITH cat_id AS (
                SELECT slug, display_name,
                       ROW_NUMBER() OVER (ORDER BY display_name) AS id
                FROM   categories
            )
            SELECT ci.id           AS category_id,
                   ci.slug         AS category_name,
                   cn.hs_chapter,
                   COALESCE(cm.chapter_title, cn.hs_chapter) AS chapter_title,
                   cn.sample_count,
                   cn.centroid::text AS centroid_txt
            FROM   category_centroids cn
            JOIN   cat_id ci ON ci.slug = cn.category_slug
            LEFT JOIN chapter_categories cm
                   ON cm.hs_chapter = cn.hs_chapter
            ORDER  BY ci.slug, cn.hs_chapter
            """
        )
        centroid_rows = list(cur.fetchall())

    if not centroid_rows:
        return {
            "token":         token,
            "embedding_dim": int(embedding.shape[0]),
            "reducer":       "pca",
            "point":         {"x": 0.0, "y": 0.0},
            "similarities":  [],
        }

    # 3. Cosine = dot product (centroids are also L2-normalized at build).
    def _parse(t: str) -> list[float]:
        inner = t.strip().lstrip("[").rstrip("]")
        return [float(x) for x in inner.split(",")] if inner else []

    matrix = np.asarray([_parse(r["centroid_txt"]) for r in centroid_rows], dtype=np.float32)
    cosines = matrix @ embedding  # (N,)

    # 4. Project the keyword into the same 2D space as /centroids.
    point = transform_point(conn, embedding)

    # 5. Overlay registry rows for this token. Same synthetic category_id
    # mapping as step 2 so the dict key matches the centroid_rows.
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            WITH cat_id AS (
                SELECT slug, ROW_NUMBER() OVER (ORDER BY display_name) AS id
                FROM   categories
            )
            SELECT ci.id AS category_id,
                   k.hs_chapter,
                   k.weight,
                   k.signal_class
            FROM   keywords k
            JOIN   chapter_categories cm USING (hs_chapter)
            JOIN   cat_id ci ON ci.slug = cm.category_slug
            WHERE  LOWER(k.keyword) = LOWER(%s)
            """,
            (token,),
        )
        registry: dict[tuple[int, str], tuple[float, str]] = {}
        for r in cur.fetchall():
            if r["weight"] is None:
                continue
            key = (int(r["category_id"]), r["hs_chapter"])
            cur_weight = registry.get(key, (-math.inf, ""))[0]
            if float(r["weight"]) > cur_weight:
                registry[key] = (float(r["weight"]), r["signal_class"])

    # 6. Zip everything together, sort by cosine desc.
    similarities = []
    for r, cos in zip(centroid_rows, cosines):
        cat_id = int(r["category_id"])
        chap   = r["hs_chapter"]
        entry  = registry.get((cat_id, chap))
        if entry is not None:
            weight, signal_class = entry
        else:
            weight, signal_class = None, None
        similarities.append({
            "category_id":      cat_id,
            "category_name":    r["category_name"],
            "hs_chapter":       chap,
            "chapter_title":    r["chapter_title"],
            "sample_count":     int(r["sample_count"]) if r["sample_count"] is not None else 0,
            "cosine":           float(cos),
            "has_registry_row": weight is not None,
            "registry_weight":  weight,
            "signal_class":     signal_class,
        })
    similarities.sort(key=lambda s: s["cosine"], reverse=True)

    return {
        "token":         token,
        "embedding_dim": int(embedding.shape[0]),
        "reducer":       point["reducer"],
        "point":         {"x": point["x"], "y": point["y"]},
        "similarities":  similarities,
    }


@router.get("/keywords/{token}/centroid-affinity")
def get_centroid_affinity(token: str) -> dict:
    """Per-keyword cosine-affinity to every centroid + 2D projection.

    Token may be any string; it does not need to exist in the registry.
    Empty / whitespace-only inputs are rejected.
    """
    if not token.strip():
        raise HTTPException(422, "token cannot be empty")
    with pooled_connection() as conn:
        try:
            return fetch_centroid_affinity(conn, token)
        except ValueError as e:
            # Raised by transform_point when no centroids exist yet.
            raise HTTPException(503, str(e))
