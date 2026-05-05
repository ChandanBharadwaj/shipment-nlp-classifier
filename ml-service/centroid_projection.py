"""
centroid_projection.py — 2D projection of category_centroids for the
/admin/api/centroids visualization, plus single-point transform for the
per-keyword affinity feature.

We project the L2-normalized 384-dim centroid vectors down to (x, y) so
the SPA can render them as a scatter plot. Two reducers are tried in
order:

    1. UMAP (umap-learn) — preserves local structure, makes related
       chapters visibly cluster.
    2. PCA (scikit-learn) — fallback when umap-learn isn't installed
       or the corpus is too small (<4 points). Always available.

The result is memoized per-process keyed on MAX(updated_at) of
category_centroids: every centroid rebuild invalidates the cache, but
identical state across requests reuses the projection.

The cache also stores the **fitted reducer object** so a new
keyword embedding can be projected into the same 2D space later via
``transform_point()`` without refitting (UMAP fits take seconds; the
transform is milliseconds).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from psycopg2.extras import RealDictCursor


# Process-local cache. Keyed by str(MAX(updated_at)) so any centroid
# rebuild bumps the key and we recompute. Each entry holds:
#   {
#     "projection":  {generated_at, reducer, points},  # what /centroids returns
#     "reducer_obj": fitted UMAP or PCA (or None if no centroids),
#     "n_comp":      1 or 2 (for PCA padding),
#   }
_CACHE: dict[str, dict[str, Any]] = {}


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _fetch_centroid_rows(conn) -> list[dict]:
    """Pull every (category, chapter) centroid plus joined display fields.

    Returns rows with: category_id, category_name, hs_chapter, chapter_title,
    sample_count, centroid (pgvector → list[float] via psycopg2 register).
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            WITH cat_id AS (
                SELECT slug, ROW_NUMBER() OVER (ORDER BY display_name) AS id
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
        return list(cur.fetchall())


def _max_updated_at(conn) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(updated_at) FROM category_centroids")
        row = cur.fetchone()
    return _isoformat(row[0] if row else None) or "empty"


def _parse_pgvector(text: str) -> list[float]:
    """pgvector text format is '[0.1,0.2,...]'."""
    inner = text.strip().lstrip("[").rstrip("]")
    if not inner:
        return []
    return [float(x) for x in inner.split(",")]


def _reduce(matrix: np.ndarray) -> tuple[str, Any, np.ndarray, int]:
    """Returns (reducer_name, fitted_reducer, 2D coords, n_components).

    UMAP if available + n>=4, else PCA. The fitted reducer is returned so
    callers can ``.transform()`` new points without refitting.
    """
    n = matrix.shape[0]
    if n >= 4:
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, max(2, n - 1)),
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            coords = reducer.fit_transform(matrix)
            return "umap", reducer, np.asarray(coords), 2
        except ImportError:
            pass
        except Exception:
            # UMAP can raise on degenerate inputs (too few neighbors after
            # filtering). Fall through to PCA so the endpoint never 500s.
            pass

    from sklearn.decomposition import PCA

    n_comp = 2 if n >= 2 else 1
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(matrix)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((n, 1))])
    return "pca", pca, np.asarray(coords), n_comp


def _ensure_cached(conn) -> dict[str, Any]:
    """Return the cache entry for the current centroid state, building it
    if missing. Used by both ``fetch_centroid_projection`` (which exposes
    only the projection) and ``transform_point`` (which needs the reducer).
    """
    key = _max_updated_at(conn)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    rows = _fetch_centroid_rows(conn)
    if not rows:
        entry: dict[str, Any] = {
            "projection": {
                "generated_at": datetime.now().isoformat(),
                "reducer":      "pca",
                "points":       [],
            },
            "reducer_obj": None,
            "n_comp":      0,
        }
        _CACHE[key] = entry
        return entry

    matrix = np.asarray(
        [_parse_pgvector(r["centroid_txt"]) for r in rows], dtype=np.float32
    )
    reducer_name, reducer_obj, coords, n_comp = _reduce(matrix)

    points = [
        {
            "category_id":   int(r["category_id"]),
            "category_name": r["category_name"],
            "hs_chapter":    r["hs_chapter"],
            "chapter_title": r["chapter_title"],
            "sample_count":  int(r["sample_count"]) if r["sample_count"] is not None else 0,
            "x":             float(coords[i, 0]),
            "y":             float(coords[i, 1]),
        }
        for i, r in enumerate(rows)
    ]
    entry = {
        "projection": {
            "generated_at": datetime.now().isoformat(),
            "reducer":      reducer_name,
            "points":       points,
        },
        "reducer_obj": reducer_obj,
        "n_comp":      n_comp,
    }
    _CACHE[key] = entry
    return entry


def fetch_centroid_projection(conn) -> dict[str, Any]:
    """Memoized 2D projection. Key invalidates whenever any centroid row's
    updated_at changes. Returns the JSON shape consumed by Centroids.vue."""
    return _ensure_cached(conn)["projection"]


def transform_point(conn, vec: np.ndarray) -> dict[str, Any]:
    """Project a single 384-dim vector into the same 2D space as the
    cached centroid projection.

    Returns ``{"reducer": "umap"|"pca", "x": float, "y": float}``. Raises
    ``ValueError`` if there are no centroids yet (nothing to project against).

    Reuses the fitted reducer from the cache — no refit, no extra DB hit
    once the cache is warm. For UMAP this is the difference between a
    multi-second fit and a millisecond transform.
    """
    entry = _ensure_cached(conn)
    reducer_obj = entry["reducer_obj"]
    if reducer_obj is None:
        raise ValueError("no centroids in DB — cannot project a point")

    arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    coords = reducer_obj.transform(arr)
    coords = np.asarray(coords)

    # PCA with n_components=1 returns a 1D coord; pad with 0 for plotting.
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((1, 1))])

    return {
        "reducer": entry["projection"]["reducer"],
        "x":       float(coords[0, 0]),
        "y":       float(coords[0, 1]),
    }


def clear_cache() -> None:
    """Test hook — clears the memoization map."""
    _CACHE.clear()
