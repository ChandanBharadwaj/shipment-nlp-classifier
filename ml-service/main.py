"""
FastAPI classification service.

Endpoints:
    GET  /health           — liveness check
    POST /classify         — classify a single shipment
    POST /classify/batch   — classify up to 500 shipments (single batched embed)
    POST /reload           — reload centroids + keywords + per-cat config from DB

Environment:
    DATABASE_URL     — Postgres connection string (see .env.example)
    EMBEDDING_MODEL  — optional; defaults to BAAI/bge-small-en-v1.5

Start:
    uvicorn main:app --host 0.0.0.0 --port 8000

Design notes:
    - A ThreadedConnectionPool in db.py feeds every request (no more global
      single connection).
    - Scoring logic lives in classifier.predict / predict_batch. main.py is
      only wiring: request validation, persistence, response shaping.
    - /reload mutates in-memory state for the current worker only. Multi-worker
      deployments need a rolling restart or a shared cache — see README.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import os

import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from classifier import (
    MODEL_NAME,
    UNCLASSIFIED_THRESHOLD_DEFAULT,
    embed_texts,
    get_metrics,
    load_category_config,
    load_centroids,
    load_chapter_titles,
    load_chapter_to_category,
    load_cross_encoder,
    load_keywords,
    predict,
    predict_batch,
)
from config import (
    MARGIN_DELTA_DEFAULT,
    RERANKER_ENABLED as CONFIG_RERANKER_ENABLED,
    RERANKER_MODEL  as CONFIG_RERANKER_MODEL,
)
from compliance import apply_compliance, load_risk_profile, prepare_risk_vectors
from db import close_pool, pooled_connection

# CCTR admin router — read-only registry browsing + discovery in Commit 2;
# governed write endpoints land in Commit 6.
from admin_routes import router as admin_router

# ── App setup ──────────────────────────────────────────────────────────────────

# In-memory state refreshed on /reload
centroids:             dict = {}
keywords:              dict = {}
category_config:       dict = {}
chapter_titles:        dict = {}
chapter_to_category:   dict = {}
category_descriptions: dict = {}
risk_vectors:          dict = {}

# Phase 3: cross-encoder reranker is opt-in. Default off → zero runtime cost
# and no model download at startup. Flip RERANKER_ENABLED=true to enable.
# Read from config.py (single source of truth for tunables).
RERANKER_ENABLED = CONFIG_RERANKER_ENABLED
RERANKER_MODEL   = CONFIG_RERANKER_MODEL


def _build_category_descriptions(keywords: dict[str, list[tuple[str, float]]]) -> dict[str, str]:
    """One short sentence per category, built from its top-weighted keywords.
    Used as the cross-encoder's class-side text. Rebuilt on /reload so prunes
    propagate to the reranker's view of each category.
    """
    out: dict[str, str] = {}
    for cat, kws in keywords.items():
        top = [kw for kw, _ in sorted(kws, key=lambda x: -x[1])[:10]]
        out[cat] = f"category: {cat} — commonly includes: " + ", ".join(top)
    return out


def _load_risk() -> dict:
    """Load risk profile JSON and embed all phrases into vectors."""
    profile = load_risk_profile()
    vectors = prepare_risk_vectors(profile, embed_texts)
    stats = vectors.get("_stats", {})
    print(
        f"Risk profile: {stats.get('global_entries', 0)} global blocked, "
        f"{stats.get('category_entries', 0)} category hard negatives, "
        f"{stats.get('safe_exception_entries', 0)} safe exceptions, "
        f"{stats.get('total_vectors', 0)} total vectors embedded"
    )
    return vectors


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load DB-backed state + risk vectors on startup,
    close the connection pool on shutdown."""
    global centroids, keywords, category_config, chapter_titles, chapter_to_category
    global category_descriptions, risk_vectors
    with pooled_connection() as conn:
        centroids           = load_centroids(conn)
        keywords            = load_keywords(conn)
        category_config     = load_category_config(conn)
        chapter_titles      = load_chapter_titles(conn)
        chapter_to_category = load_chapter_to_category(conn)
    category_descriptions = _build_category_descriptions(keywords)
    risk_vectors = _load_risk()
    if RERANKER_ENABLED:
        load_cross_encoder(RERANKER_MODEL)
        print(f"Cross-encoder reranker enabled: {RERANKER_MODEL}")
    print(
        f"Loaded {len(centroids)} category centroids "
        f"(chapter centroids total: {sum(len(v) for v in centroids.values())}), "
        f"{len(chapter_titles)} HS chapters, model={MODEL_NAME}"
    )
    yield
    close_pool()


app = FastAPI(
    title="Shipment Classifier",
    description="Multi-label NLP classification using sentence embeddings and pgvector.",
    version="1.2.0",
    lifespan=lifespan,
)

# /ui/ → the Vue SPA (web/ → built into static/dist/) when it exists,
# otherwise the legacy vanilla-JS bulk tester living in static/. The
# fallback keeps /ui/ usable while Commit 3 lands and before anyone
# runs `npm run build` on this checkout.
#
# Vue Router uses HTML5 history mode, so deep links like
# /ui/admin/tokens/battery must serve the SPA shell. Plain StaticFiles
# would 404 those (no file at that path) — Starlette mounts win over
# catch-all routes for any path under the mount, so we subclass
# StaticFiles to fall back to index.html on 404 instead. Real missing
# assets (.js, .css, .csv) keep returning 404 because the override
# only kicks in when path has no extension or matches the SPA shell.
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_DIST_DIR   = os.path.join(_STATIC_DIR, "dist")


class _SPAStaticFiles(StaticFiles):
    """StaticFiles that serves index.html on 404 for SPA deep links.

    Without this, ``GET /ui/admin/tokens/battery`` returns 404 because
    the path doesn't map to a file — even though it's a valid Vue
    Router route. We catch the 404, look at the request path, and:
      - return index.html if the path doesn't look like an asset
        (no file extension, or extension is .html);
      - re-raise the 404 otherwise so genuine missing assets still 404.
    """
    async def get_response(self, path: str, scope):
        from fastapi import HTTPException as _HTTPExc
        from starlette.exceptions import HTTPException as _StarHTTPExc
        try:
            return await super().get_response(path, scope)
        except (_HTTPExc, _StarHTTPExc) as exc:
            if exc.status_code != 404:
                raise
            # Treat any extension-less path (e.g. /admin/tokens/battery)
            # as an SPA route. Asset 404s (.js/.css/.json/.csv/...) are
            # left as-is so build mistakes surface clearly.
            tail = os.path.basename(path)
            if "." in tail:
                raise
            return await super().get_response("index.html", scope)


if os.path.isdir(_DIST_DIR):
    # Production: serve Vite build output. html=True maps /ui/ → index.html;
    # _SPAStaticFiles handles everything below it, including deep links.
    app.mount("/ui", _SPAStaticFiles(directory=_DIST_DIR, html=True), name="ui")
elif os.path.isdir(_STATIC_DIR):
    # Legacy fallback (pre-build). The old UI is a single page so the
    # SPA-fallback subclass would be no-op here, but using it keeps the
    # behaviour identical regardless of which mode the server boots in.
    app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")

# /admin/api/* — registry browsing, discovery (Commit 2), governed writes (Commit 6)
app.include_router(admin_router)


# ── Request / Response models ──────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    shipment_id:            str
    cargo_description:      str
    commodity_description:  str
    threshold:              float = Field(default=0.45, ge=0.0, le=1.0)
    unclassified_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description=(
            "Floor below which results are flagged `unclassified`. "
            f"Defaults to min({UNCLASSIFIED_THRESHOLD_DEFAULT}, threshold) so "
            "low user thresholds aren't overridden."
        ),
    )
    persist: bool = Field(
        default=False,
        description="If true, write the classification result to shipment_classifications.",
    )


class ClassifyBatchRequest(BaseModel):
    shipments: list[ClassifyRequest] = Field(..., max_length=500)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _persist_one(cur, shipment_id: str, result: dict, threshold: float) -> None:
    """INSERT one classification result. Caller manages transaction."""
    embedding = result.get("embedding")
    scores    = result.get("scores", {}) or {}

    scores_serializable = {
        cat: {k: v for k, v in s.items() if k != "embedding"}
        for cat, s in scores.items()
    }

    cur.execute("""
        INSERT INTO shipment_classifications
            (shipment_id, embedding, categories, scores, confidence_state,
             threshold_used, model_version, classified_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (shipment_id) DO UPDATE
            SET embedding        = EXCLUDED.embedding,
                categories       = EXCLUDED.categories,
                scores           = EXCLUDED.scores,
                confidence_state = EXCLUDED.confidence_state,
                threshold_used   = EXCLUDED.threshold_used,
                model_version    = EXCLUDED.model_version,
                classified_at    = now()
    """, (
        shipment_id,
        embedding.tolist() if embedding is not None else None,
        result["categories"],
        json.dumps(scores_serializable),
        result["confidence_state"],
        threshold,
        MODEL_NAME,
    ))


def _build_response(req: ClassifyRequest, result: dict, compliance: dict | None = None) -> dict:
    """Build the standard API response envelope.

    End-user lens for ``scores``:
      - ``classified``     → show only the categories that fired.
      - ``low_confidence`` → show the single top candidate we tentatively
                              assigned (matches what ``categories[]`` holds).
      - ``unclassified``   → show the top 3 candidates by ``final_score``
                              so the user sees what we considered and rejected,
                              not an empty dict.

    The full per-category breakdown (all 20 rows) is always written to the DB
    via ``_persist_one`` for offline audit and threshold tuning. The trim is
    purely an API-boundary concern.
    """
    all_scores = result.get("scores") or {}
    matched    = result.get("categories") or []

    if matched:
        # Show only the categories the model committed to.
        visible = [(c, all_scores[c]) for c in matched if c in all_scores]
    else:
        # Unclassified — surface the top 3 by final_score so the response
        # explains *why* nothing fired (the runners-up didn't clear the bar).
        visible = sorted(
            all_scores.items(),
            key=lambda kv: kv[1].get("final_score", 0.0),
            reverse=True,
        )[:3]

    scores_out = {
        cat: {k: v for k, v in s.items() if k != "embedding"}
        for cat, s in visible
    }

    response = {
        "shipment_id": req.shipment_id,
        "input": {
            "cargo_description":     req.cargo_description,
            "commodity_description": req.commodity_description,
        },
        "result": {
            "categories":       result["categories"],
            "confidence_state": result["confidence_state"],
            "threshold":        req.threshold,
            "scores":           scores_out,
        },
        "meta": {
            "model_version":    MODEL_NAME,
            "classified_at":    datetime.now(tz=timezone.utc).isoformat(),
            "total_categories": len(centroids),
            "evaluated":        len(centroids),
            # How many of the 20 categories the response actually surfaces.
            # Equals len(matched) when classified/low_confidence; <= 3 when
            # unclassified (top runners-up). Useful for caller dashboards.
            "categories_returned": len(scores_out),
            # 1 for short inputs (fast path), N for inputs that exceeded the
            # model's context window and were chunked. Surfaced for caller
            # observability — long-tail risk catches show chunks_processed > 1.
            "chunks_processed": result.get("chunks_processed", 1),
            # True when Phase 3 cross-encoder actually ran on this row (only
            # fires in `classified` state with 2+ candidates).
            "reranked": bool(result.get("reranked")),
        },
    }

    if result.get("reason"):
        response["result"]["reason"] = result["reason"]
    # Unified field name — classifier emits ``text_quality`` on every path now.
    # Fall back to the legacy ``quality`` key for any caller that pre-dates the rename.
    tq = result.get("text_quality") or result.get("quality")
    if tq:
        response["result"]["text_quality"] = tq

    # HS-code audit signal (structured, extracted before normalization). Surfaced
    # on every row — the UI compares it to `categories` to highlight mismatches.
    # Purely informational; does NOT influence `is_risky` or classifier scoring.
    response["result"]["hs_codes_extracted"]    = result.get("hs_codes_extracted") or []
    response["result"]["hs_implied_categories"] = result.get("hs_implied_categories") or []
    response["result"]["hs_text_mismatch"]      = bool(result.get("hs_text_mismatch"))

    # Compliance decision (semantic risk profile)
    if compliance:
        response["compliance"] = compliance

    return response


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/metrics")
def metrics() -> dict:
    """In-process counters for observability (single-worker semantics).

    Useful during tuning: ``margin_suppressions_total`` climbs when the
    top-margin rule is doing work; ``rerank_fires_total`` / ``rerank_drops_total``
    tell you whether the reranker is adding value. Reset by restart or
    by hitting /reload (which also rebuilds category_descriptions).
    """
    m = get_metrics()
    m["margin_delta"]     = MARGIN_DELTA_DEFAULT
    m["reranker_enabled"] = RERANKER_ENABLED
    return m


@app.get("/health")
def health() -> dict:
    stats = risk_vectors.get("_stats", {})
    return {
        "status":             "ok",
        "categories_loaded":  len(centroids),
        "chapter_centroids":  sum(len(v) for v in centroids.values()),
        "hs_chapters":        len(chapter_titles),
        "model_version":      MODEL_NAME,
        "calibrated":         any(
            c.get("platt_a") is not None for c in category_config.values()
        ),
        "risk_vectors":       stats.get("total_vectors", 0),
        "reranker_enabled":   RERANKER_ENABLED,
        "reranker_model":     RERANKER_MODEL if RERANKER_ENABLED else None,
    }


@app.post("/classify")
def classify(req: ClassifyRequest) -> dict:
    result = predict(
        req.cargo_description,
        req.commodity_description,
        centroids,
        keywords,
        category_config,
        threshold=req.threshold,
        unclassified_threshold=req.unclassified_threshold,
        chapter_titles=chapter_titles,
        chapter_to_category=chapter_to_category,
        category_descriptions=category_descriptions if RERANKER_ENABLED else None,
    )

    # Semantic compliance decision (uses the shipment embedding)
    compliance = apply_compliance(result, risk_vectors)

    if req.persist:
        try:
            with pooled_connection() as conn:
                with conn.cursor() as cur:
                    _persist_one(cur, req.shipment_id, result, req.threshold)
                conn.commit()
        except Exception as exc:
            print(f"WARNING: failed to persist {req.shipment_id}: {exc}")
            try:
                with pooled_connection() as conn:
                    conn.rollback()
            except Exception:
                pass

    return _build_response(req, result, compliance=compliance)


@app.post("/classify/batch")
def classify_batch(req: ClassifyBatchRequest) -> list[dict]:
    """
    Batched classification — single embedding call for all valid rows.
    """
    shipments = req.shipments
    inputs    = [(s.cargo_description, s.commodity_description) for s in shipments]
    thrs      = [s.threshold for s in shipments]
    unc_thrs  = [s.unclassified_threshold for s in shipments]

    results = predict_batch(
        inputs,
        centroids,
        keywords,
        category_config,
        thresholds=thrs,
        unclassified_thresholds=unc_thrs,
        chapter_titles=chapter_titles,
        chapter_to_category=chapter_to_category,
        category_descriptions=category_descriptions if RERANKER_ENABLED else None,
    )

    # Semantic compliance decisions (reuses each shipment's embedding)
    compliance_results = [
        apply_compliance(results[i], risk_vectors)
        for i in range(len(shipments))
    ]

    # Persist (single transaction for the whole batch).
    persist_indices = [i for i, s in enumerate(shipments) if s.persist]
    if persist_indices:
        try:
            with pooled_connection() as conn:
                with conn.cursor() as cur:
                    for i in persist_indices:
                        _persist_one(cur, shipments[i].shipment_id, results[i], shipments[i].threshold)
                conn.commit()
        except Exception as exc:
            print(f"WARNING: batch persist failed ({exc}); rolling back.")
            try:
                with pooled_connection() as conn:
                    conn.rollback()
            except Exception:
                pass

    return [
        _build_response(shipments[i], results[i], compliance=compliance_results[i])
        for i in range(len(shipments))
    ]


@app.post("/reload")
def reload() -> dict:
    """Reload centroids, keywords, chapter titles, HS→category map, per-category config, and risk vectors."""
    global centroids, keywords, category_config, chapter_titles, chapter_to_category
    global category_descriptions, risk_vectors
    with pooled_connection() as conn:
        centroids           = load_centroids(conn)
        keywords            = load_keywords(conn)
        category_config     = load_category_config(conn)
        chapter_titles      = load_chapter_titles(conn)
        chapter_to_category = load_chapter_to_category(conn)
    category_descriptions = _build_category_descriptions(keywords)
    risk_vectors = _load_risk()
    stats = risk_vectors.get("_stats", {})
    return {
        "status":             "reloaded",
        "categories_loaded":  len(centroids),
        "chapter_centroids":  sum(len(v) for v in centroids.values()),
        "hs_chapters":        len(chapter_titles),
        "calibrated":         any(
            c.get("platt_a") is not None for c in category_config.values()
        ),
        "risk_profile": {
            "global_entries":         stats.get("global_entries", 0),
            "category_entries":       stats.get("category_entries", 0),
            "safe_exception_entries": stats.get("safe_exception_entries", 0),
            "total_vectors":          stats.get("total_vectors", 0),
        },
    }


# SPA deep-link fallback is handled inside _SPAStaticFiles above so that
# the StaticFiles mount itself returns index.html on 404 for routes like
# /ui/admin/tokens/battery. Doing it inside the mount avoids the route
# vs. mount precedence issue (mounts always win over catch-all routes
# under the same prefix in Starlette).
