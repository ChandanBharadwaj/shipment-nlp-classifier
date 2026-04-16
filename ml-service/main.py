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

import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from classifier import (
    MODEL_NAME,
    UNCLASSIFIED_THRESHOLD_DEFAULT,
    embed_texts,
    load_category_config,
    load_centroids,
    load_chapter_titles,
    load_keywords,
    predict,
    predict_batch,
)
from compliance import apply_compliance, load_risk_profile, prepare_risk_vectors
from db import close_pool, pooled_connection

# ── App setup ──────────────────────────────────────────────────────────────────

# In-memory state refreshed on /reload
centroids:       dict = {}
keywords:        dict = {}
category_config: dict = {}
chapter_titles:  dict = {}
risk_vectors:    dict = {}


def _load_risk() -> dict:
    """Load risk profile JSON and embed all phrases into vectors."""
    profile = load_risk_profile()
    vectors = prepare_risk_vectors(profile, embed_texts)
    stats = vectors.get("_stats", {})
    print(
        f"Risk profile: {stats.get('global_entries', 0)} global blocked, "
        f"{stats.get('category_entries', 0)} category hard negatives, "
        f"{stats.get('total_vectors', 0)} total vectors embedded"
    )
    return vectors


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load DB-backed state + risk vectors on startup,
    close the connection pool on shutdown."""
    global centroids, keywords, category_config, chapter_titles, risk_vectors
    with pooled_connection() as conn:
        centroids       = load_centroids(conn)
        keywords        = load_keywords(conn)
        category_config = load_category_config(conn)
        chapter_titles  = load_chapter_titles(conn)
    risk_vectors = _load_risk()
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
    """Build the standard API response envelope."""
    scores_out = {
        cat: {k: v for k, v in s.items() if k != "embedding"}
        for cat, s in (result.get("scores") or {}).items()
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
            # 1 for short inputs (fast path), N for inputs that exceeded the
            # model's context window and were chunked. Surfaced for caller
            # observability — long-tail risk catches show chunks_processed > 1.
            "chunks_processed": result.get("chunks_processed", 1),
        },
    }

    if result.get("reason"):
        response["result"]["reason"] = result["reason"]
    if result.get("quality"):
        response["result"]["quality"] = result["quality"]

    # Compliance decision (semantic risk profile)
    if compliance:
        response["compliance"] = compliance

    return response


# ── Endpoints ──────────────────────────────────────────────────────────────────

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
    """Reload centroids, keywords, chapter titles, per-category config, and risk vectors."""
    global centroids, keywords, category_config, chapter_titles, risk_vectors
    with pooled_connection() as conn:
        centroids       = load_centroids(conn)
        keywords        = load_keywords(conn)
        category_config = load_category_config(conn)
        chapter_titles  = load_chapter_titles(conn)
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
            "global_entries":    stats.get("global_entries", 0),
            "category_entries":  stats.get("category_entries", 0),
            "total_vectors":     stats.get("total_vectors", 0),
        },
    }
