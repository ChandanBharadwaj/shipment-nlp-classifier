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
from datetime import datetime, timezone
from typing import Optional

import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from classifier import (
    MODEL_NAME,
    UNCLASSIFIED_THRESHOLD_DEFAULT,
    load_category_config,
    load_centroids,
    load_keywords,
    predict,
    predict_batch,
)
from db import close_pool, pooled_connection

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Shipment Classifier",
    description="Multi-label NLP classification using sentence embeddings and pgvector.",
    version="1.1.0",
)

# In-memory state refreshed on /reload
centroids:       dict = {}
keywords:        dict = {}
category_config: dict = {}


@app.on_event("startup")
def startup() -> None:
    global centroids, keywords, category_config
    with pooled_connection() as conn:
        centroids       = load_centroids(conn)
        keywords        = load_keywords(conn)
        category_config = load_category_config(conn)
    print(
        f"Loaded {len(centroids)} category centroids "
        f"(sub-centroids total: {sum(len(v) for v in centroids.values())}), "
        f"model={MODEL_NAME}"
    )


@app.on_event("shutdown")
def shutdown() -> None:
    close_pool()


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


def _build_response(req: ClassifyRequest, result: dict) -> dict:
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
        },
    }

    if result.get("reason"):
        response["result"]["reason"] = result["reason"]
    if result.get("quality"):
        response["result"]["quality"] = result["quality"]

    return response


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {
        "status":             "ok",
        "categories_loaded":  len(centroids),
        "sub_centroids":      sum(len(v) for v in centroids.values()),
        "model_version":      MODEL_NAME,
        "calibrated":         any(
            c.get("platt_a") is not None for c in category_config.values()
        ),
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
    )

    if req.persist:
        try:
            with pooled_connection() as conn:
                with conn.cursor() as cur:
                    _persist_one(cur, req.shipment_id, result, req.threshold)
                conn.commit()
        except Exception as exc:
            # Don't let a DB write failure break the classification response.
            # The pooled_connection context ensures the connection returns to
            # the pool; rollback is implicit on context exit for failed txns.
            print(f"WARNING: failed to persist {req.shipment_id}: {exc}")
            try:
                with pooled_connection() as conn:
                    conn.rollback()
            except Exception:
                pass

    return _build_response(req, result)


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
    )

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

    return [_build_response(shipments[i], results[i]) for i in range(len(shipments))]


@app.post("/reload")
def reload() -> dict:
    """Reload centroids, keywords, and per-category config from DB."""
    global centroids, keywords, category_config
    with pooled_connection() as conn:
        centroids       = load_centroids(conn)
        keywords        = load_keywords(conn)
        category_config = load_category_config(conn)
    return {
        "status":             "reloaded",
        "categories_loaded":  len(centroids),
        "sub_centroids":      sum(len(v) for v in centroids.values()),
        "calibrated":         any(
            c.get("platt_a") is not None for c in category_config.values()
        ),
    }
