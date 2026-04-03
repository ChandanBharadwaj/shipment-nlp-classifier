"""
FastAPI classification service.

Endpoints:
    GET  /health           — liveness check
    POST /classify         — classify a single shipment
    POST /classify/batch   — classify up to 500 shipments (optimized batch encoding)
    POST /reload           — reload centroids + keywords from DB without restart

Environment:
    DATABASE_URL — see .env.example

Start:
    uvicorn main:app --host 0.0.0.0 --port 8000

TODO (production):
    - Replace single psycopg2 connection with ThreadedConnectionPool or asyncpg.
    - /reload updates in-memory state for single-worker only. Multi-worker
      deployments need a shared cache (Redis) or a full service restart.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from classifier import (
    MODEL_NAME,
    KEYWORD_WEIGHT,
    keyword_score,
    load_centroids,
    load_keywords,
    model,
    normalize,
    predict,
    text_quality_check,
    UNCLASSIFIED_THRESHOLD,
)
from db import get_connection

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Shipment Classifier",
    description="Multi-label NLP classification using sentence embeddings and pgvector.",
    version="1.0.0",
)

conn: psycopg2.extensions.connection = None   # type: ignore[assignment]
centroids: dict = {}
keywords:  dict = {}


@app.on_event("startup")
def startup() -> None:
    global conn, centroids, keywords
    conn      = get_connection()
    centroids = load_centroids(conn)
    keywords  = load_keywords(conn)
    print(f"Loaded {len(centroids)} category centroids.")


# ── Request / Response models ──────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    shipment_id:           str
    cargo_description:     str
    commodity_description: str
    threshold:             float = Field(default=0.45, ge=0.0, le=1.0)
    persist:               bool  = Field(
        default=False,
        description="If true, write the classification result to shipment_classifications.",
    )


class ClassifyBatchRequest(BaseModel):
    shipments: list[ClassifyRequest] = Field(..., max_length=500)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _persist_result(shipment_id: str, result: dict, threshold: float) -> None:
    """Write a classification result to shipment_classifications."""
    import json
    embedding = result.get("embedding")
    scores    = result.get("scores", {})

    # Remove embedding key from scores (it's stored separately)
    scores_serializable = {
        cat: {k: v for k, v in s.items() if k != "embedding"}
        for cat, s in scores.items()
    }

    with conn.cursor() as cur:
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
    conn.commit()


def _build_response(req: ClassifyRequest, result: dict) -> dict:
    """Build the standard API response envelope."""
    # Strip embedding from scores (internal, not sent to caller)
    scores_out = {
        cat: {k: v for k, v in s.items() if k != "embedding"}
        for cat, s in result.get("scores", {}).items()
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

    return response


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {
        "status":           "ok",
        "categories_loaded": len(centroids),
        "model_version":    MODEL_NAME,
    }


@app.post("/classify")
def classify(req: ClassifyRequest) -> dict:
    result = predict(
        req.cargo_description,
        req.commodity_description,
        centroids,
        keywords,
        req.threshold,
    )

    if req.persist:
        try:
            _persist_result(req.shipment_id, result, req.threshold)
        except Exception as exc:
            # Don't let a DB write failure break the classification response
            print(f"WARNING: failed to persist {req.shipment_id}: {exc}")

    return _build_response(req, result)


@app.post("/classify/batch")
def classify_batch(req: ClassifyBatchRequest) -> list[dict]:
    """
    Optimized batch endpoint.

    Collects all texts and calls model.encode() once — significantly faster
    than calling predict() per shipment, which would embed one text at a time.
    """
    shipments = req.shipments

    # Separate valid vs. quality-filtered shipments up front
    valid_indices: list[int] = []
    results: list[dict | None] = [None] * len(shipments)

    texts: list[str] = []
    for i, s in enumerate(shipments):
        text    = f"{s.cargo_description} {s.commodity_description}".strip()
        quality = text_quality_check(text)
        if quality != "ok":
            results[i] = {
                "categories":       [],
                "confidence_state": "unclassified",
                "reason":           "insufficient_input",
                "scores":           {},
                "embedding":        None,
            }
        else:
            valid_indices.append(i)
            texts.append(text)

    # Batch-encode all valid texts in a single call
    if texts:
        embeddings = normalize(model.encode(texts, batch_size=512, show_progress_bar=False))

        for offset, idx in enumerate(valid_indices):
            s         = shipments[idx]
            embedding = embeddings[offset]

            scores: dict = {}
            max_score = 0.0
            for category, centroid in centroids.items():
                sem        = round(float(np.dot(embedding, centroid)), 4)
                kw_s, hits = keyword_score(
                    f"{s.cargo_description} {s.commodity_description}",
                    keywords.get(category, []),
                )
                final = round((1 - KEYWORD_WEIGHT) * sem + KEYWORD_WEIGHT * kw_s, 4)
                scores[category] = {
                    "semantic_score": sem,
                    "keyword_score":  kw_s,
                    "final_score":    final,
                    "matched":        False,
                    "keywords_hit":   hits,
                }
                max_score = max(max_score, final)

            if max_score < UNCLASSIFIED_THRESHOLD:
                confidence_state = "unclassified"
                reason           = "low_similarity"
                matched_cats: list[str] = []
            elif max_score < s.threshold:
                confidence_state = "low_confidence"
                reason           = None
                top_cat          = max(scores, key=lambda c: scores[c]["final_score"])
                matched_cats     = [top_cat]
                scores[top_cat]["matched"] = True
            else:
                confidence_state = "classified"
                reason           = None
                matched_cats     = [c for c, sc in scores.items() if sc["final_score"] >= s.threshold]
                for c in matched_cats:
                    scores[c]["matched"] = True

            results[idx] = {
                "categories":       matched_cats,
                "confidence_state": confidence_state,
                "reason":           reason,
                "scores":           scores,
                "embedding":        embedding,
            }

    # Persist and build responses
    responses = []
    for i, s in enumerate(shipments):
        result = results[i]
        if s.persist:
            try:
                _persist_result(s.shipment_id, result, s.threshold)
            except Exception as exc:
                print(f"WARNING: failed to persist {s.shipment_id}: {exc}")
        responses.append(_build_response(s, result))

    return responses


@app.post("/reload")
def reload() -> dict:
    """Reload centroids and keywords from DB without restarting the service."""
    global centroids, keywords
    new_centroids = load_centroids(conn)
    new_keywords  = load_keywords(conn)
    centroids = new_centroids
    keywords  = new_keywords
    return {
        "status":            "reloaded",
        "categories_loaded": len(centroids),
    }
