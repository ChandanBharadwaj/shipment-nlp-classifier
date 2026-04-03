# Shipment Classification System

Multi-label NLP classification using sentence embeddings and pgvector.
Given free-text shipment fields, automatically assign one or more categories from a fixed set of 20.

---

## Problem Statement

- Free-text fields: `cargo_document_description`, `commodity_description`
- 20 known categories — a shipment can belong to **2 or more simultaneously**
- 5M shipment records in PostgreSQL
- Existing labeled data available (keyword-match based)

---

## How It Works — Connecting the Dots

Before the schema and code, it is important to understand what the system is actually doing — especially how the labeled data helps.

### Step 1 — What an Embedding Is

An embedding model reads text and outputs a list of numbers — a coordinate in space. The key property is that **similar meaning = similar coordinates**. The model was trained on billions of sentences and learned that "PCB" and "semiconductors" live near each other in space, and far from "vegetables".

```
"Electronic PCB boards"     →  [0.12, -0.45, 0.78, 0.33, ...]  (384 numbers)
"Microchips semiconductors" →  [0.15, -0.41, 0.80, 0.29, ...]  (nearby)
"Fresh vegetables lettuce"  →  [-0.60, 0.72, -0.11, 0.55, ...]  (far away)
```

### Step 2 — How Labeled Data Shapes the Centroid

Your labeled data contains shipments already tagged with a category (by keyword rules or manually). Each labeled shipment's combined text gets embedded into a coordinate. Because shipments in the same category use similar language, their coordinates **cluster near each other in space**.

```
        semantic space (simplified to 2D)

    ^
    │                    ● ship_003 (lithium battery)
    │               ● ship_001 (PCB, semiconductors)
    │                  ● ship_002 (LED, HDMI)
    │
    │   ● ship_005 (frozen prawns)
    │      ● ship_004 (lettuce, tomatoes)
    │
    └──────────────────────────────────>
```

The **centroid** is the center of gravity of all labeled shipments in a category — the average of all their coordinates. It represents "what a typical electronics shipment looks like in semantic space".

```
ship_001 → [0.12, -0.45, 0.78, ...]
ship_002 → [0.15, -0.41, 0.80, ...]
ship_003 → [0.10, -0.48, 0.75, ...]
                  ─────────────────
centroid → [0.12, -0.44, 0.77, ...]   ← average = center of the cluster
```

After building centroids for all 20 categories you have **one coordinate per category** — your entire trained model. These are stored in `category_centroids`.

### Step 3 — Classifying a New Shipment

A new shipment arrives. Its text is embedded into the same space. You measure the distance from the new point to every centroid. The closest centroids (above a threshold) become the assigned categories.

```
        semantic space

    ^
    │            ● ship_003
    │         ★  centroid (electronics)
    │            ● ship_001
    │     ✦  NEW SHIPMENT  ← lands near electronics centroid
    │               ● ship_002
    │
    │   ★  centroid (perishables)
    │
    └──────────────────────────────────>

distance to electronics centroid  →  score 0.87  ✅  above threshold
distance to perishables centroid  →  score 0.21  ❌  below threshold
```

### Step 4 — Multi-Label Case

When a shipment contains mixed cargo, its embedding lands **between** two category clusters — close enough to both centroids to clear the threshold for both.

```
cargo:     "CPU wafers and frozen prawns"
commodity: "semiconductors seafood cold chain"

distance to electronics centroid  →  0.72  ✅
distance to perishables centroid  →  0.68  ✅
distance to chemicals centroid    →  0.15  ❌

→ assigned: ["electronics", "perishables"]
```

### Why Keyword-Based Labels Still Work

Your existing labels came from keyword rules (`IF content contains "PCB" THEN electronics`). That is fine. The label tells the system "this shipment is electronics". The system then embeds the **full content** of that shipment — not just the keyword — so the centroid captures the broader language and context around electronics, not just the trigger words.

```
keyword triggered on:  "PCB"
full content embedded: "Electronic PCB boards used in industrial control systems"
                        ↑ this richer text is what shapes the centroid
```

More labeled shipments = more votes on where the centroid sits = more stable and reliable classification.

---

## Configurable Categories

Categories and their keywords are managed in the database — not hardcoded. This allows adding, deactivating, or updating categories without code changes. Centroids are always computed by the offline job and never set manually.

### Lifecycle

```
classification_categories    →  add / deactivate categories anytime
        │
category_keywords            →  add / edit keywords anytime
        │
        ▼
centroid_builder.py runs     →  reads only active categories
        │                        rebuilds centroids from labeled shipments
        ▼
category_centroids           →  one row per active category (computed, not manual)
        │
        ▼
POST /reload                 →  service reloads active categories + centroids + keywords
        │
        ▼
/classify                    →  only active categories are evaluated
```

A category can exist with keywords before it has a centroid — useful when gathering labeled examples for a new category before it goes live.

---

## Database Schema

### `classification_categories`

Master list of categories. Decoupled from centroids so a category can exist before its centroid is built.

```sql
CREATE TABLE classification_categories (
    id            SERIAL PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,   -- internal key e.g. 'electronics'
    display_name  TEXT NOT NULL,          -- human label e.g. 'Electronics & Components'
    description   TEXT,                   -- helps reviewers understand the category
    is_active     BOOLEAN DEFAULT true,   -- deactivate without deleting
    created_at    TIMESTAMPTZ DEFAULT now(),
    updated_at    TIMESTAMPTZ DEFAULT now()
);
```

### `category_keywords`

Keywords per category. Used as a secondary confidence signal at inference time alongside the semantic score.

```sql
CREATE TABLE category_keywords (
    id           SERIAL PRIMARY KEY,
    category_id  INT NOT NULL REFERENCES classification_categories(id),
    keyword      TEXT NOT NULL,
    weight       NUMERIC(3,2) DEFAULT 1.0,  -- stronger keywords can have higher weight
    created_at   TIMESTAMPTZ DEFAULT now(),

    UNIQUE (category_id, keyword)
);

CREATE INDEX idx_kw_category ON category_keywords(category_id);
```

### `category_centroids`

One centroid vector per active category. Always computed by the offline job — never inserted manually.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE category_centroids (
    category_id   INT PRIMARY KEY REFERENCES classification_categories(id),
    centroid      vector(384) NOT NULL,
    sample_count  INT,           -- number of shipments used to build this centroid
    updated_at    TIMESTAMPTZ DEFAULT now()
);
```

### `shipment_classifications`

Stores the full classification result per shipment — assigned categories, per-category score breakdown, and the embedding used.

```sql
CREATE TABLE shipment_classifications (
    shipment_id       TEXT PRIMARY KEY,   -- FK to your existing shipments table
    embedding         vector(384),        -- embedding of combined text at classification time
    categories        TEXT[],             -- final assigned categories e.g. ARRAY['electronics','perishables']
    scores            JSONB,              -- full per-category breakdown, see structure below
    confidence_state  TEXT DEFAULT 'classified',  -- 'classified' | 'low_confidence' | 'unclassified'
    threshold_used    NUMERIC(4,3),
    model_version     TEXT,               -- track which embedding model was used
    classified_at     TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_class_categories ON shipment_classifications USING GIN(categories);
CREATE INDEX idx_class_confidence ON shipment_classifications(confidence_state);
CREATE INDEX idx_class_embedding  ON shipment_classifications
             USING hnsw(embedding vector_cosine_ops);
```

---

## Classification Output Format

### `scores` JSONB structure

The `scores` column stores the full breakdown for every category evaluated — not just the matched ones. This makes every decision explainable.

```json
{
  "electronics": {
    "semantic_score": 0.87,
    "keyword_score":  0.60,
    "final_score":    0.83,
    "matched":        true,
    "keywords_hit":   ["PCB", "semiconductors"]
  },
  "perishables": {
    "semantic_score": 0.76,
    "keyword_score":  0.40,
    "final_score":    0.71,
    "matched":        true,
    "keywords_hit":   ["cold chain"]
  },
  "chemicals": {
    "semantic_score": 0.21,
    "keyword_score":  0.00,
    "final_score":    0.21,
    "matched":        false,
    "keywords_hit":   []
  }
}
```

- `semantic_score` — cosine similarity between the shipment embedding and the category centroid
- `keyword_score` — fraction of the category's known keywords found in the shipment text
- `final_score` — weighted blend: `0.8 × semantic_score + 0.2 × keyword_score`
- `matched` — whether `final_score ≥ threshold`
- `keywords_hit` — exactly which keywords matched, for explainability

### API response from `POST /classify`

```json
{
  "shipment_id": "ship_001",
  "input": {
    "cargo_description":     "Electronic PCB boards and fresh prawns",
    "commodity_description": "semiconductors cold chain seafood"
  },
  "result": {
    "categories":       ["electronics", "perishables"],
    "confidence_state": "classified",
    "threshold":        0.45,
    "scores": {
      "electronics": {
        "semantic_score": 0.87,
        "keyword_score":  0.60,
        "final_score":    0.83,
        "keywords_hit":   ["PCB", "semiconductors"]
      },
      "perishables": {
        "semantic_score": 0.76,
        "keyword_score":  0.40,
        "final_score":    0.71,
        "keywords_hit":   ["cold chain"]
      }
    }
  },
  "meta": {
    "model_version":    "all-MiniLM-L6-v2",
    "classified_at":    "2026-04-03T10:22:00Z",
    "total_categories": 20,
    "evaluated":        20
  }
}
```

---

## Handling Unknown and Ambiguous Shipments

There are three distinct scenarios where a shipment cannot be confidently classified.

### Scenario 1 — Vague or too-short text

Some shipments have generic descriptions that carry no useful signal. These are detected **before embedding** to avoid wasting compute and silently misclassifying.

```python
GENERIC_PHRASES = {
    "cargo", "goods", "items", "as per invoice",
    "general merchandise", "various", "misc", "miscellaneous"
}

def text_quality_check(text: str) -> str:
    stripped = text.strip().lower()
    if len(stripped) < 10:
        return "too_short"
    if stripped in GENERIC_PHRASES:
        return "generic"
    return "ok"
```

If quality check fails → `confidence_state = 'unclassified'`, `reason = 'insufficient_input'`, skip embedding entirely.

### Scenario 2 — Low similarity across all categories

The text is meaningful but does not resemble any known category. All 20 cosine scores come back low.

```
"Antique furniture restoration parts"  →  all scores < 0.35
```

This is handled by two threshold bands:

```
max_score < 0.35          →  confidence_state = 'unclassified'
                               categories = []
                               reason = 'low_similarity'

max_score 0.35 – 0.45     →  confidence_state = 'low_confidence'
                               categories = [top-1 only]
                               assigned but flagged for review

max_score ≥ 0.45          →  confidence_state = 'classified'
                               categories = [all above threshold]
```

### Scenario 3 — A genuinely new category emerging

Over time new shipment types appear that don't fit any existing category. The model silently assigns the nearest category — which will be consistently wrong for a whole group of shipments.

Detection: run HDBSCAN periodically on the embeddings of all `unclassified` shipments. Any cluster that forms is a candidate new category.

```python
import hdbscan

def find_unknown_clusters(embeddings: np.ndarray):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, metric='euclidean')
    labels    = clusterer.fit_predict(embeddings)
    # label == -1  →  noise / true one-offs
    # label >= 0   →  candidate new category, surface to analyst
    return labels
```

When an analyst confirms a new category: add to `classification_categories`, label those shipments, re-run `centroid_builder.py`, call `/reload`.

### Full Decision Tree at Inference

```
New shipment arrives
        │
        ▼
Text quality check
        ├── too_short / generic
        │       └──→  unclassified (insufficient_input) → review queue
        └── ok
                │
                ▼
            Embed + cosine sim × N centroids + keyword boost
                │
                ├── max_score < 0.35
                │       └──→  unclassified (low_similarity) → review queue
                │
                ├── max_score 0.35 – 0.45
                │       └──→  low_confidence → top-1 assigned, flagged
                │
                └── max_score ≥ 0.45
                        └──→  classified → all categories above threshold assigned
```

Unclassified shipments are not failures — they are the feedback loop. They surface where the category set is incomplete and where labeled data is weak.

---

## Components

### 1. Offline Centroid Builder `centroid_builder.py`

Runs once and then on a schedule (e.g. weekly). Samples labeled shipments for active categories only, builds one centroid per category, and persists to `category_centroids`.

**Steps:**

1. Query active categories from `classification_categories`
2. For each category, sample up to 2,000 labeled shipments — combine their content columns into one string per shipment
3. Embed in batches of 512 using `all-MiniLM-L6-v2`
4. Compute incremental mean per category (memory-safe for 5M rows)
5. L2-normalise the centroid
6. UPSERT into `category_centroids`

**Incremental mean:**

```python
class IncrementalCentroid:
    def __init__(self):
        self.sum   = None
        self.count = 0

    def update(self, batch_embeddings: np.ndarray):
        self.sum   = batch_embeddings.sum(axis=0) if self.sum is None \
                     else self.sum + batch_embeddings.sum(axis=0)
        self.count += len(batch_embeddings)

    def centroid(self) -> np.ndarray:
        c = self.sum / self.count
        return c / np.linalg.norm(c)
```

**Full builder:**

```python
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from collections import defaultdict
import numpy as np, psycopg2
from pgvector.psycopg2 import register_vector

BATCH_SIZE          = 512
SAMPLE_PER_CATEGORY = 2000
MODEL_NAME          = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

def build_and_persist(conn):
    register_vector(conn)
    accumulators = defaultdict(IncrementalCentroid)

    cur = conn.cursor(name="centroid_cursor")   # server-side cursor — streams rows
    cur.execute(SAMPLING_QUERY)                 # adapt to your schema

    while True:
        rows = cur.fetchmany(BATCH_SIZE)
        if not rows:
            break

        categories = [r[0] for r in rows]
        texts      = [r[2] for r in rows]
        embeddings = normalize(model.encode(texts, batch_size=BATCH_SIZE))

        for category, emb in zip(categories, embeddings):
            accumulators[category].update(emb.reshape(1, -1))

    with conn.cursor() as wc:
        for category, acc in accumulators.items():
            centroid = acc.centroid()
            wc.execute("""
                INSERT INTO category_centroids (category_id, centroid, sample_count, updated_at)
                SELECT id, %s, %s, now()
                FROM   classification_categories
                WHERE  name = %s
                ON CONFLICT (category_id) DO UPDATE
                  SET centroid     = EXCLUDED.centroid,
                      sample_count = EXCLUDED.sample_count,
                      updated_at   = now()
            """, (centroid.tolist(), acc.count, category))

    conn.commit()
    print(f"Built centroids for {len(accumulators)} categories.")
```

---

### 2. Classification Service `classifier.py`

Loads centroids and keywords from DB once on startup. At inference, blends semantic score and keyword score.

```python
import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from collections import defaultdict

MODEL_NAME     = "all-MiniLM-L6-v2"
model          = SentenceTransformer(MODEL_NAME)
KEYWORD_WEIGHT = 0.2   # 80% semantic, 20% keyword boost

def load_centroids(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT cc.name, cen.centroid
            FROM   category_centroids cen
            JOIN   classification_categories cc ON cc.id = cen.category_id
            WHERE  cc.is_active = true
        """)
        return {row[0]: np.array(row[1]) for row in cur.fetchall()}

def load_keywords(conn) -> dict:
    keywords = defaultdict(list)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT cc.name, ck.keyword, ck.weight
            FROM   category_keywords ck
            JOIN   classification_categories cc ON cc.id = ck.category_id
            WHERE  cc.is_active = true
        """)
        for name, keyword, weight in cur.fetchall():
            keywords[name].append((keyword.lower(), float(weight)))
    return keywords

def keyword_score(text: str, keywords: list[tuple]) -> tuple[float, list[str]]:
    text_lower = text.lower()
    hits  = [kw for kw, _ in keywords if kw in text_lower]
    score = len(hits) / len(keywords) if keywords else 0.0
    return round(score, 3), hits

GENERIC_PHRASES = {
    "cargo", "goods", "items", "as per invoice",
    "general merchandise", "various", "misc", "miscellaneous"
}

def text_quality_check(text: str) -> str:
    stripped = text.strip().lower()
    if len(stripped) < 10:
        return "too_short"
    if stripped in GENERIC_PHRASES:
        return "generic"
    return "ok"

def predict(
    cargo_description: str,
    commodity_description: str,
    centroids: dict,
    keywords: dict,
    threshold: float = 0.45
) -> dict:

    text    = f"{cargo_description} {commodity_description}".strip()
    quality = text_quality_check(text)

    if quality != "ok":
        return {
            "categories":       [],
            "confidence_state": "unclassified",
            "reason":           "insufficient_input",
            "scores":           {}
        }

    embedding = normalize(model.encode([text]))[0]
    scores    = {}
    max_score = 0.0

    for category, centroid in centroids.items():
        sem        = round(float(np.dot(embedding, centroid)), 3)
        kw_s, hits = keyword_score(text, keywords.get(category, []))
        final      = round((1 - KEYWORD_WEIGHT) * sem + KEYWORD_WEIGHT * kw_s, 3)

        scores[category] = {
            "semantic_score": sem,
            "keyword_score":  kw_s,
            "final_score":    final,
            "matched":        False,
            "keywords_hit":   hits
        }
        max_score = max(max_score, final)

    if max_score < 0.35:
        confidence_state = "unclassified"
        reason           = "low_similarity"
        matched_cats     = []
    elif max_score < threshold:
        confidence_state = "low_confidence"
        reason           = None
        top_cat          = max(scores, key=lambda c: scores[c]["final_score"])
        matched_cats     = [top_cat]
        scores[top_cat]["matched"] = True
    else:
        confidence_state = "classified"
        reason           = None
        matched_cats     = [c for c, s in scores.items() if s["final_score"] >= threshold]
        for c in matched_cats:
            scores[c]["matched"] = True

    return {
        "categories":       matched_cats,
        "confidence_state": confidence_state,
        "reason":           reason,
        "scores":           scores
    }
```

---

### 3. FastAPI Service `main.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
import psycopg2, os
from classifier import load_centroids, load_keywords, predict, MODEL_NAME

app       = FastAPI()
conn      = psycopg2.connect(os.environ["DATABASE_URL"])
centroids = load_centroids(conn)
keywords  = load_keywords(conn)

class ClassifyRequest(BaseModel):
    shipment_id:           str
    cargo_description:     str
    commodity_description: str
    threshold:             float = 0.45

class ClassifyBatchRequest(BaseModel):
    shipments: list[ClassifyRequest]

@app.post("/classify")
def classify(req: ClassifyRequest):
    result = predict(req.cargo_description, req.commodity_description,
                     centroids, keywords, req.threshold)
    return {
        "shipment_id": req.shipment_id,
        "input": {
            "cargo_description":     req.cargo_description,
            "commodity_description": req.commodity_description
        },
        "result": {**result, "threshold": req.threshold},
        "meta": {
            "model_version":    MODEL_NAME,
            "total_categories": len(centroids),
            "evaluated":        len(centroids)
        }
    }

@app.post("/classify/batch")
def classify_batch(req: ClassifyBatchRequest):
    return [classify(s) for s in req.shipments]

@app.post("/reload")
def reload():
    global centroids, keywords
    centroids = load_centroids(conn)
    keywords  = load_keywords(conn)
    return {"categories_loaded": len(centroids)}

@app.get("/health")
def health():
    return {"status": "ok", "categories_loaded": len(centroids)}
```

---

### 4. Threshold Tuning `evaluate_threshold.py`

```python
import numpy as np
from classifier import load_centroids, load_keywords, predict

def evaluate(test_samples: list[dict], centroids: dict, keywords: dict, threshold: float):
    """
    test_samples: [
        {"cargo": "...", "commodity": "...", "expected": ["electronics", "perishables"]}
    ]
    """
    precision_list, recall_list = [], []

    for s in test_samples:
        result    = predict(s["cargo"], s["commodity"], centroids, keywords, threshold)
        predicted = set(result["categories"])
        expected  = set(s["expected"])
        tp = len(predicted & expected)
        precision_list.append(tp / len(predicted) if predicted else 0)
        recall_list.append(tp / len(expected)    if expected  else 0)

    print(f"threshold={threshold:.2f}  "
          f"precision={np.mean(precision_list):.3f}  "
          f"recall={np.mean(recall_list):.3f}")

centroids = load_centroids(conn)
keywords  = load_keywords(conn)
for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    evaluate(test_samples, centroids, keywords, t)
```

Pick the threshold where precision ≥ 0.80 and recall is acceptable for your use case.

---

### 5. Unknown Cluster Discovery `discover_unknowns.py`

Run periodically on `unclassified` shipments to surface candidate new categories.

```python
import hdbscan, numpy as np

def find_unknown_clusters(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT shipment_id, embedding
            FROM   shipment_classifications
            WHERE  confidence_state = 'unclassified'
        """)
        rows = cur.fetchall()

    if len(rows) < 50:
        print("Not enough unclassified shipments to cluster.")
        return

    ids        = [r[0] for r in rows]
    embeddings = np.array([r[1] for r in rows])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, metric='euclidean')
    labels    = clusterer.fit_predict(embeddings)

    for label in set(labels):
        if label == -1:
            continue   # noise — true one-offs
        cluster_ids = [ids[i] for i, l in enumerate(labels) if l == label]
        print(f"Candidate new category — cluster {label}: {len(cluster_ids)} shipments")
        print(f"  Sample IDs: {cluster_ids[:5]}")
```

When analyst confirms a new category: insert into `classification_categories`, label those shipments in your existing labeled data table, then re-run `centroid_builder.py` and call `POST /reload`.

---

## Project Structure

```
ml-service/
├── main.py                  # FastAPI app
├── classifier.py            # predict(), load_centroids(), load_keywords()
├── centroid_builder.py      # Offline job: sample → embed → upsert
├── evaluate_threshold.py    # Threshold sweep
├── discover_unknowns.py     # HDBSCAN on unclassified shipments
├── db.py                    # psycopg2 + pgvector connection helper
└── requirements.txt
```

**`requirements.txt`**
```
fastapi
uvicorn
sentence-transformers
scikit-learn
numpy
psycopg2-binary
pgvector
hdbscan
```

---

## DB Migrations

```
migrations/
├── V1__enable_pgvector.sql
├── V2__create_classification_categories.sql
├── V3__create_category_keywords.sql
├── V4__create_category_centroids.sql
└── V5__create_shipment_classifications.sql
```

---

## End-to-End Flow

### Offline (run once, then weekly)

```
classification_categories + your labeled shipment data
        │
        ▼
Sample 2,000 per active category  (~40k rows total)
        │
        ▼
Combine content columns per shipment into one string
        │
        ▼
Batch embed (SentenceTransformer, batch_size=512)
        │
        ▼
Incremental mean per category  →  N centroid vectors
        │
        ▼
UPSERT category_centroids
        │
        ▼
POST /reload  →  service picks up new centroids + keywords
```

### Online (per shipment, ~15–20ms)

```
cargo_document_description + commodity_description
        │
        ▼
Text quality check  →  too short / generic  →  unclassified
        │ ok
        ▼
Embed text  (~5ms)
        │
        ▼
Cosine similarity × N active centroids  (<1ms)
        │
        ▼
Keyword boost per category  (<1ms)
        │
        ▼
Blend scores  →  apply threshold bands
        │
        ├── max < 0.35    →  unclassified  →  review queue
        ├── max 0.35–0.45  →  low_confidence  →  top-1 assigned, flagged
        └── max ≥ 0.45    →  classified  →  all above threshold assigned
                │
                ▼
        Persist to shipment_classifications
```

### Unknown Discovery (periodic)

```
shipment_classifications WHERE confidence_state = 'unclassified'
        │
        ▼
HDBSCAN on embeddings
        │
        ├── clusters found  →  analyst reviews  →  new category OR discard
        └── no clusters     →  genuine noise / bad data

If new category confirmed:
    → INSERT classification_categories
    → label shipments in your existing labeled data
    → re-run centroid_builder.py
    → POST /reload
```

---

## Useful Queries

```sql
-- Check centroid freshness
SELECT cc.name, cen.sample_count, cen.updated_at
FROM   category_centroids cen
JOIN   classification_categories cc ON cc.id = cen.category_id
ORDER  BY cen.updated_at DESC;

-- Categories with no centroid yet (waiting for enough labeled data)
SELECT name FROM classification_categories
WHERE  is_active = true
AND    id NOT IN (SELECT category_id FROM category_centroids);

-- Find all shipments classified as electronics AND perishables
SELECT shipment_id, scores
FROM   shipment_classifications
WHERE  categories @> ARRAY['electronics', 'perishables'];

-- Unclassified shipments pending review
SELECT shipment_id, scores->>'reason' AS reason, classified_at
FROM   shipment_classifications
WHERE  confidence_state = 'unclassified'
ORDER  BY classified_at DESC;

-- Low-confidence shipments
SELECT shipment_id, categories, scores
FROM   shipment_classifications
WHERE  confidence_state = 'low_confidence'
ORDER  BY classified_at DESC;

-- Nearest neighbours to a given shipment
SELECT   sc.shipment_id,
         sc.embedding <=> ref.embedding AS distance
FROM     shipment_classifications sc,
         shipment_classifications ref
WHERE    ref.shipment_id = '<target_id>'
ORDER BY distance
LIMIT    10;
```

---

## Build Checklist

| # | Task | Notes |
|---|------|-------|
| 1 | Enable pgvector | `CREATE EXTENSION vector` |
| 2 | Run migrations V1–V5 | Creates all tables |
| 3 | Seed `classification_categories` with 20 categories | Set `is_active = true` |
| 4 | Seed `category_keywords` per category | From your existing keyword rules |
| 5 | Check labeled shipment count per category | Need ≥ 100 per category for a reliable centroid |
| 6 | Build and run `centroid_builder.py` | Verify rows in `category_centroids` |
| 7 | Build `classifier.py` + `main.py` | Test `/health` returns all categories loaded |
| 8 | Run `evaluate_threshold.py`, pick threshold | Target precision ≥ 0.80 |
| 9 | Backfill `shipment_classifications` for 5M shipments | Use `/classify/batch` |
| 10 | Schedule weekly centroid rebuild + `/reload` | Keeps centroids fresh as new labels arrive |
| 11 | Schedule periodic `discover_unknowns.py` | Surface candidate new categories |
| 12 | Monitor `unclassified` and `low_confidence` counts | Rising counts signal category gaps |
