# Shipment Classification System

Multi-label NLP classification for shipment records. Given free-text cargo and commodity descriptions, assigns one or more of **20 fixed categories** using sentence embeddings and cosine similarity against per-category centroids stored in PostgreSQL (pgvector).

---

## How It Works

1. **Embed** — combined cargo + commodity text is encoded into a 384-dimensional vector using `all-MiniLM-L6-v2`.
2. **Compare** — cosine similarity is computed against 20 pre-built category centroids.
3. **Boost** — a weighted keyword match adds a small signal on top of the semantic score.
4. **Threshold** — shipments scoring above the threshold get assigned; those below are flagged for review.

Scoring:
```
final_score = 0.8 × semantic_score + 0.2 × keyword_score
```

Confidence bands:
| max final_score | confidence_state | Action |
|---|---|---|
| < 0.35 | `unclassified` | Review queue |
| 0.35 – 0.45 | `low_confidence` | Top-1 assigned, flagged |
| ≥ 0.45 | `classified` | All above-threshold categories assigned |

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Docker Desktop | latest | Required for the PostgreSQL + pgvector container |
| Python | 3.10+ | |
| psql CLI | optional | Only needed for ad-hoc SQL queries; `init_db.py` handles migrations and seeding without it |

---

## Quick Start

### 1. Start the database

```bash
docker compose up -d
```

This starts PostgreSQL 16 with pgvector on port **5555**.
First run downloads the image (~400 MB). Wait ~10 seconds, then verify:

```bash
docker compose ps
# db   running   0.0.0.0:5555->5432/tcp
```

### 2. Set environment variables

```bash
cd ml-service
cp .env.example .env
# .env already has the correct DATABASE_URL for Docker Compose — no changes needed for local dev
```

### 3. Install Python dependencies

```bash
cd ml-service
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Windows note: hdbscan needs a C compiler OR pre-built wheels
pip install -r requirements.txt --prefer-binary
```

### 4. Initialize the database (migrations + seed data)

Run the Python init script from the repo root. It runs all migrations and seeds in order, retries if Docker is still starting, and prints a summary.

```bash
# From the repo root (reads DATABASE_URL from ml-service/.env automatically)
python init_db.py
```

Expected output:
```
Connecting to: postgresql://shipment:shipment@localhost:5555/shipment_db
Connected.

── Applying schema ───────────────────────────────────────────
  ✓  schema.sql

── Seeding data ─────────────────────────────────────────────
  ✓  seed_categories.sql
  ✓  seed_keywords.sql
  ✓  seed_shipment_labels.sql
  ✓  seed_extra_labels.sql
  ✓  seed_splits.sql

── Summary ──────────────────────────────────────────────────
  classification_categories                   20 categories
  category_keywords                          300 keywords
  shipment_labels                           1110 labeled rows (PoC — base 1000 + extras; split into train/validation/test)

Done. Next step: build centroids.
  cd ml-service && python centroid_builder.py
```

Other options:
```bash
# Schema only (skip seed data)
python init_db.py --no-seed

# Dry run — print what would run without executing
python init_db.py --dry-run

# Explicit connection string
python init_db.py --url "postgresql://shipment:shipment@localhost:5555/shipment_db"
```

### 5. Build category centroids

```bash
cd ml-service
python centroid_builder.py
```

This embeds the ~1110 labeled rows (50–80 per category), computes one L2-normalized centroid per category, and writes them to `category_centroids`. Expect ~1–3 minutes on first run (model download + embedding).

Verify:
```sql
SELECT cc.name, cen.sample_count, cen.updated_at
FROM   category_centroids cen
JOIN   classification_categories cc ON cc.id = cen.category_id
ORDER  BY cc.name;
-- Should show 20 rows with sample_count 50–80 each
```

### 6. Start the API service

```bash
cd ml-service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "categories_loaded": 20, "model_version": "all-MiniLM-L6-v2"}
```

---

### `POST /classify`

Classify a single shipment.

```bash
curl -s -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "shipment_id": "ship_001",
    "cargo_description": "Electronic PCB boards and semiconductors",
    "commodity_description": "microchips circuit boards",
    "threshold": 0.45
  }' | python -m json.tool
```

Response:
```json
{
  "shipment_id": "ship_001",
  "input": {
    "cargo_description": "Electronic PCB boards and semiconductors",
    "commodity_description": "microchips circuit boards"
  },
  "result": {
    "categories": ["electronics"],
    "confidence_state": "classified",
    "threshold": 0.45,
    "scores": {
      "electronics": {
        "semantic_score": 0.8712,
        "keyword_score": 0.45,
        "final_score": 0.787,
        "matched": true,
        "keywords_hit": ["PCB", "semiconductor", "circuit board", "microchip"]
      },
      "..."
    }
  },
  "meta": {
    "model_version": "all-MiniLM-L6-v2",
    "classified_at": "2026-04-03T10:22:00Z",
    "total_categories": 20,
    "evaluated": 20
  }
}
```

**Multi-label example** (shipment spans two categories):
```bash
curl -s -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "shipment_id": "ship_002",
    "cargo_description": "Electronic PCB boards and frozen prawns",
    "commodity_description": "semiconductors cold chain seafood",
    "threshold": 0.25
  }'
# → categories: ["electronics", "perishables"]
```

> **PoC threshold note:** With ~80 labeled examples per category, centroid scores are lower than production (where 2000+ real shipment records are used). Use `threshold: 0.25–0.35` for PoC testing. Run `evaluate_threshold.py` against a labeled test set to find the right value for your data.

**With persistence** (stores result in `shipment_classifications`):
```bash
curl -s -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "shipment_id": "ship_003",
    "cargo_description": "Industrial pump hydraulic",
    "commodity_description": "machinery compressor",
    "threshold": 0.45,
    "persist": true
  }'
```

**Unclassified example**:
```bash
curl -s -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"shipment_id": "t1", "cargo_description": "cargo", "commodity_description": "goods"}'
# → confidence_state: "unclassified", reason: "insufficient_input"
```

---

### `POST /classify/batch`

Classify up to 500 shipments in one call. Embeddings are computed in a single batch for efficiency.

```bash
curl -s -X POST http://localhost:8000/classify/batch \
  -H "Content-Type: application/json" \
  -d '{
    "shipments": [
      {"shipment_id": "b1", "cargo_description": "PCB boards", "commodity_description": "semiconductors"},
      {"shipment_id": "b2", "cargo_description": "Frozen salmon fillets", "commodity_description": "seafood cold chain"},
      {"shipment_id": "b3", "cargo_description": "cargo", "commodity_description": "goods"}
    ]
  }'
```

---

### `POST /reload`

Reload centroids and keywords from the database without restarting the service. Call this after running `centroid_builder.py` or adding new keywords.

```bash
curl -s -X POST http://localhost:8000/reload
# → {"status": "reloaded", "categories_loaded": 20}
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL connection string: `postgresql://user:pass@host:port/db` |
| `SENTENCE_TRANSFORMERS_HOME` | No | Path to local HuggingFace model cache (for offline environments) |

**Offline model caching** (air-gapped environments):
```bash
# Download once (needs internet):
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Then in .env, set:
SENTENCE_TRANSFORMERS_HOME=C:\Users\YourName\.cache\huggingface\hub
```

---

## Threshold Tuning

The threshold is the minimum `final_score` a category must reach to be assigned. It is **not** model fine-tuning — the embedding model weights are never changed. Tuning means finding the right cutoff for your centroid quality and use case.

### Why scores vary by stage

| Stage | Labeled rows per category | Expected score range | Recommended threshold |
|---|---|---|---|
| PoC (synthetic seed) | ~50–80 | 0.20 – 0.60 | 0.25 – 0.35 |
| Early production | 200–500 real records | 0.35 – 0.75 | 0.35 – 0.45 |
| Full production | 2000+ real records | 0.50 – 0.90 | 0.45 – 0.55 |

Centroids built from fewer, synthetic examples sit less precisely in semantic space — scores are lower but still usable. As real labeled data accumulates and centroids are rebuilt, scores rise and the threshold can be tightened.

### Three-step workflow

The labeled data in `shipment_labels` is split into three non-overlapping sets. Each set has a single purpose:

| Split | Rows | Used by | Purpose |
|---|---|---|---|
| `train` | 810 | `centroid_builder.py` | Build category centroids |
| `validation` | 160 | `evaluate_threshold.py --split validation` | Tune threshold (sweep all values) |
| `test` | 140 | `evaluate_threshold.py --split test --threshold X` | Confirm final threshold (run once) |

**Step 1 — Build centroids from train rows only** (already done after `init_db.py`):

```bash
cd ml-service
python centroid_builder.py
```

**Step 2 — Tune threshold on the validation set:**

```bash
python evaluate_threshold.py --split validation
```

Output:
```
 threshold   precision    recall        f1     tp     fp     fn         n
--------------------------------------------------------------------
      0.20      0.6840    0.9880    0.8082    ...
      0.25      0.7450    0.9690    0.8423    ...
      0.30      0.8010    0.9220    0.8573    ...
      0.35      0.8340    0.8910    0.8616    ...    ← best F1
      0.40      0.8590    0.8430    0.8509    ...
      ...

  Best by F1: threshold=0.35  precision=0.8340  recall=0.8910  f1=0.8616

  Per-category breakdown at threshold=0.35:
  category              tp    fp    fn    prec     rec
  -------------------------------------------------------
  electronics            7     1     1   0.875   0.875
  perishables            8     0     1   1.000   0.889
  ...

  → Recommended threshold: 0.35
  → Next step: python evaluate_threshold.py --split test --threshold 0.35
```

**Step 3 — Confirm on the held-out test set** (run once with the threshold from step 2):

```bash
python evaluate_threshold.py --split test --threshold 0.35
```

> **WARNING:** Do not sweep multiple thresholds against the test set — that leaks the test set into your threshold selection. Always tune on validation, confirm once on test.

**Alternative — external CSV** (when you don't have DB splits):

```bash
python evaluate_threshold.py --csv test_data.csv
```

CSV format (`cargo`, `commodity`, `expected_labels` columns; `expected_labels` comma-separated):
```
cargo,commodity,expected_labels
"Electronic PCB boards","semiconductors microchips","electronics"
"Frozen prawns","cold chain seafood","perishables"
"PCB boards and frozen prawns","semiconductors cold chain","electronics,perishables"
```

Target: **precision ≥ 0.80** with acceptable recall for your use case. F1 balances both.

### Applying the chosen threshold

Pass it per-request (useful for A/B testing different values):
```bash
curl -X POST http://localhost:8000/classify \
  -d '{"shipment_id": "x", "cargo_description": "...", "commodity_description": "...", "threshold": 0.35}'
```

Or update the default in `classifier.py`:
```python
def predict(..., threshold: float = 0.35):  # change default here
```

---

## Full Reset

Wipes all data (including the PostgreSQL volume) and starts from scratch.

```bash
# 1. Stop the container and delete the named volume (pgdata)
docker compose down -v

# 2. Start fresh — re-creates the container, re-runs docker/init.sql on first boot
docker compose up -d

# 3. Re-apply schema and seed data (wait ~10s for Postgres to be ready)
python init_db.py

# 4. Rebuild centroids from the fresh seed data
cd ml-service && python centroid_builder.py

# 5. Reload into the running service if it was already up
curl -s -X POST http://localhost:8000/reload
```

> The `-v` flag on `docker compose down` is what deletes the volume. Without it the data survives container removal.

---

## Centroid Rebuild

Rebuild centroids after:
- New labeled data is added to `shipment_labels`
- A new category is added and labeled
- Category keywords change significantly

```bash
# 1. Re-run the builder
python centroid_builder.py

# 2. Reload into the running service (no restart needed)
curl -X POST http://localhost:8000/reload
```

Recommended schedule: **weekly** via cron or Windows Task Scheduler.

---

## Unknown Category Discovery

Periodically run this to surface shipment types that don't fit any existing category:

```bash
python discover_unknowns.py
# or with custom cluster size:
python discover_unknowns.py --min-cluster-size 30
```

This requires unclassified shipments to exist in `shipment_classifications` (i.e., after classifying real shipments with `persist=true`).

**When a new category is confirmed by an analyst:**

```sql
-- 1. Add the category
INSERT INTO classification_categories (name, display_name, description, is_active)
VALUES ('new_category', 'New Category', 'Description of what this covers', true);

-- 2. Add keywords
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, 'keyword1', 1.0 FROM classification_categories WHERE name = 'new_category';

-- 3. Label the shipments in shipment_labels
INSERT INTO shipment_labels (shipment_id, category_name, cargo_text, commodity_text)
VALUES ('ship_xxx', 'new_category', 'cargo text here', 'commodity text here');
```

Then rebuild centroids and reload:
```bash
python centroid_builder.py
curl -X POST http://localhost:8000/reload
```

---

## Project Structure

```
shipment-classification/
│
├── docker-compose.yml                   # PostgreSQL 16 + pgvector, host port 5555
├── init_db.py                           # One-shot DB initializer: applies schema + seeds
├── schema.sql                           # Complete database schema (all tables + indexes)
├── docker/
│   └── init.sql                         # Enables pgvector on first container start
│
├── seed/
│   ├── seed_categories.sql              # 20 category definitions
│   ├── seed_keywords.sql                # ~300 weighted keywords (15–20 per category)
│   ├── seed_shipment_labels.sql         # 1000 base synthetic labeled rows (50 per category)
│   ├── seed_extra_labels.sql            # ~110 extra varied rows to improve centroid quality
│   └── seed_splits.sql                  # assigns train(70%) / validation(16%) / test(14%) splits
│
└── ml-service/
    ├── .env.example                     # Environment variable template
    ├── requirements.txt                 # Python dependencies (pinned)
    ├── db.py                            # psycopg2 + pgvector connection helper
    ├── classifier.py                    # Core ML logic: predict(), load_centroids(), load_keywords()
    ├── centroid_builder.py              # Offline job: embed labeled data → build centroids
    ├── main.py                          # FastAPI service: /classify, /classify/batch, /reload, /health
    ├── evaluate_threshold.py            # Threshold tuning: --split validation (sweep) / --split test (confirm)
    └── discover_unknowns.py             # HDBSCAN clustering on unclassified shipments
```

---

## Windows Notes

### hdbscan installation

`hdbscan` requires a C compiler. On Windows, use pre-built wheels:

```bash
pip install hdbscan --prefer-binary
```

Or with conda:
```bash
conda install -c conda-forge hdbscan
```

### psql CLI

If `psql` is not in your PATH after installing PostgreSQL, add it:
```
C:\Program Files\PostgreSQL\16\bin
```

Or use the Python init script (no psql needed at all):
```bash
python init_db.py
```

---

## Useful Queries

```sql
-- Split distribution (verify seed_splits.sql ran correctly)
-- Expected: test=140, train=810, validation=160
SELECT split, COUNT(*) FROM shipment_labels GROUP BY split ORDER BY split;

-- Labeled rows available for centroid building (train only)
SELECT category_name, COUNT(*) AS train_rows
FROM   shipment_labels
WHERE  split = 'train'
GROUP  BY category_name
ORDER  BY category_name;

-- Centroid freshness
SELECT cc.name, cen.sample_count, cen.updated_at
FROM   category_centroids cen
JOIN   classification_categories cc ON cc.id = cen.category_id
ORDER  BY cen.updated_at DESC;

-- Categories missing a centroid (need more labeled data)
SELECT name FROM classification_categories
WHERE  is_active = true
AND    id NOT IN (SELECT category_id FROM category_centroids);

-- Multi-label shipments (electronics AND perishables)
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

-- Nearest neighbours to a given shipment embedding
SELECT   sc.shipment_id,
         sc.embedding <=> ref.embedding AS distance
FROM     shipment_classifications sc,
         shipment_classifications ref
WHERE    ref.shipment_id = 'ship_001'
ORDER BY distance
LIMIT    10;
```
