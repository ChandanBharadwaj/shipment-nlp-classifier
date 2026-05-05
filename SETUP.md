# Setup — Shipment Classifier (v3)

End-to-end local setup: Postgres + pgvector, Python ML service, the data
pipeline, and the SPA.

Total time on a fresh machine: **~15 minutes** (most of it is the first-time
SentenceTransformer model download).

---

## Prerequisites

- **Python** 3.11+ with `venv`
- **Node** 20+ with `npm`
- **Docker Desktop** (for Postgres + pgvector)
- **Git**

Optional: **`ANTHROPIC_API_KEY`** if you want to regenerate categories /
CCTR rows from the LLM. Without it, the chat-generated CSVs at
`data/llm_generated/` substitute for the same output.

---

## 1. Clone + create env

```bash
git clone <repo> shipment-nlp-classifier
cd shipment-nlp-classifier

# .env at repo root (loaded by init_db.py and the pipeline scripts)
cat > .env <<EOF
DATABASE_URL=postgresql://shipment:shipment@localhost:5555/shipment_db
EOF

# .env at ml-service/.env (loaded by uvicorn at startup)
cat > ml-service/.env <<EOF
DATABASE_URL=postgresql://shipment:shipment@localhost:5555/shipment_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EOF
```

---

## 2. Database

```bash
docker compose up -d
# Postgres is on port 5555 (mapped from container's 5432). pgvector extension
# is preloaded by docker/init.sql.

# Wait for healthy (a few seconds), then apply schema + seed shipment labels.
python init_db.py
```

What this creates:

| Table | Rows after init | Populated by |
|---|---:|---|
| `categories` | 0 | step 4 below |
| `chapter_categories` | 0 | step 4 below |
| `keywords` | 0 | step 4 below |
| `category_centroids` | 0 | step 5 below |
| `public_source_descriptions` | 0 | step 4 below |
| `shipment_labels` | 15,452 | `init_db.py` (eval data) |
| `shipment_classifications` | 0 | runtime, when `/classify` is called with `persist=true` |

---

## 3. ml-service venv + dependencies

```bash
cd ml-service
python -m venv venv
venv/Scripts/pip install -r requirements.txt   # Linux/Mac: source venv/bin/activate; pip install ...
cd ..
```

This pulls SentenceTransformer (~80MB model on first use), psycopg2,
pgvector, FastAPI, etc.

---

## 4. Build the v3 data layer

This is the heart of v3. Each step is idempotent — re-running won't
duplicate rows.

```bash
# 4a. Fetch upstream public sources to local cache (USITC + UK Trade Tariff)
python scripts/fetch_public_data.py
# → data/hs/usitc_hts_2024.json (35K items)
# → data/hs/uk_chapters_raw/NN.json (96 files)

# 4b. Load USITC + UK into public_source_descriptions table (~27K rows)
python scripts/load_public_sources.py

# 4c. Load LLM-generated categories + chapter mapping
python scripts/load_llm_categories.py
# → categories table (36 rows)
# → chapter_categories table (96 rows; every active chapter mapped)
#
# Reads data/llm_generated/chapter_categories.csv. To regenerate from
# the LLM with an API key:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   python scripts/llm_classify_chapters.py    # ~$1, 2 LLM calls

# 4d. Load LLM-generated CCTR rows (anchors / suppressors / modifiers)
python scripts/load_llm_cctr.py
# → keywords table, source='chat:claude:YYYY-MM-DD' (~575 rows)
#
# Reads data/llm_generated/cctr_rows.csv. To regenerate:
#   python scripts/llm_generate_cctr.py        # ~$10-20, 192 LLM calls

# 4e. Per-chapter TF-IDF over public_source_descriptions
python scripts/build_public_keywords.py
# → keywords table, source='public_tfidf' (~6.6K rows)

# 4f. Cross-validate chat CCTR vs source text + auto-promote unused TF-IDF
python scripts/validate_chat_cctr.py
# → keywords table, source='tfidf_promoted:YYYY-MM-DD' (~1.6K rows)
# → data/llm_generated/anchor_review.csv  (suspicious chat anchors flagged)
```

---

## 5. Build centroids

```bash
python ml-service/centroid_builder.py
# → category_centroids table (96 rows; one per (category, chapter) with text)
```

Centroids are L2-normalized embedding averages of the
`public_source_descriptions` text per `(category_slug, hs_chapter)`. Wipe
and rebuild — never insert manually.

---

## 6. Start the classifier API

```bash
cd ml-service
venv/Scripts/uvicorn main:app --host 127.0.0.1 --port 8000
```

The first startup loads SentenceTransformer (~5-10s) plus the in-memory
state (centroids, keywords, chapter map). Watch for:

```
Loaded 36 category centroids (chapter centroids total: 96), 96 HS chapters,
model=sentence-transformers/all-MiniLM-L6-v2
INFO: Application startup complete.
```

Sanity:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","categories_loaded":36,"chapter_centroids":96, ...}

curl -X POST http://127.0.0.1:8000/classify -H 'content-type: application/json' \
  -d '{"shipment_id":"t1","cargo_description":"M16 rifle 5.56mm","commodity_description":"military arms"}'
# Expect: arms_ammunition, ch93
```

---

## 7. Build the SPA

```bash
cd web
npm install
npm run build
# → ml-service/static/dist/  (FastAPI serves at /ui/)
```

For HMR-enabled dev:

```bash
cd web && npm run dev
# Vite on :5173, proxies API calls to :8000
```

Open [http://localhost:8000/ui/](http://localhost:8000/ui/).

---

## Refreshing data later

The pipeline is incremental — re-run any subset:

```bash
# Refresh from upstream APIs
python scripts/fetch_public_data.py --force
python scripts/load_public_sources.py

# Re-derive everything keyword/centroid-related
python scripts/build_public_keywords.py
python scripts/validate_chat_cctr.py
python ml-service/centroid_builder.py

# Tell the running classifier to pick up the new state without restart
curl -X POST http://localhost:8000/reload
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `psycopg2.OperationalError: connection refused` | Postgres container not running / not healthy | `docker compose up -d`; `docker compose ps` to check health |
| `"chapter_categories empty"` from build_public_keywords.py | Step 4c skipped | `python scripts/load_llm_categories.py` |
| `/classify` returns `unclassified` for everything | Centroids not built | `python ml-service/centroid_builder.py` then `curl -X POST .../reload` |
| `ANTHROPIC_API_KEY not set` | Trying to use `llm_*` scripts without a key | Use the `load_llm_*` scripts instead — they read from chat-substitute CSVs |
| SPA shows grey chips with snake_case names | Catalog not loaded; dev server stale bundle | Hard-refresh (Ctrl+Shift+R) |

---

## What this setup does NOT do

- **Calibration.** v3 returns raw probabilities. There's no Platt scaling
  step. Downstream consumers interpret the score directly.
- **Manual CCTR mutation API.** v3 doesn't expose write endpoints — keywords
  are regenerated via the pipeline scripts, not edited in-flight.
- **Audit log.** Same reason: no in-flight edits, so no audit trail.
- **Collision registry.** Replaced by the source-driven CCTR pipeline (chat
  + TF-IDF promotion).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the rationale.
