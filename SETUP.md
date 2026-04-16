# Setup Guide — Shipment NLP Classifier

Step-by-step instructions to run the full system locally: PostgreSQL + pgvector database, Python ML service, classification API, and the semantic compliance layer.

Total time on a fresh machine: **~10 minutes** (most of it is the first-time model download and pip install).

---

## 0. Prerequisites

Install these once:

| Tool | Version | Notes |
|---|---|---|
| **Docker Desktop** | latest | Hosts the PostgreSQL + pgvector container |
| **Python** | 3.10 or newer | `python --version` to check |
| **Git** | any | To clone the repo |

> **Windows users:** make sure Docker Desktop is **running** (not just installed) before step 2 — the system tray icon should be green.

---

## 1. Clone the repository

```bash
git clone <repo-url> shipment-nlp-classifier
cd shipment-nlp-classifier
```

All commands from this point are run from the repo root unless noted otherwise.

---

## 2. Start the database

```bash
docker compose up -d
```

This pulls `pgvector/pgvector:pg16` (~400 MB on first run) and starts PostgreSQL on host port **5555**.

Verify:

```bash
docker compose ps
```

You should see:
```
NAME                              STATUS         PORTS
shipment-nlp-classifier-db-1      Up (healthy)   0.0.0.0:5555->5432/tcp
```

If `STATUS` says `starting`, wait ~10 seconds and re-check.

---

## 3. Configure environment variables

```bash
cd ml-service
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
```

The default `.env` already points at the Docker DB (`postgresql://shipment:shipment@localhost:5555/shipment_db`). No edits needed for local dev.

---

## 4. Create the Python virtual environment

```bash
# Still inside ml-service/
python -m venv venv

# Activate it
venv\Scripts\activate              # Windows (cmd or PowerShell)
# source venv/bin/activate         # macOS / Linux
```

Your prompt should now show `(venv)`.

---

## 5. Install Python dependencies

```bash
pip install -r requirements.txt --prefer-binary
```

Takes ~3-5 minutes on first run. Pulls `sentence-transformers`, `fastapi`, `psycopg2-binary`, `pgvector`, `numpy`, `hdbscan`, etc.

> **Windows note:** `--prefer-binary` is important because `hdbscan` would otherwise need a C compiler. If you still hit a build error, run `pip install hdbscan --prefer-binary` separately first.

---

## 6. Initialize the database (schema + seed data)

From the repo root:

```bash
cd ..                              # back to repo root
python init_db.py
```

This runs `schema.sql` then all seed files in order. Expected output ends with:

```
── Summary ──────────────────────────────────────────────────
  classification_categories                   20 categories
  category_keywords                         1936 keywords
  shipment_labels                          16778 labeled rows
  category_hs_chapters                        96 HS chapters

Done. Next step: build centroids.
  cd ml-service && python centroid_builder.py
```

Idempotent — re-running is safe.

---

## 7. Build category centroids

```bash
cd ml-service
python centroid_builder.py
```

This embeds the labeled training rows and writes one centroid per (category × HS chapter) pair into `category_centroids`. Expect **~2-4 minutes** on first run (model download `all-MiniLM-L6-v2` ~90 MB + embedding ~12k rows).

Verify:

```sql
-- via any psql/DBeaver/pgAdmin connection
SELECT COUNT(*) FROM category_centroids;
-- Should return 116 (one per chapter centroid across the 20 categories)
```

---

## 8. Fit Platt calibration (recommended)

Calibration converts raw cosine similarity into a true probability per category. Without it, the API still works but `probability` fields are uncalibrated.

```bash
python fit_calibration.py
```

Takes ~30 seconds. Writes `platt_a`/`platt_b` per category into `classification_categories`.

---

## 9. Start the API service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Startup logs should show:

```
Risk profile: 12 global blocked, 79 category hard negatives, 359 total vectors embedded
Loaded 20 category centroids (chapter centroids total: 116), 96 HS chapters, model=sentence-transformers/all-MiniLM-L6-v2
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Leave this terminal running.

---

## 10. Smoke-test the API

Open a **new terminal** (the API one is busy):

### Health check

```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "ok",
  "categories_loaded": 20,
  "chapter_centroids": 116,
  "hs_chapters": 96,
  "model_version": "sentence-transformers/all-MiniLM-L6-v2",
  "calibrated": true,
  "risk_vectors": 359
}
```

### Classify + compliance — clean shipment (should ALLOW)

```bash
curl -X POST http://localhost:8000/classify ^
  -H "Content-Type: application/json" ^
  -d "{\"shipment_id\": \"s1\", \"cargo_description\": \"LEGO building blocks educational toy\", \"commodity_description\": \"plastic toys 500 pieces\"}"
```

Look for `"compliance_decision": "allow"` in the response.

### Classify + compliance — risky shipment (should BLOCK)

```bash
curl -X POST http://localhost:8000/classify ^
  -H "Content-Type: application/json" ^
  -d "{\"shipment_id\": \"s2\", \"cargo_description\": \"depleted uranium fuel rods\", \"commodity_description\": \"nuclear material reactor grade\"}"
```

Look for `"compliance_decision": "block"` and a `decision_reasons` entry citing the IAEA-controlled phrase match.

> **Windows curl note:** the `^` is line continuation in `cmd`. In PowerShell use a backtick `` ` ``. In bash use `\`.

---

## 11. Run the compliance test suite

```bash
# Inside ml-service/ with venv active
python test_compliance.py
```

Expected: **`24 passed, 0 failed out of 24`**.

This test loads the embedding model and exercises every cascade path (global block, category block/review, semantic paraphrases, high-risk routing, low-confidence routing, allow path, multi-label edges) — no DB needed.

---

## You're done

- API: <http://localhost:8000>
- Auto-generated docs: <http://localhost:8000/docs>
- DB: `postgresql://shipment:shipment@localhost:5555/shipment_db`

---

## Common follow-up commands

| What you want | Command |
|---|---|
| Reload risk profile after editing `risk_profile.json` | `curl -X POST http://localhost:8000/reload` |
| Reload centroids after re-running `centroid_builder.py` | `curl -X POST http://localhost:8000/reload` |
| Stop the database (keep data) | `docker compose down` |
| **Wipe the database** (delete all data) | `docker compose down -v` |
| Tail API logs | The terminal running `uvicorn` |
| Tail DB logs | `docker compose logs -f db` |

---

## Troubleshooting

### "Connection refused" on port 5555
Docker Desktop isn't running. Start it from the Start menu, wait for the whale icon to stop animating, then re-run `docker compose up -d`.

### `ModuleNotFoundError: No module named 'sentence_transformers'`
Your venv isn't activated. Look for `(venv)` in your prompt. If missing, run `venv\Scripts\activate` from inside `ml-service/`.

### `psycopg2.OperationalError: FATAL: database "shipment_db" does not exist`
The container started but `docker/init.sql` didn't run (rare race condition). Wipe and restart:
```bash
docker compose down -v
docker compose up -d
# wait 15 seconds
python init_db.py
```

### Slow first request to `/classify`
The `sentence-transformers` model lazy-loads on first use. First request takes 5-10 seconds; subsequent requests are <100 ms.

### `compliance_decision` is missing from the response
Check the startup logs — if you don't see the `Risk profile: 12 global blocked, 79 category hard negatives, 359 total vectors embedded` line, the JSON file is malformed. The validator will print the offending entry. Fix `risk_profile.json` and call `POST /reload`.

### Port 8000 already in use
Either stop the existing process, or start uvicorn on a different port: `uvicorn main:app --port 8001`.

---

## What's next

- Read the [README.md](README.md) for the problem statement and architecture.
- Read `risk_profile.json` to understand the compliance rule format.
- Read `ml-service/compliance.py` to see the semantic decision cascade.
- Run `python evaluate_threshold.py --split validation` to tune the classifier threshold for your dataset.
