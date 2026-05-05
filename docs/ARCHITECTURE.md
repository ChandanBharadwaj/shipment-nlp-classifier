# Architecture (v3)

The classifier turns free-text shipment descriptions into a `(category,
hs_chapter)` decision. v3 is **source-driven**: every runtime artifact is
derived from public APIs (US ITC HTS, UK Trade Tariff) plus an LLM with
prompts committed to the repo. `shipment_labels` is used only for F1
evaluation — it never feeds keyword generation, centroid construction,
or calibration.

---

## Design principles

1. **No hand-curated rows in the runtime data path.** Categories,
   chapter assignments, keywords, anchors, suppressors, modifiers — all
   derived. The only hand-authored input is the prompt files.
2. **Reproducible from upstream.** Re-run `fetch_public_data.py
   --force` plus the loaders + builders and you get the same data
   layer (modulo TF-IDF determinism and LLM sampling).
3. **Layered evidence.** Semantic (centroids) + lexical (keywords). They
   blend at scoring time but are computed independently.
4. **Read-only admin SPA.** Mutations happen via pipeline scripts, not
   API endpoints. No audit log because no in-flight edits.

---

## Data flow

```
                ┌──────────────────────────────────────────────────────┐
                │                  Build-time (pipeline)               │
                └──────────────────────────────────────────────────────┘

  US ITC HTS API ──┐
                   │  scripts/fetch_public_data.py
                   │     │
                   ▼     ▼
            data/hs/usitc_hts_2024.json + data/hs/uk_chapters_raw/*.json
                       │
                       │  scripts/load_public_sources.py
                       ▼
              ┌─────────────────────────────┐
              │ public_source_descriptions  │  27K rows
              │ (source, hs_code, chapter,  │
              │  description)               │
              └─────────────────────────────┘
                       │
        ┌──────────────┼──────────────────────────────────────┐
        │              │                                      │
        │              ▼                                      ▼
        │   scripts/build_public_keywords.py    ml-service/centroid_builder.py
        │       (per-chapter TF-IDF)               (embed + average)
        │              │                                      │
        │              ▼                                      ▼
        │      ┌──────────────────────┐         ┌────────────────────┐
        │      │ keywords             │         │ category_centroids │
        │      │  signal:             │         │  one per           │
        │      │   public_tfidf       │         │  (cat, chapter)    │
        │      └──────────────────────┘         └────────────────────┘
        │              ▲
        │              │
        │   scripts/validate_chat_cctr.py
        │       (auto-promote unused
        │        TF-IDF as anchors)
        │              ▲
        │              │
        ▼              │
  data/llm_generated/     scripts/load_llm_categories.py
  ├── chapter_categories.csv ──┐  ────────────────────────► categories
  └── cctr_rows.csv            │                            chapter_categories
                               │
                               │  scripts/load_llm_cctr.py
                               └─────────────────────────► keywords
                                                            (anchor /
                                                             suppressor /
                                                             modifier)


                ┌──────────────────────────────────────────────────────┐
                │                       Runtime                        │
                └──────────────────────────────────────────────────────┘

  POST /classify ─► classifier.predict()
                    ├── embed text (SentenceTransformer all-MiniLM-L6-v2)
                    ├── for each (cat, chapter):
                    │     semantic = max cosine(query, centroid)
                    │     keyword  = saturating sum of (anchors + signals
                    │                                   + modifiers
                    │                                   - suppressors)
                    │     final    = 0.8 * semantic + 0.2 * keyword
                    └── return top-N above threshold

  POST /reload    ─► reloads centroids + keywords + chapter_titles into
                     in-memory state (no process restart)
```

---

## Tables (7)

| Table | Source | Rows | Lifecycle |
|---|---|---:|---|
| `public_source_descriptions` | USITC + UK API caches | ~27,000 | Refresh via `scripts/load_public_sources.py` after `fetch_public_data.py` |
| `categories` | LLM (chat or API) | 36 | Reseeded by `scripts/load_llm_categories.py` |
| `chapter_categories` | LLM (chat or API) | 96 | Same as above |
| `keywords` | TF-IDF + LLM | ~8,800 | Built by `build_public_keywords.py` + `load_llm_cctr.py` + `validate_chat_cctr.py` |
| `category_centroids` | embed + average | 96 | `ml-service/centroid_builder.py` (wipe-and-rebuild) |
| `shipment_labels` | provided | 15,452 | Static; eval-only; loaded by `init_db.py` |
| `shipment_classifications` | runtime | 0+ | Written by `/classify?persist=true` |

### Foreign keys
- `chapter_categories.category_slug` → `categories.slug` (cascade)
- `category_centroids.category_slug` → `categories.slug` (cascade)

No FK from `keywords` to `chapter_categories` — multiple keyword rows
per chapter, keyword row may exist for a chapter even if the chapter map
shifts later. Looser by design.

No FK from `shipment_labels.category_name` to `categories.slug` —
labels are immutable evaluation data; if categories change, the labels
become stale and need a one-shot SQL UPDATE (we did this in the v2→v3
migration).

---

## Code layout

```
ml-service/
├── main.py              FastAPI shell, lifespan, /classify, /reload
├── classifier.py        embed_texts + predict + predict_batch + scoring
├── centroid_builder.py  builds category_centroids from public sources
├── centroid_projection.py  UMAP/PCA reducer + memoization for /admin/api/centroids
├── admin_routes.py      /admin/api/* read endpoints
├── chunking.py          long-text chunking (used by predict_batch)
├── compliance.py        compliance/risk vector matching
├── config.py            env-driven tunables (RERANKER_ENABLED, etc.)
├── db.py                psycopg2 pool wrapper
├── preprocess.py        build_query_text (cargo + commodity normalization)
├── evaluate_threshold.py  F1 sweep over the test split
├── test_chunking.py     pure-function tests
├── test_compliance.py   pure-function tests
├── tests/               pytest suite
│   ├── test_centroid_projection.py
│   ├── test_hs_extraction.py
│   ├── test_margin_rule.py
│   ├── test_metrics.py
│   └── test_reranker.py
└── static/dist/         built SPA artifacts (served at /ui/)

scripts/
├── fetch_public_data.py        USITC + UK → local cache
├── load_public_sources.py      cache → public_source_descriptions
├── load_llm_categories.py      data/llm_generated/chapter_categories.csv → DB
├── load_llm_cctr.py            data/llm_generated/cctr_rows.csv → DB
├── build_public_keywords.py    per-chapter TF-IDF → keywords table
├── validate_chat_cctr.py       cross-check + auto-promote TF-IDF as anchors
├── llm_classify_chapters.py    [needs ANTHROPIC_API_KEY] regenerate categories
└── llm_generate_cctr.py        [needs ANTHROPIC_API_KEY] regenerate CCTR

prompts/
├── chapter_classification.md   pass-1 + pass-2 prompts for category derivation
└── cctr_generation.md          pass-1 + pass-2 prompts for anchor/suppressor/modifier

web/                  Vue 3 SPA — see web/README.md

data/
├── hs/                          public-source caches
│   ├── usitc_hts_2024.json
│   ├── uk_chapters_raw/NN.json
│   ├── uk_tariff_descriptions.csv
│   └── hs_6digit.csv
└── llm_generated/               chat-substitute CSVs + audit outputs
    ├── chapter_categories.csv
    ├── cctr_rows.csv
    └── anchor_review.csv

seed/
├── seed_keywords_v3.sql         emitted by build_public_keywords.py
└── seed_shipment_labels_v2.sql  static eval data (15K rows)

sql/
├── drift_monitor.sql            ops query for shipment_classifications
└── hnsw_reindex.sql             bulk-load helper for HNSW index
```

---

## Scoring math

### Semantic score (per category)
For each `(category_slug, hs_chapter)` centroid `c_i`:
```
sim_i = dot(query_embedding, c_i)         # both L2-normalized → cosine
semantic_score[cat] = max over chapters of sim_i
best_chapter[cat]   = argmax chapter
```

### Keyword score (per category)
The classifier walks `keywords` rows that hit the input text and
accumulates per-chapter raw scores:

```
For each keyword row (kw, weight, hs_chapter, signal_class):
  if kw matches text (word-boundary, naive plural):
    if signal_class == 'anchor':       chapter_sum[hs] += weight
    elif signal_class == 'signal':     chapter_sum[hs] += weight  (modifier-aware retyping)
    elif signal_class == 'suppressor': chapter_sum[hs] -= weight
    elif signal_class == 'modifier':   marks chapter as a target for retyping signals

chapter_score[hs] = 1 - exp(-alpha * max(0, chapter_sum[hs]))    # saturate to [0, 1]
keyword_score[cat] = max over chapters of chapter_score
```

### Final
```
final_score[cat] = 0.8 * semantic_score[cat] + 0.2 * keyword_score[cat]
probability      = final_score[cat]    # uncalibrated in v3
```

### Confidence bands
```
final_score < unclassified_threshold → unclassified
unclassified_threshold ≤ score < threshold → low_confidence
score ≥ threshold → classified
```

`threshold` defaults to 0.45; consider lowering to 0.30 for v3 since the
public-source-only centroid space produces narrower score ranges than
the legacy mixed-source centroids did.

---

## v3 trade-offs (vs the legacy pipeline)

| What you gain | What you give up |
|---|---|
| Reproducibility from public sources | Calibrated probabilities (Platt scaling dropped) |
| Refresh = one CLI command | Manual override of single-row weights via UI |
| Per-chapter chapter-pinned keywords from the start | Per-category tuning of semantic/keyword blend |
| LLM-generated CCTR (no hand maintenance) | Audit trail of who changed what |
| Clean schema, 7 tables | Backwards compat with legacy clients reading dropped tables |

The biggest empirical trade-off: centroids built from formal HS dictionary
text don't cosine-match informal trade vocabulary as strongly as the
legacy mixed-source centroids did. The CCTR anchors (`tea bags`,
`m16 rifle`, `lithium-ion battery`, …) compensate, but you'll see
classification scores cluster in the 0.30–0.55 range rather than
0.45–0.85. Adjust the threshold accordingly.

---

## Operational procedures

### Refresh data from upstream
```bash
python scripts/fetch_public_data.py --force
python scripts/load_public_sources.py
python scripts/build_public_keywords.py
python scripts/validate_chat_cctr.py
python ml-service/centroid_builder.py
curl -X POST http://localhost:8000/reload
```

### Regenerate categories (LLM)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/llm_classify_chapters.py
# Regenerates categories + chapter_categories. ~$1 cost.
# After this: rebuild keywords + centroids since chapter→category map
# changed.
```

### Regenerate CCTR (LLM)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/llm_generate_cctr.py
# 96 chapters × 2 calls each ≈ ~$10-20 cost.
# Or edit data/llm_generated/cctr_rows.csv manually + run load_llm_cctr.py.
```

### Diagnose poor predictions
1. `/admin/api/keywords/by-token/<word>` — does the word exist? In which chapter?
2. `/admin/api/keywords/<word>/centroid-affinity` — where does the embedding model put it?
3. `/admin/api/labels?prediction_status=mispredicted&category=<cat>` — see misclassifications across the test set.
4. If a regression-fix anchor is missing, edit `cctr_rows.csv` and reload.

### Drift monitoring
```bash
psql ... < sql/drift_monitor.sql
# Reports on shipment_classifications: confidence-state shifts over time,
# top-changed categories, etc.
```

---

## Why the SPA is read-only in v3

The legacy SPA had Collisions, Audit Log, and Discover views backing a
governed-write workflow: operators discovered polysemous tokens, opened
collision rules, and committed them via `apply_collision_change.py`
which stamped `keyword_audit_log`.

In v3 the equivalent of "register a new keyword" is "edit the source CSV
and re-run the loader." The pipeline is the editor. This:
- Eliminates a class of consistency bugs (audit log vs registry drift)
- Makes the keyword set reproducible from version-controlled inputs
- Removes the "who has admin access" question from the system

If you need to add an anchor for a shipment that's misclassifying:
1. Edit `data/llm_generated/cctr_rows.csv` (commit the change)
2. `python scripts/load_llm_cctr.py`
3. `curl -X POST http://localhost:8000/reload`

That's the v3 mutation flow.
