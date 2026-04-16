# Shipment NLP Classifier with Semantic Compliance Screening

A two-layer system that turns free-text shipment descriptions into **compliance decisions** at scale: classify every shipment into one of 20 trade categories *and* decide whether to **Allow**, **Block**, or **Review** it against international regulatory hard-negatives — all from the same 384-dimensional embedding, in one inference pass.

> Want to run it? See **[SETUP.md](SETUP.md)** for a 10-minute walkthrough.

---

## The Problem

Modern customs and freight brokers process **tens of thousands of shipments per day**. Every one of them needs to be:

1. **Classified** — assigned to an HS chapter and a coarse category (e.g., `electronics`, `pharmaceuticals`, `defense`) for tariff calculation, regulatory routing, and reporting.
2. **Screened for compliance** — checked against international watch-lists (IAEA nuclear materials, CWC chemical weapons, BWC biological agents, CITES protected species, DEA controlled substances, Wassenaar dual-use goods, conflict minerals, etc.) so prohibited cargo is stopped *before* it crosses a border.

Today, both jobs are typically done with one of two flawed approaches:

### Approach A: Manual analyst review
Slow, expensive, and unscalable. A trained analyst can review ~100-200 shipments per day. At 50,000 shipments/day, you'd need 250+ analysts working in parallel — and quality drifts because no two analysts apply the rules the same way.

### Approach B: Keyword regex rules
Brittle and easy to evade. A rule like `\bdepleted uranium\b` catches the literal phrase but misses **"DU penetrator"**, **"spent nuclear fuel"**, **"radioactive U-238 rods"**, and a hundred other paraphrases. Rule lists balloon to thousands of entries with overlapping spelling variants, and they still fail on every novel phrasing the model has never seen. They also can't tell context apart: the word *"acid"* is normal in a chemicals shipment and suspicious in a children's toy shipment.

Neither scales. We need something that is **fast** (CPU-viable for 50k/day), **paraphrase-resilient** (catches semantic variants without a regex maintenance treadmill), **context-aware** (the same word means different things in different categories), and **interpretable** (every Allow/Block/Review decision shows exactly which rule fired and why).

---

## The Solution

A two-stage pipeline that produces both a category prediction and a compliance decision from a **single embedding pass**:

```
        ┌─────────────────────────────────────────────────────────┐
        │  cargo_description + commodity_description (free text)  │
        └─────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                ┌────────────────────────────────────┐
                │  all-MiniLM-L6-v2 sentence encoder │   one embedding,
                │  → 384-dim L2-normalized vector    │   used twice
                └────────────────────────────────────┘
                          │                   │
                          ▼                   ▼
        ┌──────────────────────────┐   ┌──────────────────────────────┐
        │  Stage 1: Classifier     │   │  Stage 2: Compliance         │
        │  • cosine vs 116 chapter │   │  • cosine vs 359 risk vectors│
        │    centroids (20 cats)   │   │    (12 global + 79 category) │
        │  • +keyword boost (20%)  │   │  • risk-level routing        │
        │  • Platt calibration     │   │  • confidence routing        │
        │  • 3-tier confidence     │   │  • Allow / Block / Review    │
        └──────────────────────────┘   └──────────────────────────────┘
                          │                   │
                          └────────┬──────────┘
                                   ▼
                ┌────────────────────────────────────┐
                │  API response:                     │
                │  {categories, scores, compliance}  │
                └────────────────────────────────────┘
```

### Stage 1 — Multi-label classification

- **20 fixed categories** (`agriculture`, `automotive`, `chemicals`, `defense`, `electronics`, `pharmaceuticals`, …)
- **96 HS chapters** mapped underneath those categories — every category has 1-N supervised centroids, one per HS chapter
- Scoring: `0.8 × cosine(text, chapter_centroid) + 0.2 × keyword_score` per category
- **Platt-calibrated probabilities** (per-category sigmoid fitted on a held-out validation set)
- **Three-tier confidence routing:** `classified` (≥ 0.45), `low_confidence` (0.35-0.45), `unclassified` (< 0.35)
- F1 = **0.9047** on a 2,227-row held-out test set; **97% chapter accuracy** when the coarse category is correct

### Stage 2 — Semantic compliance screening

This is where this project's contribution lives. Instead of a regex list, every risk phrase in `risk_profile.json` is **embedded once at startup** with the same model the classifier uses. At request time we re-use the shipment embedding the classifier already computed and run cosine-similarity against the risk vectors. **Zero extra inference cost.**

Risk data is structured in two scopes:

| Scope | Example entries | What triggers |
|---|---|---|
| **Global blocked** (always block) | `depleted uranium`, `nerve agent`, `cluster munition`, `signal jammer` | Match anywhere → Block |
| **Category-scoped hard negatives** | `acetic anhydride` in `chemicals`, `rosewood` in `furniture`, `fentanyl` in `pharmaceuticals`, `5-axis CNC` in `machinery` | Match only when shipment is classified into that category → Block or Review |

Each entry has a `phrase` plus an `aliases` array — the matcher takes the **max cosine across phrase + aliases**, so spelling variants (`armored` / `armoured` / `armor-plated`) become a single semantic neighborhood rather than five duplicate regex entries.

The decision cascade (first match wins):

```
1. Global blocked phrase hit (cos ≥ 0.62)                    → BLOCK
2. Category hard negative w/ action=block (cos ≥ 0.55)       → BLOCK
3. Category hard negative w/ action=review (cos ≥ 0.55)      → REVIEW
4. Classified into a high-risk category                      → REVIEW
5. confidence_state in {low_confidence, unclassified}        → REVIEW
6. Otherwise (classified + medium/low risk + clean)          → ALLOW
```

Every decision returns `decision_reasons` (human-readable strings) and `hard_negative_hits` (structured matches with similarity scores) for full audit traceability.

---

## What this approach gets right

### Paraphrase resilience (the regex problem, solved)
| Shipment text | Regex would catch? | Semantic compliance |
|---|---|---|
| `"depleted uranium fuel rods"` | yes (literal phrase) | yes (sim=0.79) |
| `"spent nuclear fuel rods radioactive waste"` | **no** | **yes** (sim=0.84 via alias) |
| `"DU penetrator armor-piercing round"` | only if "DU" was in the regex | yes (sim=0.74) |
| `"GPS signal jammer device"` | yes | yes (sim=0.95) |
| `"portable cell phone RF blocker"` | **no** | **yes** (sim=0.73) |
| `"synthetic opioid analgesic similar to fentanyl"` | only if "fentanyl" appears literally | **yes** (sim=0.92) |

### Context-aware screening
Hard negatives are **scoped to categories** so context drives meaning:
- `"acetic anhydride"` is normal industrial chemistry in a paint shipment (no rule fires) but a heroin precursor in a `chemicals` shipment (Review).
- `"firearm"` is benign in a sporting goods context but a misclassification flag in a `toys` shipment.

A pure global blocklist can't make these distinctions.

### Zero extra inference cost
The shipment is encoded **once** by `all-MiniLM-L6-v2`. The classifier and compliance layer both consume that single 384-dim vector. Compliance adds only a `(N, 384) @ (384,)` matrix multiplication against ~360 pre-computed risk vectors — that's microseconds.

### CPU-viable throughput
Measured on a modern CPU:
- Embedding: ~1,000-2,000 texts/sec at batch_size=512
- 50,000 shipments → **~60-90 seconds** end-to-end (classify + compliance)
- The proposed Minerva framework's 30-minute SLA for 50k/day is met **20× over**, with no GPU.

### Interpretable, auditable decisions
Every block/review carries the matched phrase, the matched alias, the cosine similarity, the scope (global vs category), and the regulatory reason (e.g., *"CWC Schedule 2 — mustard gas precursor"*). Compliance officers see exactly *what* fired and *why*.

### No retraining to extend the rules
Adding a new risk phrase = edit `risk_profile.json` + `POST /reload`. The system embeds the new phrase and aliases in seconds. **No model fine-tuning, no labeled data required.** Compare to a fine-tuned classifier where adding a new prohibited category needs hundreds of labeled examples and a retraining cycle.

### Schema-validated rules
Risk entries are validated at startup. Missing keys, invalid actions, or wrong types fail loudly with the offending entry name — typos can never silently disable a rule.

---

## Mapping to the Minerva Hybrid Screening Framework

The Minerva proposal calls for a 3-layer screening architecture: **Layer 1 hard taxonomy**, **Layer 2 AI classifier**, **Layer 3 confidence router**. This system delivers all three:

| Minerva layer | This system | Status |
|---|---|---|
| **Layer 1: Hard taxonomy / deterministic rules** | `global_blocked` semantic blocks + per-category `hard_negatives` in `risk_profile.json` | **DONE** — semantic equivalent of keyword rules, paraphrase-resilient |
| **Layer 2: AI classifier (3-class Allow/Block/Review)** | 20-category classifier *composed* with risk levels and the compliance cascade. Allow/Block/Review is a derived view, not a separately trained model. | **DONE** — F1=0.90, calibrated, 50k/day on CPU |
| **Layer 3: Confidence router** | Three-tier confidence bands (classified / low_confidence / unclassified) + risk-level routing. High-risk categories always route to Review even on high-confidence classifications. | **DONE** |
| Sanctions/restricted-party screening | *Not in scope* — entity matching against OFAC/EU/UN lists is a distinct subsystem | Out of scope (separate project) |
| Analyst queue & SLA monitoring | *Not in scope* — workflow tooling depends on target platform | Out of scope (downstream system) |

**Key architectural choice:** Allow/Block/Review is computed as a *layered decision* on top of the 20-category prediction, not as a separate 3-class model. This means category prediction and compliance routing are decoupled — you can change risk thresholds, add new hard negatives, or re-tune risk levels without touching the classifier, and vice versa.

---

## Performance summary

| Metric | Value |
|---|---|
| Classifier F1 (20 cats, 2,227-row test set) | **0.9047** |
| HS chapter accuracy when coarse category is correct | **97%** |
| Centroids | 20 categories × N HS chapters = **116 supervised centroids** |
| Risk vectors embedded at startup | **359** (12 global + 79 category × ~4 phrases each) |
| Throughput for 50,000 shipments (CPU) | **~60-90 seconds** |
| Extra inference cost added by compliance layer | **~0** (re-uses classifier embedding) |
| Compliance test suite | **24/24 passing** |

---

## Repository layout

```
shipment-nlp-classifier/
│
├── README.md                  ← you are here
├── SETUP.md                   ← step-by-step local setup
├── docker-compose.yml         ← PostgreSQL 16 + pgvector
├── schema.sql                 ← full DB schema
├── init_db.py                 ← one-shot schema + seed
│
├── seed/                      ← 20 categories, 96 HS chapters,
│                                ~1.9k keywords, ~16k labeled rows
│
├── ml-service/
│   ├── main.py                ← FastAPI app: /classify, /classify/batch,
│   │                            /reload, /health
│   ├── classifier.py          ← embedding, scoring, calibration, confidence
│   ├── compliance.py          ← semantic compliance decision layer ★
│   ├── risk_profile.json      ← global + per-category risk phrases ★
│   ├── test_compliance.py     ← 24 smoke tests ★
│   ├── centroid_builder.py    ← rebuild centroids from labeled data
│   ├── fit_calibration.py     ← Platt sigmoid calibration
│   ├── evaluate_threshold.py  ← threshold tuning on validation/test splits
│   └── discover_unknowns.py   ← HDBSCAN clustering on unclassified shipments
│
└── scripts/                   ← data generation utilities
```

★ = the compliance layer (the focus of this project).

---

## API at a glance

```http
POST /classify
{
  "shipment_id": "ship_001",
  "cargo_description": "depleted uranium fuel rods",
  "commodity_description": "nuclear material reactor grade"
}

→ 200 OK
{
  "shipment_id": "ship_001",
  "result": {
    "categories": ["energy"],
    "confidence_state": "classified",
    "scores": {...}
  },
  "compliance": {
    "compliance_decision": "block",
    "decision_reasons": [
      "semantic match 'depleted uranium nuclear material' (global, sim=0.842, matched='spent nuclear fuel rods'): nuclear material — IAEA controlled"
    ],
    "hard_negative_hits": [...],
    "risk_levels": {"energy": "medium"}
  }
}
```

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + load stats (categories, HS chapters, risk vectors, calibration) |
| `POST` | `/classify` | Single shipment → category + compliance decision |
| `POST` | `/classify/batch` | Up to 500 shipments per call, single batched embedding |
| `POST` | `/reload` | Reload centroids, keywords, calibration, and risk vectors from disk + DB |

Auto-generated OpenAPI docs at <http://localhost:8000/docs> when the service is running.

---

## Get it running

→ **[SETUP.md](SETUP.md)** has a 10-minute step-by-step guide: Docker, venv, DB init, centroid build, calibration, API start, and smoke tests.

---

## Design references

- `ml-service/compliance.py` — the semantic decision cascade with full inline docs
- `ml-service/risk_profile.json` — the rules database, schema-validated at startup
- `ml-service/test_compliance.py` — 24 tests covering exact phrasing, semantic paraphrases, category-scoped hits, high-risk routing, low-confidence routing, allow paths, and multi-label edges
- `shipment_classification_design.md` — design notes on the classifier itself (centroid approach, calibration, threshold tuning)
