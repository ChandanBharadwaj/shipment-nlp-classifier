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

## Why semantic search beats keyword matching — the pitch

A keyword blocklist catches what you remembered to write down. Brokers — including bad-faith ones — **paraphrase**. Trade language drifts. Acronyms have a dozen alternate spellings. Every gap between the literal rule and the actual freight description is a screening hole that has to be patched by hand. The list grows; threats outpace the patching; the sieve rots.

LLM sentence encoders close that gap. A risk phrase and its aliases are embedded once into a 384-dim vector; any shipment whose description lands in the same neighborhood — **even with zero shared keywords** — fires the rule.

### Real shipment text vs. what each approach catches

Every row below is a shipment description with **no significant keyword overlap** with the rule it triggers. Cosines were measured live with `all-MiniLM-L6-v2` against the entries in [`risk_profile.json`](ml-service/risk_profile.json). Block thresholds: **0.62** (global) / **0.55** (category review).

| What the broker types | Literal-keyword regex | Semantic match (this system) | Cosine |
|---|---|---|---|
| `rocket-propelled grenade launcher RPG-7 infantry weapon` | misses (canonical rule uses `grenade launcher heavy weapon`) | ✅ **defense** — alias `"rocket propelled grenade RPG"` | **0.86** |
| `shoulder-launched surface-to-air missile system man-portable` | misses (no `MANPADS` token) | ✅ **defense block** — alias `"shoulder-fired surface-to-air missile"` | **0.85** |
| `hand-launched anti-aircraft missile shoulder-fired infantry weapon` | misses (no `MANPADS`/`Stinger` token) | ✅ **defense block** — same MANPADS rule | **0.78** |
| `used reactor fuel rods radioactive waste long-term storage casks` | misses (no `uranium`/`depleted` token) | ✅ **global block** — alias `"spent nuclear fuel rods"` of the depleted-uranium rule | **0.76** |
| `thermal imaging weapon sight uncooled microbolometer` | misses (canonical rule says `night vision military optics`) | ✅ **defense review** — alias `"thermal weapon sight"` | **0.75** |
| `simultaneous five-axis milling machine for titanium aerospace parts` | partial — `five-axis` may hit but `5-axis` / `multi-axis` evade | ✅ **machinery** — phrase `"5-axis CNC machining center advanced"` | **0.68** |
| `high-strength 18% nickel steel for solid rocket motor casings` | misses (no `maraging` token) | ✅ **metals** — alias `"rocket motor case steel"` of the maraging rule | **0.67** |
| `cruise missile guidance navigation electronics` | misses (canonical rule is `MTCR controlled missile`) | ✅ **defense block** — alias `"cruise missile"` of the guided-missile rule | **0.66** |
| `mine-resistant ambush-protected armored troop carrier` | misses (no `MRAP` token) | ✅ **automotive review** — phrase `"armored military vehicle"` | **0.66** |
| `electronic countermeasure pod radar jamming aircraft` | misses (no literal `electronic warfare`) | ✅ **electronics review** — phrase `"radar electronic warfare system"` | **0.60** |

> **Every cosine in that table clears its category's threshold without sharing a single significant keyword with the matched rule.** A regex list catches *none* of them until an analyst writes each variant by hand — and the next paraphrase ships next week.

### The critical gap, closed

| Failure mode | Keyword blocklist | This system |
|---|---|---|
| Acronym expansion (`MANPADS` → `shoulder-fired SAM`) | misses | catches |
| Brand/model swap (`Stinger` → `FIM-92`, `RPG` → `RPG-7`) | misses | catches |
| Technical paraphrase (`maraging steel` → `18% nickel rocket motor steel`) | misses | catches |
| Hyphenation drift (`5-axis` vs. `five-axis` vs. `multi-axis`) | needs every variant | one rule covers all |
| Phrasing nobody on the analyst team predicted | misses until rule is written | catches at deploy time |
| Risk phrase buried in a 2,000-token manifest | catches only if literal token survives | catches via auto-chunking + per-chunk compliance |

### Why this works (one paragraph)

The encoder learned, during pre-training, that `"MRAP"` and `"armored military vehicle"` show up in similar contexts and therefore live near each other in vector space. Same for `"thermal weapon sight"` ↔ `"infrared night vision"`, `"FIM-92"` ↔ `"Stinger"`, `"spent nuclear fuel"` ↔ `"used reactor fuel rods"`. We write **one canonical phrase plus 2–4 aliases per risk**; the model carries every other paraphrase a real broker (or a smuggler trying to evade screening) might use. Maintenance burden goes from *"enumerate every variant forever"* to *"name the concept once."*

A keyword list is a sieve with a fixed mesh. A semantic matcher is a sieve that adapts to the language people actually write.

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

## Multi-label pollution defense (three layers)

Single-token keywords like `motor`, `cell`, `park` legitimately appear in multiple categories' text. Without a defense, they push unrelated categories above the classification threshold and produce spurious secondary labels. We layer three independent mitigations — each solves a different failure mode, each is separately reversible:

### Phase 1 — Top-margin rule (runtime, always on)

In the `classified` band, secondary labels must be within `MARGIN_DELTA` (default **0.06**) of the top score. Primary decision remains: `final_score ≥ threshold` → matched; now it must *also* be within δ of the leader. Suppresses weak tail labels at zero cost (one comparison per category). Implementation: `_apply_bands` in [classifier.py](ml-service/classifier.py).

### Phase 2 — Keyword discriminativeness filter (offline)

A one-shot script that prunes single-token keywords whose embedding cosine against their assigned category's mean centroid falls below `MIN_KEYWORD_COSINE` (default **0.30**). Multi-token keywords are always kept — short phrases carry enough context to be semantically grounded. Writes a CSV snapshot before deleting; rollback is a plain `COPY … FROM csv`.

```bash
python -m diagnostics.prune_keywords                    # dry run, all categories
python -m diagnostics.prune_keywords --category toys    # preview one category
python -m diagnostics.prune_keywords --apply            # commit after review
```

### Phase 3 — Cross-encoder reranker (runtime, feature-flagged, default **off**)

When two or more labels survive Phase 1, an optional cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores each `(shipment_text, category_description)` pair jointly. Logits are squashed through sigmoid, so the margin of **0.15** is interpretable in (0, 1) probability space — same threshold is meaningful across different CE models. Category descriptions are auto-built at startup from the top-10 weighted keywords per category.

- Default off. Flip `RERANKER_ENABLED=true` to enable; `/health` reports state.
- Zero overhead when disabled — the model is never loaded.
- When enabled, only fires on rows with ≥2 post-margin candidates (~50–150 ms added on those rows).
- Fires are counted at `/metrics` so ops can see how often Phase 3 actually does anything.

### Ordering & safety

Always prune **before** enabling the reranker. Category descriptions are built from top-weighted keywords, so pruning afterwards would leave the reranker reading a stale description set until `/reload`. `/reload` rebuilds descriptions in lockstep with the keyword table, so the documented sequence is: prune → `/reload` → set `RERANKER_ENABLED=true` → restart.

### Tuning

Two decision knobs (`threshold`, `margin_delta`) are gridable cheaply with [calibrate.py](ml-service/calibrate.py), which embeds each labeled sample once and re-applies the decision logic per grid cell. A 42-cell sweep runs in the time of one full evaluation.

```bash
python calibrate.py                                     # validation split, default grid
python calibrate.py --thresholds 0.40 0.45 0.50 --deltas 0.04 0.06 0.08
```

Output is an F1 heatmap plus the top-k `(threshold, delta)` recommendations with precision/recall/TP-FP-FN alongside.

---

## What this approach gets right

### Paraphrase resilience (the regex problem, solved)
See the **[pitch section](#why-semantic-search-beats-keyword-matching--the-pitch)** above for the full table of military-context examples (MANPADS, RPG-7, MRAP, maraging steel, thermal weapon sights, spent nuclear fuel, electronic countermeasures) where shipment text shares **zero significant keywords** with the matched rule and the embedding still catches it.

### Context-aware screening
Hard negatives are **scoped to categories** so context drives meaning:
- `"acetic anhydride"` is normal industrial chemistry in a paint shipment (no rule fires) but a heroin precursor in a `chemicals` shipment (Review).
- `"firearm"` is benign in a sporting goods context but a misclassification flag in a `toys` shipment.

A pure global blocklist can't make these distinctions.

### Zero extra inference cost
The shipment is encoded **once** by `all-MiniLM-L6-v2`. The classifier and compliance layer both consume that single 384-dim vector. Compliance adds only a `(N, 384) @ (384,)` matrix multiplication against ~360 pre-computed risk vectors — that's microseconds.

### Long-input safety: automatic chunking
Sentence-transformer models silently truncate inputs past `max_seq_length` (256 tokens for MiniLM). A 2,000-token manifest with `depleted uranium` in the final sentence would be classified, scored, and approved — the risk phrase never reaching `apply_compliance`. We close that hole with token-aware chunking: long inputs are split at sentence boundaries into model-fitting windows (with 32-token overlap), each chunk is embedded, classification picks the winning chunk per category (preserving `final_score == sw·sem + kw·kw`), and compliance scores **every** chunk against the risk vectors so a tail-buried phrase still surfaces. Short inputs (the common case) take a zero-overhead fast path. The response carries `meta.chunks_processed` for observability — `> 1` indicates the input exceeded the context window and was chunked.

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
| Chunking test suite | **26/26 passing** |
| Pollution-defense unit suite (`tests/`) | **36/36 passing** |

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
│   │                            /reload, /health, /metrics
│   ├── classifier.py          ← embedding, scoring, margin rule, reranker hook
│   ├── config.py              ← all tunables + env overrides (single source) ★
│   ├── compliance.py          ← semantic compliance decision layer ★
│   ├── chunking.py            ← token-aware long-input splitter ★
│   ├── risk_profile.json      ← global + per-category risk phrases ★
│   ├── calibrate.py           ← 2D grid sweep (threshold × margin_delta) ★
│   ├── test_compliance.py     ← 24 smoke tests ★
│   ├── test_chunking.py       ← 26 chunking + tail-risk tests ★
│   ├── tests/                 ← 36 pytest unit tests — margin rule,
│   │                            reranker, prune, metrics ★
│   ├── diagnostics/
│   │   └── prune_keywords.py  ← offline keyword discriminativeness filter ★
│   ├── centroid_builder.py    ← rebuild centroids from labeled data
│   ├── fit_calibration.py     ← Platt sigmoid calibration
│   ├── evaluate_threshold.py  ← threshold/margin tuning on val/test splits
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
    "is_risky": true,
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
| `GET` | `/health` | Liveness + load stats (categories, HS chapters, risk vectors, calibration, `reranker_enabled`) |
| `GET` | `/metrics` | In-process counters: `predictions_total`, `margin_suppressions_total`, `rerank_fires_total`, `rerank_drops_total` |
| `GET` | `/risk-levels` | `{category: risk_level}` map for UI rendering — auto-refreshes on `/reload` |
| `POST` | `/classify` | Single shipment → category + compliance decision |
| `POST` | `/classify/batch` | Up to 500 shipments per call, single batched embedding |
| `POST` | `/reload` | Reload centroids, keywords, calibration, risk vectors, and category descriptions |

Auto-generated OpenAPI docs at <http://localhost:8000/docs> when the service is running.

### Bulk-test UI

A stateless single-page UI ships at <http://localhost:8000/ui>. Drag in a CSV
with columns `shipment_id,cargo_description,commodity_description` (plus
optional `threshold` / `unclassified_threshold`) and you'll get a colour-coded
results table — one chip per matched category, a `RISKY` / `clean` compliance
badge per row, and the chunk count for any input that was long enough to
trigger chunked embedding.

The UI never persists results: every batch call sets `persist: false`, and the
page sets no cookies, no `localStorage`, no `sessionStorage`. Reload to start
over. A demo CSV (`ml-service/static/sample.csv`) covers all three confidence
states plus the depleted-uranium risky path so you can verify the install in
one drag-and-drop.

> **Response shape note:** `result.scores` contains only the categories that
> fired (or the top-3 runners-up when `confidence_state == "unclassified"`,
> so the response explains *why* nothing matched). The full 20-row breakdown
> is still written to `shipment_classifications.scores` for offline audit.

---

## Get it running

→ **[SETUP.md](SETUP.md)** has a 10-minute step-by-step guide: Docker, venv, DB init, centroid build, calibration, API start, and smoke tests.

---

## Design references

- `ml-service/compliance.py` — the semantic decision cascade with full inline docs
- `ml-service/risk_profile.json` — the rules database, schema-validated at startup
- `ml-service/test_compliance.py` — 24 tests covering exact phrasing, semantic paraphrases, category-scoped hits, high-risk routing, low-confidence routing, allow paths, and multi-label edges
- `shipment_classification_design.md` — design notes on the classifier itself (centroid approach, calibration, threshold tuning)
