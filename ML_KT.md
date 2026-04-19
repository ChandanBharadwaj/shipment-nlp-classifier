# ML Knowledge-Transfer — Shipment NLP Classifier

A ground-up KT for whoever owns this classifier next. You don't need to read the full commit history. This doc covers **what's in the box, why each piece is shaped the way it is, and how to operate it**.

> Companion reads (skim after this): [README.md](README.md) for the business framing, [SETUP.md](SETUP.md) for a 10-minute local install, [`ml-service/config.py`](ml-service/config.py) for every tunable with its default.

---

## 1. What the system does

Two jobs, one embedding pass:

1. **Classify** a free-text shipment (`cargo_description` + `commodity_description`) into one of 20 trade categories, with a matched HS 2-digit chapter.
2. **Screen** it against 359 risk vectors (12 global + 79 per-category). Return `is_risky` plus auditable `decision_reasons`.

The compliance decision is **decoupled from classification confidence** — `is_risky` fires on `unclassified` rows too. That was deliberate: regex blocklists couple the two, and that's how prohibited cargo slips through.

---

## 2. Scoring model (per category, per shipment)

```
semantic_score = max_k  cosine(query_embedding, chapter_centroid_k)        # over k HS chapters
keyword_score  = 1 − exp(−α · Σ_matched_keywords weight)                   # saturating
final_score    = 0.8 · semantic_score + 0.2 · keyword_score                # defaults
probability    = sigmoid(platt_a · final + platt_b)                        # if Platt fit
```

- Centroids are L2-normalized so the dot product *is* cosine.
- Keyword matching is word-boundary with naive plural stripping (`batteries → batterie`, `cells → cell`) and case-insensitive.
- `α = 0.5` by default: 1 hit@1.0 → 0.39, 2 hits → 0.63, 3 hits → 0.78. Tuned so a single keyword doesn't dominate.
- Platt calibration is per-category. If `platt_a`/`platt_b` are NULL the probability is a passthrough of `final_score`.

### Confidence bands (gate on `final_score`, not `probability`)

| Band | Condition | Behavior |
|---|---|---|
| `unclassified` | `max_score < 0.35` | Return `[]`, reason `low_similarity` |
| `low_confidence` | `0.35 ≤ max_score < 0.45` | Return **top-1 only**, regardless of runner-ups |
| `classified` | `max_score ≥ 0.45` | Return every label ≥ threshold **and** within `MARGIN_DELTA` of the top |

We gate on `final_score` rather than Platt probability because the dataset has an 8:152 per-category imbalance — Platt sigmoids squash everything below ~0.1, which would make any reasonable threshold exclude all positives.

---

## 3. Multi-label pollution defense (the main body of recent work)

### 3.1 The problem

Single-token keywords like `motor`, `cell`, `park` appear in multiple categories' text for legitimate reasons. With ~1,936 keywords (897 of them single-token) across 20 categories, hand-curating the list doesn't scale. Symptom: a Tesla row that should classify as `automotive` also fires `toys`, `electronics`, `machinery`, `defense` because one keyword each leaks through above 0.45.

### 3.2 Three layers, each fixing a different failure mode

| Phase | Where | When it runs | What it fixes |
|---|---|---|---|
| **1. Top-margin rule** | `classifier._apply_bands` | Runtime, always on | Weak tail labels squeaking above threshold |
| **2. Keyword prune** | `diagnostics/prune_keywords.py` | Offline, one-shot | Keywords assigned to the wrong category |
| **3. Cross-encoder reranker** | `classifier._maybe_rerank` | Runtime, opt-in | Close calls between multiple legitimate candidates |

#### Phase 1 — Top-margin rule

```python
cutoff = max_score - margin_delta         # MARGIN_DELTA = 0.06
matched = [c for c in above_threshold if scores[c]["final_score"] >= cutoff]
```

Rationale: when the top score is 0.52 and a runner is at 0.47, they're semantically in the same league; when the runner is at 0.39, it's noise. δ = 0.06 is empirical (tuned on the 99-row regression CSV). Use [calibrate.py](ml-service/calibrate.py) to retune. The rule is scoped to the `classified` band — below threshold we still return exactly one top label, since "the winner by a hair" is fine when confidence is already low.

Counter: `margin_suppressions_total` — how many labels the rule dropped.

#### Phase 2 — Keyword discriminativeness filter

```
kw_vec  = embed(keyword)
cat_vec = mean of that category's chapter centroids (L2-normalized)
if  cosine(kw_vec, cat_vec) < MIN_KEYWORD_COSINE  AND  single-token  →  prune
```

Multi-token keywords (`"action figure"`, `"lithium ion battery"`) are **always kept** — a two-word phrase already carries enough context to sit close to the centroid. Single tokens are where ~90% of the pollution lives.

Ran as `python -m diagnostics.prune_keywords` — dry-run by default, `--apply` commits to DB. A CSV snapshot is written to `diagnostics/prune_backup_YYYYMMDD_HHMMSS.csv` before deletion; rollback is `COPY category_keywords FROM csv`.

Why cosine vs. TF-IDF/χ²? TF-IDF works on the corpus of keywords itself — gives you frequency-based rarity. Cosine vs. the centroid asks the stronger question: does this keyword belong to this category's *meaning*? Good first pass. If pollution returns post-prune, layering TF-IDF on top is the next move.

#### Phase 3 — Cross-encoder reranker (opt-in)

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22M params, ~15 ms/pair CPU).
- Only fires when state is `classified` **and** ≥2 candidates survived Phase 1. Nothing to rerank when there's one candidate; no point if confidence is too low.
- Pairs: `(shipment_text, category_description)`. Descriptions are auto-generated at startup from the top-10 weighted keywords per category (see `main._build_category_descriptions`).
- **CE margin is applied in sigmoid space, not on raw logits.** Logits are unbounded and model-specific; σ(logit) is in (0, 1) and comparable across models. Default `CROSS_ENCODER_MARGIN = 0.15` means "drop anything more than 15 probability-space points below the top."

Counters: `rerank_fires_total`, `rerank_drops_total`. Disabled by default (`RERANKER_ENABLED=false`); zero overhead when off — the model is never loaded.

### 3.3 Interaction rules (read once, internalize)

- **Order of introduction matters.** Prune first, then enable reranker. Category descriptions are built from top-weighted keywords; if you flip the flag on a service that hasn't yet `/reload`ed the pruned keyword table, the reranker reads stale descriptions. `/reload` rebuilds everything in lockstep.
- Phase 1 is **decision-time**; Phase 2 changes the **keyword table**; Phase 3 is a **tiebreak on Phase 1's survivors**. They compose.
- All three are independently reversible: Phase 1 via `--margin-delta 1.0` (disables it); Phase 2 via CSV replay; Phase 3 via env var.

---

## 4. Compliance layer ([compliance.py](ml-service/compliance.py))

The shipment embedding is already computed for classification — compliance **re-uses it**. No second inference pass.

- [risk_profile.json](ml-service/risk_profile.json) has two scopes: `global_blocked` (always fires) and per-category `hard_negatives`. Each entry has a canonical `phrase` plus 2–4 `aliases`; matcher takes the **max cosine across phrase + aliases** so spelling variants collapse into one rule.
- Decision cascade (first match wins): global block → category block → category review → high-risk category review → low-confidence review → Allow.
- Validated at startup. A malformed entry fails loud with the offending name.

The long-input hole (risk phrase buried in a 2,000-token manifest, truncated by the encoder's 256-token `max_seq_length`) is closed by token-aware chunking in [chunking.py](ml-service/chunking.py) — **every chunk** goes through the compliance scorer, not just the classification winner.

---

## 5. Centralized config ([config.py](ml-service/config.py))

Every tunable previously scattered across `classifier.py`/`main.py` now lives here with an env override. Restart to pick up changes (nothing is per-request).

| Constant | Env var | Default |
|---|---|---|
| `THRESHOLD_DEFAULT` | `CLASSIFY_THRESHOLD` | 0.45 |
| `UNCLASSIFIED_THRESHOLD_DEFAULT` | `UNCLASSIFIED_THRESHOLD` | 0.35 |
| `MARGIN_DELTA_DEFAULT` | `MARGIN_DELTA` | 0.06 |
| `KEYWORD_SATURATION_ALPHA` | `KEYWORD_SATURATION_ALPHA` | 0.5 |
| `MIN_KEYWORD_COSINE_DEFAULT` | `MIN_KEYWORD_COSINE` | 0.30 |
| `RERANKER_ENABLED` | `RERANKER_ENABLED` | false |
| `RERANKER_MODEL` | `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `CROSS_ENCODER_MARGIN_DEFAULT` | `CROSS_ENCODER_MARGIN` | 0.15 |

Do not re-declare these anywhere else. If you add a new knob, add it here and import it — the audit surface stays small.

---

## 6. Evaluation & tuning

### [evaluate_threshold.py](ml-service/evaluate_threshold.py)

Full-evaluation sweep on the `shipment_labels` table (has `train`/`validation`/`test` splits). Now takes `--margin-delta` so you can confirm a calibration on the test split.

```bash
python evaluate_threshold.py --split validation                      # find good defaults
python evaluate_threshold.py --split test --threshold 0.45 --margin-delta 0.06
```

### [calibrate.py](ml-service/calibrate.py)

Cheaper 2D grid search: embed each sample **once**, then sweep decision logic. A 7×6 = 42-cell grid runs in the time of a single full eval because `predict()` is called once per sample at threshold=0.0.

```bash
python calibrate.py                                   # validation split, default grid
python calibrate.py --thresholds 0.40 0.45 0.50 --deltas 0.04 0.06 0.08 0.10
```

Output: F1 heatmap + top-k recommended `(threshold, margin_delta)` with precision/recall/TP/FP/FN per cell.

**What's not in the grid:** `KEYWORD_SATURATION_ALPHA` (changes the cached final_score → invalidates the cache) and `MIN_KEYWORD_COSINE` (offline, Phase 2). Retune those manually.

**What's not calibrated from data yet:** `CROSS_ENCODER_MARGIN` — principled guess in sigmoid space, but no script picks it from labels. Future work if Phase 3 goes on by default.

### [fit_calibration.py](ml-service/fit_calibration.py)

Fits per-category Platt `(a, b)` on the validation split. Converts `final_score` into a comparable probability. Uncalibrated, the service still works — `probability` just passes through `final_score`.

---

## 7. Observability ([`/metrics`](ml-service/main.py) endpoint)

Lightweight in-process counters so ops can answer "is Phase 1 doing anything?" / "how often is the reranker firing?" without attaching a profiler.

```json
{
  "predictions_total":          12847,
  "margin_suppressions_total":  2109,    // labels Phase 1 dropped
  "rerank_fires_total":         0,       // rows where CE ran (0 if disabled)
  "rerank_drops_total":         0,       // labels CE dropped
  "margin_delta":               0.06,
  "reranker_enabled":           false
}
```

Known limits (flag these if you promote this to prod):
- Per-worker — under multi-worker uvicorn you'll see per-worker counts, not global. Fine for local/single-worker. Switch to statsd/prom when you scale.
- No latency histograms. If Phase 3 is on, add a `time.perf_counter()` around `_rerank_candidates` and emit to logs.
- Nothing alerts on drift — `margin_suppressions_total` jumping 10× overnight is a real signal, but no one's watching it.

---

## 8. Test surface

| File | Count | What it proves |
|---|---|---|
| `ml-service/test_compliance.py` | 24 | Every cascade path — global block, category block/review, paraphrase hits, high-risk routing, low-confidence routing, allow, multi-label edges |
| `ml-service/test_chunking.py` | 26 | `chunk_text` branches, aggregation, silent-truncation baseline (documents the bug), buried-tail-risk catch (proves the fix) |
| `ml-service/tests/test_margin_rule.py` | 10 | All branches of `_apply_bands`: δ boundary, unclassified/low_confidence/classified, counter increments |
| `ml-service/tests/test_reranker.py` | 12 | Sigmoid stability, short-circuit paths (disabled/single/no-descs), sigmoid-space margin drop & keep, `_maybe_rerank` gating + counters |
| `ml-service/tests/test_prune.py` | 7 | `_mean_centroid` normalization, multi-token gate shape |
| `ml-service/tests/test_metrics.py` | 3 | `reset_metrics`, `get_metrics` returns a snapshot (not a live reference), expected keys present |

```bash
python -m pytest tests/ -v          # 36 passing, ~42s (loads the ST model)
python test_compliance.py           # 24/24
python test_chunking.py             # 26/26
```

The pytest suite uses a monkey-patched stub cross-encoder — no real model download. Runs pure-logic tests; no DB needed.

Gap to close: no **end-to-end regression test**. The 99-row smoke CSV is eyeballed manually. A quick asserted regression harness (feed N shipments, compare labels/flags to a golden file) would close the loop — recommended next work if this system stays long-lived.

---

## 9. Rollout playbook (if you're turning it on)

1. **Phase 1 is already on** (default `MARGIN_DELTA = 0.06`). Inspect `/metrics` `margin_suppressions_total` after a batch — non-zero means it's catching stuff.
2. **Tune Phase 1** with `python calibrate.py`, pick top-F1 cell, update `MARGIN_DELTA` in `.env` (or `config.py`), restart, re-confirm on test split.
3. **Dry-run Phase 2**: `python -m diagnostics.prune_keywords`. Eyeball. Adjust `--min-cosine` if the cut is too aggressive (>40%) or timid (<5%).
4. **Apply Phase 2**: `python -m diagnostics.prune_keywords --apply`. CSV snapshot auto-written. `POST /reload` on the running service.
5. **(Optional) Enable Phase 3**: set `RERANKER_ENABLED=true`, restart. First startup downloads the CE model (~90 MB, one-time). `GET /health` reports `reranker_enabled: true`.
6. **Watch `/metrics`** for a week. `rerank_fires_total`/`rerank_drops_total` tell you whether Phase 3 is earning its keep.

Each phase is independently reversible.

---

## 10. Known edge cases & gotchas

- **`low_confidence` band always returns top-1.** Margin rule doesn't apply there — we already decided confidence is thin, so we commit to the single best guess rather than a speculative multi-label.
- **`embed_texts([])` returns an empty (0, dim) array** — don't call `[0]` on it. Callers guard with `if texts:`.
- **Keyword matching strips naive plurals only.** `batteries → batterie` (not `battery`). Fine for English freight text; weird on irregular plurals. Don't add singularization logic without a regression test.
- **Model fallback is silent.** If `EMBEDDING_MODEL` env var can't be loaded, we fall back to `all-MiniLM-L6-v2` with a printed warning. The warning lands in stdout — no re-raise. Intentional for frictionless local dev; noteworthy if you audit prod logs.
- **Platt sigmoids compress everything below ~0.1** for imbalanced categories. This is *why* `final_score` drives the bands, not `probability`. Don't flip to probability-based thresholding without recalibrating the bands.
- **CE descriptions are keyword-derived, not hand-written.** If you prune a category hard and only exotic keywords survive, the description degrades. Mitigation is monitoring; if it gets bad, hand-curate descriptions in DB.

---

## 11. Files you'll touch most

- **Tune a knob** → `ml-service/config.py` (or the corresponding env var).
- **Add/remove risk phrases** → `ml-service/risk_profile.json` + `POST /reload`.
- **Retune thresholds** → `python calibrate.py`, then update config.
- **Prune keywords** → `python -m diagnostics.prune_keywords [--apply]`, then `POST /reload`.
- **Rebuild centroids after relabeling** → `python centroid_builder.py`, then `POST /reload`.
- **Refit Platt calibration** → `python fit_calibration.py`, then `POST /reload`.
- **Add a test for new runtime logic** → `ml-service/tests/test_*.py` (pure-logic; keeps the suite fast).

---

## 12. What I'd build next (for the next owner)

1. **End-to-end regression harness.** 99-row CSV → golden labels → asserted diff. Closes the "unit tests pass, feature still broken" gap.
2. **Latency histograms in `/metrics`.** p50/p95/p99 for `predict` and `_rerank_candidates`. Needed before Phase 3 goes on by default.
3. **Drift alerts.** `margin_suppressions_total / predictions_total` ratio as a dashboard metric — jumps mean the keyword table shifted unexpectedly.
4. **TF-IDF / χ² keyword prune (Phase 2.5).** Cosine-vs-centroid is a blunt instrument; co-occurrence statistics would catch "`motor` legitimately belongs to multiple categories, demote it globally" cases.
5. **Calibrate `CROSS_ENCODER_MARGIN` from labels**, same pattern as `calibrate.py` but in CE space.
6. **Multi-worker metrics.** `prometheus_client` + shared dir, or switch the counters to a statsd client.
