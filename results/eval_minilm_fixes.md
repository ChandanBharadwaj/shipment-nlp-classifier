# Eval: MiniLM + P0–P4 fixes (`feature/nlp-fixes-minilm`)

Same embedding model as base, but with all correctness (P0), NLP-quality (P1), keyword-layer (P2), architecture (P3), and evaluation (P4) improvements applied. The goal was to raise F1 without swapping the model.

## Config

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Prefix strategy:** plain concatenation after preprocessing (a `"Cargo: X. Commodity: Y."` structured prefix was attempted and reverted — shared tokens compressed the category margin).
- **Centroids:** multi-centroid via k-means, but `_choose_k = max(1, min(4, n // 50))` so at current data volume every category stays at k=1. Code path activates for categories ≥ 50 samples.
- **Keyword scoring:** saturating `1 − exp(−α · Σ weights)`, α=0.5 (1 hit ≈ 0.39, 2 hits ≈ 0.63). Word-boundary regex matching. Naive plural/singular variant tolerance.
- **Quality gate:** token-based — every non-punct token must be in `GENERIC_TOKENS` frozenset OR combined text < 10 chars.
- **Calibration:** Platt sigmoid fitted per category on validation split (8 pos / 152 neg each). Stored in `classification_categories.platt_a/platt_b`. Exposed as `scores[*].probability` in API response but **not used for thresholding** (8:152 imbalance pushes probabilities below 0.1).
- **Score blend:** per-category `semantic_weight * sem + keyword_weight * kw`. Defaults 0.80 / 0.20 via new `classification_categories` columns.
- **Threshold:** 0.45 (validation sweep winner, same as base).

## Test-split headline (140 samples)

| Metric | Value |
|---|---|
| Precision | 0.9237 |
| Recall    | 0.8643 |
| F1        | **0.8930** |
| TP / FP / FN | 121 / 10 / 19 |

**vs base:** F1 +0.018, Recall **+0.064**, Precision −0.042. The system trades a small amount of precision for materially better recall — a better operating point for a multi-label classifier.

## Per-category breakdown (threshold = 0.45)

| Category | TP | FP | FN | Precision | Recall | Base → New |
|---|---:|---:|---:|---:|---:|---|
| agriculture     | 7 | 1 | 0 | 0.875 | 1.000 | 1.00 → 1.00 |
| automotive      | 7 | 1 | 0 | 0.875 | 1.000 | 0.71 → **1.00** |
| chemicals       | 7 | 1 | 0 | 0.875 | 1.000 | 0.43 → **1.00** |
| construction    | 4 | 1 | 3 | 0.800 | 0.571 | 0.57 → 0.57 |
| cosmetics       | 7 | 0 | 0 | 1.000 | 1.000 | 0.86 → **1.00** |
| defense         | 7 | 1 | 0 | 0.875 | 1.000 | 0.86 → **1.00** |
| electronics     | 3 | 0 | 4 | 1.000 | 0.429 | 0.43 → 0.43 |
| energy          | 7 | 0 | 0 | 1.000 | 1.000 | 1.00 → 1.00 |
| food_beverages  | 6 | 0 | 1 | 1.000 | 0.857 | 0.86 → 0.86 |
| furniture       | 7 | 0 | 0 | 1.000 | 1.000 | 1.00 → 1.00 |
| luxury          | 6 | 1 | 1 | 0.857 | 0.857 | 0.86 → 0.86 |
| machinery       | 4 | 0 | 3 | 1.000 | 0.571 | 0.43 → **0.57** |
| metals          | 7 | 0 | 0 | 1.000 | 1.000 | 1.00 → 1.00 |
| minerals        | 5 | 0 | 2 | 1.000 | 0.714 | 0.86 → 0.71 |
| paper           | 7 | 0 | 0 | 1.000 | 1.000 | 1.00 → 1.00 |
| perishables     | 7 | 0 | 0 | 1.000 | 1.000 | 1.00 → 1.00 |
| pharmaceuticals | 4 | 0 | 3 | 1.000 | 0.571 | 0.57 → 0.57 |
| plastics        | 7 | 2 | 0 | 0.778 | 1.000 | 0.86 → **1.00** |
| textiles        | 7 | 2 | 0 | 0.778 | 1.000 | 1.00 → 1.00 |
| toys            | 5 | 0 | 2 | 1.000 | 0.714 | 0.71 → 0.71 |

**Big wins on recall:** chemicals 0.43 → 1.00, automotive 0.71 → 1.00, plastics 0.86 → 1.00, machinery 0.43 → 0.57.
**No change on the worst case:** electronics 0.43 → 0.43 (genuinely hard without model change; batteries/sensors rarely contain category-keyword signal).
**Regression:** minerals 0.86 → 0.71 — acceptable; small-sample category swings are noisy at n=7.

## Top confusions (predicted → expected)

| Predicted | Expected | Count |
|---|---|---:|
| `<none>` | electronics     | 4 |
| `<none>` | pharmaceuticals | 3 |
| `<none>` | construction    | 2 |
| `<none>` | toys            | 2 |
| plastics | chemicals       | 1 |
| textiles | construction    | 1 |
| plastics | energy          | 1 |
| agriculture | food_beverages | 1 |
| automotive | luxury         | 1 |
| chemicals | machinery       | 1 |

The `<none>` rows are recall misses — the weak categories need more training data or explicit keywords.

## Validation sweep (F1 curve)

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.35 | 0.7730 | 0.8938 | 0.8290 |
| 0.40 | 0.8294 | 0.8812 | 0.8545 |
| **0.45** | **0.8662** | **0.8500** | **0.8580** |
| 0.50 | 0.8684 | 0.8250 | 0.8462 |
| 0.55–0.75 | 0.8733 | 0.8187 | 0.8452 (plateau) |

## P0–P4 correctness verifications

- **Token-based quality gate:** `POST /classify {"cargo_description":"cargo goods","commodity_description":"various items"}` → `quality: "generic"`, `reason: "insufficient_input"`.
- **Word-boundary keyword match:** `"placid lake painting"` no longer matches `chemicals.keyword "acid"` — `chemicals.keyword_score = 0.0`.
- **Industrial pump** (machinery recall was 0.43): classifies correctly as `[machinery]` with keyword hit `"pump"`.
- **Multi-label** (`"refrigerated vaccine … cold chain"`): correctly returns `[perishables, pharmaceuticals]`.
- **Persist rollback** + **batch commit** in `main.py` — one commit at end of `/classify/batch`, try/except with `conn.rollback()` around write.
- **`unclassified_threshold` in request body** — accepted, always clamped to `min(unclassified_threshold, threshold)` so it can never exceed the main threshold.

## Notes

- Saturating keyword alpha was tuned from 1.2 → 0.5 during A/B. α=1.2 (1 hit ≈ 0.70) pushed too many wrong-category scores over 0.45 with a single coincidental keyword hit.
- Multi-centroid with `k = n//20` hurt this dataset (validation F1 0.858 → 0.842) — sub-centroids drift into neighboring categories' topic space and burn precision. `k = n//50` (effectively k=1 for current volume) chosen; the code path activates automatically when categories grow.
- BGE-small-en-v1.5 was A/B'd as a replacement model and rejected — narrow cosine range compressed separability. See `experiment/e5-small-v2` for the retry with an asymmetric-trained model.
- Platt calibration is fit and exposed in API responses for UI confidence display, but not used for the classify/unclassify decision due to class-imbalance compression.
- Runtime: 140-sample test eval in ~6 s on CPU.
- Branch tip: `feature/nlp-fixes-minilm`.
