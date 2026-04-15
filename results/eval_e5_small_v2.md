# Eval: E5-small-v2 + P0–P4 fixes (`experiment/e5-small-v2`)

A/B of `intfloat/e5-small-v2` against MiniLM with everything else held constant
(same P0–P4 fixes, same 384-dim schema, same keyword/blend code paths). E5 is
asymmetric-trained and requires explicit `"query: "` / `"passage: "` prefixes;
the branch plumbs that through `build_query_text(cargo, commodity, role=...)`
with `role="passage"` at centroid-build time and `role="query"` at inference
and calibration-fit time.

## Config

- **Embedding model:** `intfloat/e5-small-v2` (384-dim, Microsoft)
- **Prefix strategy:** `"query: {cargo} {commodity}"` at inference, `"passage: {cargo} {commodity}"` at centroid build. Prefix prepended automatically by `preprocess.build_query_text()` when `EMBEDDING_MODEL` starts with `intfloat/e5`; MiniLM code path unchanged.
- **Centroids:** single centroid per category (`_choose_k = n // 50` → k=1 at current data volume).
- **Keyword scoring:** unchanged from MiniLM branch (α=0.5 saturating, word-boundary regex).
- **Quality gate:** unchanged (token-based `GENERIC_TOKENS`).
- **Calibration:** Platt sigmoid fitted per category on validation, stored in `classification_categories.platt_a/platt_b`. Not used for thresholding (same 8:152 class imbalance).
- **Score blend:** unchanged defaults 0.80 sem / 0.20 kw.
- **Threshold:** **0.78** (validation sweep winner — far higher than MiniLM's 0.45 because E5 cosines run hot).

## Test-split headline (140 samples)

| Metric | Value |
|---|---|
| Precision | 0.8298 |
| Recall    | 0.8357 |
| F1        | **0.8327** |
| TP / FP / FN | 117 / 24 / 23 |

**vs base (MiniLM, original code):** F1 −0.042, P −0.136, R +0.036.
**vs MiniLM + P0–P4 fixes:** F1 **−0.060**, P −0.094, R −0.029.

E5 underperforms both the base and the fixes branch on every metric except a
marginal recall gain over base. The asymmetric-training hypothesis (centroid as
passage, incoming shipment as query → better separability than a symmetric
model) did not materialize on this dataset.

## Per-category breakdown (threshold = 0.78)

| Category | TP | FP | FN | Precision | Recall | MiniLM-fix → E5 (recall) |
|---|---:|---:|---:|---:|---:|---|
| agriculture     | 5 | 2 | 2 | 0.714 | 0.714 | 1.000 → **0.714** |
| automotive      | 7 | 3 | 0 | 0.700 | 1.000 | 1.000 → 1.000 |
| chemicals       | 5 | 2 | 2 | 0.714 | 0.714 | 1.000 → **0.714** |
| construction    | 5 | 2 | 2 | 0.714 | 0.714 | 0.571 → **0.714** |
| cosmetics       | 7 | 0 | 0 | 1.000 | 1.000 | 1.000 → 1.000 |
| defense         | 5 | 1 | 2 | 0.833 | 0.714 | 1.000 → **0.714** |
| electronics     | 5 | 0 | 2 | 1.000 | 0.714 | 0.429 → **0.714** |
| energy          | 6 | 0 | 1 | 1.000 | 0.857 | 1.000 → **0.857** |
| food_beverages  | 5 | 1 | 2 | 0.833 | 0.714 | 0.857 → **0.714** |
| furniture       | 7 | 0 | 0 | 1.000 | 1.000 | 1.000 → 1.000 |
| luxury          | 7 | 1 | 0 | 0.875 | 1.000 | 0.857 → **1.000** |
| machinery       | 1 | 2 | 6 | 0.333 | 0.143 | 0.571 → **0.143** |
| metals          | 7 | 3 | 0 | 0.700 | 1.000 | 1.000 → 1.000 |
| minerals        | 5 | 1 | 2 | 0.833 | 0.714 | 0.714 → 0.714 |
| paper           | 7 | 0 | 0 | 1.000 | 1.000 | 1.000 → 1.000 |
| perishables     | 7 | 1 | 0 | 0.875 | 1.000 | 1.000 → 1.000 |
| pharmaceuticals | 7 | 0 | 0 | 1.000 | 1.000 | 0.571 → **1.000** |
| plastics        | 7 | 2 | 0 | 0.778 | 1.000 | 1.000 → 1.000 |
| textiles        | 6 | 3 | 1 | 0.667 | 0.857 | 1.000 → **0.857** |
| toys            | 6 | 0 | 1 | 1.000 | 0.857 | 0.714 → **0.857** |

**E5 wins:** electronics recall 0.43 → 0.71 (the hard category on MiniLM),
pharmaceuticals 0.57 → 1.00, construction 0.57 → 0.71, toys 0.71 → 0.86,
luxury 0.86 → 1.00 — all cases where MiniLM struggled with category-keyword-poor
descriptions.
**E5 losses:** machinery collapses to 0.143 recall (6 of 7 missed, even at
threshold 0.78 most scored above 0.78 for the *wrong* category), agriculture
1.00 → 0.71, chemicals 1.00 → 0.71, defense 1.00 → 0.71, food_beverages
0.86 → 0.71.

The precision hit is broad — 7 categories carry ≥2 FPs, vs only 4 on the MiniLM
branch. E5's compressed cosine range is the root cause: unrelated texts score
0.70+ against every passage-prefixed centroid, so a single threshold struggles
to separate any category cleanly.

## Top confusions (predicted → expected)

| Predicted | Expected | Count |
|---|---|---:|
| chemicals        | machinery     | 2 |
| food_beverages   | agriculture   | 1 |
| textiles         | agriculture   | 1 |
| plastics         | chemicals     | 1 |
| agriculture      | chemicals     | 1 |
| automotive       | construction  | 1 |
| textiles         | construction  | 1 |
| construction     | defense       | 1 |
| automotive       | defense       | 1 |
| machinery        | electronics   | 1 |
| automotive       | electronics   | 1 |
| plastics         | energy        | 1 |
| machinery        | energy        | 1 |
| agriculture      | food_beverages | 1 |
| perishables      | food_beverages | 1 |

Confusions are spread across many category pairs rather than concentrated on a
few — classic symptom of a compressed-similarity regime where nothing stands
out strongly against anything else.

## Validation sweep (F1 curve)

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.35 – 0.55 | 0.0500 | 1.0000 | 0.0952 *(saturated — everything passes)* |
| 0.60 | 0.0500 | 1.0000 | 0.0953 |
| 0.65 | 0.2078 | 1.0000 | 0.3441 |
| 0.70 | 0.7202 | 0.8688 | 0.7875 |
| 0.75 | 0.7931 | 0.8625 | 0.8263 |
| **0.78** | **0.8375** | **0.8375** | **0.8375** |
| 0.80 – 0.92 | 0.8375 | 0.8375 | 0.8375 *(plateau — score distribution has a hard gap above 0.78)* |

**Score distribution diagnosis:** at threshold ≤ 0.55 every (sample, category)
pair passes — 160 samples × 20 categories / 20 positives = 3040 FPs vs 160 TPs.
The usable range opens up only above 0.65, and the plateau above 0.78 means
pushing the threshold higher can't recover the remaining 26 misclassifications
because the score gap between correct and incorrect predictions has already
closed. This is a model-level property, not a tuning issue.

## P0–P4 correctness verifications

The P0–P4 fixes are inherited unchanged from `feature/nlp-fixes-minilm` — all
code paths (token-based gate, word-boundary keywords, batch commit, rollback,
`unclassified_threshold` clamping) are active and behave identically. Only the
embedding model and the preprocess prefix hook are different.

## Verdict: **revert to MiniLM**

E5-small-v2 test F1 = 0.8327 loses to both:
- baseline MiniLM (0.8750, −0.042)
- MiniLM + P0–P4 fixes (0.8930, −0.060)

Keep `feature/nlp-fixes-minilm` as the promotion candidate. This branch stays
as a documented negative result so the choice doesn't need to be relitigated.

## Notes

- Runtime: centroid build ~30 s on CPU (first run downloads ~130 MB of weights),
  test eval ~7 s.
- The `build_query_text(cargo, commodity, role="query"|"passage")` plumbing is
  model-agnostic — MiniLM simply ignores the role and returns the plain concat,
  so this branch's preprocess changes are safe to cherry-pick to MiniLM if a
  future asymmetric model is tried.
- BGE-small-en-v1.5 and E5-small-v2 have now both been A/B'd and rejected on
  this dataset. Two independent model swaps losing to MiniLM suggests the
  embedding layer is not the bottleneck — remaining FNs (electronics,
  pharmaceuticals on MiniLM) are likely data-coverage limitations that more
  training rows or explicit keywords would fix faster than model hunting.
- Branch tip: `experiment/e5-small-v2` (forked from `feature/nlp-fixes-minilm` @ `7b5851e`).
