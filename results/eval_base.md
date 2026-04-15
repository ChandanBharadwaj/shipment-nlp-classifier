# Eval: Base (original repo on master)

Baseline against which every later change is measured. Original code as shipped in the initial commit — no preprocessing module, substring keyword matching, single centroid per category, exact-match generic-phrase gate, no calibration, 80/20 hard-coded semantic/keyword blend.

## Config

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Prefix strategy:** plain concatenation `f"{cargo} {commodity}"`
- **Centroids:** single centroid per category (incremental mean, L2-normalized)
- **Keyword scoring:** `hits_weight / total_weight`, substring match (`if kw in text_lower`)
- **Quality gate:** exact-match against a small frozenset of generic phrases
- **Calibration:** none
- **Score blend:** fixed `0.8 * semantic + 0.2 * keyword`
- **Threshold:** 0.45 (validation sweep winner)
- **Training:** 810 rows split train, 160 validation, 140 test

## Test-split headline (140 samples)

| Metric | Value |
|---|---|
| Precision | 0.9655 |
| Recall    | 0.8000 |
| F1        | **0.8750** |
| TP / FP / FN | 112 / 4 / 28 |

## Per-category breakdown (threshold = 0.45)

| Category | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| agriculture     | 7 | 0 | 0 | 1.000 | 1.000 |
| automotive      | 5 | 0 | 2 | 1.000 | 0.714 |
| chemicals       | 3 | 0 | 4 | 1.000 | **0.429** |
| construction    | 4 | 0 | 3 | 1.000 | 0.571 |
| cosmetics       | 6 | 0 | 1 | 1.000 | 0.857 |
| defense         | 6 | 1 | 1 | 0.857 | 0.857 |
| electronics     | 3 | 0 | 4 | 1.000 | **0.429** |
| energy          | 7 | 0 | 0 | 1.000 | 1.000 |
| food_beverages  | 6 | 1 | 1 | 0.857 | 0.857 |
| furniture       | 7 | 0 | 0 | 1.000 | 1.000 |
| luxury          | 6 | 0 | 1 | 1.000 | 0.857 |
| machinery       | 3 | 0 | 4 | 1.000 | **0.429** |
| metals          | 7 | 0 | 0 | 1.000 | 1.000 |
| minerals        | 6 | 0 | 1 | 1.000 | 0.857 |
| paper           | 7 | 0 | 0 | 1.000 | 1.000 |
| perishables     | 7 | 0 | 0 | 1.000 | 1.000 |
| pharmaceuticals | 4 | 0 | 3 | 1.000 | 0.571 |
| plastics        | 6 | 1 | 1 | 0.857 | 0.857 |
| textiles        | 7 | 1 | 0 | 0.875 | 1.000 |
| toys            | 5 | 0 | 2 | 1.000 | 0.714 |

Three categories at 0.429 recall — **chemicals, electronics, machinery** — are the clearest weak spots.

## Top confusions

N/A — confusion matrix tracking was added in P4. Master's `evaluate_threshold.py` only prints per-category TP/FP/FN.

## Validation sweep (F1 curve)

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.20 | 0.2761 | 0.8750 | 0.4198 |
| 0.25 | 0.4317 | 0.8688 | 0.5768 |
| 0.30 | 0.6618 | 0.8562 | 0.7466 |
| 0.35 | 0.8110 | 0.8313 | 0.8210 |
| 0.40 | 0.8523 | 0.7937 | 0.8220 |
| **0.45** | **0.8803** | **0.7812** | **0.8278** |
| 0.50 | 0.8794 | 0.7750 | 0.8239 |
| 0.55 | 0.8794 | 0.7750 | 0.8239 |
| 0.60 | 0.8794 | 0.7750 | 0.8239 |

## Notes

- Known correctness bugs at this baseline: `"cargo goods"` slips through the quality gate (only `"cargo"` and `"goods"` match the exact-match gate individually, not in combination); `"placid"` matches `chemicals.keyword "acid"` via substring; batch endpoint commits once per shipment; no rollback on persist failure; `unclassified_threshold` floor can silently override caller's `threshold`. See P0 fixes on `feature/nlp-fixes-minilm`.
- Runtime: 140-sample test eval completes in ~6 s on CPU.
- Branch tip: `master` (`a547e25`).
