# Evaluation — v2 data + hierarchical taxonomy

**Date:** 2026-04-16
**Branch:** `feature/data-v2-hierarchy`
**Model:** all-MiniLM-L6-v2 (384-dim)
**Threshold:** 0.70 (recommended by validation sweep, best F1)

## Summary

| | Validation | Test |
|---|---|---|
| **Samples** | 2,277 | 2,227 |
| **Precision** | 0.9141 | 0.9200 |
| **Recall** | 0.8871 | 0.8899 |
| **F1** | 0.9004 | **0.9047** |

## What changed from prior baseline

| Aspect | Prior (eval_minilm_fixes.md) | This eval |
|---|---|---|
| Training rows | 810 | 12,151 |
| Test rows | 140 | 2,227 |
| Categories | 20 | 20 |
| Centroids | ~30 (k-means, unsupervised) | 116 (supervised per-chapter) |
| HS chapters mapped | 0 | 96 |
| Keywords | ~400 hand-curated | 1,936 (hand + TF-IDF merged) |
| Calibration | fit on 160 validation rows | fit on 2,277 validation rows |
| Fine-grained chapter accuracy | n/a | 97.0% |

## Validation threshold sweep

```
 threshold   precision    recall        f1     tp     fp     fn      n
--------------------------------------------------------------------
      0.35      0.2241    0.9641    0.3637   2255   7807     84   2277
      0.40      0.3545    0.9547    0.5170   2233   4066    106   2277
      0.45      0.5091    0.9440    0.6615   2208   2129    131   2277
      0.50      0.6478    0.9320    0.7644   2180   1185    159   2277
      0.55      0.7682    0.9196    0.8371   2151    649    188   2277
      0.60      0.8556    0.9021    0.8783   2110    356    229   2277
      0.65      0.8997    0.8931    0.8964   2089    233    250   2277
      0.70      0.9141    0.8871    0.9004   2075    195    264   2277
      0.75      0.9171    0.8799    0.8981   2058    186    281   2277
```

Best by F1: **0.70** (precision=0.9141, recall=0.8871).

## Test set — per-category breakdown (threshold=0.70)

```
  category                tp    fp    fn    prec     rec
  -------------------------------------------------------
  agriculture             91     8    14   0.919   0.867
  automotive              76    15    11   0.835   0.874
  chemicals              132     5    19   0.964   0.874
  construction           104    17    17   0.860   0.860
  cosmetics               81     2     8   0.976   0.910
  defense                 82     1     4   0.988   0.953
  electronics            124     9    34   0.932   0.785
  energy                  89     4     4   0.957   0.957
  food_beverages          96    11     9   0.897   0.914
  furniture               72     7     5   0.911   0.935
  luxury                  88     7     3   0.926   0.967
  machinery              171    14    31   0.924   0.847
  metals                  96    20     9   0.828   0.914
  minerals                83    22     5   0.790   0.943
  paper                   75     6     2   0.926   0.974
  perishables            120     5    17   0.960   0.876
  pharmaceuticals        122     3    16   0.976   0.884
  plastics                85     4    16   0.955   0.842
  textiles               159    16    11   0.909   0.935
  toys                    90     1    17   0.989   0.841
```

### Strongest categories (F1 > 0.94)
- **defense** — 0.988 precision, 0.953 recall
- **energy** — 0.957 precision, 0.957 recall
- **cosmetics** — 0.976 precision, 0.910 recall
- **paper** — 0.926 precision, 0.974 recall
- **luxury** — 0.926 precision, 0.967 recall

### Weakest categories (by recall)
- **electronics** — 0.785 recall. Top confusions: machinery→electronics (8), automotive→electronics (6). EV motors, industrial controllers are genuinely ambiguous.
- **toys** — 0.841 recall. Gets missed entirely (`<none>`) when descriptions are too generic.
- **plastics** — 0.842 recall. Borderline items with chemicals.

### Weakest categories (by precision)
- **minerals** — 0.790 precision. Over-predicts on construction aggregates (stone, gravel) and agriculture-adjacent (soil minerals).
- **metals** — 0.828 precision. Some construction metals (rebar, structural steel) get tagged metals instead of construction.

## Top confusions (test)

```
  <none>               -> perishables           15
  <none>               -> electronics           13
  <none>               -> automotive            9
  <none>               -> textiles              9
  machinery            -> electronics           8
  textiles             -> construction          6
  automotive           -> electronics           6
  <none>               -> pharmaceuticals       6
  <none>               -> chemicals             6
  minerals             -> construction          6
```

The dominant failure mode is `<none>` — the model doesn't clear the 0.70 threshold at all for those rows. These are typically short or ambiguous descriptions.

## Fine-grained HS-chapter accuracy (test, coarse-correct subset)

When the coarse category prediction is correct, how often is `best_chapter` also correct?

```
  category                hit   miss     acc
  ---------------------------------------------
  agriculture              84      3   0.966
  automotive               74      0   1.000
  chemicals               126      1   0.992
  construction             99      3   0.971
  cosmetics                78      0   1.000
  defense                  79      0   1.000
  electronics             121      2   0.984
  energy                   84      0   1.000
  food_beverages           87      4   0.956
  furniture                63      3   0.955
  luxury                   81      1   0.988
  machinery               165      3   0.982
  metals                   85      5   0.944
  minerals                 81      0   1.000
  paper                    58     11   0.841
  perishables             113      0   1.000
  pharmaceuticals         118      0   1.000
  plastics                 78      1   0.987
  textiles                130     22   0.855
  toys                     86      0   1.000
  OVERALL                1890     59   0.970
```

**97.0% chapter accuracy** — essentially free HS 2-digit metadata at zero extra inference cost.

Paper (0.841) and textiles (0.855) are the weakest because of sibling-chapter confusion within those categories (e.g., HS 48 printed matter vs 49 books; HS 61 knitted vs 62 woven garments).

## Centroid geometry

116 centroids total across 20 categories.

**Cross-category pairs with cosine > 0.85** (potential confusion zones):

| Cosine | Left | Right | Note |
|---|---|---|---|
| 0.993 | agriculture/13 | chemicals/13 | Chapter 13 (gums/resins) shared multi-label |
| 0.972 | luxury/42 | textiles/42 | Chapter 42 (leather goods) shared multi-label |
| 0.954 | perishables/30 | pharmaceuticals/30 | Chapter 30 (pharma) shared multi-label |
| 0.901 | agriculture/24 | food_beverages/24 | Chapter 24 (tobacco) shared multi-label |
| 0.895 | electronics/90 | pharmaceuticals/90 | Chapter 90 (instruments) shared multi-label |

All five are **genuine multi-label edges** from confusable training rows — not model defects. The taxonomy correctly assigns these chapters to one primary category but the multi-label confusables teach the model both are valid.

Only 1 intra-category pair above 0.90: textiles/61 ↔ textiles/62 (knitted vs woven garments, cos=0.961).

## Data coverage

- All 96 HS chapters covered, every chapter >= 20 train rows
- Median Jaccard similarity: 0.040–0.135 per category (good vocabulary diversity)
- Single-chapter-dominated categories (expected): automotive (87), cosmetics (33), defense (93), energy (27), minerals (26)

## Pipeline

```
python init_db.py                                              # schema + seeds
python centroid_builder.py                                     # 116 supervised centroids
python fit_calibration.py                                      # Platt on 2,277 validation rows
python evaluate_threshold.py --split validation                # sweep → threshold=0.70
python evaluate_threshold.py --split test --threshold 0.70     # final: F1=0.9047
python evaluate_threshold.py --split test --threshold 0.70 --fine  # chapter accuracy: 97%
```
