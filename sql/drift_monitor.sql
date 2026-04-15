-- Drift-monitoring queries for shipment_classifications.
--
-- Run these on a schedule (daily/weekly) to detect when centroids start
-- losing fit to the real-world distribution. Early signals:
--   - rising share of `unclassified`/`low_confidence` shipments over time
--   - shrinking max-score histogram (mass moving left)
--   - categories that no longer receive any predictions
--
-- When any of these drift, rebuild centroids from refreshed labeled data
-- (centroid_builder.py) and re-fit calibration (fit_calibration.py).

-- ── 1. Daily confidence-state distribution (last 30 days) ──────────────
SELECT DATE_TRUNC('day', classified_at) AS day,
       confidence_state,
       COUNT(*) AS n,
       ROUND(100.0 * COUNT(*)::numeric /
             SUM(COUNT(*)) OVER (PARTITION BY DATE_TRUNC('day', classified_at)), 2
       ) AS pct
FROM   shipment_classifications
WHERE  classified_at >= now() - interval '30 days'
GROUP  BY 1, 2
ORDER  BY 1 DESC, 2;

-- ── 2. Max-score histogram (last 7 days) ──────────────────────────────
-- scores is JSONB; we extract the largest final_score per row. If the
-- distribution shifts left over time, centroids need a rebuild.
SELECT
    CASE
        WHEN max_score < 0.20 THEN '00-0.20'
        WHEN max_score < 0.30 THEN '0.20-0.30'
        WHEN max_score < 0.40 THEN '0.30-0.40'
        WHEN max_score < 0.50 THEN '0.40-0.50'
        WHEN max_score < 0.60 THEN '0.50-0.60'
        WHEN max_score < 0.70 THEN '0.60-0.70'
        WHEN max_score < 0.80 THEN '0.70-0.80'
        WHEN max_score < 0.90 THEN '0.80-0.90'
        ELSE                          '0.90+'
    END AS bucket,
    COUNT(*) AS n
FROM (
    SELECT classified_at,
           (SELECT MAX((value->>'final_score')::float)
            FROM jsonb_each(scores)) AS max_score
    FROM   shipment_classifications
    WHERE  classified_at >= now() - interval '7 days'
) t
GROUP BY bucket
ORDER BY bucket;

-- ── 3. Category-level traffic (last 30 days) ───────────────────────────
-- Unbalanced prediction volumes are often the first sign that a category
-- centroid no longer represents reality.
SELECT   category,
         COUNT(*) AS predictions
FROM (
    SELECT   UNNEST(categories) AS category
    FROM     shipment_classifications
    WHERE    classified_at >= now() - interval '30 days'
) t
GROUP BY category
ORDER BY predictions DESC;

-- ── 4. Growth of unclassified queue ────────────────────────────────────
-- Feed into discover_unknowns.py when this accumulates.
SELECT DATE_TRUNC('week', classified_at) AS week,
       COUNT(*) FILTER (WHERE confidence_state = 'unclassified') AS unclassified,
       COUNT(*) AS total,
       ROUND(100.0 *
             COUNT(*) FILTER (WHERE confidence_state = 'unclassified')::numeric /
             COUNT(*), 2) AS pct_unclassified
FROM   shipment_classifications
WHERE  classified_at >= now() - interval '90 days'
GROUP  BY 1
ORDER  BY 1 DESC;

-- ── 5. Stale centroids check ───────────────────────────────────────────
-- If any centroid hasn't been updated in > 14 days, consider a rebuild.
SELECT cc.name,
       cen.cluster_id,
       cen.sample_count,
       cen.updated_at,
       now() - cen.updated_at AS staleness
FROM   category_centroids cen
JOIN   classification_categories cc ON cc.id = cen.category_id
ORDER  BY cen.updated_at ASC;
