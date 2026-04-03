-- Assigns train / validation / test splits to all seeded shipment_labels rows.
-- Run this after seed_shipment_labels.sql and seed_extra_labels.sql.
-- Safe to re-run (idempotent UPDATE).
--
-- Split strategy for base rows (xx_001 – xx_050, 50 per category):
--   rows 001–035  → train       (35/50 = 70%)
--   rows 036–043  → validation  ( 8/50 = 16%)
--   rows 044–050  → test        ( 7/50 = 14%)
--
-- Extra rows (xx_e01 – xx_eNN) → all train
-- (variety-boosting rows, not held out)
--
-- Global result across 20 categories:
--   train:       700 base  + 110 extra = 810 rows
--   validation:  160 rows
--   test:        140 rows

-- ── Reset all to train first (handles re-runs) ────────────────────────────────
UPDATE shipment_labels SET split = 'train';

-- ── Validation: base rows 036–043 per category ───────────────────────────────
UPDATE shipment_labels
SET    split = 'validation'
WHERE  shipment_id ~ '^[a-z]{2}_0(3[6-9]|4[0-3])$';

-- ── Test: base rows 044–050 per category ─────────────────────────────────────
UPDATE shipment_labels
SET    split = 'test'
WHERE  shipment_id ~ '^[a-z]{2}_0(4[4-9]|50)$';

-- ── Verify the distribution ───────────────────────────────────────────────────
-- Run this SELECT to confirm counts look right:
-- SELECT split, COUNT(*) FROM shipment_labels GROUP BY split ORDER BY split;
-- Expected:
--   test        140
--   train       810
--   validation  160
