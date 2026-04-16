-- Assigns train / validation / test splits to LEGACY shipment_labels rows only.
-- Legacy = shipment_id matching '[a-z]{2}_###' (xx_001 .. xx_050).
-- v2-generated rows ('v2_', 'v2c_' prefixes) already carry split inline and
-- must not be touched here.
-- Run this after seed_shipment_labels.sql. Safe to re-run (idempotent UPDATE).
--
-- Split strategy for base rows (xx_001 – xx_050, 50 per category):
--   rows 001–035  → train       (35/50 = 70%)
--   rows 036–043  → validation  ( 8/50 = 16%)
--   rows 044–050  → test        ( 7/50 = 14%)
--
-- Global result across 20 categories (legacy only):
--   train:       700 rows
--   validation:  160 rows
--   test:        140 rows
--
-- v2-generated rows keep their inline split from seed_shipment_labels_v2.sql.

-- ── Reset legacy rows to train (handles re-runs; skips v2 rows) ──────────────
UPDATE shipment_labels
SET    split = 'train'
WHERE  shipment_id ~ '^[a-z]{2}_\d{3}$';

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
