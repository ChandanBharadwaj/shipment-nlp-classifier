-- HNSW bulk-load helper for shipment_classifications.embedding
--
-- Problem: HNSW builds each vector's graph edges incrementally on INSERT. On
-- a 1M-row backfill this turns an O(n) bulk load into O(n log n) with a large
-- constant factor — can take hours instead of minutes.
--
-- Workflow:
--   1. Drop the HNSW index.
--   2. Run your bulk insert / COPY.
--   3. Recreate the index in one pass (HNSW build is much faster in bulk mode).
--   4. Resume API traffic.
--
-- Safe to run multiple times — both the DROP and CREATE are IF [NOT] EXISTS.

-- Step 1: drop before bulk load
DROP INDEX IF EXISTS idx_class_embedding;

-- ...run your backfill here...
--
--  e.g. \copy shipment_classifications(shipment_id, embedding, categories, ...) FROM 'backfill.csv'
--       or INSERT INTO shipment_classifications SELECT ... FROM source_table;

-- Step 2: recreate after backfill
CREATE INDEX IF NOT EXISTS idx_class_embedding
    ON shipment_classifications
    USING hnsw (embedding vector_cosine_ops);

-- Optional: ANALYZE after bulk loads so the planner picks the index.
ANALYZE shipment_classifications;
