-- Shipment Classification — complete database schema.
-- Idempotent: safe to re-run (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS).
--
-- Tables created:
--   classification_categories  — master category list
--   category_keywords          — weighted keyword signals per category
--   category_centroids         — one 384-dim vector centroid per category
--   shipment_labels            — labeled training data (train/validation/test splits)
--   shipment_classifications   — classification results written by the API

-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ── classification_categories ─────────────────────────────────────────────────
-- Master list of categories. Decoupled from centroids — a category can exist
-- before its centroid is built.
CREATE TABLE IF NOT EXISTS classification_categories (
    id            SERIAL PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,      -- internal key e.g. 'electronics'
    display_name  TEXT NOT NULL,             -- human label e.g. 'Electronics & Components'
    description   TEXT,
    is_active     BOOLEAN DEFAULT true,
    created_at    TIMESTAMPTZ DEFAULT now(),
    updated_at    TIMESTAMPTZ DEFAULT now()
);

-- ── category_keywords ─────────────────────────────────────────────────────────
-- Secondary confidence signal used alongside the semantic score at inference.
CREATE TABLE IF NOT EXISTS category_keywords (
    id           SERIAL PRIMARY KEY,
    category_id  INT NOT NULL REFERENCES classification_categories(id) ON DELETE CASCADE,
    keyword      TEXT NOT NULL,
    weight       NUMERIC(3,2) DEFAULT 1.0,
    created_at   TIMESTAMPTZ DEFAULT now(),

    UNIQUE (category_id, keyword)
);

CREATE INDEX IF NOT EXISTS idx_kw_category ON category_keywords(category_id);

-- ── category_centroids ────────────────────────────────────────────────────────
-- L2-normalized centroid vectors per category. Built by centroid_builder.py
-- from train-split rows only. Never inserted manually.
--
-- Each category can have multiple sub-centroids (k-means clusters inside the
-- category). Score at inference = max cosine similarity across sub-centroids.
-- Categories with cluster_id=0 only are the single-centroid fallback.
CREATE TABLE IF NOT EXISTS category_centroids (
    category_id   INT NOT NULL REFERENCES classification_categories(id) ON DELETE CASCADE,
    cluster_id    INT NOT NULL DEFAULT 0,
    centroid      vector(384) NOT NULL,
    sample_count  INT,
    updated_at    TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (category_id, cluster_id)
);

-- Migration: on existing DBs created before multi-centroid support, the PK was
-- (category_id) only. Switch it to (category_id, cluster_id) idempotently.
DO $$
BEGIN
    -- Add cluster_id if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'category_centroids' AND column_name = 'cluster_id'
    ) THEN
        ALTER TABLE category_centroids ADD COLUMN cluster_id INT NOT NULL DEFAULT 0;
    END IF;

    -- If the old single-column PK is still in place, drop and re-add
    IF EXISTS (
        SELECT 1
        FROM   pg_constraint pc
        JOIN   pg_attribute  pa ON pa.attrelid = pc.conrelid AND pa.attnum = ANY(pc.conkey)
        WHERE  pc.conname = 'category_centroids_pkey'
        GROUP  BY pc.conname
        HAVING COUNT(*) = 1
    ) THEN
        ALTER TABLE category_centroids DROP CONSTRAINT category_centroids_pkey;
        ALTER TABLE category_centroids ADD PRIMARY KEY (category_id, cluster_id);
    END IF;
END $$;

-- ── classification_categories: tuning/calibration columns ─────────────────────
-- Added additively so existing rows get defaults.
--   semantic_weight, keyword_weight: per-category blend (default 0.8 / 0.2)
--   platt_a, platt_b: Platt-scaling calibration fit on validation split
--                     (probability = sigmoid(platt_a * final_score + platt_b))
--                     NULL until fit_calibration.py has been run.
--   threshold: optional per-category threshold override (NULL → use request)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='classification_categories' AND column_name='semantic_weight') THEN
        ALTER TABLE classification_categories ADD COLUMN semantic_weight NUMERIC(3,2) NOT NULL DEFAULT 0.80;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='classification_categories' AND column_name='keyword_weight') THEN
        ALTER TABLE classification_categories ADD COLUMN keyword_weight  NUMERIC(3,2) NOT NULL DEFAULT 0.20;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='classification_categories' AND column_name='platt_a') THEN
        ALTER TABLE classification_categories ADD COLUMN platt_a DOUBLE PRECISION;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='classification_categories' AND column_name='platt_b') THEN
        ALTER TABLE classification_categories ADD COLUMN platt_b DOUBLE PRECISION;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='classification_categories' AND column_name='threshold') THEN
        ALTER TABLE classification_categories ADD COLUMN threshold NUMERIC(4,3);
    END IF;
END $$;

-- ── shipment_labels ───────────────────────────────────────────────────────────
-- Labeled training data. split column controls which rows are used by each tool:
--   train      → centroid_builder.py builds centroids
--   validation → evaluate_threshold.py --split validation (sweep threshold)
--   test       → evaluate_threshold.py --split test (confirm once)
CREATE TABLE IF NOT EXISTS shipment_labels (
    id             SERIAL PRIMARY KEY,
    shipment_id    TEXT NOT NULL UNIQUE,
    category_name  TEXT NOT NULL REFERENCES classification_categories(name) ON DELETE CASCADE,
    cargo_text     TEXT NOT NULL,
    commodity_text TEXT NOT NULL,
    split          TEXT NOT NULL DEFAULT 'train'
                     CHECK (split IN ('train', 'validation', 'test'))
);

CREATE INDEX IF NOT EXISTS idx_sl_split    ON shipment_labels(split);
CREATE INDEX IF NOT EXISTS idx_sl_category ON shipment_labels(category_name);

-- ── shipment_classifications ──────────────────────────────────────────────────
-- Stores the full classification result for each shipment. Written by the API
-- when persist=true is passed. Used by discover_unknowns.py for cluster analysis.
--
-- NOTE on the HNSW index: building it on millions of rows during a bulk backfill
-- is slow. For large backfills, comment the CREATE INDEX line out, run the
-- backfill, then create it once afterwards.
CREATE TABLE IF NOT EXISTS shipment_classifications (
    shipment_id       TEXT PRIMARY KEY,
    embedding         vector(384),
    categories        TEXT[],
    scores            JSONB,
    confidence_state  TEXT DEFAULT 'classified'
                        CHECK (confidence_state IN ('classified', 'low_confidence', 'unclassified')),
    threshold_used    NUMERIC(4,3),
    model_version     TEXT,
    classified_at     TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_class_categories ON shipment_classifications USING GIN(categories);
CREATE INDEX IF NOT EXISTS idx_class_confidence ON shipment_classifications(confidence_state);
CREATE INDEX IF NOT EXISTS idx_class_embedding  ON shipment_classifications
    USING hnsw(embedding vector_cosine_ops);
