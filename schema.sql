-- Shipment Classification — complete database schema.
-- Idempotent: safe to re-run (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS).
--
-- Tables created:
--   classification_categories  — master category list
--   category_hs_chapters       — coarse category ↔ HS 2-digit chapter edges
--   category_keywords          — weighted keyword signals per category
--   category_centroids         — 384-dim centroid per (category, hs_chapter)
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

-- ── category_hs_chapters ──────────────────────────────────────────────────────
-- 2-level hierarchical taxonomy: every active HS 2-digit chapter (01..97,
-- excluding 77 which is WCO-reserved) is assigned to exactly one coarse
-- category. A single category has 1..N chapters. The UNIQUE(hs_chapter)
-- constraint enforces the "each chapter belongs to exactly one category"
-- invariant the classifier relies on.
CREATE TABLE IF NOT EXISTS category_hs_chapters (
    category_id    INT  NOT NULL REFERENCES classification_categories(id) ON DELETE CASCADE,
    hs_chapter     TEXT NOT NULL,        -- '01' .. '97' (zero-padded)
    chapter_title  TEXT NOT NULL,
    is_primary     BOOL NOT NULL DEFAULT false,  -- primary anchor vs absorbed
    PRIMARY KEY (category_id, hs_chapter),
    UNIQUE (hs_chapter)
);

CREATE INDEX IF NOT EXISTS idx_chc_chapter  ON category_hs_chapters(hs_chapter);
CREATE INDEX IF NOT EXISTS idx_chc_category ON category_hs_chapters(category_id);

-- ── category_keywords ─────────────────────────────────────────────────────────
-- Secondary confidence signal used alongside the semantic score at inference.
--
-- CCTR (Context-Conditional Token Resolution) extension:
--   hs_chapter    pins each keyword to a specific HS 2-digit chapter, so the
--                 same word can carry different weights in different chapters
--                 (battery in 85/87/95/30 each get their own row).
--   signal_class  declares the keyword's role in scoring:
--                   'anchor'     — strong positive; rebalancer cannot lower
--                   'signal'     — weighted positive; the default class
--                   'suppressor' — negative evidence for this chapter
--                   'modifier'   — context-shifter that retypes neighbour signals
--   source        provenance tag: 'manual' | 'tfidf' | 'merged' | 'discovered'
--   notes         justification / owner / ticket — required by governance for
--                 anchor / suppressor / modifier rows.
CREATE TABLE IF NOT EXISTS category_keywords (
    id           SERIAL PRIMARY KEY,
    category_id  INT NOT NULL REFERENCES classification_categories(id) ON DELETE CASCADE,
    keyword      TEXT NOT NULL,
    weight       NUMERIC(3,2) DEFAULT 1.0,
    created_at   TIMESTAMPTZ DEFAULT now(),

    UNIQUE (category_id, keyword)
);

CREATE INDEX IF NOT EXISTS idx_kw_category ON category_keywords(category_id);

-- CCTR migration: add hs_chapter / signal_class / source / notes columns and
-- swap the legacy UNIQUE (category_id, keyword) for a chapter+class composite.
-- The new partial unique index allows rows without hs_chapter (legacy) to
-- coexist while the migration script populates them. Once the migration has
-- run and all rows carry an hs_chapter, the legacy NULL-permitting branch is
-- a no-op.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='category_keywords' AND column_name='hs_chapter') THEN
        ALTER TABLE category_keywords ADD COLUMN hs_chapter TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='category_keywords' AND column_name='signal_class') THEN
        ALTER TABLE category_keywords
            ADD COLUMN signal_class TEXT NOT NULL DEFAULT 'signal'
            CHECK (signal_class IN ('anchor','signal','suppressor','modifier'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='category_keywords' AND column_name='source') THEN
        ALTER TABLE category_keywords ADD COLUMN source TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='category_keywords' AND column_name='notes') THEN
        ALTER TABLE category_keywords ADD COLUMN notes TEXT;
    END IF;

    -- Drop legacy UNIQUE (category_id, keyword) — same word can now appear in
    -- multiple chapters and signal classes for the same category.
    IF EXISTS (SELECT 1 FROM pg_constraint
               WHERE conname = 'category_keywords_category_id_keyword_key') THEN
        ALTER TABLE category_keywords
            DROP CONSTRAINT category_keywords_category_id_keyword_key;
    END IF;
END $$;

-- New composite uniqueness. Partial so legacy rows (hs_chapter NULL) don't
-- block re-runs before the migration script has been applied.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_category_keywords_cctr
    ON category_keywords (category_id, hs_chapter, keyword, signal_class)
    WHERE hs_chapter IS NOT NULL;

-- Legacy idempotency: pre-migration rows (hs_chapter IS NULL) still need to
-- behave like the dropped UNIQUE (category_id, keyword) so that re-running
-- seed_keywords*.sql is a no-op. This index disappears in effect once every
-- row has been assigned a chapter by migrate_keywords_to_per_chapter.py — the
-- partial predicate then matches zero rows.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_category_keywords_legacy
    ON category_keywords (category_id, keyword)
    WHERE hs_chapter IS NULL;

CREATE INDEX IF NOT EXISTS idx_kw_chapter      ON category_keywords(hs_chapter);
CREATE INDEX IF NOT EXISTS idx_kw_signal_class ON category_keywords(signal_class);

-- ── token_collisions ──────────────────────────────────────────────────────────
-- Declarative ambiguity registry. One row per known polysemy pattern. The
-- inference engine consults this table only for tokens actually present in
-- the request text (set membership), so its size is irrelevant to per-request
-- cost. See the CCTR plan for the resolution-rule grammar.
CREATE TABLE IF NOT EXISTS token_collisions (
    id             SERIAL PRIMARY KEY,
    token          TEXT NOT NULL UNIQUE,
    home_chapters  TEXT[] NOT NULL,                   -- chapters where this token legitimately appears
    risk_tier      TEXT NOT NULL CHECK (risk_tier IN ('low','medium','high')),
    resolution     JSONB NOT NULL,                    -- ordered list of resolver rules
    owner          TEXT,
    test_case      TEXT,                              -- pytest case that pins this rule
    notes          TEXT,
    created_at     TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_collisions_token ON token_collisions(token);
CREATE INDEX IF NOT EXISTS idx_collisions_tier  ON token_collisions(risk_tier);

-- ── keyword_audit_log ─────────────────────────────────────────────────────────
-- Append-only audit trail for changes to category_keywords / token_collisions.
-- Written exclusively by scripts/apply_collision_change.py — direct INSERT/
-- UPDATE/DELETE on the two governed tables is allowed but discouraged because
-- it bypasses this log. See CCTR plan §5c (governance).
CREATE TABLE IF NOT EXISTS keyword_audit_log (
    id          SERIAL PRIMARY KEY,
    table_name  TEXT NOT NULL CHECK (table_name IN ('category_keywords','token_collisions')),
    operation   TEXT NOT NULL CHECK (operation IN ('insert','update','delete')),
    row_pk      TEXT NOT NULL,                        -- string repr of the affected row's PK
    before      JSONB,
    after       JSONB,
    actor       TEXT NOT NULL,
    ticket      TEXT,
    reason      TEXT,
    occurred_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_table_time ON keyword_audit_log(table_name, occurred_at);

-- ── category_centroids ────────────────────────────────────────────────────────
-- L2-normalized centroid vectors. Built by centroid_builder.py from train-split
-- rows only. Never inserted manually.
--
-- Hierarchical centroids: one row per (category_id, hs_chapter) — supervised
-- sub-centroids replacing the prior unsupervised k-means cluster_id scheme.
-- Score at inference = max cosine similarity across a category's child chapter
-- centroids; the winning chapter is reported as best_chapter on the response.
--
-- Migration from the prior (category_id, cluster_id) schema: centroids are a
-- derived artifact (centroid_builder.py wipes and rebuilds), so on any row
-- where the legacy PK still names cluster_id we drop and recreate. This avoids
-- having to rewrite the PK in place.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints tc
        JOIN   information_schema.key_column_usage kcu
               ON tc.constraint_name = kcu.constraint_name
        WHERE  tc.table_name = 'category_centroids'
          AND  tc.constraint_type = 'PRIMARY KEY'
          AND  kcu.column_name = 'cluster_id'
    ) THEN
        DROP TABLE category_centroids;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS category_centroids (
    category_id   INT NOT NULL REFERENCES classification_categories(id) ON DELETE CASCADE,
    hs_chapter    TEXT NOT NULL,
    cluster_id    INT DEFAULT 0,
    centroid      vector(384) NOT NULL,
    sample_count  INT,
    updated_at    TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (category_id, hs_chapter)
);

CREATE INDEX IF NOT EXISTS idx_centroids_chapter ON category_centroids(category_id, hs_chapter);

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
    shipment_id    TEXT NOT NULL,
    category_name  TEXT NOT NULL REFERENCES classification_categories(name) ON DELETE CASCADE,
    cargo_text     TEXT NOT NULL,
    commodity_text TEXT NOT NULL,
    hs_chapter     TEXT,
    split          TEXT NOT NULL DEFAULT 'train'
                     CHECK (split IN ('train', 'validation', 'test')),
    UNIQUE (shipment_id, category_name)
);

-- Migration from the prior single-label schema where UNIQUE(shipment_id)
-- blocked multi-label rows. Drop that constraint if present and add
-- hs_chapter column if missing.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'shipment_labels' AND column_name = 'hs_chapter'
    ) THEN
        ALTER TABLE shipment_labels ADD COLUMN hs_chapter TEXT;
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE  conname = 'shipment_labels_shipment_id_key'
    ) THEN
        ALTER TABLE shipment_labels DROP CONSTRAINT shipment_labels_shipment_id_key;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE  conname = 'shipment_labels_shipment_id_category_name_key'
    ) THEN
        ALTER TABLE shipment_labels
            ADD CONSTRAINT shipment_labels_shipment_id_category_name_key
            UNIQUE (shipment_id, category_name);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_sl_split    ON shipment_labels(split);
CREATE INDEX IF NOT EXISTS idx_sl_category ON shipment_labels(category_name);
CREATE INDEX IF NOT EXISTS idx_sl_chapter  ON shipment_labels(category_name, hs_chapter);

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
                        CHECK (confidence_state IN ('classified', 'low_confidence', 'unclassified', 'ambiguous_high_risk')),
    threshold_used    NUMERIC(4,3),
    model_version     TEXT,
    classified_at     TIMESTAMPTZ DEFAULT now()
);

-- CCTR additions: requires_review surfaces high-risk deferrals to a review queue;
-- expand the confidence_state CHECK to include 'ambiguous_high_risk' for installs
-- predating CCTR.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='shipment_classifications' AND column_name='requires_review') THEN
        ALTER TABLE shipment_classifications
            ADD COLUMN requires_review BOOLEAN NOT NULL DEFAULT false;
    END IF;

    -- Replace the legacy CHECK constraint if it exists with the expanded enum.
    IF EXISTS (
        SELECT 1 FROM pg_constraint c
        JOIN   pg_class t ON c.conrelid = t.oid
        WHERE  t.relname = 'shipment_classifications'
          AND  c.contype = 'c'
          AND  c.conname = 'shipment_classifications_confidence_state_check'
    ) THEN
        -- Probe whether the existing constraint already lists ambiguous_high_risk.
        -- pg_get_constraintdef gives the DDL text; cheap substring check.
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint c
            JOIN   pg_class t ON c.conrelid = t.oid
            WHERE  t.relname = 'shipment_classifications'
              AND  c.conname = 'shipment_classifications_confidence_state_check'
              AND  pg_get_constraintdef(c.oid) LIKE '%ambiguous_high_risk%'
        ) THEN
            ALTER TABLE shipment_classifications
                DROP CONSTRAINT shipment_classifications_confidence_state_check;
            ALTER TABLE shipment_classifications
                ADD CONSTRAINT shipment_classifications_confidence_state_check
                CHECK (confidence_state IN ('classified','low_confidence','unclassified','ambiguous_high_risk'));
        END IF;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_class_requires_review ON shipment_classifications(requires_review)
    WHERE requires_review = true;

CREATE INDEX IF NOT EXISTS idx_class_categories ON shipment_classifications USING GIN(categories);
CREATE INDEX IF NOT EXISTS idx_class_confidence ON shipment_classifications(confidence_state);
CREATE INDEX IF NOT EXISTS idx_class_embedding  ON shipment_classifications
    USING hnsw(embedding vector_cosine_ops);
