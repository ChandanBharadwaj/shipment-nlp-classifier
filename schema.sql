-- Shipment Classification — v3 schema (source-driven pipeline).
--
-- Tables:
--   public_source_descriptions  — USITC + UK reference text per (source, hs_code)
--   categories                  — LLM-derived business categories
--   chapter_categories          — chapter -> category mapping (LLM-derived)
--   keywords                    — chapter-pinned keywords (TF-IDF + LLM CCTR)
--   category_centroids          — (category, hs_chapter) embedding average
--   shipment_labels             — labeled shipments for F1 evaluation
--   shipment_classifications    — classification results written by /classify
--
-- Idempotent: every CREATE uses IF NOT EXISTS; every ALTER guards on
-- information_schema. Safe to re-run.

CREATE EXTENSION IF NOT EXISTS vector;

-- ── public_source_descriptions ────────────────────────────────────────────
-- Reference text from public taxonomies (USITC HTS API, UK Trade Tariff API).
-- One row per (source, hs_code). Loaded by scripts/load_public_sources.py.
CREATE TABLE IF NOT EXISTS public_source_descriptions (
    source       TEXT NOT NULL,
    hs_code      TEXT NOT NULL,
    hs_chapter   TEXT NOT NULL,
    description  TEXT NOT NULL,
    fetched_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source, hs_code)
);
CREATE INDEX IF NOT EXISTS idx_public_source_chapter ON public_source_descriptions(hs_chapter);

-- ── categories ────────────────────────────────────────────────────────────
-- LLM-derived business category set. Slug is the URL-safe identifier;
-- display_name is the natural-language form for UI.
CREATE TABLE IF NOT EXISTS categories (
    slug         TEXT PRIMARY KEY,
    display_name TEXT NOT NULL UNIQUE,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── chapter_categories ────────────────────────────────────────────────────
-- LLM-derived mapping of HS 2-digit chapter to v3 category.
CREATE TABLE IF NOT EXISTS chapter_categories (
    hs_chapter    TEXT PRIMARY KEY,
    category_slug TEXT NOT NULL REFERENCES categories(slug) ON DELETE CASCADE,
    chapter_title TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chapter_cat_slug ON chapter_categories(category_slug);

-- ── keywords ──────────────────────────────────────────────────────────────
-- Per-chapter keyword registry. signal_class values:
--   'signal'     — TF-IDF over public source text (the bulk)
--   'anchor'     — LLM-generated; dispositive evidence for the chapter
--   'suppressor' — LLM-generated; pushes a different category's score down
--   'modifier'   — LLM-generated; re-routes a head_noun to target_chapter
--
-- Type-specific fields are first-class columns instead of free-text notes:
--   target_chapter    — modifiers only
--   suppress_category — suppressors only (refs categories.slug)
--   head_noun         — modifiers only
CREATE TABLE IF NOT EXISTS keywords (
    id                SERIAL PRIMARY KEY,
    keyword           TEXT NOT NULL,
    hs_chapter        TEXT NOT NULL,
    hs_code           TEXT,
    signal_class      TEXT NOT NULL CHECK (signal_class IN ('signal','anchor','suppressor','modifier')),
    weight            NUMERIC(3,2) NOT NULL,
    target_chapter    TEXT,
    suppress_category TEXT,
    head_noun         TEXT,
    source            TEXT NOT NULL,
    generated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (keyword, hs_chapter, signal_class, source)
);
CREATE INDEX IF NOT EXISTS idx_keywords_chapter      ON keywords(hs_chapter);
CREATE INDEX IF NOT EXISTS idx_keywords_signal_class ON keywords(signal_class);
CREATE INDEX IF NOT EXISTS idx_keywords_source       ON keywords(source);

-- ── category_centroids ────────────────────────────────────────────────────
-- L2-normalized embedding average per (category, hs_chapter). Built by
-- ml-service/centroid_builder.py from public_source_descriptions text.
-- Wipe-and-rebuild semantics — never insert manually.
CREATE TABLE IF NOT EXISTS category_centroids (
    category_slug  TEXT NOT NULL REFERENCES categories(slug) ON DELETE CASCADE,
    hs_chapter     TEXT NOT NULL,
    centroid       vector(384) NOT NULL,
    sample_count   INT,
    updated_at     TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (category_slug, hs_chapter)
);
CREATE INDEX IF NOT EXISTS idx_centroids_chapter ON category_centroids(hs_chapter);

-- ── shipment_labels ───────────────────────────────────────────────────────
-- Labeled shipment data, used ONLY for F1 evaluation in v3 (not for
-- training/centroid construction). category_name holds the v3 slug.
CREATE TABLE IF NOT EXISTS shipment_labels (
    id             SERIAL PRIMARY KEY,
    shipment_id    TEXT NOT NULL,
    category_name  TEXT NOT NULL,
    cargo_text     TEXT NOT NULL,
    commodity_text TEXT NOT NULL,
    hs_chapter     TEXT,
    split          TEXT NOT NULL DEFAULT 'train'
                     CHECK (split IN ('train', 'validation', 'test')),
    UNIQUE (shipment_id, category_name)
);
CREATE INDEX IF NOT EXISTS idx_sl_split    ON shipment_labels(split);
CREATE INDEX IF NOT EXISTS idx_sl_category ON shipment_labels(category_name);
CREATE INDEX IF NOT EXISTS idx_sl_chapter  ON shipment_labels(category_name, hs_chapter);

-- ── shipment_classifications ──────────────────────────────────────────────
-- Stores classification results written by /classify when persist=true.
CREATE TABLE IF NOT EXISTS shipment_classifications (
    shipment_id       TEXT PRIMARY KEY,
    embedding         vector(384),
    categories        TEXT[],
    scores            JSONB,
    confidence_state  TEXT DEFAULT 'classified'
                        CHECK (confidence_state IN ('classified','low_confidence','unclassified','ambiguous_high_risk')),
    threshold_used    NUMERIC(4,3),
    model_version     TEXT,
    requires_review   BOOLEAN NOT NULL DEFAULT false,
    classified_at     TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_class_categories      ON shipment_classifications USING GIN(categories);
CREATE INDEX IF NOT EXISTS idx_class_confidence      ON shipment_classifications(confidence_state);
CREATE INDEX IF NOT EXISTS idx_class_requires_review ON shipment_classifications(requires_review)
    WHERE requires_review = true;
CREATE INDEX IF NOT EXISTS idx_class_embedding       ON shipment_classifications
    USING hnsw(embedding vector_cosine_ops);
