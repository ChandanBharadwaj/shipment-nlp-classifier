// types.ts — handcrafted mirrors of the JSON shapes returned by
// ml-service/admin_routes.py. Kept here (not generated from OpenAPI)
// because the API surface is small and the mapping is straightforward.
//
// Numeric quirk: psycopg2 returns NUMERIC columns as strings, so
// `weight`, `semantic_weight`, `keyword_weight`, `threshold` come over
// the wire as strings. Helper `num()` below coerces — every view that
// wants to do math must call it instead of trusting the typed field.

export type SignalClass = "anchor" | "signal" | "suppressor" | "modifier";
export type RiskTier    = "low" | "medium" | "high";
export type AuditOp     = "insert" | "update" | "delete";

/** psycopg2 returns NUMERIC as a string; coerce to number where needed. */
export function num(v: string | number | null | undefined): number {
  if (v === null || v === undefined) return 0;
  return typeof v === "number" ? v : Number(v);
}

// ── Overview ─────────────────────────────────────────────────────────────

export interface ClassCount    { signal_class: SignalClass; n: number }
export interface CategoryCount { category: string;          n: number }
export interface ChapterCount  { hs_chapter: string;        n: number }
export interface PolysemyHit   {
  keyword: string;
  n_chapters: number;
  chapters: string[];
}

export interface Overview {
  total_keywords:   number;
  total_collisions: number;
  last_audit_ts:    string | null;
  counts: {
    by_class:    ClassCount[];
    by_category: CategoryCount[];
    by_chapter:  ChapterCount[];
  };
  top_polysemous: PolysemyHit[];
}

// ── Categories ───────────────────────────────────────────────────────────

export interface CategoryChapter {
  hs_chapter:    string;
  chapter_title: string;
  is_primary:    boolean;
  keyword_count: number;
}

export interface Category {
  id:               number;
  name:             string;
  display_name:     string;
  semantic_weight:  string | number;
  keyword_weight:   string | number;
  threshold:        string | number | null;
  chapters:         CategoryChapter[];
}

// ── Chapters (flat HS-chapter list with rollups) ─────────────────────────

export interface ChapterRow {
  hs_chapter:       string;
  chapter_title:    string;
  is_primary:       boolean;
  category_id:      number;
  category_name:    string;
  keyword_count:    number;
  label_count:      number;
  centroid_present: boolean;
  sample_count:     number | null;
}

// ── Labels ───────────────────────────────────────────────────────────────

export type LabelSplit = "train" | "validation" | "test";

export interface LabelRow {
  id:                 number;
  shipment_id:        string;
  category_name:      string;        // ground truth
  predicted_category: string | null; // classifier output; null when the input failed quality gates
  cargo_text:         string;
  commodity_text:     string;
  hs_chapter:         string | null;
  split:              LabelSplit;
}

export interface LabelPage {
  rows:   LabelRow[];
  total:  number;
  limit:  number;
  offset: number;
}

// ── Centroids (UMAP/PCA projection) ──────────────────────────────────────

export interface CentroidPoint {
  category_id:   number;
  category_name: string;
  hs_chapter:    string;
  chapter_title: string;
  sample_count:  number;
  x:             number;
  y:             number;
}

export interface CentroidProjection {
  generated_at: string;
  reducer:      "umap" | "tsne" | "pca";
  points:       CentroidPoint[];
}

/**
 * One row of /admin/api/keywords/{token}/centroid-affinity — the cosine
 * score between the token's embedding and a single (category, chapter)
 * centroid, plus whether the registry already has a row there.
 */
export interface CentroidAffinityRow {
  category_id:      number;
  category_name:    string;
  hs_chapter:       string;
  chapter_title:    string;
  sample_count:     number;
  cosine:           number;          // [-1, 1]; embeddings are L2-normalized
  has_registry_row: boolean;
  registry_weight:  number | null;   // populated iff has_registry_row
  signal_class:     string | null;   // anchor|signal|suppressor|modifier of the winning row, or null
}

/**
 * Response shape of /admin/api/keywords/{token}/centroid-affinity. The
 * `point` field is the keyword's coordinates in the same UMAP/PCA space
 * as /admin/api/centroids — overlay it on that scatter directly.
 */
export interface CentroidAffinity {
  token:         string;
  embedding_dim: number;
  reducer:       "umap" | "pca";
  point:         { x: number; y: number };
  similarities:  CentroidAffinityRow[];   // pre-sorted desc by cosine
}

// ── Keywords ─────────────────────────────────────────────────────────────

export interface KeywordRow {
  id:            number;
  category_id:   number;
  category_name: string;
  hs_chapter:    string | null;
  keyword:       string;
  weight:        string | number;
  signal_class:  SignalClass;
  source:        string | null;
  notes:         string | null;
  created_at:    string | null;
}

export interface KeywordPage {
  rows:   KeywordRow[];
  total:  number;
  limit:  number;
  offset: number;
}

export interface KeywordByToken {
  token: string;
  rows:  KeywordRow[];
}

// v3 removed: Collision, AuditEntry, DiscoverCandidate types — those backed
// the manual collision registry, governed audit log, and discovery scan
// views, which don't apply to the source-driven pipeline.
