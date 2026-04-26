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

// ── Collisions ───────────────────────────────────────────────────────────

export interface ResolutionArm {
  if?:   string;
  then?: string;
  // Forward-compat: registry may grow extra keys. The UI renders any
  // `if`/`then` pair and shows the rest in a tooltip.
  [k: string]: string | undefined;
}

export interface Collision {
  id:             number;
  token:          string;
  home_chapters:  string[];
  risk_tier:      RiskTier;
  resolution:     ResolutionArm[];
  owner:          string | null;
  test_case:      string | null;
  notes:          string | null;
  created_at:     string | null;
}

// ── Audit log ────────────────────────────────────────────────────────────

export interface AuditEntry {
  id:         number;
  ts:         string;
  table_name: string;
  pk:         string | null;
  operation:  AuditOp;
  actor:      string;
  ticket:     string | null;
  reason:     string | null;
  before:     unknown;
  after:      unknown;
}

export interface AuditPage {
  rows:   AuditEntry[];
  total:  number;
  limit:  number;
  offset: number;
}

// ── Discover ─────────────────────────────────────────────────────────────

export interface DiscoverCandidate {
  keyword:        string;
  n_chapters:     number;
  home_chapters:  string;       // comma-separated
  categories:     string;       // comma-separated
  max_weight:     string;
  n_rows:         number;
  suggested_tier: RiskTier;
}

export interface DiscoverResult {
  candidates:           DiscoverCandidate[];
  n_already_registered: number;
  min_chapters:         number;
  min_weight:           number;
}
