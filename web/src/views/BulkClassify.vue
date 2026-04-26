<script setup lang="ts">
/**
 * BulkClassify.vue — Vue port of the legacy ml-service/static/app.js
 * CSV bulk tester. Behaviour is intentionally byte-equivalent to the
 * old vanilla-JS implementation:
 *
 *   1. User drops a CSV → PapaParse → header validation → POST
 *      /classify/batch with persist:false on every row.
 *   2. Render results into a table; no client storage, no DB writes.
 *
 * Risk is per-shipment via `compliance.is_risky` (boolean). HS↔text
 * mismatches and reranks are surfaced inline. Logic mirrors app.js
 * line-for-line; differences are confined to Vue-isms (refs, reactive
 * state, <CategoryChip>) and have no behavioural effect.
 */
import { ref, computed } from "vue";
import Papa from "papaparse";
import CategoryChip from "@/components/CategoryChip.vue";

// ── Config ─────────────────────────────────────────────────────────────
const REQUIRED_COLUMNS = ["shipment_id", "cargo_description", "commodity_description"] as const;
const OPTIONAL_COLUMNS = ["threshold", "unclassified_threshold"] as const;
const MAX_ROWS  = 500;   // matches /classify/batch server-side cap
const TRUNC_LEN = 80;

// ── Types — only the fields the table cares about. ────────────────────
interface ScoreEntry { final_score?: number }
interface Compliance {
  is_risky?: boolean;
  decision_reasons?: string[];
}
interface ClassifyResult {
  confidence_state?: "classified" | "low_confidence" | "unclassified";
  categories?: string[];
  scores?: Record<string, ScoreEntry>;
  hs_codes_extracted?: string[];
  hs_implied_categories?: string[];
  hs_text_mismatch?: boolean;
}
interface ResultRow {
  shipment_id?: string;
  input?: { cargo_description?: string; commodity_description?: string };
  result?: ClassifyResult;
  compliance?: Compliance;
  meta?: { chunks_processed?: number; reranked?: boolean };
}

// ── State ──────────────────────────────────────────────────────────────
const fileInput = ref<HTMLInputElement | null>(null);
const dragOver  = ref(false);
const banner    = ref<{ kind: "error" | "warn" | "info"; text: string } | null>(null);
const status    = ref<string | null>(null);   // when set, spinner shows
const results   = ref<ResultRow[] | null>(null);
const elapsed   = ref<string>("");

const summary = computed(() => {
  const rows = results.value;
  if (!rows) return "";
  let classified = 0, lowConf = 0, unclassified = 0;
  let risky = 0, mismatch = 0, reranked = 0;
  for (const r of rows) {
    const conf = r.result?.confidence_state ?? "unclassified";
    if (conf === "classified") classified++;
    else if (conf === "low_confidence") lowConf++;
    else unclassified++;
    if (r.compliance?.is_risky === true) risky++;
    if (r.result?.hs_text_mismatch === true) mismatch++;
    if (r.meta?.reranked === true) reranked++;
  }
  const n = rows.length;
  let s =
    `${n} shipment${n === 1 ? "" : "s"} in ${elapsed.value}s — ` +
    `${classified} classified, ${lowConf} low_confidence, ${unclassified} unclassified · ` +
    `${risky} flagged risky · ${mismatch} HS↔text mismatch${mismatch === 1 ? "" : "es"}`;
  if (reranked > 0) s += ` · ${reranked} reranked`;
  return s;
});

// ── Drop / pick handlers ───────────────────────────────────────────────
function pick(): void { fileInput.value?.click(); }
function onPicked(ev: Event): void {
  const f = (ev.target as HTMLInputElement).files?.[0];
  if (f) handleFile(f);
}
function onDrop(ev: DragEvent): void {
  ev.preventDefault();
  ev.stopPropagation();
  dragOver.value = false;
  const f = ev.dataTransfer?.files?.[0];
  if (f) handleFile(f);
}
function onDragEnter(ev: DragEvent): void { ev.preventDefault(); ev.stopPropagation(); dragOver.value = true;  }
function onDragLeave(ev: DragEvent): void { ev.preventDefault(); ev.stopPropagation(); dragOver.value = false; }

// ── Pipeline ───────────────────────────────────────────────────────────
function handleFile(file: File): void {
  banner.value = null;
  results.value = null;
  status.value = `Parsing ${file.name}…`;

  Papa.parse<Record<string, string>>(file, {
    header: true,
    skipEmptyLines: true,
    transformHeader: (h: string) => (h || "").trim(),
    complete: (parsed) => onParsed(parsed),
    error: (err) => {
      status.value = null;
      banner.value = { kind: "error", text: `CSV parse error: ${err.message || String(err)}` };
    },
  });
}

function onParsed(parsed: Papa.ParseResult<Record<string, string>>): void {
  const rows = parsed.data ?? [];
  const headers = parsed.meta?.fields ?? [];

  const missing = REQUIRED_COLUMNS.filter((c) => !headers.includes(c));
  if (missing.length) {
    status.value = null;
    banner.value = {
      kind: "error",
      text:
        `CSV is missing required column(s): ${missing.map((m) => `"${m}"`).join(", ")}. ` +
        `Required: ${REQUIRED_COLUMNS.join(", ")}. Optional: ${OPTIONAL_COLUMNS.join(", ")}.`,
    };
    return;
  }

  if (!rows.length) {
    status.value = null;
    banner.value = { kind: "warn", text: "CSV had headers but no data rows." };
    return;
  }

  let working = rows;
  let warning: string | null = null;
  if (rows.length > MAX_ROWS) {
    working = rows.slice(0, MAX_ROWS);
    warning = `CSV had ${rows.length} rows; processing the first ${MAX_ROWS} (server-side batch limit).`;
  }

  const shipments = working.map(buildShipment);
  void classifyBatch(shipments, warning);
}

interface ShipmentBody {
  shipment_id: string;
  cargo_description: string;
  commodity_description: string;
  persist: false;
  threshold?: number;
  unclassified_threshold?: number;
}

function buildShipment(row: Record<string, string>): ShipmentBody {
  const ship: ShipmentBody = {
    shipment_id:           String(row.shipment_id ?? "").trim(),
    cargo_description:     String(row.cargo_description ?? "").trim(),
    commodity_description: String(row.commodity_description ?? "").trim(),
    persist: false,        // hard guarantee of statelessness
  };
  if (row.threshold !== undefined && row.threshold !== "") {
    const t = Number(row.threshold);
    if (!Number.isNaN(t)) ship.threshold = t;
  }
  if (row.unclassified_threshold !== undefined && row.unclassified_threshold !== "") {
    const u = Number(row.unclassified_threshold);
    if (!Number.isNaN(u)) ship.unclassified_threshold = u;
  }
  return ship;
}

async function classifyBatch(shipments: ShipmentBody[], warning: string | null): Promise<void> {
  status.value = `Classifying ${shipments.length} shipment${shipments.length === 1 ? "" : "s"}…`;
  try {
    const t0 = performance.now();
    const resp = await fetch("/classify/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ shipments }),
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${text.slice(0, 300)}`);
    }
    const data = (await resp.json()) as ResultRow[];
    elapsed.value = ((performance.now() - t0) / 1000).toFixed(2);
    status.value = null;
    if (warning) banner.value = { kind: "warn", text: warning };
    results.value = data;
  } catch (err) {
    status.value = null;
    banner.value = {
      kind: "error",
      text: `Classification failed: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

// ── Render helpers ─────────────────────────────────────────────────────
function trunc(text: string | undefined): string {
  const s = text ?? "";
  return s.length > TRUNC_LEN ? s.slice(0, TRUNC_LEN) + "…" : s;
}

function runnerUps(scores: Record<string, ScoreEntry> | undefined): Array<[string, number]> {
  if (!scores) return [];
  return Object.entries(scores)
    .map(([cat, s]) => [cat, s?.final_score ?? 0] as [string, number])
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);
}
</script>

<template>
  <section class="page">
    <header class="page-header">
      <h2>Bulk Classify</h2>
      <p class="subtitle">
        Drop a CSV. We classify in batch and show only the labels that fired.
        Stateless — reload the page and everything clears.
      </p>
      <div class="actions">
        <a href="/sample.csv" download class="link-btn">Download sample CSV</a>
      </div>
    </header>

    <section
      class="dropzone"
      :class="{ dragover: dragOver }"
      @click="pick"
      @drop="onDrop"
      @dragenter="onDragEnter"
      @dragover="onDragEnter"
      @dragleave="onDragLeave"
    >
      <input
        ref="fileInput"
        type="file"
        accept=".csv,text/csv"
        hidden
        @change="onPicked"
      />
      <div class="dz-inner">
        <p class="dz-title">Drop a CSV here</p>
        <p class="dz-sub">
          or
          <button type="button" class="link-btn inline" @click.stop="pick">choose a file</button>
        </p>
        <p class="dz-hint">
          Required columns: <code>shipment_id</code>, <code>cargo_description</code>, <code>commodity_description</code>.
          Optional: <code>threshold</code>, <code>unclassified_threshold</code>.
          Max 500 rows per upload.
        </p>
      </div>
    </section>

    <div v-if="banner" :class="['banner', banner.kind]">{{ banner.text }}</div>

    <section v-if="status" class="status">
      <span class="spinner" />
      <span>{{ status }}</span>
    </section>

    <section v-if="results" class="results">
      <div class="results-header">
        <h3>Results</h3>
        <div class="legend">
          <span class="badge risky">RISKY</span>
          <span class="badge clean">clean</span>
          <span class="badge mismatch">HS &ne; text</span>
        </div>
      </div>
      <div class="summary">{{ summary }}</div>

      <div class="table-wrap">
        <table class="results-table">
          <thead>
            <tr>
              <th>Shipment ID</th>
              <th>Cargo</th>
              <th>Commodity</th>
              <th>Labels</th>
              <th>HS Signal</th>
              <th>Confidence</th>
              <th>Compliance</th>
              <th>Chunks</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(row, i) in results"
              :key="row.shipment_id || i"
              :class="{
                'risky-row':       row.compliance?.is_risky === true,
                'hs-mismatch-row': row.result?.hs_text_mismatch === true,
              }"
            >
              <td class="cell-id">{{ row.shipment_id || "" }}</td>
              <td class="cell-text">
                <span class="truncate" :title="row.input?.cargo_description">
                  {{ trunc(row.input?.cargo_description) }}
                </span>
              </td>
              <td class="cell-text">
                <span class="truncate" :title="row.input?.commodity_description">
                  {{ trunc(row.input?.commodity_description) }}
                </span>
              </td>

              <!-- Labels -->
              <td class="cell-labels">
                <template v-if="row.result?.categories?.length">
                  <CategoryChip
                    v-for="cat in row.result.categories"
                    :key="cat"
                    :category="cat"
                    :score="row.result.scores?.[cat]?.final_score"
                  />
                </template>
                <template v-else>
                  <span class="chip unclassified">unclassified</span>
                  <CategoryChip
                    v-for="[cat, s] in runnerUps(row.result?.scores)"
                    :key="'r-' + cat"
                    :category="cat"
                    :score="s"
                    runner-up
                  />
                </template>
              </td>

              <!-- HS signal -->
              <td
                class="cell-hs"
                :class="{ mismatch: row.result?.hs_text_mismatch === true }"
              >
                <template v-if="row.result?.hs_codes_extracted?.length">
                  <span
                    v-for="code in row.result.hs_codes_extracted"
                    :key="code"
                    class="chip hs-code"
                  >{{ code }}</span>
                  <template v-if="row.result?.hs_implied_categories?.length">
                    <span class="hs-arrow">→</span>
                    <CategoryChip
                      v-for="cat in row.result.hs_implied_categories"
                      :key="'h-' + cat"
                      :category="cat"
                    />
                  </template>
                  <span
                    v-if="row.result?.hs_text_mismatch === true"
                    class="badge mismatch"
                  >HS ≠ text</span>
                </template>
                <span v-else class="hs-none">—</span>
              </td>

              <!-- Confidence -->
              <td class="cell-conf">
                <span :class="['confidence', row.result?.confidence_state || 'unclassified']">
                  {{ row.result?.confidence_state || "unclassified" }}
                </span>
              </td>

              <!-- Compliance -->
              <td class="cell-compl">
                <span :class="['badge', row.compliance?.is_risky ? 'risky' : 'clean']">
                  {{ row.compliance?.is_risky ? "RISKY" : "clean" }}
                </span>
                <span
                  v-if="row.compliance?.is_risky && row.compliance?.decision_reasons?.length"
                  class="compl-reasons"
                  :title="row.compliance.decision_reasons.join('\n')"
                >{{ row.compliance.decision_reasons.slice(0, 2).join(' · ') }}</span>
              </td>

              <td class="cell-chunks">
                {{ (row.meta?.chunks_processed ?? 0) > 1 ? row.meta?.chunks_processed : "" }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </section>
</template>

<style scoped>
.page-header {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 12px 24px;
  margin-bottom: 20px;
}
.page-header h2 { margin: 0; font-size: 20px; grid-column: 1; }
.page-header .subtitle {
  margin: 0;
  color: var(--text-muted);
  max-width: 720px;
  grid-column: 1;
}
.page-header .actions {
  grid-column: 2;
  grid-row: 1 / span 2;
  align-self: start;
  display: flex;
  gap: 8px;
}

.dropzone {
  border: 2px dashed var(--border-2);
  background: var(--surface);
  border-radius: var(--radius);
  padding: 40px 24px;
  text-align: center;
  transition: background 120ms, border-color 120ms;
  cursor: pointer;
}
.dropzone.dragover {
  border-color: var(--accent);
  background: var(--accent-soft);
}
.dz-title { margin: 0 0 4px; font-size: 16px; font-weight: 600; }
.dz-sub   { margin: 0 0 12px; color: var(--text-muted); }
.dz-hint  { margin: 0; color: var(--text-muted); font-size: 12px; }

.status {
  display: flex; align-items: center; gap: 10px;
  margin-top: 16px; padding: 12px 16px;
  border-radius: var(--radius);
  background: var(--surface);
  border: 1px solid var(--border);
  color: var(--text-muted);
}

.results { margin-top: 24px; }
.results-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 12px; flex-wrap: wrap; gap: 12px;
}
.results-header h3 { margin: 0; font-size: 18px; }
.legend { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.summary { margin-bottom: 12px; color: var(--text-muted); font-size: 13px; }

.table-wrap {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: auto;
  box-shadow: var(--shadow);
}
.results-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.results-table thead th {
  text-align: left;
  background: var(--surface-2);
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  font-weight: 600;
  color: var(--text-muted);
  white-space: nowrap;
  position: sticky;
  top: 0;
}
.results-table tbody td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
.results-table tbody tr:last-child td { border-bottom: none; }
.results-table tbody tr.risky-row td:first-child {
  border-left: 4px solid var(--risky);
}
.results-table tbody tr.hs-mismatch-row td:first-child {
  box-shadow: inset 3px 0 0 0 #b54708;
}
.results-table tbody tr.hs-mismatch-row:not(.risky-row) {
  background: rgba(181, 71, 8, 0.04);
}

.cell-id   {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: nowrap;
}
.cell-text { max-width: 240px; }
.cell-text .truncate {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.cell-labels { min-width: 240px; }
.cell-hs     { min-width: 200px; }
.cell-hs .hs-none  { color: var(--text-muted); }
.cell-hs .hs-arrow { color: var(--text-muted); margin: 0 4px; font-size: 12px; }
.cell-hs .chip.hs-code {
  background: var(--surface-2);
  color: var(--text);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  display: inline-flex;
  align-items: center;
  padding: 3px 9px;
  border-radius: 999px;
  margin: 2px 4px 2px 0;
  border: 1px solid rgba(0, 0, 0, 0.06);
}
.cell-conf   { white-space: nowrap; }
.cell-compl  { white-space: nowrap; }
.cell-chunks { text-align: right; color: var(--text-muted); white-space: nowrap; }

.chip.unclassified {
  display: inline-flex;
  align-items: center;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
  margin: 2px 4px 2px 0;
  border: 1px solid rgba(0, 0, 0, 0.06);
  background: var(--grey-soft);
  color: var(--text-muted);
}

.confidence {
  font-weight: 500;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.confidence.classified     { color: var(--clean); }
.confidence.low_confidence { color: var(--warn);  }
.confidence.unclassified   { color: var(--text-muted); }

.badge {
  display: inline-block;
  padding: 3px 9px;
  border-radius: var(--radius-sm);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.05em;
}
.badge.risky {
  background: var(--risky-soft);
  color: var(--risky);
  animation: pulse 2.4s ease-in-out infinite;
}
.badge.clean    { background: var(--clean-soft); color: var(--clean); }
.badge.mismatch { background: rgba(181, 71, 8, 0.12); color: #b54708; margin-left: 6px; }
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(217, 45, 32, 0.35); }
  50%      { box-shadow: 0 0 0 6px rgba(217, 45, 32, 0.00); }
}

.compl-reasons {
  display: block;
  font-size: 11px;
  font-weight: 400;
  color: var(--text-muted);
  margin-top: 4px;
  max-width: 200px;
}
</style>
