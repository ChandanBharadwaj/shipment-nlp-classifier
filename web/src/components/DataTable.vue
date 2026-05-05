<script setup lang="ts" generic="T extends object">
/**
 * DataTable — generic paginated table for read-only views.
 *
 * Columns drive the header + default cell rendering. Each column also
 * exposes a named slot `cell-<key>` so views can swap in a chip/badge/
 * link without re-implementing pagination, sticky headers, or the
 * "Showing N–M of T" footer.
 *
 * Filter state lives in URL (via useRouteQuery) — this component only
 * cares about offset/limit, which it emits as `update:offset` /
 * `update:limit`. The parent owns the source of truth.
 */
import { computed } from "vue";

interface Column {
  key: string;
  label: string;
  align?: "left" | "right" | "center";
  width?: string;
  /** Optional formatter for the default cell. Slots override this. */
  format?: (value: unknown, row: T) => string;
  /** Optional plain-English tooltip shown on hover of the column header. */
  tip?: string;
}

const props = defineProps<{
  columns: Column[];
  rows: T[];
  total?: number;
  limit?: number;
  offset?: number;
  loading?: boolean;
  emptyMessage?: string;
  rowKey?: (row: T, i: number) => string | number;
}>();

const emit = defineEmits<{
  (e: "update:offset", v: number): void;
  (e: "update:limit", v: number): void;
  (e: "row-click", row: T): void;
}>();

const limit  = computed(() => props.limit  ?? props.rows.length);
const offset = computed(() => props.offset ?? 0);
const total  = computed(() => props.total  ?? props.rows.length);

const showingFrom = computed(() => (total.value === 0 ? 0 : offset.value + 1));
const showingTo   = computed(() => Math.min(offset.value + props.rows.length, total.value));
const canPrev     = computed(() => offset.value > 0);
const canNext     = computed(() => offset.value + limit.value < total.value);

function prev(): void {
  if (!canPrev.value) return;
  emit("update:offset", Math.max(0, offset.value - limit.value));
}
function next(): void {
  if (!canNext.value) return;
  emit("update:offset", offset.value + limit.value);
}
function changeLimit(ev: Event): void {
  const n = Number((ev.target as HTMLSelectElement).value);
  if (!Number.isNaN(n) && n > 0) {
    emit("update:limit", n);
    emit("update:offset", 0);
  }
}

function defaultCell(col: Column, row: T): string {
  const v = (row as Record<string, unknown>)[col.key];
  if (col.format) return col.format(v, row);
  if (v === null || v === undefined) return "";
  return String(v);
}
</script>

<template>
  <div class="dt-wrap">
    <div class="dt-table-wrap">
      <table class="dt-table">
        <thead>
          <tr>
            <th
              v-for="c in columns"
              :key="c.key"
              :style="{ width: c.width, textAlign: c.align || 'left' }"
              :title="c.tip"
              :class="{ 'has-tip': !!c.tip }"
            >{{ c.label }}<span v-if="c.tip" class="dt-tip-mark" aria-hidden="true">ⓘ</span></th>
          </tr>
        </thead>
        <tbody>
          <tr v-if="loading && !rows.length">
            <td :colspan="columns.length" class="dt-empty">
              <span class="spinner" /> loading…
            </td>
          </tr>
          <tr v-else-if="!rows.length">
            <td :colspan="columns.length" class="dt-empty">
              <slot name="empty">{{ emptyMessage || "No rows match the current filters." }}</slot>
            </td>
          </tr>
          <tr
            v-for="(row, i) in rows"
            :key="rowKey ? rowKey(row, i) : i"
            @click="emit('row-click', row)"
          >
            <td
              v-for="c in columns"
              :key="c.key"
              :style="{ textAlign: c.align || 'left' }"
            >
              <slot
                :name="`cell-${c.key}`"
                :row="row"
                :value="(row as Record<string, unknown>)[c.key]"
              >
                {{ defaultCell(c, row) }}
              </slot>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="dt-footer">
      <div class="dt-counts">
        <span v-if="total > 0">Showing {{ showingFrom }}–{{ showingTo }} of {{ total }}</span>
        <span v-else>0 results</span>
        <span v-if="loading && rows.length" class="dt-loading-inline">
          <span class="spinner" /> updating…
        </span>
      </div>
      <div class="dt-pager">
        <label class="dt-page-size">
          Rows
          <select :value="limit" @change="changeLimit">
            <option :value="25">25</option>
            <option :value="50">50</option>
            <option :value="100">100</option>
            <option :value="200">200</option>
          </select>
        </label>
        <button :disabled="!canPrev" @click="prev">‹ Prev</button>
        <button :disabled="!canNext" @click="next">Next ›</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.dt-wrap {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  overflow: hidden;
}
.dt-table-wrap {
  overflow: auto;
  max-height: calc(100vh - 320px);
}
.dt-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.dt-table thead th {
  text-align: left;
  background: var(--surface-2);
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  font-weight: 600;
  color: var(--text-muted);
  white-space: nowrap;
  position: sticky;
  top: 0;
  z-index: 1;
}
.dt-table tbody td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
.dt-table tbody tr:hover td {
  background: rgba(37, 99, 235, 0.04);
}
.dt-table tbody tr:last-child td { border-bottom: none; }

.dt-empty {
  text-align: center;
  padding: 32px 12px !important;
  color: var(--text-muted);
}
.dt-loading-inline {
  margin-left: 12px;
  color: var(--text-muted);
}

.dt-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  background: var(--surface-2);
  border-top: 1px solid var(--border);
  font-size: 12px;
  color: var(--text-muted);
  flex-wrap: wrap;
  gap: 8px;
}
.dt-pager {
  display: flex;
  align-items: center;
  gap: 8px;
}
.dt-pager button {
  background: var(--surface);
  color: var(--text);
  border: 1px solid var(--border-2);
  border-radius: var(--radius-sm);
  padding: 4px 10px;
  font-size: 12px;
  font-family: inherit;
  cursor: pointer;
}
.dt-pager button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.dt-pager button:not(:disabled):hover {
  border-color: var(--accent);
  color: var(--accent);
}
.dt-page-size select {
  background: var(--surface);
  color: var(--text);
  border: 1px solid var(--border-2);
  border-radius: var(--radius-sm);
  padding: 3px 6px;
  font-size: 12px;
  margin-left: 4px;
  font-family: inherit;
}

.dt-table thead th.has-tip { cursor: help; }
.dt-tip-mark {
  display: inline-block;
  margin-left: 4px;
  color: var(--text-muted);
  font-size: 10px;
  opacity: 0.7;
}
.dt-table thead th.has-tip:hover .dt-tip-mark { opacity: 1; color: var(--accent); }
</style>
