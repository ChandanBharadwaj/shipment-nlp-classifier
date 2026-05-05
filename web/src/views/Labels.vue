<script setup lang="ts">
/**
 * Labels.vue — paginated browser over shipment_labels.
 *
 * Splits matter: train rows feed centroid_builder, validation rows are
 * used by evaluate_threshold, test rows confirm the final pick. The
 * filter row exposes split + category + chapter + free-text search,
 * URL-synced like the Tokens view.
 */
import { ref, computed, watch, onMounted } from "vue";
import api from "@/api";
import type { LabelPage, LabelRow, LabelSplit } from "@/types";
import { useCatalogStore } from "@/stores/catalog";
import { useRouteQuery } from "@/composables/useRouteQuery";
import FilterBar from "@/components/FilterBar.vue";
import DataTable from "@/components/DataTable.vue";
import CategoryChip from "@/components/CategoryChip.vue";
import InfoTip from "@/components/InfoTip.vue";

const catalog = useCatalogStore();

const split             = useRouteQuery<string>("split", "");
const category          = useRouteQuery<string>("category", "");
const chapter           = useRouteQuery<string>("chapter", "");
const search            = useRouteQuery<string>("search", "");
const predictionStatus  = useRouteQuery<string>("prediction_status", "");
const offset            = useRouteQuery<number>("offset", 0);
const limit             = useRouteQuery<number>("limit", 100);

const data    = ref<LabelPage | null>(null);
const loading = ref(false);
const error   = ref<string | null>(null);

// Same cross-chapter probe as Tokens.vue: when 0 rows come back with both
// category+chapter set, check whether the category has labels in other
// chapters and surface a "Clear chapter filter" hint.
const crossChapterTotal = ref<number | null>(null);

async function probeCrossChapter(): Promise<void> {
  if (!data.value || data.value.rows.length > 0
      || !category.value || !chapter.value) {
    crossChapterTotal.value = null;
    return;
  }
  try {
    const probe = await api.labels({
      split:             split.value || undefined,
      category:          category.value,
      search:            search.value || undefined,
      prediction_status: predictionStatus.value || undefined,
      limit:             1,
      offset:            0,
    });
    crossChapterTotal.value = probe.total;
  } catch {
    crossChapterTotal.value = null;
  }
}

async function load(): Promise<void> {
  loading.value = true;
  error.value = null;
  try {
    data.value = await api.labels({
      split:             split.value || undefined,
      category:          category.value || undefined,
      chapter:           chapter.value || undefined,
      search:            search.value || undefined,
      prediction_status: predictionStatus.value || undefined,
      limit:             limit.value,
      offset:            offset.value,
    });
    await probeCrossChapter();
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  } finally {
    loading.value = false;
  }
}

onMounted(async () => {
  await Promise.all([catalog.ensure(), catalog.ensureChapters()]);
  await load();
});

watch(
  () => [split.value, category.value, chapter.value, search.value, predictionStatus.value],
  () => { if (offset.value !== 0) offset.value = 0; else void load(); },
);
watch(() => [offset.value, limit.value], () => void load());

let sTimer: number | undefined;
const searchDraft = ref(search.value);
watch(search, v => { searchDraft.value = v; });
function onSearchInput(ev: Event): void {
  const v = (ev.target as HTMLInputElement).value;
  searchDraft.value = v;
  if (sTimer) window.clearTimeout(sTimer);
  sTimer = window.setTimeout(() => { search.value = v; }, 300);
}

const SPLITS: LabelSplit[] = ["train", "validation", "test"];

// Sorted (chapter_num, chapter_title) for the dropdown — non-ML users
// can't read bare chapter codes, so we render "85 — Electrical machinery".
const chapterOptions = computed<{ value: string; label: string }[]>(() => {
  const seen = new Map<string, string>();
  for (const c of catalog.chapters) {
    if (!seen.has(c.hs_chapter)) seen.set(c.hs_chapter, c.chapter_title);
  }
  return Array.from(seen.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([value, title]) => ({ value, label: `${value} — ${title}` }));
});

const columns = [
  { key: "shipment_id",        label: "Shipment",  width: "16%" },
  { key: "split",              label: "Split",     width: "100px" },
  { key: "category_name",      label: "Category",  width: "14%" },
  { key: "predicted_category", label: "Predicted", width: "14%" },
  { key: "hs_chapter",         label: "Ch",        width: "60px",  align: "center" as const },
  { key: "cargo_text",         label: "Cargo" },
  { key: "commodity_text",     label: "Commodity" },
];

function isMispredicted(row: LabelRow): boolean {
  return row.predicted_category !== null
      && row.predicted_category !== row.category_name;
}

function trunc(s: string | null | undefined, n = 80): string {
  const v = s ?? "";
  return v.length > n ? v.slice(0, n) + "…" : v;
}

const totalDisplay = computed(() => data.value?.total ?? 0);
</script>

<template>
  <section class="page">
    <header class="page-header">
      <h2>
        Labels
        <InfoTip side="below">
          <p>The <strong>training data</strong>: real shipment descriptions
          where a human (or upstream system) has tagged the correct
          category and HS chapter.</p>
          <p><strong>train</strong> rows build the centroids. <strong>validation</strong>
          tunes the confidence threshold. <strong>test</strong> rows are held
          out for the final accuracy check — they're never used to fit
          anything, so the score is honest.</p>
        </InfoTip>
      </h2>
      <p class="subtitle">
        Labeled shipment rows used to build centroids and tune thresholds.
        <strong>{{ totalDisplay.toLocaleString() }}</strong> rows match the
        current filter.
      </p>
    </header>

    <div v-if="error" class="banner error">{{ error }}</div>

    <FilterBar>
      <div class="filter-field">
        <label>Split</label>
        <select v-model="split">
          <option value="">All splits</option>
          <option v-for="s in SPLITS" :key="s" :value="s">{{ s }}</option>
        </select>
      </div>
      <div class="filter-field">
        <label>Category</label>
        <select v-model="category">
          <option value="">All categories</option>
          <option v-for="c in catalog.categories" :key="c.id" :value="c.name">
            {{ c.display_name }}
          </option>
        </select>
      </div>
      <div class="filter-field">
        <label>HS chapter</label>
        <select v-model="chapter">
          <option value="">All chapters</option>
          <option v-for="ch in chapterOptions" :key="ch.value" :value="ch.value">
            {{ ch.label }}
          </option>
        </select>
      </div>
      <div class="filter-field">
        <label>
          Prediction
          <InfoTip text="Filter to rows where the live classifier agrees / disagrees with the ground-truth label. Server-side filter — pagination respects the filtered total, but classifying every matching row can take a few seconds on large filtered sets." />
        </label>
        <select v-model="predictionStatus">
          <option value="">All rows</option>
          <option value="correct">Correct</option>
          <option value="mispredicted">Mispredicted only</option>
        </select>
      </div>
      <div class="filter-field" style="flex: 1; min-width: 220px;">
        <label>Search</label>
        <input
          :value="searchDraft"
          @input="onSearchInput"
          placeholder="Substring on cargo / commodity / shipment_id…"
        />
      </div>
    </FilterBar>

    <DataTable
      :columns="columns"
      :rows="data?.rows ?? []"
      :total="data?.total ?? 0"
      :limit="limit"
      :offset="offset"
      :loading="loading"
      :row-key="(r: LabelRow) => r.id"
      empty-message="No labels match these filters."
      @update:offset="offset = $event"
      @update:limit="limit = $event"
    >
      <template #empty>
        <template v-if="(crossChapterTotal ?? 0) > 0">
          <strong>{{ category }}</strong> has {{ crossChapterTotal!.toLocaleString() }}
          label<template v-if="crossChapterTotal !== 1">s</template>
          matching the other filters across all chapters.
          <button class="link-btn" @click="chapter = ''">Clear chapter filter</button>
        </template>
        <template v-else>
          No labels match these filters.
        </template>
      </template>
      <template #cell-shipment_id="{ row }">
        <code class="ship-id">{{ row.shipment_id }}</code>
      </template>
      <template #cell-split="{ row }">
        <span :class="['split-pill', row.split]">{{ row.split }}</span>
      </template>
      <template #cell-category_name="{ row }">
        <CategoryChip :category="row.category_name" />
      </template>
      <template #cell-predicted_category="{ row }">
        <span v-if="row.predicted_category === null" class="pred-empty" title="Classifier punted — likely failed input quality gate.">—</span>
        <CategoryChip
          v-else
          :category="row.predicted_category"
          :class="{ 'pred-mispredicted': isMispredicted(row) }"
        />
      </template>
      <template #cell-hs_chapter="{ row }">
        <span class="ch-mono">{{ row.hs_chapter ?? "—" }}</span>
      </template>
      <template #cell-cargo_text="{ row }">
        <span class="text-cell" :title="row.cargo_text">{{ trunc(row.cargo_text) }}</span>
      </template>
      <template #cell-commodity_text="{ row }">
        <span class="text-cell" :title="row.commodity_text">{{ trunc(row.commodity_text) }}</span>
      </template>
    </DataTable>
  </section>
</template>

<style scoped>
.page-header { margin-bottom: 16px; }
.page-header h2 { margin: 0 0 4px; font-size: 20px; }
.subtitle { margin: 0; color: var(--text-muted); max-width: 720px; }
.subtitle strong { color: var(--text); }

.ship-id {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  color: var(--text);
}

.split-pill {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border: 1px solid transparent;
}
.split-pill.train      { background: var(--accent-soft); color: var(--accent); border-color: var(--accent); }
.split-pill.validation { background: var(--warn-soft);   color: var(--warn);   border-color: var(--warn); }
.split-pill.test       { background: var(--clean-soft);  color: var(--clean);  border-color: var(--clean); }

.ch-mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  color: var(--text-muted);
}
.text-cell {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  font-size: 12px;
  max-width: 360px;
}

.pred-empty {
  color: var(--text-muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}
.pred-mispredicted {
  outline: 1px solid var(--risky);
  outline-offset: 1px;
  border-radius: 999px;
}

.link-btn {
  background: none;
  border: none;
  padding: 0;
  margin-left: 6px;
  color: var(--accent);
  font: inherit;
  cursor: pointer;
  text-decoration: underline;
}
.link-btn:hover { color: var(--text); }
</style>
