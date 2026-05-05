<script setup lang="ts">
/**
 * Tokens.vue — paginated keyword browser. Wraps /admin/api/keywords with
 * a FilterBar (category, chapter, signal_class, search) and a DataTable.
 * All filter state lives in URL query params via useRouteQuery so a
 * deep-link like /admin/tokens?category=defense&signal_class=anchor
 * recreates the exact same view.
 */
import { ref, computed, watch, onMounted } from "vue";
import { useRouter } from "vue-router";
import api from "@/api";
import type { KeywordPage, KeywordRow, SignalClass } from "@/types";
import { num } from "@/types";
import { useCatalogStore } from "@/stores/catalog";
import { useRouteQuery } from "@/composables/useRouteQuery";
import FilterBar from "@/components/FilterBar.vue";
import DataTable from "@/components/DataTable.vue";
import CategoryChip from "@/components/CategoryChip.vue";
import SignalClassBadge from "@/components/SignalClassBadge.vue";
import InfoTip from "@/components/InfoTip.vue";

const router  = useRouter();
const catalog = useCatalogStore();

const category     = useRouteQuery<string>("category", "");
const chapter      = useRouteQuery<string>("chapter", "");
const signalClass  = useRouteQuery<string>("signal_class", "");
const search       = useRouteQuery<string>("search", "");
const offset       = useRouteQuery<number>("offset", 0);
const limit        = useRouteQuery<number>("limit", 50);

const data    = ref<KeywordPage | null>(null);
const loading = ref(false);
const error   = ref<string | null>(null);

// When the primary fetch returns 0 rows AND both category+chapter are set,
// probe the same query without the chapter filter — if the category does
// have keywords (in other chapters or chapter-agnostic NULL-chapter rows),
// surface a hint with a "Clear chapter filter" button.
const crossChapterTotal = ref<number | null>(null);

async function probeCrossChapter(): Promise<void> {
  if (!data.value || data.value.rows.length > 0
      || !category.value || !chapter.value) {
    crossChapterTotal.value = null;
    return;
  }
  try {
    const probe = await api.keywords({
      category:     category.value,
      signal_class: signalClass.value || undefined,
      search:       search.value || undefined,
      limit:        1,
      offset:       0,
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
    data.value = await api.keywords({
      category:     category.value || undefined,
      chapter:      chapter.value || undefined,
      signal_class: signalClass.value || undefined,
      search:       search.value || undefined,
      limit:        limit.value,
      offset:       offset.value,
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

// Reload whenever any filter changes. Reset offset on filter changes
// (but not on offset itself).
watch(
  () => [category.value, chapter.value, signalClass.value, search.value],
  () => {
    if (offset.value !== 0) offset.value = 0;
    else void load();
  },
);
watch(() => [offset.value, limit.value], () => void load());

// Debounce search input — reload via the watch above when value lands.
let searchTimer: number | undefined;
const searchDraft = ref(search.value);
watch(search, (v) => { searchDraft.value = v; });
function onSearchInput(ev: Event): void {
  const v = (ev.target as HTMLInputElement).value;
  searchDraft.value = v;
  if (searchTimer) window.clearTimeout(searchTimer);
  searchTimer = window.setTimeout(() => { search.value = v; }, 300);
}

const SIGNAL_CLASSES: SignalClass[] = ["anchor", "signal", "suppressor", "modifier"];

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
  { key: "keyword",       label: "Token",    width: "20%",
    tip: "The exact word/phrase the classifier matches against shipment text. Click to see all chapters this word lives in." },
  { key: "category_name", label: "Category", width: "14%",
    tip: "The business category this row pulls toward (e.g. electronics, defense)." },
  { key: "hs_chapter",    label: "Ch",       width: "60px", align: "center" as const,
    tip: "Harmonised System chapter — the 2-digit international tariff code (85 = electrical, 87 = vehicles, etc)." },
  { key: "signal_class",  label: "Class",    width: "100px",
    tip: "How the classifier uses this word. anchor = decisive on its own. signal = soft evidence. suppressor = vetoes a category. modifier = re-routes another word." },
  { key: "weight",        label: "Weight",   width: "80px", align: "right" as const,
    tip: "Strength of evidence, 0–1.5. ≥0.9 = anchor (one match classifies). 0.4–0.9 = normal. <0.4 = weak hint." },
  { key: "source",        label: "Source",   width: "120px",
    tip: "Where this row came from: seed = hand-curated baseline; discovered = surfaced by the discovery scan; manual = added by an operator." },
  { key: "notes",         label: "Notes" },
];

function gotoToken(token: string): void {
  router.push({ path: `/admin/tokens/${encodeURIComponent(token)}` });
}

function fmtWeight(w: string | number | null | undefined): string {
  if (w === null || w === undefined) return "—";
  return num(w).toFixed(2);
}

const totalDisplay = computed(() => data.value?.total ?? 0);
</script>

<template>
  <section class="page">
    <header class="page-header">
      <h2>
        Tokens
        <InfoTip side="below">
          <p>Each row is one <strong>keyword</strong> tied to a
          <strong>(category, HS chapter)</strong> pair, with a <strong>weight</strong>
          (0–1.5) saying how strongly that word pulls a shipment toward
          that category.</p>
          <p>The same word can appear in many rows — that's polysemy, e.g.
          <code>battery</code> shows up under chapters 85, 87, 95.</p>
        </InfoTip>
      </h2>
      <p class="subtitle">
        Browse the {{ totalDisplay.toLocaleString() }} keyword rows in the registry.
        Filters are URL-synced so a deep link reproduces the exact view.
      </p>
      <p class="page-tip">
        <strong>Tip:</strong> click any token to see <em>every</em> chapter it
        appears in, plus how the embedding model thinks it fits.
        Hover the column headers (<span class="tip-mark-inline">ⓘ</span>) for
        plain-English explanations.
      </p>
    </header>

    <div v-if="error" class="banner error">{{ error }}</div>

    <FilterBar>
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
          Signal class
          <InfoTip side="below">
            <p><strong>anchor</strong> — high-confidence give-away (e.g.
            <code>tank</code> → defense). One match alone classifies.</p>
            <p><strong>signal</strong> — softer evidence (e.g.
            <code>battery</code>). Needs to add up with others.</p>
            <p><strong>suppressor</strong> — keyword that <em>rules out</em>
            a category (e.g. <code>toy</code> in the defense bucket).</p>
            <p><strong>modifier</strong> — a phrase that redirects another
            keyword to a specific chapter (e.g. <code>li-ion</code>
            sends <code>battery</code> to chapter 85, not 87).</p>
          </InfoTip>
        </label>
        <select v-model="signalClass">
          <option value="">Any class</option>
          <option v-for="sc in SIGNAL_CLASSES" :key="sc" :value="sc">{{ sc }}</option>
        </select>
      </div>
      <div class="filter-field" style="flex: 1; min-width: 220px;">
        <label>Search</label>
        <input
          type="text"
          :value="searchDraft"
          @input="onSearchInput"
          placeholder="Substring match on keyword…"
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
      :row-key="(r: KeywordRow) => r.id"
      empty-message="No keywords match these filters."
      @update:offset="offset = $event"
      @update:limit="limit = $event"
    >
      <template #empty>
        <template v-if="(crossChapterTotal ?? 0) > 0">
          <strong>{{ category }}</strong> has {{ crossChapterTotal!.toLocaleString() }}
          keyword<template v-if="crossChapterTotal !== 1">s</template>
          matching the other filters across all chapters (or chapter-agnostic).
          <button class="link-btn" @click="chapter = ''">Clear chapter filter</button>
        </template>
        <template v-else>
          No keywords match these filters.
        </template>
      </template>
      <template #cell-keyword="{ row }">
        <a class="token-link" @click.prevent="gotoToken(row.keyword)" href="#">
          {{ row.keyword }}
        </a>
      </template>
      <template #cell-category_name="{ row }">
        <CategoryChip :category="row.category_name" />
      </template>
      <template #cell-hs_chapter="{ row }">
        <span class="ch-mono">{{ row.hs_chapter ?? "—" }}</span>
      </template>
      <template #cell-signal_class="{ row }">
        <SignalClassBadge :value="row.signal_class" compact />
      </template>
      <template #cell-weight="{ row }">
        <span class="num-cell">{{ fmtWeight(row.weight) }}</span>
      </template>
      <template #cell-notes="{ row }">
        <span class="notes-cell" :title="row.notes ?? ''">
          {{ row.notes ?? "" }}
        </span>
      </template>
    </DataTable>
  </section>
</template>

<style scoped>
.page-header { margin-bottom: 16px; }
.page-header h2 { margin: 0 0 4px; font-size: 20px; }
.subtitle { margin: 0 0 8px; color: var(--text-muted); max-width: 720px; }
.page-tip {
  margin: 0;
  font-size: 12px;
  color: var(--text-muted);
  background: var(--accent-soft, #eff6ff);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 6px 10px;
  display: inline-block;
}
.page-tip strong { color: var(--text); }
.tip-mark-inline {
  display: inline-block;
  padding: 0 4px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 999px;
  font-size: 10px;
}

.token-link {
  color: var(--accent);
  cursor: pointer;
  text-decoration: none;
  font-weight: 500;
}
.token-link:hover { text-decoration: underline; }

.ch-mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  color: var(--text-muted);
}
.num-cell {
  font-variant-numeric: tabular-nums;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.notes-cell {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  color: var(--text-muted);
  font-size: 12px;
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
