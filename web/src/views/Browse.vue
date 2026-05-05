<script setup lang="ts">
/**
 * Browse.vue — two-pane category ↔ chapter explorer.
 *
 * Left:  the 20 active categories as a vertical list with chapter count.
 * Right: chapters of the selected category as cards, each showing
 *        chapter title, primary marker, keyword count, and label count.
 *        Cards link out to the Tokens browser pre-filtered to that
 *        (category, chapter) so this page is the "where do I want to
 *        zoom in" map.
 *
 * Both data sets come from the catalog store so revisiting the page is
 * cache-cheap. The selected category lives in the URL query (`?cat=…`)
 * so deep-links work.
 */
import { computed, onMounted } from "vue";
import { useRouter } from "vue-router";
import { useCatalogStore } from "@/stores/catalog";
import { useRouteQuery } from "@/composables/useRouteQuery";
import CategoryChip from "@/components/CategoryChip.vue";
import InfoTip from "@/components/InfoTip.vue";
import type { Category, ChapterRow } from "@/types";

const router  = useRouter();
const catalog = useCatalogStore();
const selected = useRouteQuery<string>("cat", "");

onMounted(async () => {
  await Promise.all([catalog.ensure(), catalog.ensureChapters()]);
  // Default to the first category if none selected
  if (!selected.value && catalog.categories.length) {
    selected.value = catalog.categories[0].name;
  }
});

const selectedCategory = computed<Category | null>(() => {
  if (!selected.value) return null;
  return catalog.byName[selected.value] ?? null;
});

const selectedChapters = computed<ChapterRow[]>(() => {
  if (!selectedCategory.value) return [];
  return catalog.chapters.filter(c => c.category_id === selectedCategory.value!.id);
});

function selectCategory(name: string): void {
  selected.value = name;
}

function gotoTokens(category: string, chapter?: string): void {
  const query: Record<string, string> = { category };
  if (chapter) query.chapter = chapter;
  router.push({ path: "/admin/tokens", query });
}

// `chapters` is optional on Category in some response shapes — guard
// rather than assume the embedded array is always present so this view
// can't crash on an older /categories payload.
function chapterCountFor(cat: Category): number {
  return (cat.chapters ?? []).length;
}
function totalKeywordsFor(cat: Category): number {
  return (cat.chapters ?? []).reduce((acc, ch) => acc + (ch.keyword_count ?? 0), 0);
}
</script>

<template>
  <section class="page">
    <header class="page-header">
      <h2>
        Browse
        <InfoTip side="below">
          <p><strong>Categories</strong> are the 20 business buckets we
          classify a shipment into (e.g. <code>electronics</code>).</p>
          <p><strong>HS chapters</strong> are the 96 internationally
          standardised 2-digit codes for goods (e.g. <code>85</code>
          = electrical machinery). Each category owns one or more
          chapters; the <strong>★ primary</strong> chapter is the one
          most representative of the category.</p>
        </InfoTip>
      </h2>
      <p class="subtitle">
        Categories on the left, the HS chapters mapped to each on the right.
        Click a chapter card to open its keywords.
      </p>
    </header>

    <div v-if="catalog.error" class="banner error">{{ catalog.error }}</div>

    <div class="browse-shell">
      <aside class="cat-list">
        <div class="cat-list-title">Categories ({{ catalog.categories.length }})</div>
        <button
          v-for="cat in catalog.categories"
          :key="cat.id"
          :class="['cat-row', { active: selected === cat.name }]"
          :title="cat.display_name"
          @click="selectCategory(cat.name)"
        >
          <span class="cat-row-text">
            <CategoryChip :category="cat.name" />
            <span class="cat-display-name">{{ cat.display_name }}</span>
          </span>
          <span class="cat-meta">
            {{ chapterCountFor(cat) }}ch · {{ totalKeywordsFor(cat) }}kw
          </span>
        </button>
      </aside>

      <div class="cat-detail">
        <template v-if="selectedCategory">
          <div class="cat-detail-header">
            <CategoryChip :category="selectedCategory.name" />
            <h3>{{ selectedCategory.display_name }}</h3>
            <button class="link-btn" @click="gotoTokens(selectedCategory.name)">
              Open all keywords →
            </button>
          </div>

          <div class="cat-meta-row">
            <span><strong>{{ selectedChapters.length }}</strong> chapters</span>
            <span>
              semantic_weight=<strong>{{ selectedCategory.semantic_weight }}</strong>
              <InfoTip text="How much the meaning of the text matters when scoring this category. 0.8 = mostly meaning, 0.2 = mostly keywords." />
            </span>
            <span>
              keyword_weight=<strong>{{ selectedCategory.keyword_weight }}</strong>
              <InfoTip text="How much keyword matches matter (the complement of semantic_weight). They sum to 1.0." />
            </span>
            <span v-if="selectedCategory.threshold !== null">
              threshold=<strong>{{ selectedCategory.threshold }}</strong>
              <InfoTip text="Minimum confidence required to call a shipment this category. Null = use the request-level default." />
            </span>
          </div>

          <div class="chapter-grid">
            <article
              v-for="ch in selectedChapters"
              :key="ch.hs_chapter"
              class="chapter-card"
              @click="gotoTokens(selectedCategory!.name, ch.hs_chapter)"
            >
              <header class="chapter-card-header">
                <span class="chapter-num">{{ ch.hs_chapter }}</span>
                <span v-if="ch.is_primary" class="primary-star" title="Primary chapter">★</span>
              </header>
              <p class="chapter-title">{{ ch.chapter_title }}</p>
              <footer class="chapter-card-footer">
                <span class="badge-count" title="Keywords"><b>{{ ch.keyword_count }}</b> kw</span>
                <span class="badge-count" title="Labeled shipments"><b>{{ ch.label_count }}</b> lbl</span>
                <span
                  v-if="ch.centroid_present"
                  class="badge-count centroid"
                  :title="`centroid · n=${ch.sample_count ?? '?'}`"
                >●</span>
                <span v-else class="badge-count missing" title="No centroid yet">○</span>
              </footer>
            </article>
          </div>

          <p v-if="!selectedChapters.length && catalog.chaptersLoaded" class="empty-hint">
            No chapters mapped to this category.
          </p>
        </template>
        <p v-else class="empty-hint">Select a category on the left.</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.page-header { margin-bottom: 16px; }
.page-header h2 { margin: 0 0 4px; font-size: 20px; }
.subtitle { margin: 0; color: var(--text-muted); max-width: 720px; }

.browse-shell {
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 16px;
  align-items: start;
}
@media (max-width: 900px) {
  .browse-shell { grid-template-columns: 1fr; }
}

.cat-list {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 8px;
  position: sticky;
  top: calc(var(--topbar-h) + 12px);
  max-height: calc(100vh - var(--topbar-h) - 24px);
  overflow: auto;
}
.cat-list-title {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  padding: 8px 8px 6px;
}
.cat-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  background: transparent;
  border: 1px solid transparent;
  padding: 6px 8px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  text-align: left;
  font-family: inherit;
}
.cat-row:hover { background: var(--surface-2); }
.cat-row.active {
  background: var(--accent-soft);
  border-color: var(--accent);
}
.cat-meta {
  color: var(--text-muted);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  margin-left: 8px;
  flex-shrink: 0;
}
.cat-row-text {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  flex: 1;
}
.cat-display-name {
  font-size: 12px;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
}
.cat-row.active .cat-display-name { color: var(--accent); font-weight: 500; }

.cat-detail {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 16px;
}
.cat-detail-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}
.cat-detail-header h3 { margin: 0; font-size: 18px; flex: 1; }

.cat-meta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  color: var(--text-muted);
  font-size: 12px;
  margin-bottom: 16px;
}
.cat-meta-row strong { color: var(--text); }

.chapter-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 12px;
}
.chapter-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px;
  cursor: pointer;
  transition: border-color 120ms, box-shadow 120ms, transform 120ms;
}
.chapter-card:hover {
  border-color: var(--accent);
  box-shadow: var(--shadow);
  transform: translateY(-1px);
}
.chapter-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 4px;
}
.chapter-num {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 14px;
  font-weight: 600;
  color: var(--accent);
}
.primary-star { color: var(--warn); font-size: 14px; }
.chapter-title {
  margin: 0 0 8px;
  font-size: 13px;
  color: var(--text);
  line-height: 1.35;
  min-height: 36px;
}
.chapter-card-footer {
  display: flex;
  gap: 8px;
  align-items: center;
  font-size: 11px;
  color: var(--text-muted);
  border-top: 1px solid var(--border);
  padding-top: 8px;
}
.badge-count b { color: var(--text); font-variant-numeric: tabular-nums; }
.badge-count.centroid { color: var(--clean); margin-left: auto; }
.badge-count.missing  { color: var(--grey);  margin-left: auto; }

.empty-hint { color: var(--text-muted); margin: 12px 0; }

.link-btn {
  background: none;
  border: 1px solid var(--border-2);
  color: var(--accent);
  padding: 4px 10px;
  border-radius: var(--radius-sm);
  font-size: 12px;
  font-family: inherit;
  cursor: pointer;
}
.link-btn:hover { border-color: var(--accent); background: var(--accent-soft); }
</style>
