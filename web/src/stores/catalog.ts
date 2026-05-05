// catalog.ts — single fetch of /admin/api/categories and /admin/api/chapters
// cached for the app lifetime. Browse, Tokens, Labels, and Centroids all
// want these maps and re-fetching them on every navigation would burn a
// DB query for static data. Views call `ensure()` / `ensureChapters()`
// which are idempotent and concurrency-safe.
import { defineStore } from "pinia";
import { ref, computed } from "vue";
import api from "@/api";
import type { Category, ChapterRow } from "@/types";

export const useCatalogStore = defineStore("catalog", () => {
  const categories = ref<Category[]>([]);
  const loading    = ref(false);
  const error      = ref<string | null>(null);
  const loaded     = ref(false);

  const chapters        = ref<ChapterRow[]>([]);
  const chaptersLoading = ref(false);
  const chaptersError   = ref<string | null>(null);
  const chaptersLoaded  = ref(false);

  /** Map name → Category for O(1) chip lookup. */
  const byName = computed<Record<string, Category>>(() => {
    const out: Record<string, Category> = {};
    for (const c of categories.value) out[c.name] = c;
    return out;
  });

  /** Distinct hs_chapter list (sorted), useful for filter dropdowns. */
  const distinctChapters = computed<string[]>(() => {
    const set = new Set<string>();
    for (const c of chapters.value) set.add(c.hs_chapter);
    return Array.from(set).sort();
  });

  async function ensure(): Promise<void> {
    if (loaded.value || loading.value) return;
    loading.value = true;
    error.value = null;
    try {
      categories.value = await api.categories();
      loaded.value = true;
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err);
    } finally {
      loading.value = false;
    }
  }

  async function ensureChapters(): Promise<void> {
    if (chaptersLoaded.value || chaptersLoading.value) return;
    chaptersLoading.value = true;
    chaptersError.value = null;
    try {
      chapters.value = await api.chapters();
      chaptersLoaded.value = true;
    } catch (err) {
      chaptersError.value = err instanceof Error ? err.message : String(err);
    } finally {
      chaptersLoading.value = false;
    }
  }

  function invalidate(): void {
    loaded.value = false;
    chaptersLoaded.value = false;
  }

  return {
    categories, loading, error, loaded, byName,
    chapters, chaptersLoading, chaptersError, chaptersLoaded, distinctChapters,
    ensure, ensureChapters, invalidate,
  };
});
