// catalog.ts — single fetch of /admin/api/categories cached for the app
// lifetime. Browse and Tokens both want the category → chapters map and
// re-fetching it on every navigation would burn a DB query for static
// data. After a write lands in Commit 6, the affected view will call
// `invalidate()` so the next mount re-loads.
import { defineStore } from "pinia";
import { ref, computed } from "vue";
import api from "@/api";
import type { Category } from "@/types";

export const useCatalogStore = defineStore("catalog", () => {
  const categories = ref<Category[]>([]);
  const loading    = ref(false);
  const error      = ref<string | null>(null);
  const loaded     = ref(false);

  /** Map name → Category for O(1) chip lookup. */
  const byName = computed<Record<string, Category>>(() => {
    const out: Record<string, Category> = {};
    for (const c of categories.value) out[c.name] = c;
    return out;
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

  function invalidate(): void {
    loaded.value = false;
  }

  return { categories, loading, error, loaded, byName, ensure, invalidate };
});
