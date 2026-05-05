<script setup lang="ts">
/**
 * Centroids.vue — UMAP/PCA scatter of every (category, hs_chapter)
 * centroid. The backend memoizes the projection per centroid rebuild,
 * so the second visit is instant.
 *
 * Visual choices:
 *   - One point per (category, chapter) — ~96 dots for our corpus.
 *   - Colour encodes category (shared palette with CategoryChip).
 *   - Size encodes sample_count (sqrt-scaled so a 5x bump in samples
 *     reads as ~2x dot, not 5x — keeps small chapters legible).
 *   - Right-side legend is the canonical category list; clicking a chip
 *     toggles series visibility.
 *   - Click a point → /admin/tokens?chapter=… so the user lands on the
 *     keyword surface that produced that centroid.
 *
 * Reducer banner: when the backend falls back to PCA (umap-learn
 * missing, or n<4), we surface that so the operator knows the layout
 * is linear and not local-structure-preserving.
 */
import { ref, computed, onMounted } from "vue";
import { useRouter } from "vue-router";
import api, { ApiError } from "@/api";
import type { CentroidProjection, CentroidPoint, CentroidAffinity } from "@/types";
import { useCatalogStore } from "@/stores/catalog";
import VChart from "@/components/VChart.vue";
import CategoryChip from "@/components/CategoryChip.vue";
import InfoTip from "@/components/InfoTip.vue";
import { paletteFor } from "@/charts/palette";

const router = useRouter();
const catalog = useCatalogStore();

const data    = ref<CentroidProjection | null>(null);
const loading = ref(false);
const error   = ref<string | null>(null);

// Categories the user has hidden via the legend. Set so toggle is O(1).
const hidden = ref<Set<string>>(new Set());

// Free-text overlay: a user-entered word gets embedded server-side and
// projected into this same space, then drawn as a gold diamond. Stays
// transient — clears on a new search or via the "Clear" button.
const overlayInput  = ref<string>("");
const overlay       = ref<CentroidAffinity | null>(null);
const overlayBusy   = ref<boolean>(false);
const overlayError  = ref<string | null>(null);

onMounted(async () => {
  loading.value = true;
  try {
    await catalog.ensure();
    data.value = await api.centroids();
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  } finally {
    loading.value = false;
  }
});

const points = computed<CentroidPoint[]>(() => data.value?.points ?? []);

// Visible categories for the legend, sourced from the catalog so the
// list is stable even when a category has zero centroids built yet.
const allCategories = computed<string[]>(() =>
  catalog.categories.map(c => c.name).sort(),
);

const visiblePoints = computed<CentroidPoint[]>(() =>
  points.value.filter(p => !hidden.value.has(p.category_name)),
);

// Sqrt-scale sample_count → 8..28 px symbol size, clamped both ends.
function symbolSize(p: CentroidPoint): number {
  const n = Math.max(1, p.sample_count);
  const scaled = 6 + Math.sqrt(n) * 1.4;
  return Math.max(8, Math.min(28, scaled));
}

const chartOption = computed(() => {
  const byCat = new Map<string, CentroidPoint[]>();
  for (const p of visiblePoints.value) {
    const arr = byCat.get(p.category_name) ?? [];
    arr.push(p);
    byCat.set(p.category_name, arr);
  }

  const series = Array.from(byCat.entries()).map(([cat, pts]) => ({
    name: cat,
    type: "scatter",
    data: pts.map(p => ({
      value: [p.x, p.y],
      meta: p,
      symbolSize: symbolSize(p),
    })),
    itemStyle: {
      color: paletteFor(cat),
      borderColor: "rgba(0,0,0,0.25)",
      borderWidth: 0.5,
    },
    emphasis: {
      itemStyle: {
        borderColor: "#1a1f2b",
        borderWidth: 1.5,
      },
      scale: 1.2,
    },
  }));

  // Append a separate "overlay" series for the user-entered keyword so
  // it sits on top of every category series and survives legend toggles.
  if (overlay.value) {
    series.push({
      name: `★ ${overlay.value.token}`,
      type: "scatter",
      data: [{
        value: [overlay.value.point.x, overlay.value.point.y],
        meta: { __overlay: true, token: overlay.value.token } as unknown as CentroidPoint,
        symbolSize: 28,
      }],
      symbol: "diamond",
      itemStyle: {
        color: "#FFB000",
        borderColor: "#1a1f2b",
        borderWidth: 1.5,
      },
      emphasis: { scale: 1.15 },
      // Force this series to draw above the category scatters.
      z: 10,
    } as unknown as typeof series[number]);
  }

  return {
    grid: { left: 30, right: 30, top: 16, bottom: 30 },
    tooltip: {
      trigger: "item",
      backgroundColor: "rgba(26,31,43,0.95)",
      borderWidth: 0,
      textStyle: { color: "#fff", fontSize: 12 },
      formatter: (params: { data?: { meta?: CentroidPoint & { __overlay?: boolean; token?: string } } }) => {
        const m = params.data?.meta;
        if (!m) return "";
        if (m.__overlay) {
          return `<div style="font-weight:600;">"${m.token}"</div>
                  <div style="opacity:.85;font-size:11px;">your overlay (★)</div>`;
        }
        return [
          `<div style="font-weight:600;margin-bottom:4px;">${m.category_name} · ch ${m.hs_chapter}</div>`,
          `<div style="opacity:.85;">${m.chapter_title}</div>`,
          `<div style="margin-top:4px;font-family:ui-monospace,monospace;font-size:11px;">samples: ${m.sample_count.toLocaleString()}</div>`,
        ].join("");
      },
    },
    xAxis: {
      type: "value",
      scale: true,
      splitLine: { lineStyle: { color: "rgba(0,0,0,0.05)" } },
      axisLabel: { show: false },
      axisTick:  { show: false },
      axisLine:  { show: false },
    },
    yAxis: {
      type: "value",
      scale: true,
      splitLine: { lineStyle: { color: "rgba(0,0,0,0.05)" } },
      axisLabel: { show: false },
      axisTick:  { show: false },
      axisLine:  { show: false },
    },
    series,
  };
});

function onChartClick(params: { data?: unknown }): void {
  // ECharts forwards the full datum object on `params.data`. We attached
  // a `meta: CentroidPoint` to every point in `chartOption`, so reading
  // it back here is robust to series re-orderings (an earlier version
  // indexed into a filtered array which broke when categories were
  // toggled).
  const meta = (params.data as { meta?: CentroidPoint & { __overlay?: boolean; token?: string } } | undefined)?.meta;
  if (!meta) return;
  // Overlay marker: clicking jumps to that token's detail page.
  if (meta.__overlay && meta.token) {
    router.push({ path: `/admin/tokens/${encodeURIComponent(meta.token)}` });
    return;
  }
  router.push({ path: "/admin/tokens", query: { chapter: meta.hs_chapter } });
}

async function runOverlay(): Promise<void> {
  const t = overlayInput.value.trim();
  if (!t) return;
  overlayBusy.value = true;
  overlayError.value = null;
  try {
    overlay.value = await api.centroidAffinity(t);
  } catch (err) {
    overlay.value = null;
    if (err instanceof ApiError && err.status === 503) {
      overlayError.value = "Centroids not built yet — overlay unavailable.";
    } else {
      overlayError.value = err instanceof Error ? err.message : String(err);
    }
  } finally {
    overlayBusy.value = false;
  }
}

function clearOverlay(): void {
  overlay.value = null;
  overlayInput.value = "";
  overlayError.value = null;
}

function toggleCategory(cat: string): void {
  const next = new Set(hidden.value);
  if (next.has(cat)) next.delete(cat); else next.add(cat);
  hidden.value = next;
}

function showOnly(cat: string): void {
  // Right-click / shift-click handler — isolates one category.
  const others = allCategories.value.filter(c => c !== cat);
  hidden.value = new Set(others);
}

function showAll(): void {
  hidden.value = new Set();
}

const reducerLabel = computed(() => {
  const r = data.value?.reducer;
  if (r === "umap") return "UMAP";
  if (r === "pca")  return "PCA";
  if (r === "tsne") return "t-SNE";
  return "—";
});

const generatedDisplay = computed(() => {
  const ts = data.value?.generated_at;
  if (!ts) return "—";
  try { return new Date(ts).toLocaleString(); } catch { return ts; }
});

const usingFallback = computed(() => data.value?.reducer === "pca");

const totalPoints   = computed(() => points.value.length);
const visibleCount  = computed(() => visiblePoints.value.length);
</script>

<template>
  <section class="page">
    <header class="page-header">
      <h2>
        Centroids
        <InfoTip side="below">
          <p>A <strong>centroid</strong> is the "average meaning" of all
          shipments tagged with one (category, HS chapter) combo —
          stored as a 384-number fingerprint.</p>
          <p>To draw it on a 2D map we use <strong>UMAP</strong>: chapters
          whose meaning is similar end up close together, dissimilar
          ones drift apart. The exact x/y numbers don't mean anything;
          only the <em>relative distances</em> do.</p>
          <p>An outlier (a dot far from its colour cluster) is a hint
          that the label set for that chapter may have drifted or be
          mis-tagged.</p>
        </InfoTip>
      </h2>
      <p class="subtitle">
        Each dot is one <code>(category, hs_chapter)</code> centroid
        projected to 2D. Related chapters cluster; outliers may flag a
        mis-categorised label set. <strong>Click a dot</strong> to inspect that
        chapter's keywords. <strong>Dot size</strong> = how many labels built
        the centroid.
      </p>
    </header>

    <div v-if="error" class="banner error">{{ error }}</div>
    <div v-if="usingFallback && !loading" class="banner info">
      <strong>PCA fallback:</strong> <code>umap-learn</code> isn't installed
      or the corpus is too small. Layout is linear, not local-structure
      preserving — install umap-learn for a richer projection.
    </div>

    <div class="meta-bar">
      <span><strong>{{ visibleCount }}</strong> / {{ totalPoints }} centroids</span>
      <span class="sep">·</span>
      <span>reducer: <code>{{ reducerLabel }}</code></span>
      <span class="sep">·</span>
      <span>generated: {{ generatedDisplay }}</span>
    </div>

    <div class="layout">
      <div class="chart-pane">
        <VChart
          :option="chartOption"
          :loading="loading"
          height="640px"
          @click="onChartClick"
        />
      </div>

      <aside class="legend-pane">
        <div class="overlay-box">
          <div class="overlay-header">
            <span>Overlay a word</span>
            <InfoTip text="Type any word — even one not in the registry. We embed it server-side and place it on the same map. Useful for probing where a discovery candidate would land." />
          </div>
          <form class="overlay-form" @submit.prevent="runOverlay">
            <input
              v-model="overlayInput"
              type="text"
              placeholder="e.g. graphene, panel, drone…"
              :disabled="overlayBusy"
              autocomplete="off"
            />
            <button type="submit" class="overlay-btn primary" :disabled="overlayBusy || !overlayInput.trim()">
              <span v-if="overlayBusy"><span class="spinner" /></span>
              <span v-else>Map</span>
            </button>
          </form>
          <div v-if="overlay" class="overlay-status">
            <span class="kw-marker">◆</span>
            <code>{{ overlay.token }}</code>
            <button class="link-btn" @click="clearOverlay" title="Remove overlay">clear</button>
          </div>
          <div v-if="overlayError" class="overlay-error">{{ overlayError }}</div>
        </div>

        <div class="legend-header">
          <span>Categories</span>
          <button class="link-btn" @click="showAll" :disabled="hidden.size === 0">
            show all
          </button>
        </div>
        <ul class="legend-list">
          <li
            v-for="cat in allCategories"
            :key="cat"
            :class="['legend-item', { dimmed: hidden.has(cat) }]"
          >
            <button
              class="legend-toggle"
              :title="hidden.has(cat) ? 'Show' : 'Hide'"
              @click="toggleCategory(cat)"
            >
              <span class="dot" :style="{ background: paletteFor(cat) }" />
              <CategoryChip :category="cat" />
            </button>
            <button
              class="link-btn solo"
              title="Show only this category"
              @click="showOnly(cat)"
            >solo</button>
          </li>
        </ul>
      </aside>
    </div>
  </section>
</template>

<style scoped>
.page-header { margin-bottom: 16px; }
.page-header h2 { margin: 0 0 4px; font-size: 20px; }
.subtitle { margin: 0; color: var(--text-muted); max-width: 720px; }
.subtitle code {
  background: var(--surface-2);
  padding: 1px 6px;
  border-radius: var(--radius-sm);
  font-size: 12px;
}

.banner {
  border-radius: var(--radius);
  padding: 10px 14px;
  margin: 8px 0;
  font-size: 13px;
}
.banner.info {
  background: var(--accent-soft);
  color: var(--text);
  border: 1px solid var(--accent);
}
.banner.info code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}

.meta-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 10px;
}
.meta-bar code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  color: var(--text);
}
.meta-bar .sep { opacity: 0.5; }
.meta-bar strong { color: var(--text); }

.layout {
  display: grid;
  grid-template-columns: 1fr 240px;
  gap: 16px;
}
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
}
.chart-pane {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 8px;
  box-shadow: var(--shadow);
}

.legend-pane {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px;
  box-shadow: var(--shadow);
  max-height: 640px;
  overflow-y: auto;
}
.legend-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 11px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 8px;
}
.legend-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.legend-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
}
.legend-item.dimmed .legend-toggle { opacity: 0.35; }

.legend-toggle {
  background: none;
  border: 0;
  padding: 2px 0;
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  font: inherit;
  flex: 1;
  text-align: left;
}
.legend-toggle .dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.link-btn {
  background: none;
  border: 0;
  color: var(--text-muted);
  font-size: 11px;
  cursor: pointer;
  font-family: inherit;
  padding: 1px 6px;
  border-radius: var(--radius-sm);
}
.link-btn:hover:not(:disabled) { color: var(--accent); background: var(--accent-soft); }
.link-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.link-btn.solo { font-size: 10px; opacity: 0.7; }

/* ── Overlay box (free-text keyword) ────────────────────────────────── */

.overlay-box {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 10px;
  margin-bottom: 12px;
}
.overlay-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  margin-bottom: 6px;
}
.overlay-form {
  display: flex;
  gap: 6px;
}
.overlay-form input {
  flex: 1;
  min-width: 0;
  padding: 5px 8px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  background: var(--surface);
  font: inherit;
  font-size: 12px;
}
.overlay-form input:focus {
  outline: none;
  border-color: var(--accent);
}
.overlay-btn.primary {
  background: var(--accent);
  color: #fff;
  border: 0;
  padding: 5px 12px;
  border-radius: var(--radius-sm);
  font-size: 12px;
  font-family: inherit;
  font-weight: 500;
  cursor: pointer;
}
.overlay-btn.primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.overlay-status {
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--text);
}
.overlay-status code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  background: var(--surface);
  padding: 1px 6px;
  border-radius: 3px;
  border: 1px solid var(--border);
}
.overlay-status .kw-marker {
  color: #FFB000;
  font-size: 14px;
  text-shadow: 0 0 1px #1a1f2b;
}
.overlay-status .link-btn { margin-left: auto; }
.overlay-error {
  margin-top: 6px;
  font-size: 11px;
  color: var(--risky, #b91c1c);
}
</style>
