<script setup lang="ts">
/**
 * TokenDetail.vue — every chapter row for a single keyword, plus the
 * collision-registry entry if one exists.
 *
 * Two GETs in parallel: /admin/api/keywords/by-token/{token} (always)
 * and /admin/api/collisions/{token} (404-tolerant). The collision panel
 * only renders on 200; a missing entry is the common case.
 */
import { ref, computed, watch, onMounted } from "vue";
import { useRoute, useRouter } from "vue-router";
import api, { ApiError } from "@/api";
import type {
  KeywordByToken, KeywordRow,
  CentroidAffinity, CentroidAffinityRow,
  CentroidProjection, CentroidPoint,
} from "@/types";
import { num } from "@/types";
import StatCard from "@/components/StatCard.vue";
import CategoryChip from "@/components/CategoryChip.vue";
import SignalClassBadge from "@/components/SignalClassBadge.vue";
import RiskTierBadge from "@/components/RiskTierBadge.vue";
import JsonBlock from "@/components/JsonBlock.vue";
import VChart from "@/components/VChart.vue";
import InfoTip from "@/components/InfoTip.vue";
import { paletteFor, KEYWORD_COLOR } from "@/charts/palette";

const route   = useRoute();
const router  = useRouter();

const detail    = ref<KeywordByToken | null>(null);
const loading   = ref(false);
const error     = ref<string | null>(null);

// Affinity + centroid map load in parallel but on their own track — the
// Semantic Neighborhood card is supplementary and must not block the
// primary view if the embedding model or centroid cache is cold/missing.
const affinity        = ref<CentroidAffinity | null>(null);
const centroidMap     = ref<CentroidProjection | null>(null);
const affinityLoading = ref(false);
const affinityError   = ref<string | null>(null);

const token = computed(() => decodeURIComponent(String(route.params.token ?? "")));

async function load(): Promise<void> {
  if (!token.value) return;
  loading.value = true;
  error.value = null;
  detail.value = null;
  try {
    detail.value = await api.keywordByToken(token.value);
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  } finally {
    loading.value = false;
  }
}

async function loadAffinity(): Promise<void> {
  if (!token.value) return;
  affinityLoading.value = true;
  affinityError.value = null;
  affinity.value = null;
  // Don't reset centroidMap — same payload across tokens, the backend
  // already memoizes it; just refresh on the first load.
  try {
    const [aff, cm] = await Promise.all([
      api.centroidAffinity(token.value),
      centroidMap.value ? Promise.resolve(centroidMap.value) : api.centroids(),
    ]);
    affinity.value = aff;
    centroidMap.value = cm;
  } catch (err) {
    // 503 (no centroids yet) or model-not-loaded → degrade silently with
    // an inline note. Other errors surface so we know something's broken.
    if (err instanceof ApiError && err.status === 503) {
      affinityError.value = "Centroids not built yet — embedding map unavailable.";
    } else {
      affinityError.value = err instanceof Error ? err.message : String(err);
    }
  } finally {
    affinityLoading.value = false;
  }
}

onMounted(() => { void load(); void loadAffinity(); });
watch(token, () => { void load(); void loadAffinity(); });

const rows = computed<KeywordRow[]>(() => detail.value?.rows ?? []);
// Drop nulls before counting — a single chapter-less row would otherwise
// show up as one "extra chapter" called "".
const nChapters  = computed(() => new Set(
  rows.value.map(r => r.hs_chapter).filter((c): c is string => !!c),
).size);
const nCategories = computed(() => new Set(rows.value.map(r => r.category_name)).size);
const maxWeight = computed(() =>
  rows.value.length ? Math.max(...rows.value.map(r => num(r.weight))) : 0,
);

const weightChart = computed(() => {
  const sorted = [...rows.value].sort((a, b) => num(b.weight) - num(a.weight));
  return {
    grid: { left: 110, right: 36, top: 10, bottom: 28 },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (params: unknown) => {
        const arr = params as Array<{ name: string; value: number; dataIndex: number }>;
        const p = arr[0];
        const r = sorted[p.dataIndex];
        return `<b>${r.category_name}</b> · ch${r.hs_chapter ?? "—"}<br/>
                weight = ${p.value.toFixed(2)} · ${r.signal_class}`;
      },
    },
    xAxis: { type: "value", min: 0, max: 1.5, axisLabel: { fontSize: 11 } },
    yAxis: {
      type: "category",
      data: sorted.map(r => `${r.category_name} · ${r.hs_chapter ?? "—"}`),
      axisLabel: { fontSize: 11 },
      inverse: true,
    },
    series: [{
      type: "bar",
      data: sorted.map(r => num(r.weight)),
      itemStyle: { color: "#2563eb", borderRadius: [0, 4, 4, 0] },
      label: { show: true, position: "right", fontSize: 11, formatter: (p: { value: number }) => p.value.toFixed(2) },
    }],
  };
});

function gotoChapter(ch: string | null): void {
  if (!ch) return;
  router.push({ path: "/admin/tokens", query: { chapter: ch } });
}
function gotoTokensFiltered(category: string, chapter: string): void {
  router.push({ path: "/admin/tokens", query: { category, chapter } });
}
function back(): void {
  router.push({ path: "/admin/tokens" });
}
function fmtWeight(w: string | number | null | undefined): string {
  if (w === null || w === undefined) return "—";
  return num(w).toFixed(2);
}

// ── Semantic neighborhood (centroid affinity) ────────────────────────────
// Palette + KEYWORD_COLOR are imported from charts/palette.ts. Note that
// this view's historical fallback was "#9aa3b1" (darker than Centroids' —
// it sits better on bar charts), so we pass it explicitly to paletteFor.
const PALETTE_FALLBACK = "#9aa3b1";

const TOP_N = 15;

// Disagreement detector tuning. Tweak here when the embedding model or
// registry shifts. Used by the `disagreements` computed below.
const OVER_WEIGHT_REGISTRY_MIN = 0.7; // registry weight at/above which we expect strong embedding agreement
const OVER_WEIGHT_COSINE_MAX   = 0.3; // embedding cosine below which a strong registry row looks suspicious
const DISCOVERY_COSINE_MIN     = 0.6; // embedding cosine above which a missing-registry token is a candidate

const topAffinities = computed<CentroidAffinityRow[]>(() =>
  (affinity.value?.similarities ?? []).slice(0, TOP_N),
);

/** Top-15 cosines as a horizontal bar chart, sorted desc, colored by category. */
const affinityBarChart = computed(() => {
  const rows = topAffinities.value;
  return {
    grid: { left: 150, right: 60, top: 8, bottom: 28 },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (params: unknown) => {
        const arr = params as Array<{ dataIndex: number; value: number }>;
        const r = rows[arr[0].dataIndex];
        if (!r) return "";
        const reg = r.has_registry_row
          ? `<br/>registered: weight = ${(r.registry_weight ?? 0).toFixed(2)}`
          : `<br/><span style="color:#fbbf24;">no registry row</span>`;
        return `<b>${r.category_name}</b> · ch ${r.hs_chapter}<br/>
                ${r.chapter_title}<br/>
                cosine = ${r.cosine.toFixed(3)}${reg}`;
      },
    },
    xAxis: {
      type: "value", min: -0.2, max: 1.0,
      axisLabel: { fontSize: 11 },
      splitLine: { lineStyle: { color: "rgba(0,0,0,0.05)" } },
    },
    yAxis: {
      type: "category",
      data: rows.map(r =>
        `${r.has_registry_row ? "✓ " : "✗ "}${r.category_name} · ${r.hs_chapter}`,
      ),
      axisLabel: { fontSize: 11 },
      inverse: true,
    },
    series: [{
      type: "bar",
      data: rows.map(r => ({
        value: r.cosine,
        itemStyle: {
          color: paletteFor(r.category_name, PALETTE_FALLBACK),
          opacity: r.has_registry_row ? 1.0 : 0.55,
          borderRadius: [0, 4, 4, 0],
        },
        meta: r,
      })),
      label: {
        show: true, position: "right", fontSize: 11,
        formatter: (p: { value: number }) => p.value.toFixed(2),
      },
    }],
  };
});

/**
 * Mini scatter: every centroid plotted at its real UMAP coords, dimmed.
 * The embedded keyword is overlaid as a gold diamond at its real (x, y).
 * Top-5 nearest centroids get their category color so the eye finds them
 * quickly; the rest are grey.
 *
 * Both the centroid map and the keyword point come from the SAME UMAP
 * fit (cached server-side), so positions are directly comparable.
 */
const affinityScatterChart = computed(() => {
  if (!affinity.value || !centroidMap.value) return {};

  const sims    = affinity.value.similarities;
  const cosByKey = new Map<string, number>();
  for (const s of sims) cosByKey.set(`${s.category_id}|${s.hs_chapter}`, s.cosine);

  const HIGHLIGHT = new Set(sims.slice(0, 5).map(s => `${s.category_id}|${s.hs_chapter}`));

  const centroidPoints = centroidMap.value.points.map((p: CentroidPoint) => {
    const key  = `${p.category_id}|${p.hs_chapter}`;
    const isHi = HIGHLIGHT.has(key);
    const cos  = cosByKey.get(key);
    return {
      value: [p.x, p.y],
      symbolSize: isHi ? 14 : 8,
      itemStyle: {
        color:       isHi ? paletteFor(p.category_name, PALETTE_FALLBACK) : "#cbd2d9",
        opacity:     isHi ? 0.95 : 0.45,
        borderColor: "rgba(0,0,0,0.15)",
        borderWidth: 0.5,
      },
      meta: { ...p, cosine: cos },
    };
  });

  return {
    grid: { left: 16, right: 16, top: 16, bottom: 16 },
    tooltip: {
      trigger: "item",
      backgroundColor: "rgba(26,31,43,0.95)",
      borderWidth: 0,
      textStyle: { color: "#fff", fontSize: 12 },
      formatter: (params: { data?: { meta?: (CentroidPoint & { cosine?: number }) | { token: string } } }) => {
        const m = params.data?.meta;
        if (!m) return "";
        if ("token" in m) {
          return `<div style="font-weight:600;">"${m.token}"</div>
                  <div style="opacity:.85;font-size:11px;">your keyword (★)</div>`;
        }
        const cosLine = typeof m.cosine === "number"
          ? `<div style="margin-top:4px;font-family:ui-monospace,monospace;font-size:11px;">
               cosine = ${m.cosine.toFixed(3)}
             </div>`
          : "";
        return `<div style="font-weight:600;">${m.category_name} · ch ${m.hs_chapter}</div>
                <div style="opacity:.85;">${m.chapter_title}</div>${cosLine}`;
      },
    },
    xAxis: { type: "value", scale: true, show: false },
    yAxis: { type: "value", scale: true, show: false },
    series: [
      { type: "scatter", name: "centroids", data: centroidPoints },
      {
        type:        "scatter",
        name:        "keyword",
        symbol:      "diamond",
        symbolSize:  22,
        z:           10,
        data: [{
          value:     [affinity.value.point.x, affinity.value.point.y],
          itemStyle: { color: KEYWORD_COLOR, borderColor: "#1a1f2b", borderWidth: 1.5 },
          meta:      { token: affinity.value.token },
        }],
      },
    ],
  };
});

function onAffinityBarClick(params: { data?: unknown }): void {
  const meta = (params.data as { meta?: CentroidAffinityRow } | undefined)?.meta;
  if (!meta) return;
  gotoTokensFiltered(meta.category_name, meta.hs_chapter);
}

/**
 * Disagreement detector: flags rows where the keyword registry says one
 * thing and the embedding model says another. Two flavours surfaced:
 *   - registry STRONG, embedding WEAK  → maybe an over-weighted entry
 *   - embedding STRONG, registry MISSING → discovery candidate
 * Thresholds are deliberately conservative; tuning is cheap if noise.
 */
const disagreements = computed(() => {
  const sims = affinity.value?.similarities ?? [];
  const weakOnEmbedding   = sims.filter(s =>
    s.has_registry_row
    && (s.registry_weight ?? 0) >= OVER_WEIGHT_REGISTRY_MIN
    && s.cosine < OVER_WEIGHT_COSINE_MAX,
  );
  const missingFromRegistry = sims.filter(s =>
    !s.has_registry_row && s.cosine > DISCOVERY_COSINE_MIN,
  );
  return { weakOnEmbedding, missingFromRegistry };
});

const hasDisagreement = computed(() =>
  disagreements.value.weakOnEmbedding.length > 0 ||
  disagreements.value.missingFromRegistry.length > 0,
);

// The Semantic neighborhood section is collapsed by default to keep the
// page from feeling overwhelming. We auto-open it the first time a
// disagreement shows up — that's the case where it's actually worth
// looking at — and remember the user's choice within the page lifetime.
const showSemantic = ref<boolean>(false);
watch(hasDisagreement, (isNow, was) => {
  if (isNow && !was) showSemantic.value = true;
});
</script>

<template>
  <section class="page">
    <header class="page-header">
      <button class="back-btn" @click="back">← All tokens</button>
      <h2>{{ token }}</h2>
      <p class="subtitle">
        Everything the classifier knows about this single word.
      </p>
    </header>

    <div v-if="error" class="banner error">{{ error }}</div>

    <div v-if="loading && !detail" class="status"><span class="spinner" /> loading…</div>

    <template v-else-if="detail">
      <div class="page-intro">
        <p>
          The classifier sees this word in
          <strong>{{ rows.length }}</strong> place{{ rows.length === 1 ? "" : "s" }}
          across the registry — listed below with the weight and signal class
          attached to each. Polysemous words (registered in &gt;1 chapter)
          can be candidates for a collision rule.
        </p>
      </div>

      <div class="stat-grid">
        <StatCard
          label="Chapters"
          :value="nChapters"
          tone="accent"
          hint="distinct HS chapters this word is registered in"
        />
        <StatCard
          label="Categories"
          :value="nCategories"
          tone="accent"
          hint="distinct business categories"
        />
        <StatCard
          label="Max weight"
          :value="maxWeight ? maxWeight.toFixed(2) : '—'"
          :tone="maxWeight >= 0.9 ? 'clean' : 'default'"
          :hint="maxWeight >= 0.9
            ? 'anchor-strength (one match classifies)'
            : maxWeight >= 0.4
              ? 'normal signal (adds up across matches)'
              : 'weak evidence only'"
        />
      </div>

      <section v-if="rows.length" class="card">
        <h3>Where this word is registered ({{ rows.length }} row{{ rows.length === 1 ? "" : "s" }})</h3>
        <p class="chart-hint">
          Each row says <em>"if you see this word in shipment text, treat it as evidence for this (category, chapter) at this strength."</em>
          Click a chapter code to see all keywords for that chapter.
        </p>
        <div class="table-wrap">
          <table class="rows-table">
            <thead>
              <tr>
                <th>Category</th>
                <th>Chapter</th>
                <th>
                  Class
                  <InfoTip text="anchor = single-word give-away (one match classifies). signal = soft evidence (adds up across matches). suppressor = veto word (subtracts from a category). modifier = re-router (sends another word to a chapter)." />
                </th>
                <th class="num">
                  Weight
                  <InfoTip text="Strength of this evidence. 0.9–1.5 = anchor (decisive). 0.4–0.9 = normal signal. <0.4 = weak hint only." />
                </th>
                <th>
                  Source
                  <InfoTip text="Where this row came from: seed = hand-curated baseline; discovered = surfaced by the discovery scan; manual = added later by an operator." />
                </th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in rows" :key="r.id">
                <td><CategoryChip :category="r.category_name" /></td>
                <td>
                  <a class="ch-link" @click.prevent="gotoChapter(r.hs_chapter)" href="#">
                    {{ r.hs_chapter ?? "—" }}
                  </a>
                </td>
                <td><SignalClassBadge :value="r.signal_class" compact /></td>
                <td class="num">{{ fmtWeight(r.weight) }}</td>
                <td class="src">{{ r.source ?? "" }}</td>
                <td class="notes" :title="r.notes ?? ''">{{ r.notes ?? "" }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section v-if="rows.length > 1" class="card">
        <h3>Weight comparison</h3>
        <p class="chart-hint">
          Same rows as above, sorted high → low so you can see at a glance which
          (category, chapter) gets the strongest vote when this word appears.
          The longer the bar, the more decisive the match.
        </p>
        <VChart :option="weightChart" :height="`${Math.max(180, rows.length * 28 + 60)}px`" />
      </section>

      <div class="banner info">
        Registry is read-only from this view. To register, edit, or delete
        keywords and collisions, use
        <code>scripts/apply_collision_change.py</code> (CCTR-audited).
        See <code>docs/UI_GUIDE.md</code> for context.
      </div>

      <!-- Semantic neighborhood: ML-heavy section, collapsed by default to
           keep the page calm for users who only want the registry view.
           Auto-opens when a disagreement is detected (handled in the
           script). -->
      <section class="card semantic-card">
        <button class="semantic-toggle" @click="showSemantic = !showSemantic">
          <span class="caret">{{ showSemantic ? "▾" : "▸" }}</span>
          <span class="semantic-title">
            Compare with the embedding model
            <InfoTip side="below">
              <p>An optional second opinion. We feed "{{ token }}" through the
              <strong>embedding model</strong> — the same one that builds
              centroids — and ask: "out of all
              <em>(category, chapter)</em> bins, which ones use the most
              similar language to this word?"</p>
              <p>If the model's answer matches the registry rows above,
              that's reassuring. If they disagree, it's a hint to look
              closer — maybe a weight is too high, or there's a chapter
              the registry missed.</p>
              <p>This is purely informational — it doesn't change any
              classification. The registry is the source of truth.</p>
            </InfoTip>
          </span>
          <span v-if="hasDisagreement && !showSemantic" class="semantic-badge">
            ⚠ disagreement found
          </span>
        </button>

        <div v-if="showSemantic" class="semantic-body">
          <p class="chart-hint">
            <strong>What this answers:</strong> "if I forgot the registry
            existed, where would the language model put this word?" Useful
            for spotting over-weighted rows or chapters the registry
            hasn't covered yet.
          </p>

          <div v-if="affinityLoading && !affinity" class="status">
            <span class="spinner" /> embedding "{{ token }}"…
          </div>
          <div v-else-if="affinityError" class="banner info">
            <strong>Not available:</strong> {{ affinityError }}
          </div>
          <template v-else-if="affinity">
            <div v-if="hasDisagreement" class="disagree-banner">
              <div v-if="disagreements.weakOnEmbedding.length" class="disagree-row">
                <span class="disagree-tag warn">might be over-weighted</span>
                The registry gives <strong>"{{ token }}"</strong> a high weight
                in
                <template v-for="(s, i) in disagreements.weakOnEmbedding.slice(0, 3)" :key="`w-${s.category_id}-${s.hs_chapter}`">
                  <code>{{ s.category_name }} · ch {{ s.hs_chapter }}</code><span v-if="i < Math.min(2, disagreements.weakOnEmbedding.length - 1)">, </span>
                </template><span v-if="disagreements.weakOnEmbedding.length > 3"> and {{ disagreements.weakOnEmbedding.length - 3 }} more</span>,
                but the language model doesn't see a strong link there.
                Worth a sanity check.
              </div>
              <div v-if="disagreements.missingFromRegistry.length" class="disagree-row">
                <span class="disagree-tag info">possible discovery</span>
                The language model thinks this word fits
                <template v-for="(s, i) in disagreements.missingFromRegistry.slice(0, 3)" :key="`m-${s.category_id}-${s.hs_chapter}`">
                  <code>{{ s.category_name }} · ch {{ s.hs_chapter }}</code><span v-if="i < Math.min(2, disagreements.missingFromRegistry.length - 1)">, </span>
                </template><span v-if="disagreements.missingFromRegistry.length > 3"> and {{ disagreements.missingFromRegistry.length - 3 }} more</span>,
                but there's no registry row for it there yet —
                possibly worth adding.
              </div>
            </div>

            <!-- Bar chart first: it's the answer to "where does the model
                 think this word fits?" Most users won't need to look at the
                 map at all, so it sits as a follow-up. -->
            <div class="semantic-block">
              <div class="block-header">
                <h4>Closest chapter matches</h4>
                <span class="legend-inline">
                  <span class="legend-mark check">✓</span> registry already has a row here
                  <span class="legend-mark cross">✗</span> not in registry
                </span>
              </div>
              <p class="block-hint">
                Each bar is one (category, chapter). Bar length =
                <em>cosine similarity</em>: how close the language of "{{ token }}"
                is to the average language of past shipments labelled there.
                <strong>1.0 = a perfect linguistic match, 0 = unrelated.</strong>
                Click a bar to inspect that chapter's keywords.
              </p>
              <VChart
                :option="affinityBarChart"
                :height="`${Math.max(220, topAffinities.length * 22 + 40)}px`"
                @click="onAffinityBarClick"
              />
            </div>

            <details class="semantic-block map-block">
              <summary>
                Show on the centroid map
                <span class="reducer-pill" :title="`projection: ${affinity.reducer.toUpperCase()}`">
                  {{ affinity.reducer.toUpperCase() }}
                </span>
              </summary>
              <p class="block-hint">
                The same chapter map as the
                <strong>Centroids</strong> page. Each grey dot is one
                (category, chapter); your word is the
                <span class="kw-marker">◆</span> gold diamond.
                The 5 closest chapters keep their category colour. If the
                diamond lands inside a coloured cluster, that cluster's
                language is what the model thinks "{{ token }}" sounds
                like.
              </p>
              <VChart :option="affinityScatterChart" height="340px" />
            </details>
          </template>
        </div>
      </section>

      <p v-if="!rows.length" class="empty-hint">
        No keyword rows for this token. It may have been removed after the
        page was deep-linked.
      </p>
    </template>
  </section>
</template>

<style scoped>
.page-header { margin-bottom: 16px; }
.page-header h2 {
  margin: 4px 0 4px;
  font-size: 22px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: var(--text);
}
.subtitle { margin: 0; color: var(--text-muted); max-width: 720px; }
.back-btn {
  background: none; border: 0; color: var(--text-muted);
  cursor: pointer; padding: 0; font-size: 12px; font-family: inherit;
}
.back-btn:hover { color: var(--accent); }

.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 14px 16px;
  margin-bottom: 16px;
}
.card h3 { margin: 0 0 8px; font-size: 14px; display: flex; align-items: center; gap: 12px; }
.chart-hint { margin: 0 0 8px; color: var(--text-muted); font-size: 12px; }

.table-wrap { overflow: auto; }
.rows-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.rows-table thead th {
  text-align: left;
  background: var(--surface-2);
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  font-weight: 600;
  color: var(--text-muted);
  white-space: nowrap;
}
.rows-table tbody td {
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
.rows-table tbody tr:last-child td { border-bottom: none; }
.rows-table .num {
  text-align: right;
  font-variant-numeric: tabular-nums;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.rows-table .src { color: var(--text-muted); font-size: 12px; }
.rows-table .notes {
  color: var(--text-muted);
  font-size: 12px;
  max-width: 400px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ch-link {
  color: var(--accent);
  text-decoration: none;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}
.ch-link:hover { text-decoration: underline; }

.collision-card { border-left: 4px solid var(--warn); }
.collision-meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px 24px;
  margin-bottom: 12px;
  font-size: 13px;
}
.collision-meta .meta-label {
  display: block;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  margin-bottom: 2px;
}
.ch-strip { display: flex; flex-wrap: wrap; gap: 4px; }
.ch-chip {
  background: var(--surface-2);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--border);
}

details summary {
  cursor: pointer;
  color: var(--text-muted);
  font-size: 12px;
  margin-bottom: 8px;
}
details summary:hover { color: var(--accent); }

.empty-hint { color: var(--text-muted); }
.status { display: flex; align-items: center; gap: 8px; color: var(--text-muted); }

/* ── Page intro paragraph ───────────────────────────────────────────── */

.page-intro {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  margin-bottom: 14px;
}
.page-intro p {
  margin: 0;
  font-size: 13px;
  color: var(--text);
  line-height: 1.5;
}

/* ── Semantic neighborhood card (collapsible) ───────────────────────── */

.semantic-card {
  padding: 0;
  overflow: hidden;
}
.semantic-toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  background: none;
  border: 0;
  text-align: left;
  padding: 14px 16px;
  cursor: pointer;
  font: inherit;
  color: var(--text);
}
.semantic-toggle:hover { background: var(--surface-2); }
.semantic-toggle .caret {
  color: var(--text-muted);
  font-size: 12px;
  width: 14px;
  flex-shrink: 0;
}
.semantic-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  font-weight: 600;
}
.semantic-badge {
  margin-left: auto;
  background: #fef3c7;
  color: #92400e;
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
}
.semantic-body {
  padding: 0 16px 14px;
  border-top: 1px solid var(--border);
}
.semantic-body > .chart-hint:first-child {
  margin-top: 12px;
}

.semantic-block {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  margin-top: 12px;
}
.semantic-block .block-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 4px;
  flex-wrap: wrap;
}
.semantic-block h4 {
  margin: 0;
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
}
.semantic-block .block-hint {
  margin: 0 0 10px;
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.5;
}
.semantic-block.map-block {
  padding: 0;
}
.semantic-block.map-block summary {
  cursor: pointer;
  padding: 12px 14px;
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  list-style: none;
  display: flex;
  align-items: center;
  gap: 8px;
}
.semantic-block.map-block summary::before {
  content: "▸";
  color: var(--text-muted);
  font-size: 11px;
}
.semantic-block.map-block[open] summary::before { content: "▾"; }
.semantic-block.map-block summary:hover { background: var(--surface); }
.semantic-block.map-block[open] > .block-hint,
.semantic-block.map-block[open] > .v-chart-wrap {
  margin: 0 14px;
}
.semantic-block.map-block[open] > .block-hint {
  margin-top: 4px;
  margin-bottom: 10px;
}
.semantic-block.map-block[open] {
  padding-bottom: 14px;
}

.legend-inline {
  font-size: 11px;
  display: flex;
  gap: 12px;
  align-items: center;
  color: var(--text-muted);
}
.legend-mark {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-weight: 700;
  margin-right: 2px;
}
.legend-mark.check { color: #16a34a; }
.legend-mark.cross { color: #94a3b8; }

.reducer-pill {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 10px;
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 1px 6px;
  border-radius: 999px;
  color: var(--text);
  text-transform: none;
  letter-spacing: 0;
  margin-left: auto;
}
.kw-marker {
  color: #FFB000;
  font-size: 14px;
  text-shadow: 0 0 1px #1a1f2b;
}

.disagree-banner {
  background: var(--accent-soft, #eff6ff);
  border: 1px solid var(--border);
  border-left: 4px solid var(--warn, #f59e0b);
  border-radius: var(--radius-sm);
  padding: 10px 12px;
  margin-bottom: 12px;
  font-size: 13px;
  color: var(--text);
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.disagree-row code {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 0 4px;
  font-size: 12px;
}
.disagree-tag {
  display: inline-block;
  font-size: 10px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-weight: 600;
  padding: 1px 6px;
  border-radius: 3px;
  margin-right: 6px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.disagree-tag.warn { background: #fef3c7; color: #92400e; }
.disagree-tag.info { background: #dbeafe; color: #1e3a8a; }

.banner.info {
  background: var(--accent-soft, #eff6ff);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px 14px;
  font-size: 13px;
}
</style>
