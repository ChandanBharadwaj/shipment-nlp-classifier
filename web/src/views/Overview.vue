<script setup lang="ts">
/**
 * Overview.vue — landing dashboard for the registry. Single GET to
 * /admin/api/overview returns four datasets in one round-trip; we render
 * them as four StatCards (totals + last_audit + class breakdown), two
 * bar charts (keywords by category, by chapter), and a top-10
 * polysemous-token table.
 *
 * Bars and table rows deep-link to /admin/tokens with pre-populated
 * filters so this page acts as the "where do I want to go next" map.
 */
import { ref, computed, onMounted } from "vue";
import { useRouter } from "vue-router";
import api from "@/api";
import type { Overview, ClassCount } from "@/types";
import StatCard from "@/components/StatCard.vue";
import VChart from "@/components/VChart.vue";
import DataTable from "@/components/DataTable.vue";
import SignalClassBadge from "@/components/SignalClassBadge.vue";
import InfoTip from "@/components/InfoTip.vue";

const CLASS_HINTS: Record<string, string> = {
  anchor:     "Strong, single-word give-aways. One match is enough to classify (e.g. tank → defense).",
  signal:     "Soft evidence. Adds up across multiple matches; rarely decisive alone.",
  suppressor: "Veto words. Their presence pulls a category's score down — useful for ruling things OUT.",
  modifier:   "Re-routers. They redirect another keyword to a specific chapter (e.g. 'li-ion' sends 'battery' to ch 85).",
};

const router = useRouter();
const data    = ref<Overview | null>(null);
const loading = ref(false);
const error   = ref<string | null>(null);

onMounted(async () => {
  loading.value = true;
  try { data.value = await api.overview(); }
  catch (err) { error.value = err instanceof Error ? err.message : String(err); }
  finally { loading.value = false; }
});

const lastAuditDisplay = computed(() => {
  const ts = data.value?.last_audit_ts;
  if (!ts) return "—";
  try { return new Date(ts).toLocaleString(); } catch { return ts; }
});

const totalKeywords   = computed(() => data.value?.total_keywords   ?? 0);
const totalCollisions = computed(() => data.value?.total_collisions ?? 0);

const classBreakdown = computed<ClassCount[]>(() => data.value?.counts.by_class ?? []);

const categoryChart = computed(() => {
  const rows = data.value?.counts.by_category ?? [];
  return {
    grid: { left: 110, right: 24, top: 12, bottom: 28 },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    xAxis: { type: "value", axisLabel: { fontSize: 11 } },
    yAxis: {
      type: "category",
      data: rows.map(r => r.category),
      axisLabel: { fontSize: 11, color: "#5a6478" },
      inverse: true,
    },
    series: [{
      name: "Keywords",
      type: "bar",
      data: rows.map(r => r.n),
      itemStyle: { color: "#2563eb", borderRadius: [0, 4, 4, 0] },
      label: { show: true, position: "right", fontSize: 11 },
    }],
  };
});

const chapterChart = computed(() => {
  const rows = data.value?.counts.by_chapter ?? [];
  return {
    grid: { left: 36, right: 16, top: 12, bottom: 60 },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    xAxis: {
      type: "category",
      data: rows.map(r => r.hs_chapter),
      axisLabel: { fontSize: 10, rotate: 60, color: "#5a6478" },
    },
    yAxis: { type: "value", axisLabel: { fontSize: 11 } },
    series: [{
      name: "Keywords",
      type: "bar",
      data: rows.map(r => r.n),
      itemStyle: { color: "#499894", borderRadius: [4, 4, 0, 0] },
    }],
  };
});

const polyColumns = [
  { key: "keyword",   label: "Token", width: "30%" },
  { key: "n_chapters", label: "# chapters", align: "right" as const, width: "12%" },
  { key: "chapters",  label: "Chapters" },
  { key: "actions",   label: "", width: "80px", align: "right" as const },
];

function gotoCategory(cat: string): void {
  router.push({ path: "/admin/tokens", query: { category: cat } });
}
function gotoChapter(ch: string): void {
  router.push({ path: "/admin/tokens", query: { chapter: ch } });
}
function gotoToken(token: string): void {
  router.push({ path: `/admin/tokens/${encodeURIComponent(token)}` });
}

function onCategoryBarClick(params: { name?: string }): void {
  if (params?.name) gotoCategory(params.name);
}
function onChapterBarClick(params: { name?: string }): void {
  if (params?.name) gotoChapter(params.name);
}
</script>

<template>
  <section class="page">
    <header class="page-header">
      <h2>
        Overview
        <InfoTip side="below">
          <p>This dashboard shows the <strong>shape of the registry</strong> — how
          many keywords we have, how they split across categories and HS chapters,
          and which words live in more than one chapter (and so risk being
          misclassified).</p>
          <p>Every bar and token is a deep-link: click to jump straight into the
          underlying rows in the Tokens browser.</p>
        </InfoTip>
      </h2>
      <p class="subtitle">
        A snapshot of the keyword and collision registries. Click any bar or
        polysemous token to drill into the underlying rows.
      </p>
    </header>

    <div v-if="error" class="banner error">{{ error }}</div>

    <div class="stat-grid">
      <StatCard
        label="Total keywords"
        :value="totalKeywords.toLocaleString()"
        hint="across all categories and chapters"
        tone="accent"
      />
      <StatCard
        label="Collisions registered"
        :value="totalCollisions.toLocaleString()"
        hint="polysemous tokens with explicit resolution rules"
        tone="warn"
      />
      <StatCard
        label="Last audit entry"
        :value="lastAuditDisplay"
        hint="most recent registry change"
      />
      <StatCard
        label="Signal class mix"
        :value="classBreakdown.length ? classBreakdown.map(c => c.n).reduce((a,b)=>a+b, 0).toLocaleString() : '—'"
        hint="anchor + signal + suppressor + modifier"
        tone="clean"
      >
      </StatCard>
    </div>

    <section class="class-row">
      <div class="class-row-title">
        Signal class breakdown
        <InfoTip side="below">
          <p>Every keyword belongs to one of four <strong>signal classes</strong>
          that tell the classifier <em>how</em> to use it:</p>
          <p><strong>anchor</strong> — strong, single-word give-aways
          (e.g. <code>tank</code> → defense). One match is enough.</p>
          <p><strong>signal</strong> — soft evidence that adds up across
          matches (e.g. <code>steel</code>, <code>cotton</code>). Rarely
          decisive alone.</p>
          <p><strong>suppressor</strong> — veto words that pull a category's
          score <em>down</em> when present (used to rule things OUT).</p>
          <p><strong>modifier</strong> — re-routers that send another
          keyword to a specific chapter (e.g. <code>li-ion</code> sends
          <code>battery</code> to chapter 85).</p>
        </InfoTip>
      </div>
      <div class="class-row-items">
        <div v-for="c in classBreakdown" :key="c.signal_class" class="class-row-item">
          <SignalClassBadge :value="c.signal_class" />
          <span class="class-row-count">{{ c.n.toLocaleString() }}</span>
          <InfoTip :text="CLASS_HINTS[c.signal_class] ?? ''" side="below" />
        </div>
      </div>
    </section>

    <section class="chart-grid">
      <article class="chart-card">
        <h3>Keywords by category</h3>
        <p class="chart-hint">Click a bar to filter the Tokens browser.</p>
        <VChart
          :option="categoryChart"
          :loading="loading"
          height="420px"
          @click="onCategoryBarClick"
        />
      </article>

      <article class="chart-card">
        <h3>Keywords by HS chapter</h3>
        <p class="chart-hint">Click a bar to filter by chapter.</p>
        <VChart
          :option="chapterChart"
          :loading="loading"
          height="420px"
          @click="onChapterBarClick"
        />
      </article>
    </section>

    <section class="poly-section">
      <h3>
        Top polysemous tokens
        <InfoTip side="below">
          <p><strong>Polysemous</strong> = a single word that means different
          things in different contexts. <code>battery</code> can be a phone
          battery (ch 85), a car battery (ch 85), or a torch battery (ch 85)
          — but <code>tank</code> can be a fuel tank, a fish tank, or an
          armored vehicle, each in a different chapter.</p>
          <p>These are the words that appear in the <em>most</em> HS chapters
          across the registry — they're the prime candidates for the
          <strong>collision registry</strong>, where we add explicit "if X
          also appears, route to chapter Y" rules.</p>
        </InfoTip>
      </h3>
      <p class="chart-hint">
        Signal-class tokens appearing in two or more HS chapters — these
        are candidates for the collision registry.
      </p>
      <DataTable
        :columns="polyColumns"
        :rows="data?.top_polysemous ?? []"
        :loading="loading"
        empty-message="No multi-chapter tokens found."
      >
        <template #cell-keyword="{ row }">
          <a class="token-link" @click.prevent="gotoToken(String(row.keyword))" href="#">
            {{ row.keyword }}
          </a>
        </template>
        <template #cell-chapters="{ row }">
          <span
            v-for="ch in (row.chapters as string[])"
            :key="ch"
            class="chip-chapter"
            @click="gotoChapter(ch)"
          >{{ ch }}</span>
        </template>
        <template #cell-actions="{ row }">
          <a class="token-link" @click.prevent="gotoToken(String(row.keyword))" href="#">view →</a>
        </template>
      </DataTable>
    </section>
  </section>
</template>

<style scoped>
.page-header { margin-bottom: 20px; }
.page-header h2 { margin: 0 0 4px; font-size: 20px; }
.subtitle { margin: 0; color: var(--text-muted); max-width: 720px; }

.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}

.class-row {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 12px 16px;
  margin-bottom: 18px;
}
.class-row-title {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.class-row-items {
  display: flex;
  flex-wrap: wrap;
  gap: 18px;
  align-items: center;
}
.class-row-item {
  display: flex;
  align-items: center;
  gap: 8px;
}
.class-row-count {
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.chart-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 18px;
}
@media (max-width: 1100px) {
  .chart-grid { grid-template-columns: 1fr; }
}
.chart-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 14px 16px;
}
.chart-card h3 { margin: 0 0 4px; font-size: 14px; }
.chart-hint { margin: 0 0 10px; color: var(--text-muted); font-size: 12px; }

.poly-section h3 { margin: 0 0 4px; font-size: 14px; }
.poly-section .chart-hint { margin-bottom: 12px; }

.token-link {
  color: var(--accent);
  cursor: pointer;
  text-decoration: none;
  font-weight: 500;
}
.token-link:hover { text-decoration: underline; }

.chip-chapter {
  display: inline-block;
  background: var(--surface-2);
  color: var(--text);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 999px;
  margin: 2px 4px 2px 0;
  border: 1px solid var(--border);
  cursor: pointer;
}
.chip-chapter:hover {
  border-color: var(--accent);
  color: var(--accent);
}
</style>
