<script setup lang="ts">
/**
 * CategoryChip.vue — coloured pill for a v3 classification category.
 *
 * Color: looked up from `paletteFor(slug)` in @/charts/palette (36 v3
 * categories, stable hex per slug). Falls back to neutral grey when a
 * slug isn't in the map.
 *
 * Label: prefer `display_name` from the catalog store (natural language
 * like "Vehicles, Aircraft & Marine Vessels"); fall back to the slug
 * itself before the catalog has loaded.
 *
 * Foreground: WCAG-AA-ish luminance pick so the same chip works on any
 * palette without hardcoding text colour per entry.
 */
import { computed } from "vue";
import { paletteFor } from "@/charts/palette";
import { useCatalogStore } from "@/stores/catalog";

const props = defineProps<{
  category: string;
  /** Optional final score; surfaced as a tooltip when present. */
  score?: number;
  /** When true, render a smaller "considered: …" runner-up styling. */
  runnerUp?: boolean;
}>();

const catalog = useCatalogStore();

function pickTextColour(hex: string): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16) / 255;
  const g = parseInt(h.slice(2, 4), 16) / 255;
  const b = parseInt(h.slice(4, 6), 16) / 255;
  const lin = (c: number) => (c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
  const L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  return L > 0.5 ? "#1a1f2b" : "#ffffff";
}

const bg = computed(() => paletteFor(props.category));
const fg = computed(() => pickTextColour(bg.value));

/** Resolve display_name from the catalog. Falls back to the slug if the
 * catalog hasn't loaded yet (or the slug isn't in the catalog). */
const displayName = computed(() => {
  const cat = catalog.byName[props.category];
  return cat?.display_name ?? props.category;
});

const tooltip = computed(() => {
  const score = props.score !== undefined
    ? `final_score=${props.score.toFixed(3)}`
    : null;
  // Show the slug in the tooltip too — useful when the display name is
  // long and you want to copy/paste the URL-safe slug.
  const slug = props.category !== displayName.value ? `(${props.category})` : null;
  return [score, slug].filter(Boolean).join(" ") || undefined;
});

const label = computed(() => {
  if (props.runnerUp && props.score !== undefined) {
    return `considered: ${displayName.value} (${props.score.toFixed(2)})`;
  }
  return displayName.value;
});
</script>

<template>
  <span
    :class="['chip', { 'runner-up': runnerUp }]"
    :style="runnerUp ? undefined : { background: bg, color: fg }"
    :title="tooltip"
  >{{ label }}</span>
</template>

<style scoped>
.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
  margin: 2px 4px 2px 0;
  border: 1px solid rgba(0, 0, 0, 0.08);
  white-space: nowrap;
}
.chip.runner-up {
  background: var(--surface-2);
  color: var(--text-muted);
  font-weight: 400;
  font-size: 11px;
  padding: 2px 7px;
}
</style>
