<script setup lang="ts">
/**
 * CategoryChip.vue — coloured pill for one of the 20 classification
 * categories. Looks up the colour from --cat-{name} in tokens.css and
 * picks a WCAG-AA-ish text colour at runtime so the same chip works on
 * any palette tweak without duplicating the tint as text.
 *
 * The pickTextColour helper is the same one from the legacy app.js;
 * kept identical so the migrated bulk classifier renders pixel-similar.
 */
import { computed } from "vue";

const props = defineProps<{
  category: string;
  /** Optional final score; surfaced as a tooltip when present. */
  score?: number;
  /** When true, render a smaller "considered: …" runner-up styling. */
  runnerUp?: boolean;
}>();

// Mirror of the CSS palette in styles/tokens.css. Inlined so the
// component can compute foreground colour without hitting the DOM.
const PALETTE: Record<string, string> = {
  agriculture:     "#4E79A7",
  automotive:      "#A0CBE8",
  chemicals:       "#F28E2B",
  construction:    "#FFBE7D",
  cosmetics:       "#59A14F",
  defense:         "#8CD17D",
  electronics:     "#B6992D",
  energy:          "#F1CE63",
  food_beverages:  "#499894",
  furniture:       "#86BCB6",
  luxury:          "#E15759",
  machinery:       "#FF9D9A",
  metals:          "#79706E",
  minerals:        "#BAB0AC",
  paper:           "#D37295",
  perishables:     "#FABFD2",
  pharmaceuticals: "#B07AA1",
  plastics:        "#D4A6C8",
  textiles:        "#9D7660",
  toys:            "#D7B5A6",
};
const FALLBACK = "#cbd2d9";

function pickTextColour(hex: string): string {
  // Relative luminance per WCAG. Returns black on light backgrounds,
  // white on dark — same logic as legacy app.js:382-391.
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16) / 255;
  const g = parseInt(h.slice(2, 4), 16) / 255;
  const b = parseInt(h.slice(4, 6), 16) / 255;
  const lin = (c: number) => (c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
  const L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  return L > 0.5 ? "#1a1f2b" : "#ffffff";
}

const bg = computed(() => PALETTE[props.category] ?? FALLBACK);
const fg = computed(() => pickTextColour(bg.value));
const tooltip = computed(() =>
  props.score !== undefined ? `final_score=${props.score.toFixed(3)}` : undefined,
);
const label = computed(() =>
  props.runnerUp && props.score !== undefined
    ? `considered: ${props.category} (${props.score.toFixed(2)})`
    : props.category,
);
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
  border: 1px solid rgba(0, 0, 0, 0.06);
}
.chip.runner-up {
  background: var(--surface-2);
  color: var(--text-muted);
  font-weight: 400;
  font-size: 11px;
  padding: 2px 7px;
}
</style>
