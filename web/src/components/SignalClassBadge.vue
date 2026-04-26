<script setup lang="ts">
/**
 * SignalClassBadge — visual marker for the four CCTR signal classes.
 *
 *   anchor     green   (strong category evidence)
 *   signal     blue    (soft category evidence)
 *   suppressor red     (counter-evidence)
 *   modifier   purple  (context shifter)
 *
 * Colour vocabulary lives in styles/tokens.css under --sig-* so the
 * palette can be re-tinted without touching components.
 */
import type { SignalClass } from "@/types";
defineProps<{ value: SignalClass; compact?: boolean }>();
</script>

<template>
  <span :class="['sig-badge', value, { compact }]">{{ value }}</span>
</template>

<style scoped>
.sig-badge {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border: 1px solid transparent;
}
.sig-badge.compact { padding: 1px 6px; font-size: 10px; }

.sig-badge.anchor     { background: var(--sig-anchor-soft);     color: var(--sig-anchor);     border-color: var(--sig-anchor); }
.sig-badge.signal     { background: var(--sig-signal-soft);     color: var(--sig-signal);     border-color: var(--sig-signal); }
.sig-badge.suppressor { background: var(--sig-suppressor-soft); color: var(--sig-suppressor); border-color: var(--sig-suppressor); }
.sig-badge.modifier   { background: var(--sig-modifier-soft);   color: var(--sig-modifier);   border-color: var(--sig-modifier); }
</style>
