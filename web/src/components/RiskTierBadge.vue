<script setup lang="ts">
/**
 * RiskTierBadge — visual marker for collision risk tiers (low/medium/high).
 * Mirrors SignalClassBadge so the two read consistently in the same row.
 * Colour vocabulary lives in tokens.css under --risk-*.
 */
import type { RiskTier } from "@/types";
defineProps<{ value: RiskTier; compact?: boolean }>();
</script>

<template>
  <span :class="['risk-badge', value, { compact }]">{{ value }}</span>
</template>

<style scoped>
.risk-badge {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border: 1px solid transparent;
}
.risk-badge.compact { padding: 1px 6px; font-size: 10px; }

.risk-badge.low    { background: var(--risk-low-soft);    color: var(--risk-low);    border-color: var(--risk-low); }
.risk-badge.medium { background: var(--risk-medium-soft); color: var(--risk-medium); border-color: var(--risk-medium); }
.risk-badge.high   { background: var(--risk-high-soft);   color: var(--risk-high);   border-color: var(--risk-high); }
</style>
