<script setup lang="ts">
/**
 * StatCard — single big-number tile. Used across the Overview dashboard
 * and any view that needs a "headline metric" treatment. Tone maps to
 * the existing status-colour vocabulary in tokens.css so the same card
 * can be neutral, "good", "warn", or "risky" without bespoke styling.
 */
defineProps<{
  label: string;
  value: string | number;
  hint?: string;
  tone?: "default" | "warn" | "risky" | "clean" | "accent";
}>();
</script>

<template>
  <div :class="['stat-card', tone || 'default']">
    <div class="value">{{ value }}</div>
    <div class="label">{{ label }}</div>
    <div v-if="hint" class="hint">{{ hint }}</div>
  </div>
</template>

<style scoped>
.stat-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left-width: 4px;
  border-left-color: var(--border-2);
  border-radius: var(--radius);
  padding: 14px 16px;
  box-shadow: var(--shadow);
  min-width: 0;
}
.value {
  font-size: 26px;
  font-weight: 600;
  color: var(--text);
  line-height: 1.1;
  margin-bottom: 4px;
}
.label {
  color: var(--text-muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.hint {
  margin-top: 6px;
  color: var(--text-muted);
  font-size: 12px;
}

.stat-card.accent { border-left-color: var(--accent); }
.stat-card.clean  { border-left-color: var(--clean);  }
.stat-card.warn   { border-left-color: var(--warn);   }
.stat-card.risky  { border-left-color: var(--risky);  }
</style>
