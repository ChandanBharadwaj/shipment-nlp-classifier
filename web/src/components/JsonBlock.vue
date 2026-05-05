<script setup lang="ts">
/**
 * JsonBlock — pretty-prints any JSON-serialisable value as a monospace
 * <pre> block. Used by:
 *   - Collisions view, to show the resolution rule grammar
 *   - AuditLog view, to show before/after JSONB diffs
 *
 * Kept dep-free; for the audit diff we render two side-by-side blocks
 * with red/green tints rather than computing a real diff.
 */
import { computed } from "vue";

const props = defineProps<{
  value: unknown;
  tone?: "default" | "added" | "removed";
}>();

const formatted = computed(() => {
  if (props.value === null || props.value === undefined) return "null";
  try {
    return JSON.stringify(props.value, null, 2);
  } catch {
    return String(props.value);
  }
});
</script>

<template>
  <pre :class="['json-block', tone || 'default']">{{ formatted }}</pre>
</template>

<style scoped>
.json-block {
  margin: 0;
  padding: 10px 12px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  line-height: 1.5;
  color: var(--text);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 360px;
  overflow: auto;
}
.json-block.added   { background: rgba(2, 122, 72, 0.06);  border-color: var(--clean); }
.json-block.removed { background: rgba(217, 45, 32, 0.06); border-color: var(--risky); }
</style>
