<script setup lang="ts">
/**
 * FilterBar — slot-based row container for view filter inputs. Just
 * enforces consistent layout, padding, and border treatment so every
 * paginated view (Tokens, Labels, AuditLog, Discover) looks identical.
 *
 * Children are expected to be label+input pairs wrapped in `.filter-field`,
 * but the slot is unopinionated about content.
 */
</script>

<template>
  <div class="filter-bar">
    <slot />
  </div>
</template>

<style scoped>
.filter-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 16px;
  align-items: flex-end;
  padding: 12px 16px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  margin-bottom: 16px;
}
:deep(.filter-field) {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
:deep(.filter-field > label) {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  font-weight: 600;
}
:deep(.filter-field input),
:deep(.filter-field select) {
  background: var(--surface);
  color: var(--text);
  border: 1px solid var(--border-2);
  border-radius: var(--radius-sm);
  padding: 6px 10px;
  font-size: 13px;
  min-width: 140px;
  font-family: inherit;
}
:deep(.filter-field input:focus),
:deep(.filter-field select:focus) {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}
:deep(.filter-actions) {
  display: flex;
  gap: 8px;
  margin-left: auto;
}
</style>
