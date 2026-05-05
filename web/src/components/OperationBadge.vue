<script setup lang="ts">
/**
 * OperationBadge — insert/update/delete marker for the audit log.
 * Reuses the status-colour vocabulary (clean=insert, accent=update,
 * risky=delete) so the pill reads as "good change / change / removal".
 */
import type { AuditOp } from "@/types";
defineProps<{ value: AuditOp; compact?: boolean }>();
</script>

<template>
  <span :class="['op-badge', value, { compact }]">{{ value }}</span>
</template>

<style scoped>
.op-badge {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border: 1px solid transparent;
}
.op-badge.compact { padding: 1px 6px; font-size: 10px; }

.op-badge.insert { background: var(--clean-soft);  color: var(--clean);  border-color: var(--clean); }
.op-badge.update { background: var(--accent-soft); color: var(--accent); border-color: var(--accent); }
.op-badge.delete { background: var(--risky-soft);  color: var(--risky);  border-color: var(--risky); }
</style>
