<script setup lang="ts">
/**
 * InfoTip.vue — small "?" icon that reveals a plain-English
 * explanation on hover/focus. Designed for non-ML users who land on
 * dashboards full of jargon (signal class, centroid, polysemous, etc.)
 * without forcing them to leave the page for the docs.
 *
 * Two surfaces:
 *   1. The icon itself — keyboard-focusable, aria-described.
 *   2. The popover — `<slot>` for rich content; falls back to `:text`
 *      so callers can pass a one-liner without ceremony.
 *
 * Pure CSS, no JS state — :hover and :focus-within drive visibility.
 * No portal / popper — the parent is responsible for not clipping
 * overflow if the tip lands near a scroll boundary (every place we
 * use it is either a header or a card with overflow: visible).
 */
defineProps<{
  /** One-line explanation; ignored when the default slot is used. */
  text?: string;
  /** Aria label on the trigger; defaults to "More info". */
  label?: string;
  /** "right" (default) | "left" | "below" — popover anchor side. */
  side?: "right" | "left" | "below";
}>();
</script>

<template>
  <span :class="['info-tip', `tip-${side ?? 'right'}`]">
    <button
      type="button"
      class="info-trigger"
      :aria-label="label ?? 'More info'"
      tabindex="0"
    >?</button>
    <span class="info-pop" role="tooltip">
      <slot>{{ text }}</slot>
    </span>
  </span>
</template>

<style scoped>
.info-tip {
  position: relative;
  display: inline-flex;
  align-items: center;
  margin-left: 6px;
  vertical-align: middle;
}
.info-trigger {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  border: 1px solid var(--border-2);
  background: var(--surface-2);
  color: var(--text-muted);
  font-size: 10px;
  font-weight: 600;
  font-family: inherit;
  line-height: 1;
  padding: 0;
  cursor: help;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.info-trigger:hover,
.info-trigger:focus-visible {
  background: var(--accent-soft);
  color: var(--accent);
  border-color: var(--accent);
  outline: none;
}

.info-pop {
  position: absolute;
  z-index: 50;
  background: #1a1f2b;
  color: #f5f7fa;
  font-size: 12px;
  font-weight: 400;
  line-height: 1.45;
  padding: 8px 10px;
  border-radius: var(--radius-sm);
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.18);
  width: max-content;
  max-width: 320px;
  pointer-events: none;
  opacity: 0;
  transform: translateY(-2px);
  transition: opacity 120ms, transform 120ms;
}

/* Side anchors. Default is to the right of the trigger. */
.tip-right .info-pop {
  left: calc(100% + 8px);
  top: 50%;
  transform: translateY(-50%) translateX(-2px);
}
.tip-left .info-pop {
  right: calc(100% + 8px);
  top: 50%;
  transform: translateY(-50%) translateX(2px);
}
.tip-below .info-pop {
  top: calc(100% + 6px);
  left: 50%;
  transform: translateX(-50%) translateY(-2px);
}

.info-tip:hover .info-pop,
.info-tip:focus-within .info-pop {
  opacity: 1;
  pointer-events: auto;
}
.tip-right:hover .info-pop,
.tip-right:focus-within .info-pop {
  transform: translateY(-50%) translateX(0);
}
.tip-left:hover .info-pop,
.tip-left:focus-within .info-pop {
  transform: translateY(-50%) translateX(0);
}
.tip-below:hover .info-pop,
.tip-below:focus-within .info-pop {
  transform: translateX(-50%) translateY(0);
}

/* Use `code` inside slot for inline tokens. */
.info-pop :deep(code) {
  background: rgba(255, 255, 255, 0.1);
  padding: 1px 5px;
  border-radius: 3px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
}
.info-pop :deep(strong) { color: #fff; }
.info-pop :deep(p) { margin: 0 0 6px; }
.info-pop :deep(p:last-child) { margin-bottom: 0; }
</style>
