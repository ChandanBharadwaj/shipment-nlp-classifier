<script setup lang="ts">
/**
 * ComingSoon.vue — placeholder for views landing in Commits 4–6.
 * Keeps the sidebar links navigable so the shell layout can be
 * exercised end-to-end before the real implementations arrive.
 */
import { useRoute } from "vue-router";
const route = useRoute();
</script>

<template>
  <section class="coming-soon">
    <h2>{{ route.meta.title || "Coming soon" }}</h2>
    <p>This view ships in a later commit of the Vue admin UI rollout.</p>
    <ul>
      <li>Overview / Browse / Tokens — Commit 4</li>
      <li>Collisions / Audit Log / Discover — Commit 5</li>
      <li>Inline edits + ChangeModal write flows — Commit 6</li>
    </ul>
    <p>The backend endpoints already work — try them at
      <a href="/docs" target="_blank" rel="noopener"><code>/docs</code></a>.</p>
  </section>
</template>

<style scoped>
.coming-soon {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px 32px;
  max-width: 720px;
  box-shadow: var(--shadow);
}
.coming-soon h2 { margin: 0 0 8px; font-size: 18px; }
.coming-soon p  { margin: 0 0 12px; color: var(--text-muted); }
.coming-soon ul {
  margin: 8px 0 16px 18px;
  color: var(--text-muted);
  font-size: 13px;
}
</style>
