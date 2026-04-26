<script setup lang="ts">
/**
 * App.vue — the persistent shell: topbar (brand + operator chip) and
 * sidebar (route links), with a <RouterView> in the main pane.
 *
 * Operator gating — on first load, if no operator name is stored in
 * localStorage, we open a blocking <dialog> that forces the user to
 * supply one. This is the same identifier `apply_collision_change.py
 * --actor` requires; the UI mirrors that contract so writes added in
 * Commit 6 already have an audit-stamping name to send.
 */
import { ref, onMounted, computed } from "vue";
import { useOperatorStore } from "@/stores/operator";
import { useToastStore } from "@/stores/toast";

const op = useOperatorStore();
const toast = useToastStore();

const dialogRef = ref<HTMLDialogElement | null>(null);
const draftName = ref("");
const canSubmit = computed(() => draftName.value.trim().length >= 2);

function openDialog(): void {
  draftName.value = op.name;
  dialogRef.value?.showModal();
}
function submit(): void {
  if (!canSubmit.value) return;
  op.set(draftName.value);
  dialogRef.value?.close();
}

onMounted(() => {
  if (!op.name) openDialog();
});
</script>

<template>
  <div class="app-shell">
    <header class="app-topbar">
      <div class="brand">
        <span class="dot" />
        Shipment Classifier
      </div>
      <div class="ops">
        <a href="/docs" target="_blank" rel="noopener" class="link-btn">API docs</a>
        <span
          class="operator-chip"
          :title="op.name ? 'Click to change operator name' : 'No operator set'"
          @click="openDialog"
        >{{ op.name || "set operator…" }}</span>
      </div>
    </header>

    <aside class="app-sidebar">
      <nav>
        <div class="group">Tools</div>
        <RouterLink to="/classify">Bulk Classify</RouterLink>

        <div class="group">Registry</div>
        <RouterLink to="/admin/overview">Overview</RouterLink>
        <RouterLink to="/admin/browse">Browse</RouterLink>
        <RouterLink to="/admin/tokens">Tokens</RouterLink>
        <RouterLink to="/admin/collisions">Collisions</RouterLink>

        <div class="group">Governance</div>
        <RouterLink to="/admin/audit-log">Audit Log</RouterLink>
        <RouterLink to="/admin/discover">Discover</RouterLink>
      </nav>
    </aside>

    <main class="app-main">
      <RouterView />
    </main>

    <!-- Toast tray; richer styling lands in Commit 7 -->
    <div class="toast-tray" v-if="toast.items.length">
      <div
        v-for="t in toast.items"
        :key="t.id"
        :class="['toast', t.kind]"
        @click="toast.dismiss(t.id)"
      >{{ t.text }}</div>
    </div>

    <dialog ref="dialogRef" class="op-dialog">
      <h2>Operator name</h2>
      <p>Recorded on every change to the keyword and collision registries.
         Stored in this browser's localStorage.</p>
      <input
        v-model="draftName"
        placeholder="e.g. jane.doe"
        @keydown.enter.prevent="submit"
        autofocus
      />
      <div class="row">
        <button class="primary" :disabled="!canSubmit" @click="submit">Save</button>
      </div>
    </dialog>
  </div>
</template>

<style scoped>
.toast-tray {
  position: fixed;
  bottom: 16px;
  right: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 100;
}
.toast {
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-left-width: 4px;
  border-left-color: var(--accent);
  border-radius: var(--radius-sm);
  box-shadow: var(--shadow);
  padding: 10px 14px;
  font-size: 13px;
  max-width: 360px;
  cursor: pointer;
}
.toast.success { border-left-color: var(--clean); }
.toast.warn    { border-left-color: var(--warn);  }
.toast.error   { border-left-color: var(--risky); }
</style>
