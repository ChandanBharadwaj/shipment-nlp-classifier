// operator.ts — persists the current operator's name in localStorage so
// every governed write can stamp keyword_audit_log.actor without a login
// system. This is *not* auth (no signature, no server check); it's the
// equivalent of the --actor flag on apply_collision_change.py — a
// trust-based identifier that survives reloads.
//
// App.vue blocks the UI behind a one-time prompt if `name` is empty.
import { defineStore } from "pinia";
import { ref } from "vue";

const LS_KEY = "adminOperator";

export const useOperatorStore = defineStore("operator", () => {
  const name = ref<string>(localStorage.getItem(LS_KEY) ?? "");

  function set(newName: string): void {
    const trimmed = newName.trim();
    if (!trimmed) return;
    name.value = trimmed;
    localStorage.setItem(LS_KEY, trimmed);
  }

  function clear(): void {
    name.value = "";
    localStorage.removeItem(LS_KEY);
  }

  return { name, set, clear };
});
