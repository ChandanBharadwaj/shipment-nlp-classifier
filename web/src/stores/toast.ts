// toast.ts — minimal toast queue. Components push messages; <App.vue>
// renders the queue (Commit 7 polishes the UI). Kept here so write
// flows added in Commit 6 can call into a stable API now.
import { defineStore } from "pinia";
import { ref } from "vue";

export type ToastKind = "info" | "success" | "warn" | "error";
export interface Toast {
  id: number;
  kind: ToastKind;
  text: string;
}

let nextId = 1;

export const useToastStore = defineStore("toast", () => {
  const items = ref<Toast[]>([]);

  function push(text: string, kind: ToastKind = "info", ttlMs = 4000): void {
    const id = nextId++;
    items.value.push({ id, kind, text });
    if (ttlMs > 0) {
      setTimeout(() => dismiss(id), ttlMs);
    }
  }

  function dismiss(id: number): void {
    items.value = items.value.filter((t) => t.id !== id);
  }

  return { items, push, dismiss };
});
