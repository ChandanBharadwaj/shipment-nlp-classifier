// useRouteQuery.ts — bind a writable computed to a single URL query
// parameter. Lets a view treat a filter the same way it would treat a
// `ref`, while keeping the URL as the source of truth so deep-links
// and browser-back work for free.
//
// Usage:
//   const search = useRouteQuery("search", "");
//   const offset = useRouteQuery("offset", 0, Number);
//
// Writes are flushed via router.replace (no history entry), which is
// what feels right for a filter — a back-button shouldn't undo a single
// keystroke. Use `push: true` to opt into a real history entry.
import { computed, type WritableComputedRef } from "vue";
import { useRoute, useRouter } from "vue-router";

export function useRouteQuery<T extends string | number | boolean>(
  key: string,
  defaultValue: T,
  parse?: (raw: string) => T,
  options: { push?: boolean } = {},
): WritableComputedRef<T> {
  const route  = useRoute();
  const router = useRouter();

  const coerce = (raw: unknown): T => {
    if (raw === undefined || raw === null || raw === "") return defaultValue;
    const s = String(raw);
    if (parse) return parse(s);
    if (typeof defaultValue === "number")  return Number(s) as T;
    if (typeof defaultValue === "boolean") return (s === "true") as T;
    return s as T;
  };

  return computed<T>({
    get: () => coerce(route.query[key]),
    set: (v: T) => {
      const next = { ...route.query };
      const isDefault =
        v === defaultValue ||
        (typeof v === "string" && v === "") ||
        v === undefined ||
        v === null;
      if (isDefault) delete next[key];
      else next[key] = String(v);
      const nav = { path: route.path, query: next };
      if (options.push) router.push(nav);
      else router.replace(nav);
    },
  });
}
