// router.ts — single source of truth for SPA routes.
//
// History mode (createWebHistory) means deep links like /ui/admin/tokens/battery
// must be served the SPA shell by FastAPI; main.py installs a catch-all under
// /ui/{path:path} that returns index.html for exactly that reason.
//
// We pass `import.meta.env.BASE_URL` so the router honours the same `/ui/` base
// that vite.config.ts sets — keeps dev (`/`) and prod (`/ui/`) consistent
// without conditional code.
import { createRouter, createWebHistory, type RouteRecordRaw } from "vue-router";

// Lazy chunks per view. Bulk Classify is the only fully-built view in
// Commit 3; the other entries are placeholders that resolve to a shared
// ComingSoon stub until Commits 4–6 land their real implementations.
const ComingSoon = () => import("@/views/ComingSoon.vue");

const routes: RouteRecordRaw[] = [
  { path: "/",          redirect: "/classify" },
  {
    path: "/classify",
    name: "bulk-classify",
    component: () => import("@/views/BulkClassify.vue"),
    meta: { title: "Bulk Classify" },
  },
  {
    path: "/admin/overview",
    name: "overview",
    component: ComingSoon,
    meta: { title: "Overview" },
  },
  {
    path: "/admin/browse",
    name: "browse",
    component: ComingSoon,
    meta: { title: "Browse" },
  },
  {
    path: "/admin/tokens",
    name: "tokens",
    component: ComingSoon,
    meta: { title: "Tokens" },
  },
  {
    path: "/admin/tokens/:token",
    name: "token-detail",
    component: ComingSoon,
    meta: { title: "Token detail" },
  },
  {
    path: "/admin/collisions",
    name: "collisions",
    component: ComingSoon,
    meta: { title: "Collisions" },
  },
  {
    path: "/admin/audit-log",
    name: "audit-log",
    component: ComingSoon,
    meta: { title: "Audit Log" },
  },
  {
    path: "/admin/discover",
    name: "discover",
    component: ComingSoon,
    meta: { title: "Discover" },
  },
  // Catch-all → bulk classify so a stale link can't 404 the SPA.
  { path: "/:pathMatch(.*)*", redirect: "/classify" },
];

export const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
});

router.afterEach((to) => {
  const t = to.meta?.title as string | undefined;
  document.title = t ? `${t} · Shipment Classifier` : "Shipment Classifier";
});
