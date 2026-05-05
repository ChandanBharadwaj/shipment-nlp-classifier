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

// Lazy chunks per view. Each route loads its own component bundle so
// we don't pay for unused views on first paint.
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
    component: () => import("@/views/Overview.vue"),
    meta: { title: "Overview" },
  },
  {
    path: "/admin/browse",
    name: "browse",
    component: () => import("@/views/Browse.vue"),
    meta: { title: "Browse" },
  },
  {
    path: "/admin/tokens",
    name: "tokens",
    component: () => import("@/views/Tokens.vue"),
    meta: { title: "Tokens" },
  },
  {
    path: "/admin/tokens/:token",
    name: "token-detail",
    component: () => import("@/views/TokenDetail.vue"),
    meta: { title: "Token detail" },
  },
  // v3 removed: collisions / audit-log / discover. The source-driven pipeline
  // doesn't have a manual collision registry, governed audit log, or
  // discovery scan to populate.
  {
    path: "/admin/labels",
    name: "labels",
    component: () => import("@/views/Labels.vue"),
    meta: { title: "Labels" },
  },
  {
    path: "/admin/centroids",
    name: "centroids",
    component: () => import("@/views/Centroids.vue"),
    meta: { title: "Centroids" },
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
