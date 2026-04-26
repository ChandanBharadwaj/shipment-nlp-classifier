// vite.config.ts — build the Vue SPA into ml-service/static/dist/ so FastAPI
// can serve it under /ui/ without any post-build copy step. The dev server
// proxies API calls to the FastAPI backend on :8000 so hot-reload works
// against a real classifier without CORS gymnastics.
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { fileURLToPath, URL } from "node:url";

// Tweak this if the backend lives elsewhere during dev. The proxy only
// matters for `npm run dev` — the prod build is served by FastAPI itself.
const BACKEND = process.env.VITE_BACKEND_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [vue()],

  // The SPA is served at /ui/ in production. Vite must emit asset URLs
  // rooted at /ui/ or the built index.html will 404 trying to fetch
  // /assets/*.js when the page itself lives under /ui/.
  base: "/ui/",

  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },

  build: {
    // Direct output into the FastAPI static tree; main.py prefers
    // static/dist/ when it exists.
    outDir: fileURLToPath(new URL("../ml-service/static/dist", import.meta.url)),
    emptyOutDir: true,
    sourcemap: true,
  },

  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      // All backend surfaces the SPA needs.
      "/admin":   { target: BACKEND, changeOrigin: true },
      "/classify":{ target: BACKEND, changeOrigin: true },
      "/health":  { target: BACKEND, changeOrigin: true },
      "/reload":  { target: BACKEND, changeOrigin: true },
      "/metrics": { target: BACKEND, changeOrigin: true },
      "/docs":    { target: BACKEND, changeOrigin: true },
      "/openapi.json": { target: BACKEND, changeOrigin: true },
    },
  },
});
