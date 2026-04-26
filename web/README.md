# web/ — Vue 3 admin SPA

Vue 3 + Vite + TypeScript single-page app served by FastAPI at `/ui/`.

The app contains two halves on one shell:

- **Bulk Classify** — drop a CSV, hit `/classify/batch`, see results (1:1 port of the legacy `ml-service/static/app.js`).
- **Admin** — Overview / Browse / Tokens / Collisions / Audit Log / Discover, all backed by `ml-service/admin_routes.py` (`/admin/api/*`). The non-classify views are stubbed in this commit and filled in by Commits 4–6 of the rollout plan.

## Quick start

```bash
# 1. install deps
cd web
npm install

# 2. run the FastAPI backend in another shell (port 8000)
cd ../ml-service && uvicorn main:app --reload

# 3. dev server with HMR (proxies /admin /classify /health to :8000)
npm run dev
# open http://localhost:5173/ui/
```

## Production build

```bash
cd web
npm run build
# → emits ../ml-service/static/dist/
```

FastAPI's `main.py` serves `static/dist/` at `/ui/` when present and falls back to the legacy `static/` directory otherwise. After `npm run build` the SPA replaces the old vanilla-JS tester transparently — no FastAPI restart required during dev (StaticFiles re-reads on every request).

## Layout

```
web/
├── index.html               # Vite shell — single <div id="app"/>
├── vite.config.ts           # outDir → ../ml-service/static/dist; dev proxy
├── public/sample.csv        # served at /ui/sample.csv (download link)
└── src/
    ├── main.ts              # createApp + Pinia + Router
    ├── App.vue              # topbar + sidebar + <RouterView/>
    ├── router.ts            # all routes, history mode
    ├── api.ts               # fetch wrapper, ApiError, actor injection
    ├── stores/              # operator, toast, (catalog later)
    ├── styles/              # tokens.css + base.css (no UI lib)
    ├── components/          # CategoryChip, more later
    └── views/               # BulkClassify, ComingSoon (Overview/etc. follow)
```

## Operator identity

Every governed write to `category_keywords` or `token_collisions` must carry an `actor` so it can be stamped into `keyword_audit_log`. The SPA persists the operator name in `localStorage.adminOperator` and prompts on first load. Click the chip in the topbar to change.

This is **not** authentication — it mirrors the `--actor` flag on `apply_collision_change.py`. The service stays on a trusted network for now.
