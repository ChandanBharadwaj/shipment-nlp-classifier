# web/ — Vue 3 admin SPA

Vue 3 + Vite + TypeScript single-page app served by FastAPI at `/ui/`.
Read-only inspector for the v3 classifier's data layer.

## Pages

- **Bulk Classify** — drop a CSV, hit `/classify/batch`, view results.
- **Overview** — counts of keywords by signal_class / category / chapter, top polysemous tokens.
- **Browse** — categories ↔ chapters tree.
- **Tokens** — paginated keyword list with filters.
- **Labels** — 15K eval labels with predictions side-by-side.
- **Centroids** — UMAP scatter of (category, chapter) centroids.

## Quick start

```bash
# 1. install deps
cd web
npm install

# 2. run the FastAPI backend in another shell (port 8000)
cd ../ml-service && venv/Scripts/uvicorn main:app --port 8000

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

FastAPI's `main.py` serves `static/dist/` at `/ui/` when present. After
`npm run build` the SPA is live; no FastAPI restart needed.

## Layout

```
web/
├── index.html               Vite shell — single <div id="app"/>
├── vite.config.ts           outDir → ../ml-service/static/dist; dev proxy
├── public/sample.csv        served at /ui/sample.csv (download link)
└── src/
    ├── main.ts              createApp + Pinia + Router
    ├── App.vue              topbar + sidebar + <RouterView/>
    ├── router.ts            routes (history mode)
    ├── api.ts               fetch wrapper, ApiError, actor injection
    ├── stores/              operator, toast, catalog
    ├── styles/              tokens.css + base.css (no UI lib)
    ├── charts/palette.ts    36-color v3 category palette
    ├── components/          CategoryChip, DataTable, FilterBar, …
    └── views/               BulkClassify, Overview, Browse, Tokens, TokenDetail, Labels, Centroids
```

## v3 changes from earlier versions

- Removed `Collisions`, `AuditLog`, `Discover` views — those backed
  manual-curation flows that don't exist in the source-driven pipeline.
- Removed write endpoints (no CCTR mutations through the SPA).
- Palette regenerated for 36 LLM-derived categories (was 20 hard-coded).
- `CategoryChip` uses `display_name` from the catalog store, not raw slug.

## Operator identity

The topbar chip stores an operator name in `localStorage.adminOperator`.
This was used by the legacy write endpoints to stamp `keyword_audit_log`
entries; v3 has no writes so the chip is currently decorative.
