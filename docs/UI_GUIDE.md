# Admin UI Guide (v3)

A walkthrough of the admin SPA for non-ML engineers inspecting the v3
classifier's data layer. Hover any `?` icon in the UI for inline help.

---

## The big picture

The system classifies a free-text shipment description (e.g. *"5 pallets
of lithium-ion batteries for handheld radios"*) into:
- one of **36 business categories** (LLM-derived, e.g. `electronics_electrical`,
  `pharmaceuticals`, `vehicles_aircraft_marine`)
- one of **96 HS chapters** (e.g. `85 — Electrical machinery`)

It blends two signals:

1. **Keyword evidence** — every word in the input is looked up in the
   `keywords` table (~8.8K rows, all chapter-pinned) and contributes a
   weighted vote toward one or more (category, chapter) buckets.
2. **Semantic evidence** — the input is embedded into a 384-dimensional
   vector and compared against a **centroid** (the average vector of
   public-source descriptions for that chapter).

The SPA is **read-only**. It exists for inspection, debugging, and visual
sanity-checking of the v3 data layer. Mutations happen via the pipeline
scripts, not via the UI.

---

## Pages

### Bulk Classify (`/ui/classify`)
Drop a CSV of shipments, hit `/classify/batch`, see results in a table
with confidence states + per-row category chips. Stateless — reload to
clear.

CSV columns: `shipment_id`, `cargo_description`, `commodity_description`.
Optional: `threshold`, `unclassified_threshold`.

### Overview (`/ui/admin/overview`)
Counts of keywords by signal class + by category + by chapter. The "Top
polysemous tokens" panel surfaces signal-class words that span 2+
chapters — useful for understanding cross-category vocabulary.

### Browse (`/ui/admin/browse`)
Categories on the left, their chapters on the right. Each chapter shows
its keyword count (link to Tokens) and label count (link to Labels).

### Tokens (`/ui/admin/tokens`)
Paginated keyword list. Filters: category, chapter, signal_class, search.
Click a token → TokenDetail with every chapter it lives in plus a
centroid-affinity scatter (where would this token land semantically?).

### Labels (`/ui/admin/labels`)
The 15K labeled shipments used for F1 evaluation. Shows ground truth
(`category_name`) plus the live classifier's `predicted_category` for
side-by-side comparison. Filter by `prediction_status=mispredicted` to
see only the disagreements (the most useful slice).

### Centroids (`/ui/admin/centroids`)
UMAP/PCA scatter of all 96 (category, hs_chapter) centroids. Colored by
category (36-color palette). Type a free-text token in the input to
overlay it as a gold diamond — shows where the embedding model would put
that word in the centroid space.

---

## Signal classes

Every keyword in the registry belongs to one of four classes:

| Class | Typical count | What it does |
|---|---:|---|
| **anchor** | ~2,200 (548 chat + 1,609 TF-IDF promoted) | Strong evidence for a chapter. Trade-vocabulary first-class names like `tea bags` (ch09) or `m16 rifle` (ch93). |
| **signal** | ~6,600 | Soft evidence. TF-IDF over public source text — `sulphuric acid`, `polypropylene`, `wristwatch`. |
| **suppressor** | ~19 | Negative evidence. `toy gun` lives in ch95 (toys) and pushes `arms_ammunition` *down*. |
| **modifier** | ~8 | Re-routing rules. `li-ion` modifies `battery` to route ch85 (electronics). |

### anchor (weight ~1.0)
A word so specific to one chapter that seeing it is essentially proof.
Examples:
- `tea bags` → ch09 (Coffee/Tea/Spices)
- `m16 rifle` → ch93 (Arms)
- `lithium-ion battery` → ch85 (Electronics)

If an anchor matches, the classifier short-circuits hard toward that
chapter. Anchors are rare and high-value.

### signal (weight 0.8–1.5, rank-based)
The bread-and-butter class. TF-IDF tokens distinctive to a chapter's
public text: `sulphuric acid` (ch28), `polypropylene` (ch39), `wristwatch`
(ch91). Each one is suggestive but not conclusive on its own.

### suppressor (weight 0.7, applied negatively)
Words that **subtract** from a category's score. Used to rule things
*out*. Example: when `toy gun` is seen (lives in ch95 — toys), suppress
`arms_ammunition` so the classifier doesn't route it as a real weapon.

### modifier
Words that change *which chapter* another word routes to. Example:
- `battery` alone → ambiguous (ch85, ch87, ch95)
- `li-ion battery` → modifier `li-ion` re-routes to ch85

---

## What the UI does NOT do (changed from earlier versions)

- **No write endpoints.** The CCTR registry is generated, not edited.
  To change keywords, edit `data/llm_generated/cctr_rows.csv` and run
  `python scripts/load_llm_cctr.py`.
- **No collision registry.** Replaced by source-driven CCTR generation.
- **No audit log.** No mutations to audit.
- **No discovery scan.** Polysemous tokens still surface on the Overview
  page, but the manual "register-from-discovery" flow is gone.

---

## Color coding

Each of the 36 categories gets a stable hex color from
`web/src/charts/palette.ts`. Same color across Centroids scatter,
Browse, Tokens, Labels, and Bulk Classify chips. To regenerate (e.g.
after categories change):

```bash
docker exec ... psql ... -c "SELECT slug FROM categories ORDER BY slug" \
  | python -c "..."  # see palette.ts header for the snippet
```
