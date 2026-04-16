# Taxonomy notes — HS chapter → category mapping

**Date:** 2026-04-16
**Source:** `data/chapter_to_category.csv` (96 rows)

## Design principles

1. **Every active HS 2-digit chapter (01–97, minus 77 WCO-reserved) assigned exactly once.** The `UNIQUE(hs_chapter)` constraint in `category_hs_chapters` enforces this. No gaps, no overlaps.

2. **Two levels only.** Coarse category (20) → HS 2-digit chapter (96). We explicitly do not add 4-digit or 6-digit children — at ~12k training rows, further splitting starves leaves of data.

3. **Primary vs absorbed.** Each category has one or more "primary" chapters (the semantic core) and zero or more "absorbed" chapters (pragmatic assignments for chapters that don't have their own category). The `is_primary` flag in `category_hs_chapters` distinguishes them.

4. **Multi-label handled at the training level**, not the taxonomy level. A chapter belongs to exactly one category in the taxonomy, but individual training rows (confusables) can be labeled with multiple categories. This teaches the classifier the boundary without breaking the clean tree structure.

## Full mapping

| Category | Primary chapters | Absorbed chapters | Notes |
|---|---|---|---|
| **agriculture** | 06 (plants), 10 (cereals), 12 (oilseeds), 14 (vegetable plaiting), 23 (animal feed) | 01 (live animals), 05 (animal products nec), 13 (gums/resins), 41 (raw hides) | Ch 41 raw hides are an agricultural commodity pre-tanning; tanned leather → luxury/42 |
| **automotive** | 87 (vehicles) | — | Single-chapter category. Bicycles (8712) get multi-label → toys |
| **chemicals** | 28 (inorganic), 29 (organic), 31 (fertilizers), 32 (tanning/dye), 34 (soap/wax), 35 (glues/enzymes), 38 (misc chemical) | 36 (explosives/pyrotechnics) | Ch 36 absorbed as industrial chemicals rather than defense (consumer fireworks dominate trade) |
| **construction** | 25 (salt/stone/cement), 44 (wood), 45 (cork), 68 (stone/cement articles), 69 (ceramics), 70 (glass) | — | Ch 25 shared semantically with minerals (ores vs aggregates); resolved at heading level in training data |
| **cosmetics** | 33 (perfumery/cosmetics) | — | Single-chapter category. Medicated skincare gets multi-label → pharmaceuticals |
| **defense** | 93 (arms/ammunition) | — | Single-chapter. Airsoft/replicas → toys via multi-label confusables |
| **electronics** | 85 (electrical machinery), 37 (photographic/cinema) | — | Ch 37 absorbed: modern "photography" is digital electronics. Ch 90 (instruments) → pharmaceuticals as primary |
| **energy** | 27 (mineral fuels/oils) | — | Single-chapter. Lubricating oils for machinery get multi-label training |
| **food_beverages** | 04 (dairy), 09 (coffee/tea/spices), 11 (milling), 15 (fats/oils), 16 (meat/fish prep), 17 (sugar), 18 (cocoa), 19 (cereal prep), 20 (veg/fruit prep), 21 (misc food), 22 (beverages) | 24 (tobacco) | Ch 24 absorbed — no standalone tobacco category at 20-cat granularity |
| **furniture** | 94 (furniture/lighting) | 46 (straw/basketwork) | Ch 46 absorbed as decorative furnishings |
| **luxury** | 71 (precious metals/stones) | 42 (leather articles), 43 (furskins), 91 (clocks/watches), 97 (art/antiques) | Leather handbags (42), fur coats (43), Swiss watches (91) are luxury goods in trade context |
| **machinery** | 84 (nuclear reactors, boilers, machinery) | 86 (railway), 88 (aircraft), 89 (ships) | Transport vehicles-as-machinery. Ch 87 (road vehicles) stays automotive |
| **metals** | 72 (iron/steel), 73 (iron/steel articles), 74 (copper), 75 (nickel), 76 (aluminium), 78 (lead), 79 (zinc), 80 (tin), 81 (other base metals) | 82 (tools), 83 (misc metal articles) | Full base metals complex. Tools (82) and metal fittings (83) absorbed |
| **minerals** | 26 (ores/slag/ash) | — | Single-chapter. Ch 25 (stone/cement) → construction |
| **paper** | 47 (wood pulp), 48 (paper/paperboard), 49 (printed matter) | — | Clean three-chapter cluster |
| **perishables** | 02 (meat), 03 (fish/seafood), 07 (vegetables), 08 (fruit/nuts) | — | Fresh/frozen cold-chain goods. Canned/preserved versions → food_beverages |
| **pharmaceuticals** | 30 (pharmaceutical products) | 90 (optical/medical instruments) | Ch 90 (surgical instruments, diagnostic devices) is pharma-adjacent in trade |
| **plastics** | 39 (plastics), 40 (rubber) | — | Synthetic polymers cluster |
| **textiles** | 50–63 (silk through other textile articles) | 64 (footwear), 65 (headgear), 66 (umbrellas), 67 (feathers/artificial flowers) | Full textile complex. Footwear (64) absorbed as textile-constructed goods |
| **toys** | 95 (toys/games/sports) | 92 (musical instruments), 96 (miscellaneous manufactured) | Musical instruments and misc manufactured goods absorbed as consumer/recreational |

## Judgment calls worth revisiting

1. **Ch 90 → pharmaceuticals** (not electronics). Medical instruments and optical devices could go either way. Chose pharma because the trade context is healthcare procurement. If consumer optics (cameras, binoculars) dominate traffic, consider moving to electronics.

2. **Ch 41 → agriculture** (not textiles/luxury). Raw hides before tanning are an agricultural commodity. Tanned leather articles (ch 42) → luxury. This split is semantically correct but means the "leather supply chain" spans two categories.

3. **Ch 37 → electronics**. Historical photography was chemical (film), but modern trade in ch 37 is dominated by digital imaging. If chemical film imports matter, consider chemicals.

4. **Ch 24 (tobacco) → food_beverages**. No standalone tobacco category at 20-cat granularity. Could be split out as a 21st category if tobacco shipments are significant in the user's traffic.

5. **Ch 36 (explosives) → chemicals** (not defense). Consumer pyrotechnics (fireworks) dominate ch 36 trade volume. Military explosives are classified under ch 93 (defense) in practice.

## Centroid statistics

- 116 total centroids (20 extras from cross-category multi-label confusables)
- Single-centroid categories: cosmetics (33), defense (93), minerals (26)
- Largest category by centroids: textiles (20 chapters)
- Largest category by training rows: machinery (944), textiles (980)

## Cross-category centroid overlaps (cosine > 0.85)

These are genuine shared-domain pairs, not defects:

| Pair | Cosine | Explanation |
|---|---|---|
| agriculture/13 ↔ chemicals/13 | 0.993 | Gums/resins straddle both |
| luxury/42 ↔ textiles/42 | 0.972 | Leather goods are both luxury and textile-constructed |
| perishables/30 ↔ pharma/30 | 0.954 | Vaccines (cold-chain pharma) are perishable |
| agriculture/24 ↔ food_bev/24 | 0.901 | Tobacco is both agricultural and consumable |
| electronics/90 ↔ pharma/90 | 0.895 | Medical instruments span both domains |
