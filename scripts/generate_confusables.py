"""
Generate hand-curated confusable training rows.

These target specific confusion pairs observed in
results/eval_minilm_fixes.md — shipments that belong to category A but use
vocabulary that the embedding model would pull toward category B (or
vice versa). Adding labeled examples of these edges teaches the classifier
where the real boundary sits.

Output:
    seed/seed_confusables.sql   ~250 train-only rows with hs_chapter

Pairs covered (by count of observed confusions):
    plastics ↔ chemicals
    textiles ↔ construction
    electronics ↔ energy ↔ automotive
    agriculture ↔ food_beverages
    minerals ↔ construction
    chemicals ↔ machinery
    pharmaceuticals ↔ cosmetics (real-world overlap; personal care vs OTC)
    luxury ↔ textiles (handbags, apparel)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
OUT_SQL = ROOT / "seed" / "seed_confusables.sql"


# Each entry: (category, hs_chapter, cargo, commodity, extra_categories_for_multilabel)
# The confusable is written as the plain shipment text; the second (target)
# category is only added when the shipment genuinely belongs in both.
ROWS: list[tuple[str, str, str, str, list[str]]] = [

    # ── plastics vs chemicals ────────────────────────────────────────────────
    ("plastics", "39", "hdpe resin granules virgin polymer",                  "polyethylene film-grade pellets; raw plastics material", []),
    ("plastics", "39", "pvc compound flexible wire-grade",                    "polyvinyl chloride resin; extrusion grade", []),
    ("plastics", "39", "polypropylene homopolymer injection grade",           "pp pellets for moulding; thermoplastic resin", []),
    ("plastics", "39", "polystyrene gpps crystal grade pellets",              "expandable polystyrene beads; packaging foam precursor", []),
    ("plastics", "39", "abs resin natural pellets",                           "acrylonitrile butadiene styrene; housing-grade plastic", []),
    ("plastics", "40", "sbr rubber bales synthetic",                          "styrene butadiene rubber; tyre manufacturing precursor", []),
    ("plastics", "40", "silicone rubber compound curable",                    "liquid silicone rubber; moulding grade", []),
    ("chemicals", "28", "sodium hydroxide caustic soda industrial grade",     "naoh flakes 99%; inorganic base chemical", []),
    ("chemicals", "28", "hydrochloric acid technical grade 31%",              "hcl drums industrial reagent", []),
    ("chemicals", "29", "methanol industrial grade bulk",                     "methyl alcohol tanker shipment; organic solvent", []),
    ("chemicals", "29", "ethylene glycol monomer industrial",                 "meg bulk liquid chemical precursor", []),
    ("chemicals", "29", "phthalic anhydride flakes",                          "pa technical grade; plasticiser intermediate", []),
    ("chemicals", "38", "mixed solvent blend industrial cleaner",             "organic solvent mixture; degreasing chemical", []),
    ("chemicals", "32", "titanium dioxide pigment rutile grade",              "tio2 white pigment; paint coating raw material", []),
    ("chemicals", "38", "activated carbon pellets industrial grade",          "charcoal adsorbent; water treatment chemical", []),

    # ── textiles vs construction ─────────────────────────────────────────────
    ("textiles",    "59", "geotextile woven polypropylene rolls",              "non-woven fabric for soil reinforcement; civil engineering textile", ["construction"]),
    ("textiles",    "56", "fibreglass mat insulation rolls",                   "glass fibre reinforced mat; industrial textile", []),
    ("textiles",    "62", "cotton twill workwear trousers",                    "men's cotton work pants; industrial uniform apparel", []),
    ("construction","68", "asbestos-free cement fibreboard panels",            "autoclaved cement-fibre boards; drywall alternative", []),
    ("construction","68", "glass wool insulation batts",                       "mineral wool thermal insulation rolls; building insulation", []),
    ("construction","70", "double-glazed laminated glass panels",              "architectural glazing; tempered window glass", []),
    ("construction","69", "glazed ceramic floor tiles",                        "porcelain tiles 60x60; vitrified flooring", []),
    ("construction","44", "osb engineered wood panels 12mm",                   "oriented strand board sheathing; structural panel", []),
    ("construction","25", "portland cement type i grey 50kg bags",             "opc bagged cement; building material", []),
    ("construction","25", "dolomitic hydrated lime powder",                    "building lime; plaster and mortar material", []),

    # ── electronics vs energy vs automotive ──────────────────────────────────
    ("electronics","85", "lithium-ion battery pack 3.7v 2600mah",              "rechargeable cells for consumer devices; li-ion pack", []),
    ("electronics","85", "lead-acid ups battery 12v 100ah",                    "sealed vrla battery for ups backup; stationary battery", []),
    ("electronics","85", "photovoltaic solar panels 400w monocrystalline",     "pv module silicon cells; rooftop solar array", ["energy"]),
    ("electronics","85", "wind turbine generator components stator rotor",     "permanent magnet generator parts; wind power electrical machinery", ["energy"]),
    ("electronics","85", "electric vehicle traction motor 150kw",              "bldc motor assembly; ev drivetrain electrical component", ["automotive"]),
    ("electronics","85", "ev battery management system bms board",             "li-ion bms pcb with mosfets; ev electronics", ["automotive"]),
    ("energy",     "27", "diesel fuel automotive grade ulsd",                  "ultra-low sulphur diesel bulk; road fuel hydrocarbon", []),
    ("energy",     "27", "lpg cooking gas cylinders",                          "propane butane mix; household fuel", []),
    ("energy",     "27", "lubricating oil industrial gear 220",                "mineral oil base lubricant; machinery lube oil", []),
    ("automotive", "87", "hybrid electric vehicle battery module for toyota",  "nimh traction battery for hev car; automotive oem part", []),
    ("automotive", "87", "oem alternator 12v 90a replacement unit",            "car alternator for sedan; automotive electrical part", []),

    # ── agriculture vs food_beverages ────────────────────────────────────────
    ("agriculture","10", "yellow corn maize #2 grade bulk 25tonne",            "animal-feed grade maize kernels; raw cereal grain", []),
    ("agriculture","10", "durum wheat semolina grade",                         "hard wheat grain for pasta milling; agri commodity", []),
    ("agriculture","12", "soybean seeds for crushing oil extraction",          "gmo-free soy beans bulk; oilseed raw commodity", []),
    ("agriculture","12", "sunflower seeds bulk oilseed",                       "hybrid sunflower seeds; oil extraction raw material", []),
    ("agriculture","23", "soybean meal 48% protein animal feed",               "solvent-extracted soy meal pellets; livestock feed", []),
    ("food_beverages","11", "wheat flour 50kg bags bakery grade",              "milled wheat flour; bread and pastry ingredient", []),
    ("food_beverages","19", "instant noodles cup packs assorted flavours",     "msg-seasoned wheat noodles retail packaging", []),
    ("food_beverages","15", "refined sunflower oil bottled retail",            "deodorised edible oil; cooking oil retail packaging", []),
    ("food_beverages","22", "bottled mineral water sparkling 500ml",           "spring water carbonated pet bottles; beverage retail", []),
    ("food_beverages","17", "refined white sugar 50kg bags",                   "icumsa-45 white sugar; sucrose food grade", []),

    # ── minerals vs construction ─────────────────────────────────────────────
    ("minerals",   "26", "iron ore fines fe-62 bulk vessel",                   "hematite iron ore pellet feed; mining commodity", []),
    ("minerals",   "26", "copper concentrate 28% cu",                          "flotation concentrate ore shipment; mining output", []),
    ("minerals",   "26", "bauxite ore lump grade for alumina",                 "aluminium ore red bauxite; mining raw material", []),
    ("construction","25", "crushed granite aggregate 20mm",                    "road-base aggregate stone; civil construction material", []),
    ("construction","25", "marble blocks quarry raw unprocessed",              "dimension stone blocks for cutting; quarry product", []),
    ("construction","25", "silica sand construction grade",                    "fine construction sand washed; concrete aggregate", []),

    # ── chemicals vs machinery ───────────────────────────────────────────────
    ("chemicals", "38", "industrial coolant concentrate for cnc",              "synthetic metalworking fluid; machining chemical", []),
    ("chemicals", "27", "hydraulic oil iso 46 anti-wear",                      "hydraulic system lubricant; industrial fluid", []),
    ("chemicals", "34", "industrial degreaser alkaline concentrate",           "metal-parts cleaning chemical; solvent preparation", []),
    ("machinery", "84", "hydraulic pump piston axial variable",                "parker-type hydraulic piston pump; industrial fluid-power machinery", []),
    ("machinery", "84", "centrifugal water pump 5hp cast iron",                "end-suction centrifugal pump; industrial water-handling machinery", []),
    ("machinery", "84", "rotary screw air compressor 22kw",                    "industrial air compressor with vfd; pneumatic machinery", []),
    ("machinery", "84", "cnc vertical machining center 3-axis",                "vmc milling machine; metal-cutting machinery", []),
    ("machinery", "84", "industrial centrifuge separator decanter",            "continuous decanter separator; process-industry machinery", []),

    # ── pharmaceuticals vs cosmetics ─────────────────────────────────────────
    ("pharmaceuticals","30", "paracetamol 500mg tablets blister packs",        "acetaminophen oral analgesic retail blister; otc pharma", []),
    ("pharmaceuticals","30", "amoxicillin 500mg capsules antibiotics",         "oral antibiotic prescription; broad-spectrum pharma", []),
    ("pharmaceuticals","30", "insulin glargine injection pens cold chain",     "refrigerated diabetes medication; biologic pharma", ["perishables"]),
    ("cosmetics","33", "anti-aging serum retinol 30ml bottles",                "skincare cosmetic with active ingredient; retail beauty", []),
    ("cosmetics","33", "spf 50 sunscreen lotion 200ml tubes",                  "uv-protection cosmetic; personal-care toiletry", []),
    ("cosmetics","33", "medicated acne gel benzoyl peroxide 2.5%",             "cosmetic acne treatment otc; skincare product", ["pharmaceuticals"]),
    ("cosmetics","33", "hair dye permanent color kit boxes retail",            "oxidative hair colorant; cosmetic kit retail packaging", []),
    ("pharmaceuticals","30", "oral rehydration salts sachets who formula",     "ors packets for dehydration; pharma retail", []),

    # ── luxury vs textiles ──────────────────────────────────────────────────
    ("luxury",   "42", "leather handbags designer brand 200 pcs",              "premium leather tote bags; luxury accessories", ["textiles"]),
    ("luxury",   "42", "crocodile-skin wallets embossed",                      "exotic leather small goods; luxury leather accessory", []),
    ("luxury",   "71", "gold jewelry 22k chains and bangles 500g",             "precious metal jewelry retail; high-value luxury", []),
    ("luxury",   "71", "diamond loose stones gia-certified",                   "natural polished diamond stones; jewelry precursor", []),
    ("luxury",   "91", "swiss mechanical watches automatic retail",            "haute-horlogerie wristwatches; luxury retail timepieces", []),
    ("textiles", "62", "men's wool suits retail mto",                          "tailored business suits wool blend; garment apparel", []),
    ("textiles", "61", "cashmere sweaters knitted luxury-grade",               "100% cashmere knitwear; high-end apparel", ["luxury"]),
    ("textiles", "64", "synthetic sports sneakers retail pairs",               "athletic footwear pu sole; casual shoes retail", []),
    ("textiles", "64", "luxury leather dress shoes italian-made",              "handmade oxford shoes leather sole; premium footwear", ["luxury"]),

    # ── food_beverages vs perishables ───────────────────────────────────────
    ("perishables","03", "frozen shrimp headless shell-on 26/30 block",         "iqf shrimp block; cold-chain seafood", []),
    ("perishables","02", "frozen boneless beef striploin cuts",                 "frozen primal beef cuts; cold-chain meat", []),
    ("perishables","07", "fresh tomatoes for export pallet",                    "greenhouse-grown fresh tomatoes; perishable produce", []),
    ("food_beverages","16", "canned tuna in oil retail cans",                   "shelf-stable canned seafood; ambient retail food", []),
    ("food_beverages","20", "canned sweet corn retail cans",                    "shelf-stable canned vegetable; ambient retail food", []),
    ("food_beverages","04", "uht milk cartons long-life shelf-stable",          "long-life milk tetra packs; ambient dairy retail", []),

    # ── toys vs electronics ─────────────────────────────────────────────────
    ("toys",       "95", "remote control toy cars retail boxed",                "battery-powered rc toys with transmitter; kids toys retail", []),
    ("toys",       "95", "plush stuffed animals assorted 40cm",                 "soft toys for children; retail packaging", []),
    ("electronics","85", "bluetooth wireless earbuds consumer",                 "tws bluetooth earphones with charging case; consumer electronics", []),
    ("electronics","85", "smart watch wearable fitness tracker",                "bt-enabled fitness smartwatch; wearable consumer electronics", []),
    ("toys",       "95", "educational electronic learning tablet kids",         "children's edu tablet with preloaded content; toys retail", ["electronics"]),

    # ── defense vs toys (airsoft/replica) ────────────────────────────────────
    ("defense",    "93", "7.62mm nato ammunition cartridges military",          "small-arms ammunition military-grade cartridges; defense", []),
    ("defense",    "93", "ballistic helmets level iiia kevlar",                 "tactical helmets for security forces; defense body armor", []),
    ("toys",       "95", "airsoft pellet gun replica 6mm",                      "non-lethal bb gun for recreational use; sporting toys", []),

    # ── plastics vs construction ─────────────────────────────────────────────
    ("plastics",    "39", "pvc pipes 4-inch schedule 40",                       "rigid pvc plumbing pipes; building fitting", ["construction"]),
    ("plastics",    "39", "hdpe water tank 5000l",                              "rotational-moulded water storage tank; polymer container", []),
    ("construction","68", "concrete roof tiles terracotta colour",              "precast concrete roofing tiles; building envelope", []),
    ("construction","68", "gypsum plasterboard 12mm panels",                    "drywall board for partitions; building finish", []),
]


SQL_HEADER = """\
-- Generated by scripts/generate_confusables.py. Do not edit by hand.
-- Hand-curated counterfactuals targeting known confusion pairs. All train split.
-- Shipment IDs prefixed 'v2c_' to distinguish from generated HS rows and legacy.

"""


def _esc(s: str) -> str:
    return s.replace("'", "''")


def main() -> int:
    OUT_SQL.parent.mkdir(parents=True, exist_ok=True)

    # Batch into INSERTs of ~100 rows each.
    out_rows: list[tuple[str, str, str, str, str]] = []  # (sid, cat, cargo, com, chapter)
    for i, (cat, chap, cargo, com, extras) in enumerate(ROWS, start=1):
        sid = f"v2c_{i:04d}"
        out_rows.append((sid, cat, cargo, com, chap))
        for extra in extras:
            out_rows.append((sid, extra, cargo, com, chap))

    with OUT_SQL.open("w", encoding="utf-8", newline="\n") as f:
        f.write(SQL_HEADER)
        BATCH = 100
        for i in range(0, len(out_rows), BATCH):
            batch = out_rows[i:i + BATCH]
            f.write(
                "INSERT INTO shipment_labels "
                "(shipment_id, category_name, cargo_text, commodity_text, hs_chapter, split) VALUES\n"
            )
            lines = [
                f"    ('{sid}', '{cat}', '{_esc(cargo)}', '{_esc(com)}', '{chap}', 'train')"
                for sid, cat, cargo, com, chap in batch
            ]
            f.write(",\n".join(lines))
            f.write("\nON CONFLICT (shipment_id, category_name) DO NOTHING;\n\n")

    print(f"Emitted {len(out_rows)} confusable rows ({len(ROWS)} shipments) → {OUT_SQL}")
    # Breakdown
    from collections import Counter
    by_cat = Counter(r[1] for r in out_rows)
    for cat in sorted(by_cat):
        print(f"  {cat:<18} {by_cat[cat]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
