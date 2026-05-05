/**
 * v3 category color palette. Auto-generated from `categories` table —
 * 36 LLM-derived business categories, each assigned a stable color from
 * a fixed 36-palette in alphabetical-slug order.
 *
 * Re-derive with:
 *   docker exec ... psql ... -c "SELECT slug FROM categories ORDER BY slug"
 * then map index → palette[i].
 */

export const CATEGORY_PALETTE: Readonly<Record<string, string>> = {
  animal_feed_food_residues:     "#4E79A7",
  apparel_finished_textiles:     "#F28E2B",
  arms_ammunition:               "#E15759",
  art_antiques_collectibles:     "#76B7B2",
  beverages_spirits_tobacco:     "#59A14F",
  cereals_grains_oilseeds:       "#EDC948",
  clocks_watches_musical:        "#B07AA1",
  coffee_tea_spices_herbs:       "#FF9DA7",
  cosmetics_personal_care:       "#9C755F",
  electronics_electrical:        "#BAB0AC",
  explosives_pyrotechnics:       "#A0CBE8",
  fats_and_oils:                 "#FFBE7D",
  fertilisers:                   "#8CD17D",
  footwear_headgear_accessories: "#86BCB6",
  fresh_produce_vegetables:      "#FF9D9A",
  furniture_lighting:            "#D7B5A6",
  industrial_chemicals:          "#D4A6C8",
  industrial_machinery:          "#FABFD2",
  iron_steel_metal_articles:     "#79706E",
  jewellery_precious_metals:     "#B6992D",
  leather_hides_furs:            "#499894",
  live_animals_animal_products:  "#F1CE63",
  mining_minerals_fuels:         "#D37295",
  misc_manufactured:             "#9D7660",
  non_ferrous_metals:            "#1F77B4",
  optical_medical_precision:     "#FF7F0E",
  paints_dyes_pigments:          "#2CA02C",
  paper_pulp_print:              "#D62728",
  pharmaceuticals:               "#9467BD",
  plastics_rubber:               "#8C564B",
  prepared_foods_confectionery:  "#E377C2",
  stone_cement_glass_ceramics:   "#7F7F7F",
  textile_materials_yarns:       "#BCBD22",
  toys_games_sports:             "#17BECF",
  vehicles_aircraft_marine:      "#AEC7E8",
  wood_cork_basketwork:          "#FFBB78",
};

export const PALETTE_FALLBACK = "#cbd2d9";
export const KEYWORD_COLOR    = "#FFB000"; // gold star — TokenDetail keyword highlight

export function paletteFor(name: string, fallback: string = PALETTE_FALLBACK): string {
  return CATEGORY_PALETTE[name] ?? fallback;
}
