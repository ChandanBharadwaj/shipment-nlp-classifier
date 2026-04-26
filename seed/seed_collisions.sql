-- ─────────────────────────────────────────────────────────────────────────────
-- seed_collisions.sql — initial token_collisions registry.
-- ─────────────────────────────────────────────────────────────────────────────
--
-- Each row is one polysemy we have decided to handle deliberately. The
-- inference engine consults this table only for tokens that appear in the
-- request text (set membership), so registry size has no effect on per-
-- request latency for unrelated shipments.
--
-- Resolution rules use a tiny, fixed grammar (see ml-service/collision_resolver.py
-- _match_if):
--     if "hs_code_present"             — caller extracted an HS code
--     if "modifier_hit"                — any modifier fired in the typed scorer
--     if "modifier_hit:tok1|tok2"      — one of the listed modifiers fired
--     if "anchor_hit_in_home"          — an anchor row hit in one of home_chapters
--     if "margin < X"                  — top1.final - top2.final < X
--     if "default"                     — catch-all, always last
--
-- Resolution outcomes are a finite enum the engine knows how to apply:
--     "use_per_chapter_weights"   — no override; let the scorer's top1 stand
--     "use_hs_chapter"            — use the chapter implied by the HS code
--     "use_modifier_chapter"      — use the chapter selected by an active modifier
--     "force_chapter:NN"          — pin to the named chapter
--     "use_anchor_chapter"        — use any anchor's chapter that fired
--     "defer_ambiguous_high_risk" — high-risk: refuse to guess, surface review
--     "low_confidence"            — keep top1 but mark confidence_state=low_confidence
--
-- Risk tiers control what happens when no rule resolves:
--     'low'    → pick top chapter even with thin margin
--     'medium' → pick top; mark `low_confidence` if margin tight
--     'high'   → defer (`ambiguous_high_risk`, requires_review=true)
--
-- Each row's `test_case` references a pytest case in
-- ml-service/tests/test_collisions.py that pins the expected outcome.
-- scripts/audit_collisions.py runs that case to catch drift.
--
-- Governance (CCTR plan §5c): direct INSERT/UPDATE/DELETE on this table is
-- discouraged. Use scripts/apply_collision_change.py so the change is
-- recorded in keyword_audit_log.
--
-- Safe to re-run: ON CONFLICT (token) DO NOTHING.
-- ─────────────────────────────────────────────────────────────────────────────

INSERT INTO token_collisions (token, home_chapters, risk_tier, resolution, owner, test_case, notes)
VALUES
-- ── battery ───────────────────────────────────────────────────────────────
-- Low risk: wrong answer is an accuracy bug, not compliance.
-- Falls through to per-chapter weights almost always; the modifier path
-- catches "toy battery" by routing to the chapter the toys-side modifier
-- endorses.
('battery',
 ARRAY['85','87','95','30'],
 'low',
 '[
    {"if": "hs_code_present",        "then": "use_hs_chapter"},
    {"if": "modifier_hit:toy|plush|nerf|lego|rc",
                                     "then": "use_modifier_chapter"},
    {"if": "default",                "then": "use_per_chapter_weights"}
  ]'::jsonb,
 'cctr-team',
 'test_cctr015_toy_battery_does_not_resolve_to_electronics',
 'Polysemous across electronics/auto/toys/pharma; no compliance stake.'),

-- ── gun ───────────────────────────────────────────────────────────────────
-- High risk: toy gun classified as weapon (or weapon as toy) is a
-- compliance failure. The `toy` modifier wins decisively when active; if
-- neither modifier fires AND the top-2 margin is thin, we defer rather
-- than guess.
('gun',
 ARRAY['93','95'],
 'high',
 '[
    {"if": "hs_code_present",        "then": "use_hs_chapter"},
    {"if": "modifier_hit:toy|plush|nerf|lego|rc|scale model",
                                     "then": "force_chapter:95"},
    {"if": "modifier_hit:military|tactical|military-grade|live-fire|mil-spec",
                                     "then": "force_chapter:93"},
    {"if": "anchor_hit_in_home",     "then": "use_anchor_chapter"},
    {"if": "margin < 0.05",          "then": "defer_ambiguous_high_risk"},
    {"if": "default",                "then": "use_per_chapter_weights"}
  ]'::jsonb,
 'cctr-team',
 'test_cctr004_toy_gun_does_not_resolve_to_defense',
 'Toys/defense overlap; high compliance impact.'),

-- ── tank ──────────────────────────────────────────────────────────────────
-- Same pattern as `gun` but with industrial overlap (water tank, fuel tank).
-- 73 covers iron/steel storage tanks; 87 trains; 84 industrial; 95 toys;
-- 93 military.
('tank',
 ARRAY['93','95','73','87','84'],
 'high',
 '[
    {"if": "hs_code_present",        "then": "use_hs_chapter"},
    {"if": "modifier_hit:toy|plush|nerf|lego|rc|scale model",
                                     "then": "force_chapter:95"},
    {"if": "modifier_hit:military|tactical|military-grade|mil-spec",
                                     "then": "force_chapter:93"},
    {"if": "anchor_hit_in_home",     "then": "use_anchor_chapter"},
    {"if": "margin < 0.05",          "then": "defer_ambiguous_high_risk"},
    {"if": "default",                "then": "use_per_chapter_weights"}
  ]'::jsonb,
 'cctr-team',
 'test_cctr_tank_high_tier_defers_when_ambiguous',
 'Toys/military/industrial overlap; high compliance impact.'),

-- ── helicopter ────────────────────────────────────────────────────────────
-- High risk: ch88 (real aircraft) vs ch95 (toy/RC helicopter) — wrong call
-- changes export-control treatment.
('helicopter',
 ARRAY['88','95'],
 'high',
 '[
    {"if": "hs_code_present",        "then": "use_hs_chapter"},
    {"if": "modifier_hit:toy|plush|nerf|lego|rc|scale model",
                                     "then": "force_chapter:95"},
    {"if": "anchor_hit_in_home",     "then": "use_anchor_chapter"},
    {"if": "margin < 0.05",          "then": "defer_ambiguous_high_risk"},
    {"if": "default",                "then": "use_per_chapter_weights"}
  ]'::jsonb,
 'cctr-team',
 'test_cctr008_toy_helicopter_is_toys_not_machinery',
 'Aircraft/toy overlap; high compliance impact.'),

-- ── aircraft ──────────────────────────────────────────────────────────────
('aircraft',
 ARRAY['88','95'],
 'high',
 '[
    {"if": "hs_code_present",        "then": "use_hs_chapter"},
    {"if": "modifier_hit:toy|plush|nerf|lego|rc|scale model",
                                     "then": "force_chapter:95"},
    {"if": "anchor_hit_in_home",     "then": "use_anchor_chapter"},
    {"if": "margin < 0.05",          "then": "defer_ambiguous_high_risk"},
    {"if": "default",                "then": "use_per_chapter_weights"}
  ]'::jsonb,
 'cctr-team',
 'test_cctr007_turbofan_resolves_to_machinery',
 'Same pattern as helicopter; aircraft is the more frequent token.'),

-- ── vehicle / car ─────────────────────────────────────────────────────────
('vehicle',
 ARRAY['87','95','88','89'],
 'medium',
 '[
    {"if": "hs_code_present",        "then": "use_hs_chapter"},
    {"if": "modifier_hit:toy|plush|nerf|lego|rc|scale model",
                                     "then": "force_chapter:95"},
    {"if": "anchor_hit_in_home",     "then": "use_anchor_chapter"},
    {"if": "margin < 0.05",          "then": "low_confidence"},
    {"if": "default",                "then": "use_per_chapter_weights"}
  ]'::jsonb,
 'cctr-team',
 'test_cctr006_toy_car_is_toys_not_automotive',
 'Auto/toy/aircraft/ship overlap; medium because civilian.'),

('car',
 ARRAY['87','95'],
 'medium',
 '[
    {"if": "hs_code_present",        "then": "use_hs_chapter"},
    {"if": "modifier_hit:toy|plush|nerf|lego|rc|scale model",
                                     "then": "force_chapter:95"},
    {"if": "anchor_hit_in_home",     "then": "use_anchor_chapter"},
    {"if": "margin < 0.05",          "then": "low_confidence"},
    {"if": "default",                "then": "use_per_chapter_weights"}
  ]'::jsonb,
 'cctr-team',
 'test_cctr006_toy_car_is_toys_not_automotive',
 'Common toy-car phrase; defer to modifiers.'),

-- ── chemical products ─────────────────────────────────────────────────────
-- Medium risk: routing impact rather than compliance. Drives the worst
-- regression from the 99-row test ("Chemical products, nos" → 5 categories).
-- HS code is the deterministic prior when available; otherwise fall back to
-- per-chapter weights but mark low_confidence on thin margins.
('chemical products',
 ARRAY['28','29','32','38'],
 'medium',
 '[
    {"if": "hs_code_present",        "then": "use_hs_chapter"},
    {"if": "anchor_hit_in_home",     "then": "use_anchor_chapter"},
    {"if": "margin < 0.05",          "then": "low_confidence"},
    {"if": "default",                "then": "use_per_chapter_weights"}
  ]'::jsonb,
 'cctr-team',
 'test_cctr_chemical_products_nos_low_confidence',
 'Vague descriptor; almost always HS-code dependent.'),

-- ── tea ────────────────────────────────────────────────────────────────────
-- Low risk: the "Tea, bags" → electronics regression. The anchor in ch09
-- already wins decisively in the typed scorer; the registry row exists
-- mainly to pin the regression with a test case so it can never silently
-- come back.
('tea',
 ARRAY['09'],
 'low',
 '[
    {"if": "anchor_hit_in_home",     "then": "use_anchor_chapter"},
    {"if": "default",                "then": "use_per_chapter_weights"}
  ]'::jsonb,
 'cctr-team',
 'test_cctr013_tea_bags_resolves_to_food_beverages',
 'Pinned regression: SID 50 "Tea, bags" used to land in electronics.'),

-- ── bag ────────────────────────────────────────────────────────────────────
-- Low risk; very polysemous. ch42 leather goods, ch63 textile bags, ch48
-- paper bags, ch39 plastic bags, ch73 metal bags. The anchor path wins
-- when one is present.
('bag',
 ARRAY['42','63','48','39','73'],
 'low',
 '[
    {"if": "hs_code_present",        "then": "use_hs_chapter"},
    {"if": "anchor_hit_in_home",     "then": "use_anchor_chapter"},
    {"if": "default",                "then": "use_per_chapter_weights"}
  ]'::jsonb,
 'cctr-team',
 'test_cctr_bag_falls_through_to_per_chapter_weights',
 'Frequent and benign; let per-chapter weights decide.'),

-- ── cell ───────────────────────────────────────────────────────────────────
('cell',
 ARRAY['85','30'],
 'low',
 '[
    {"if": "hs_code_present",        "then": "use_hs_chapter"},
    {"if": "modifier_hit:toy|plush|nerf|lego|rc",
                                     "then": "use_modifier_chapter"},
    {"if": "anchor_hit_in_home",     "then": "use_anchor_chapter"},
    {"if": "default",                "then": "use_per_chapter_weights"}
  ]'::jsonb,
 'cctr-team',
 'test_cctr_cell_falls_through_to_per_chapter_weights',
 'Battery cell vs biological cell; HS code or anchor decides.'),

-- ── core ───────────────────────────────────────────────────────────────────
('core',
 ARRAY['84','85','48','73'],
 'low',
 '[
    {"if": "hs_code_present",        "then": "use_hs_chapter"},
    {"if": "anchor_hit_in_home",     "then": "use_anchor_chapter"},
    {"if": "default",                "then": "use_per_chapter_weights"}
  ]'::jsonb,
 'cctr-team',
 'test_cctr_core_falls_through_to_per_chapter_weights',
 'Engine core, paper core, transformer core. HS or anchor.')

ON CONFLICT (token) DO NOTHING;
