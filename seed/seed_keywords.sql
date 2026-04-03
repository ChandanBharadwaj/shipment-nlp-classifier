-- Seed keywords per category.
-- Uses sub-selects on category name to avoid hardcoded IDs.
-- Weights: 1.0 = standard, 1.5 = highly distinctive, 0.8 = common but relevant.
-- Safe to re-run (ON CONFLICT DO NOTHING).

-- ── electronics ──────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('PCB', 1.5),
    ('semiconductor', 1.5),
    ('circuit board', 1.5),
    ('microchip', 1.5),
    ('transistor', 1.2),
    ('LED', 1.0),
    ('HDMI', 1.0),
    ('lithium battery', 1.2),
    ('capacitor', 1.0),
    ('diode', 1.0),
    ('resistor', 1.0),
    ('motherboard', 1.2),
    ('GPU', 1.5),
    ('CPU', 1.5),
    ('power supply unit', 1.0),
    ('heat sink', 0.8),
    ('soldering', 0.8),
    ('fiber optic', 1.0),
    ('USB cable', 0.8),
    ('antenna', 0.8)
) AS t(kw, wt)
WHERE name = 'electronics'
ON CONFLICT DO NOTHING;

-- ── perishables ───────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('frozen', 1.5),
    ('cold chain', 1.5),
    ('refrigerated', 1.5),
    ('chilled', 1.2),
    ('perishable', 1.5),
    ('fresh produce', 1.2),
    ('seafood', 1.2),
    ('prawns', 1.0),
    ('temperature controlled', 1.2),
    ('reefer', 1.5),
    ('dairy', 1.0),
    ('meat', 1.0),
    ('poultry', 1.0),
    ('live animals', 1.5),
    ('flowers', 0.8),
    ('mushrooms', 0.8),
    ('cold storage', 1.2),
    ('ice pack', 1.0),
    ('fresh fish', 1.2),
    ('lettuce', 0.8)
) AS t(kw, wt)
WHERE name = 'perishables'
ON CONFLICT DO NOTHING;

-- ── chemicals ─────────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('hazmat', 1.5),
    ('flammable', 1.5),
    ('corrosive', 1.5),
    ('toxic', 1.5),
    ('MSDS', 1.5),
    ('solvent', 1.2),
    ('acid', 1.2),
    ('chemical', 1.0),
    ('reagent', 1.2),
    ('industrial chemical', 1.2),
    ('bleach', 1.0),
    ('ammonia', 1.2),
    ('oxidizer', 1.5),
    ('explosive', 1.5),
    ('UN number', 1.5),
    ('DG cargo', 1.5),
    ('compressed gas', 1.2),
    ('caustic', 1.2),
    ('methanol', 1.2),
    ('ethanol', 1.0)
) AS t(kw, wt)
WHERE name = 'chemicals'
ON CONFLICT DO NOTHING;

-- ── textiles ──────────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('fabric', 1.2),
    ('garment', 1.2),
    ('yarn', 1.2),
    ('cotton', 1.0),
    ('polyester', 1.0),
    ('woven', 1.2),
    ('knitted', 1.2),
    ('denim', 1.0),
    ('apparel', 1.2),
    ('footwear', 1.2),
    ('silk', 1.0),
    ('linen', 1.0),
    ('wool', 1.0),
    ('synthetic fiber', 1.2),
    ('spandex', 1.0),
    ('nylon fabric', 1.0),
    ('embroidery', 0.8),
    ('sportswear', 1.0),
    ('uniforms', 0.8),
    ('textile roll', 1.2)
) AS t(kw, wt)
WHERE name = 'textiles'
ON CONFLICT DO NOTHING;

-- ── machinery ─────────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('turbine', 1.5),
    ('compressor', 1.5),
    ('conveyor', 1.2),
    ('hydraulic', 1.2),
    ('pump', 1.0),
    ('generator', 1.2),
    ('crane', 1.2),
    ('lathe', 1.5),
    ('drill press', 1.2),
    ('CNC machine', 1.5),
    ('industrial robot', 1.5),
    ('excavator', 1.5),
    ('forklift', 1.2),
    ('boiler', 1.2),
    ('heat exchanger', 1.2),
    ('gearbox', 1.0),
    ('motor', 0.8),
    ('valve', 0.8),
    ('industrial equipment', 1.0),
    ('heavy machinery', 1.2)
) AS t(kw, wt)
WHERE name = 'machinery'
ON CONFLICT DO NOTHING;

-- ── pharmaceuticals ───────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('API', 1.5),
    ('active pharmaceutical ingredient', 1.5),
    ('vaccine', 1.5),
    ('capsule', 1.2),
    ('tablet', 1.0),
    ('sterile', 1.2),
    ('medical device', 1.2),
    ('syringe', 1.2),
    ('GMP', 1.5),
    ('pharma', 1.2),
    ('drug', 1.0),
    ('excipient', 1.5),
    ('vial', 1.2),
    ('ampoule', 1.2),
    ('clinical trial', 1.5),
    ('biologics', 1.5),
    ('diagnostic kit', 1.2),
    ('implant', 1.2),
    ('surgical instrument', 1.2),
    ('controlled substance', 1.5)
) AS t(kw, wt)
WHERE name = 'pharmaceuticals'
ON CONFLICT DO NOTHING;

-- ── automotive ────────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('spare part', 1.2),
    ('tyre', 1.2),
    ('engine', 1.2),
    ('gearbox', 1.2),
    ('chassis', 1.5),
    ('OEM', 1.2),
    ('vehicle', 1.0),
    ('brake pad', 1.2),
    ('suspension', 1.2),
    ('exhaust', 1.0),
    ('radiator', 1.2),
    ('catalytic converter', 1.5),
    ('transmission', 1.2),
    ('automotive', 1.2),
    ('car parts', 1.0),
    ('windshield', 1.0),
    ('alternator', 1.2),
    ('shock absorber', 1.2),
    ('fuel injector', 1.5),
    ('EV battery pack', 1.5)
) AS t(kw, wt)
WHERE name = 'automotive'
ON CONFLICT DO NOTHING;

-- ── food_beverages ────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('beverage', 1.2),
    ('canned food', 1.2),
    ('packaged food', 1.2),
    ('condiment', 1.0),
    ('snack', 1.0),
    ('bottled water', 1.0),
    ('juice', 1.0),
    ('cereal', 1.0),
    ('confectionery', 1.2),
    ('chocolate', 1.0),
    ('sugar', 0.8),
    ('flour', 0.8),
    ('cooking oil', 1.0),
    ('instant noodles', 1.2),
    ('sauce', 0.8),
    ('spice', 0.8),
    ('processed food', 1.0),
    ('energy drink', 1.2),
    ('wine', 1.0),
    ('spirits', 1.2)
) AS t(kw, wt)
WHERE name = 'food_beverages'
ON CONFLICT DO NOTHING;

-- ── furniture ─────────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('sofa', 1.2),
    ('mattress', 1.2),
    ('wardrobe', 1.2),
    ('cabinet', 1.0),
    ('desk', 1.0),
    ('chair', 1.0),
    ('bed frame', 1.2),
    ('upholstery', 1.2),
    ('dining table', 1.2),
    ('bookshelf', 1.0),
    ('recliner', 1.2),
    ('office furniture', 1.2),
    ('cushion', 0.8),
    ('lamp', 0.8),
    ('curtain', 0.8),
    ('rug', 0.8),
    ('flooring', 1.0),
    ('kitchen cabinet', 1.2),
    ('sectional sofa', 1.2),
    ('home decor', 1.0)
) AS t(kw, wt)
WHERE name = 'furniture'
ON CONFLICT DO NOTHING;

-- ── plastics ──────────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('HDPE', 1.5),
    ('PVC', 1.5),
    ('polypropylene', 1.5),
    ('resin', 1.2),
    ('polymer', 1.2),
    ('rubber', 1.0),
    ('elastomer', 1.2),
    ('nylon', 1.0),
    ('polyethylene', 1.5),
    ('ABS plastic', 1.5),
    ('polystyrene', 1.2),
    ('plastic granules', 1.2),
    ('synthetic rubber', 1.2),
    ('foam', 0.8),
    ('plastic pipe', 1.0),
    ('PET', 1.2),
    ('acrylic', 1.0),
    ('silicone', 1.2),
    ('thermoplastic', 1.2),
    ('injection molding', 1.5)
) AS t(kw, wt)
WHERE name = 'plastics'
ON CONFLICT DO NOTHING;

-- ── metals ────────────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('steel', 1.2),
    ('aluminium', 1.2),
    ('copper', 1.2),
    ('iron ore', 1.5),
    ('alloy', 1.2),
    ('pipe', 0.8),
    ('bar', 0.8),
    ('sheet metal', 1.2),
    ('coil', 1.0),
    ('stainless steel', 1.5),
    ('galvanized', 1.2),
    ('titanium', 1.5),
    ('zinc', 1.0),
    ('nickel', 1.2),
    ('brass', 1.0),
    ('wire rod', 1.2),
    ('structural steel', 1.2),
    ('metal scrap', 1.0),
    ('cast iron', 1.2),
    ('ingot', 1.2)
) AS t(kw, wt)
WHERE name = 'metals'
ON CONFLICT DO NOTHING;

-- ── paper ─────────────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('paper roll', 1.5),
    ('cardboard', 1.2),
    ('corrugated', 1.2),
    ('printing paper', 1.2),
    ('tissue', 1.0),
    ('newsprint', 1.2),
    ('kraft paper', 1.5),
    ('paperboard', 1.2),
    ('offset paper', 1.2),
    ('coated paper', 1.2),
    ('packaging paper', 1.0),
    ('magazine paper', 1.0),
    ('wood pulp', 1.5),
    ('recycled paper', 1.0),
    ('stationery', 0.8),
    ('envelope', 0.8),
    ('label stock', 1.0),
    ('laminated paper', 1.2),
    ('wrapping paper', 1.0),
    ('paper bag', 1.0)
) AS t(kw, wt)
WHERE name = 'paper'
ON CONFLICT DO NOTHING;

-- ── minerals ──────────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('ore', 1.5),
    ('limestone', 1.5),
    ('granite', 1.2),
    ('aggregate', 1.2),
    ('sand', 1.0),
    ('gravel', 1.0),
    ('coal', 1.5),
    ('bauxite', 1.5),
    ('silica', 1.2),
    ('marble', 1.2),
    ('quartz', 1.2),
    ('feldspar', 1.2),
    ('talc', 1.2),
    ('gypsum', 1.2),
    ('phosphate', 1.2),
    ('mineral ore', 1.5),
    ('rock salt', 1.2),
    ('potash', 1.2),
    ('chromite', 1.5),
    ('manganese ore', 1.5)
) AS t(kw, wt)
WHERE name = 'minerals'
ON CONFLICT DO NOTHING;

-- ── energy ────────────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('crude oil', 1.5),
    ('petroleum', 1.5),
    ('LNG', 1.5),
    ('LPG', 1.5),
    ('diesel', 1.2),
    ('lubricant', 1.2),
    ('fuel', 1.0),
    ('kerosene', 1.2),
    ('natural gas', 1.5),
    ('bunker fuel', 1.5),
    ('bitumen', 1.5),
    ('coal energy', 1.2),
    ('jet fuel', 1.5),
    ('petrol', 1.2),
    ('transformer oil', 1.2),
    ('hydraulic oil', 1.0),
    ('naphtha', 1.5),
    ('refinery', 1.2),
    ('oil tanker', 1.5),
    ('petroleum products', 1.5)
) AS t(kw, wt)
WHERE name = 'energy'
ON CONFLICT DO NOTHING;

-- ── agriculture ───────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('grain', 1.2),
    ('wheat', 1.2),
    ('corn', 1.2),
    ('fertilizer', 1.5),
    ('pesticide', 1.5),
    ('seeds', 1.0),
    ('livestock', 1.5),
    ('cotton', 1.0),
    ('soybean', 1.5),
    ('rice', 1.0),
    ('barley', 1.2),
    ('palm oil', 1.2),
    ('agricultural produce', 1.2),
    ('crop', 1.0),
    ('herbicide', 1.5),
    ('animal feed', 1.2),
    ('horticulture', 1.2),
    ('cocoa beans', 1.2),
    ('coffee beans', 1.2),
    ('tobacco', 1.2)
) AS t(kw, wt)
WHERE name = 'agriculture'
ON CONFLICT DO NOTHING;

-- ── cosmetics ─────────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('skincare', 1.2),
    ('perfume', 1.2),
    ('lotion', 1.0),
    ('shampoo', 1.0),
    ('makeup', 1.2),
    ('cosmetic', 1.2),
    ('fragrance', 1.2),
    ('sunscreen', 1.2),
    ('lipstick', 1.2),
    ('foundation', 1.0),
    ('moisturizer', 1.2),
    ('hair care', 1.0),
    ('nail polish', 1.2),
    ('body wash', 1.0),
    ('deodorant', 1.0),
    ('serum', 1.2),
    ('essential oil', 1.0),
    ('face cream', 1.2),
    ('hair dye', 1.2),
    ('beauty product', 1.0)
) AS t(kw, wt)
WHERE name = 'cosmetics'
ON CONFLICT DO NOTHING;

-- ── toys ──────────────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('toy', 1.2),
    ('doll', 1.2),
    ('board game', 1.2),
    ('sporting goods', 1.2),
    ('bicycle', 1.0),
    ('playground equipment', 1.2),
    ('LEGO', 1.5),
    ('action figure', 1.2),
    ('puzzle', 1.0),
    ('stuffed animal', 1.0),
    ('remote control car', 1.2),
    ('video game', 1.0),
    ('sports equipment', 1.0),
    ('scooter', 1.0),
    ('skateboard', 1.0),
    ('fishing gear', 1.0),
    ('camping equipment', 1.0),
    ('gym equipment', 1.0),
    ('children toy', 1.2),
    ('game console', 1.5)
) AS t(kw, wt)
WHERE name = 'toys'
ON CONFLICT DO NOTHING;

-- ── construction ──────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('cement', 1.5),
    ('timber', 1.2),
    ('rebar', 1.5),
    ('brick', 1.2),
    ('glass', 1.0),
    ('roofing', 1.2),
    ('insulation', 1.2),
    ('concrete', 1.5),
    ('plywood', 1.2),
    ('structural steel', 1.2),
    ('building material', 1.2),
    ('tiles', 1.0),
    ('scaffolding', 1.2),
    ('waterproofing', 1.2),
    ('prefab', 1.2),
    ('drywall', 1.2),
    ('window frame', 1.0),
    ('flooring material', 1.0),
    ('grout', 1.0),
    ('adhesive sealant', 1.0)
) AS t(kw, wt)
WHERE name = 'construction'
ON CONFLICT DO NOTHING;

-- ── defense ───────────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('military', 1.5),
    ('ammunition', 1.5),
    ('ballistic', 1.5),
    ('surveillance', 1.2),
    ('tactical', 1.5),
    ('armour', 1.5),
    ('defense equipment', 1.5),
    ('weapons system', 1.5),
    ('radar', 1.5),
    ('night vision', 1.5),
    ('body armor', 1.5),
    ('drone', 1.2),
    ('UAV', 1.5),
    ('military vehicle', 1.5),
    ('explosive ordnance', 1.5),
    ('communication system', 1.0),
    ('security equipment', 1.0),
    ('classified cargo', 1.5),
    ('end user certificate', 1.5),
    ('ITAR', 1.5)
) AS t(kw, wt)
WHERE name = 'defense'
ON CONFLICT DO NOTHING;

-- ── luxury ────────────────────────────────────────────────────────────────────
INSERT INTO category_keywords (category_id, keyword, weight)
SELECT id, kw, wt FROM classification_categories,
(VALUES
    ('jewelry', 1.5),
    ('watch', 1.2),
    ('diamond', 1.5),
    ('gold', 1.2),
    ('designer', 1.2),
    ('handbag', 1.2),
    ('luxury', 1.5),
    ('gemstone', 1.5),
    ('platinum', 1.5),
    ('silver jewelry', 1.2),
    ('haute couture', 1.5),
    ('branded goods', 1.2),
    ('fine art', 1.5),
    ('antique', 1.2),
    ('collectible', 1.2),
    ('high-end', 1.0),
    ('precious stone', 1.5),
    ('luxury vehicle', 1.5),
    ('wine fine', 1.2),
    ('bespoke', 1.2)
) AS t(kw, wt)
WHERE name = 'luxury'
ON CONFLICT DO NOTHING;
