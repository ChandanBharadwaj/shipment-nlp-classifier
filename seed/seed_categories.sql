-- Seed 20 shipment classification categories.
-- Safe to re-run (ON CONFLICT DO NOTHING).
INSERT INTO classification_categories (name, display_name, description, is_active) VALUES
    ('electronics',      'Electronics & Components',         'PCBs, semiconductors, consumer electronics, cables, batteries, displays', true),
    ('perishables',      'Perishable Goods',                 'Temperature-sensitive goods requiring cold chain: seafood, dairy, fresh produce', true),
    ('chemicals',        'Chemicals & Hazardous Materials',  'Industrial chemicals, solvents, acids, hazmat goods, MSDS-regulated cargo', true),
    ('textiles',         'Textiles & Apparel',               'Fabrics, garments, yarn, footwear, woven and knitted materials', true),
    ('machinery',        'Industrial Machinery & Equipment', 'Heavy equipment, industrial tools, turbines, compressors, conveyors', true),
    ('pharmaceuticals',  'Pharmaceuticals & Medical Devices','Drugs, vaccines, APIs, surgical instruments, sterile medical equipment', true),
    ('automotive',       'Automotive Parts & Vehicles',      'Vehicles, spare parts, tyres, engines, gearboxes, OEM components', true),
    ('food_beverages',   'Food & Beverages',                 'Packaged food, beverages, condiments, snacks, canned and bottled goods', true),
    ('furniture',        'Furniture & Home Goods',           'Household furniture, mattresses, decor items, bedding, upholstery', true),
    ('plastics',         'Plastics & Rubber',                'Raw plastic resin, rubber goods, polymers, elastomers, HDPE/PVC/PP products', true),
    ('metals',           'Metals & Metal Products',          'Steel, aluminium, copper, iron, alloys, pipes, bars, sheet metal, coils', true),
    ('paper',            'Paper & Printing Materials',       'Paper rolls, cardboard, corrugated sheets, printed matter, kraft paper', true),
    ('minerals',         'Minerals & Mining Products',       'Ores, limestone, granite, aggregates, sand, gravel, coal, bauxite', true),
    ('energy',           'Energy & Fuel Products',           'Crude oil, petroleum, LNG, LPG, diesel, lubricants, fuel, kerosene', true),
    ('agriculture',      'Agricultural Products',            'Grains, wheat, corn, fertilizers, pesticides, seeds, livestock, cotton', true),
    ('cosmetics',        'Cosmetics & Personal Care',        'Skincare, makeup, perfume, shampoo, toiletries, sunscreen, fragrances', true),
    ('toys',             'Toys & Sporting Goods',            'Toys, dolls, board games, bicycles, sports equipment, playground gear', true),
    ('construction',     'Construction Materials',           'Cement, timber, rebar, bricks, glass, roofing, insulation, plywood', true),
    ('defense',          'Defense & Security Equipment',     'Military hardware, surveillance equipment, tactical gear, ammunition', true),
    ('luxury',           'Luxury Goods & Jewelry',           'Watches, jewelry, diamonds, gold, designer handbags, luxury apparel', true)
ON CONFLICT (name) DO NOTHING;
