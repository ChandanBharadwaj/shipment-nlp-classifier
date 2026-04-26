-- ─────────────────────────────────────────────────────────────────────────────
-- seed_synthetic_labels.sql — hand-curated chapter-locked shipment rows.
-- ─────────────────────────────────────────────────────────────────────────────
--
-- WHY THIS FILE EXISTS
-- --------------------
-- centroid_builder.py builds one centroid per (category, hs_chapter) from
-- shipment_labels rows in the train split. When real-world rows for a
-- chapter are noisy or thin, the resulting centroid drifts toward whatever
-- terms happen to dominate by sheer count (e.g., "battery" ends up close
-- to ch95's centroid because seed_shipment_labels_v2.sql under-represents
-- toys-with-batteries).
--
-- These hand-curated rows fix that. Each is:
--   - chapter-locked (commodity_text never collides with another category)
--   - rich in the chapter's distinctive vocabulary
--   - paired with a plausible cargo_text (the kind of free-text shippers
--     actually write on bills of lading)
--
-- COVERAGE
-- --------
-- The plan's "top 5 error chapters" first: 95, 93, 87, 88, 28. ~50 rows
-- each. Other chapters get hand-curated coverage as failures are found.
--
-- ROW IDENTITY
-- ------------
-- shipment_id prefix 'syn_' distinguishes these rows from the generated
-- v2_ corpus and the legacy unprefixed rows. Re-runs are idempotent
-- because shipment_id is the primary key.
--
-- SPLIT POLICY
-- ------------
-- All rows go to the 'train' split — these labels exist to shape the
-- centroid, not to evaluate the classifier. Test/validation must come
-- from real shipments so the evaluation isn't tautological.
--
-- INVARIANT
-- ---------
-- Every commodity_text below MUST be unique to its category — the
-- chapter-lock invariant from CCTR P1. test_chapter_locked_labels.py
-- asserts this on the merged corpus.
-- ─────────────────────────────────────────────────────────────────────────────


-- ── ch95 — toys, games, sports requisites ────────────────────────────────────
INSERT INTO shipment_labels (shipment_id, category_name, cargo_text, commodity_text, hs_chapter, split) VALUES
    ('syn_95_001', 'toys', 'plush teddy bears, soft polyester filling, kids 3+',                         'Plush toy panda — stuffed animal toy, polyester fibre filling',           '95', 'train'),
    ('syn_95_002', 'toys', 'wooden building blocks set, 100 pieces, ages 2-5',                           'Wooden building block toy set — children educational toy',                '95', 'train'),
    ('syn_95_003', 'toys', 'fashion dolls assorted, plastic, 12-inch',                                   'Fashion doll — plastic toy figurine, 12 inch',                            '95', 'train'),
    ('syn_95_004', 'toys', 'jigsaw puzzles, 1000-piece, cardboard',                                      'Jigsaw puzzle — cardboard recreational toy, 1000 pieces',                 '95', 'train'),
    ('syn_95_005', 'toys', 'children tricycle, plastic frame, ages 2-4',                                 'Tricycle — children riding toy, plastic frame',                           '95', 'train'),
    ('syn_95_006', 'toys', 'rocking horse wooden, hand-painted',                                         'Rocking horse — wooden ride-on toy for toddlers',                         '95', 'train'),
    ('syn_95_007', 'toys', 'remote-control toy car, 1:24 scale, RC',                                     'Toy car — RC scale model recreational toy',                               '95', 'train'),
    ('syn_95_008', 'toys', 'toy helicopter, RC, indoor model',                                           'Toy helicopter — RC indoor recreational toy',                             '95', 'train'),
    ('syn_95_009', 'toys', 'toy train set, electric, plastic track',                                     'Toy train set — recreational scale-model toy',                            '95', 'train'),
    ('syn_95_010', 'toys', 'lego brick set, 500 pieces, ages 5+',                                        'Lego construction toy set — interlocking plastic blocks',                 '95', 'train'),
    ('syn_95_011', 'toys', 'plush dinosaur toy, soft, polyester',                                        'Plush dinosaur toy — stuffed animal',                                     '95', 'train'),
    ('syn_95_012', 'toys', 'baby rattles, BPA-free plastic, ages 0+',                                    'Baby rattle toy — infant plastic toy, BPA-free',                          '95', 'train'),
    ('syn_95_013', 'toys', 'board game, family edition, with dice',                                      'Board game — family recreational game with dice',                         '95', 'train'),
    ('syn_95_014', 'toys', 'card game deck, collectible trading',                                        'Trading card game deck — recreational',                                   '95', 'train'),
    ('syn_95_015', 'toys', 'plastic action figures, articulated, kids',                                  'Action figure toy — articulated plastic figurine',                        '95', 'train'),
    ('syn_95_016', 'toys', 'toy musical instrument, plastic xylophone',                                  'Toy xylophone — children musical toy',                                    '95', 'train'),
    ('syn_95_017', 'toys', 'kite assorted, nylon, recreational',                                         'Kite — recreational sporting requisite, nylon',                           '95', 'train'),
    ('syn_95_018', 'toys', 'inflatable beach ball, PVC, sports',                                         'Inflatable beach ball — sports requisite',                                '95', 'train'),
    ('syn_95_019', 'toys', 'football match ball, leather, professional',                                 'Soccer football — leather sports requisite',                              '95', 'train'),
    ('syn_95_020', 'toys', 'tennis racket, graphite frame, sports',                                      'Tennis racket — graphite sporting goods',                                 '95', 'train'),
    ('syn_95_021', 'toys', 'cricket bat, willow wood, ICC standard',                                     'Cricket bat — willow wood sporting requisite',                            '95', 'train'),
    ('syn_95_022', 'toys', 'badminton shuttlecock, feather, sports',                                     'Shuttlecock — feather sporting requisite for badminton',                  '95', 'train'),
    ('syn_95_023', 'toys', 'fishing rod, telescopic, recreational',                                      'Fishing rod — recreational sporting goods',                               '95', 'train'),
    ('syn_95_024', 'toys', 'bicycle, kids size, training wheels',                                        'Children bicycle — recreational toy with training wheels',                '95', 'train'),
    ('syn_95_025', 'toys', 'skateboard, deck and trucks, sports',                                        'Skateboard — sporting requisite, recreational',                           '95', 'train'),
    ('syn_95_026', 'toys', 'roller skates pair, kids size',                                              'Roller skates — children sporting requisite',                             '95', 'train'),
    ('syn_95_027', 'toys', 'toy soldiers plastic, set of 50',                                            'Toy soldier set — plastic figurine recreational toy',                     '95', 'train'),
    ('syn_95_028', 'toys', 'nerf blaster, foam darts, kids',                                             'Nerf blaster toy — foam dart recreational toy',                           '95', 'train'),
    ('syn_95_029', 'toys', 'plastic toy gun, water pistol',                                              'Toy water pistol — plastic recreational toy',                             '95', 'train'),
    ('syn_95_030', 'toys', 'plush bunny, easter, soft toy',                                              'Plush bunny — stuffed animal toy',                                        '95', 'train'),
    ('syn_95_031', 'toys', 'toy kitchen set, plastic, kids',                                             'Toy kitchen play set — plastic',                                          '95', 'train'),
    ('syn_95_032', 'toys', 'tea party play set, plastic',                                                'Toy tea party set — plastic',                                             '95', 'train'),
    ('syn_95_033', 'toys', 'wooden train set with tracks',                                               'Wooden toy train and track',                                              '95', 'train'),
    ('syn_95_034', 'toys', 'water gun toy, plastic, kids summer',                                        'Toy water gun — kids recreational',                                       '95', 'train'),
    ('syn_95_035', 'toys', 'toy aircraft scale model, 1:72',                                             'Toy aircraft scale model — recreational',                                 '95', 'train'),
    ('syn_95_036', 'toys', 'toy tank, plastic, kids',                                                    'Plastic toy tank — kids recreational toy',                                '95', 'train'),
    ('syn_95_037', 'toys', 'rc helicopter mini, indoor toy',                                             'RC mini helicopter toy — indoor recreational',                            '95', 'train'),
    ('syn_95_038', 'toys', 'plush stuffed elephant, large, soft',                                        'Plush elephant toy — stuffed animal',                                     '95', 'train'),
    ('syn_95_039', 'toys', 'puzzle 500 piece nature scenes',                                             'Jigsaw puzzle — 500 piece recreational',                                  '95', 'train'),
    ('syn_95_040', 'toys', 'children drum set, plastic, toy',                                            'Toy drum set — children musical toy',                                     '95', 'train'),
    ('syn_95_041', 'toys', 'toy battery powered car, kids',                                              'Toy battery-powered car — children ride-on',                              '95', 'train'),
    ('syn_95_042', 'toys', 'plush koala stuffed animal, soft',                                           'Plush koala toy — stuffed animal',                                        '95', 'train'),
    ('syn_95_043', 'toys', 'kids scooter, foldable, sports',                                             'Children scooter — sporting requisite',                                   '95', 'train'),
    ('syn_95_044', 'toys', 'toy fishing game, magnetic, kids',                                           'Toy fishing game — magnetic recreational',                                '95', 'train'),
    ('syn_95_045', 'toys', 'sports yoga mat, recreational',                                              'Yoga mat — sporting requisite',                                           '95', 'train'),
    ('syn_95_046', 'toys', 'snowboard, recreational, sports',                                            'Snowboard — sporting requisite',                                          '95', 'train'),
    ('syn_95_047', 'toys', 'lego architecture series toy set',                                           'Lego architecture toy set — interlocking blocks',                         '95', 'train'),
    ('syn_95_048', 'toys', 'plush dragon stuffed soft toy',                                              'Plush dragon toy — stuffed animal',                                       '95', 'train'),
    ('syn_95_049', 'toys', 'baseball bat, aluminum, sports',                                             'Baseball bat — aluminum sporting requisite',                              '95', 'train'),
    ('syn_95_050', 'toys', 'toy musical keyboard, kids electronic',                                      'Toy musical keyboard — children musical toy',                             '95', 'train')
ON CONFLICT (shipment_id, category_name) DO NOTHING;


-- ── ch93 — arms and ammunition ───────────────────────────────────────────────
INSERT INTO shipment_labels (shipment_id, category_name, cargo_text, commodity_text, hs_chapter, split) VALUES
    ('syn_93_001', 'defense', 'm16 rifles, military issue, 5.56mm',                                      'M16 rifle — military firearm, 5.56mm NATO calibre',                       '93', 'train'),
    ('syn_93_002', 'defense', 'ak-47 assault rifle, 7.62mm, military',                                   'AK-47 — military assault rifle, 7.62mm calibre',                          '93', 'train'),
    ('syn_93_003', 'defense', 'live ammunition 5.56mm NATO, cartridges',                                 'Live ammunition — 5.56mm NATO cartridge, military',                       '93', 'train'),
    ('syn_93_004', 'defense', '9mm pistol cartridges, mil-spec',                                         '9mm pistol cartridge — mil-spec ammunition',                              '93', 'train'),
    ('syn_93_005', 'defense', 'rocket launcher, military, anti-tank',                                    'Rocket launcher — anti-tank military weapon',                             '93', 'train'),
    ('syn_93_006', 'defense', 'grenade fragmentation, military issue',                                   'Fragmentation grenade — military explosive ordnance',                     '93', 'train'),
    ('syn_93_007', 'defense', 'mortar shells 81mm, live, military',                                      '81mm mortar shell — live military ordnance',                              '93', 'train'),
    ('syn_93_008', 'defense', 'shotgun shells 12-gauge, live ammo',                                      '12-gauge shotgun shell — live ammunition',                                '93', 'train'),
    ('syn_93_009', 'defense', 'sniper rifle bolt-action, military',                                      'Sniper rifle — military bolt-action firearm',                             '93', 'train'),
    ('syn_93_010', 'defense', 'machine gun belt-fed, mil-spec',                                          'Machine gun — mil-spec belt-fed firearm',                                 '93', 'train'),
    ('syn_93_011', 'defense', 'firearm parts barrels, mil-spec',                                         'Firearm barrel — mil-spec spare part',                                    '93', 'train'),
    ('syn_93_012', 'defense', 'gunpowder propellant, military grade',                                    'Gunpowder propellant — military-grade explosive',                         '93', 'train'),
    ('syn_93_013', 'defense', 'tank ammunition 120mm, live',                                             '120mm tank ammunition — live ordnance',                                   '93', 'train'),
    ('syn_93_014', 'defense', 'artillery shells 155mm, live military',                                   '155mm artillery shell — live military ordnance',                          '93', 'train'),
    ('syn_93_015', 'defense', 'pistol holsters tactical, mil-spec',                                      'Tactical pistol holster — mil-spec accessory',                            '93', 'train'),
    ('syn_93_016', 'defense', 'rifle scopes military issue, optical',                                    'Military rifle scope — optical sighting accessory',                       '93', 'train'),
    ('syn_93_017', 'defense', 'bayonet, m9, mil-spec',                                                   'M9 bayonet — mil-spec firearm accessory',                                 '93', 'train'),
    ('syn_93_018', 'defense', 'firearm cleaning kit, military',                                          'Firearm cleaning kit — military maintenance accessory',                   '93', 'train'),
    ('syn_93_019', 'defense', 'rifle magazines 30-round, mil-spec',                                      'Rifle magazine — 30-round mil-spec accessory',                            '93', 'train'),
    ('syn_93_020', 'defense', 'tracer rounds 7.62mm, military',                                          '7.62mm tracer round — military ammunition',                               '93', 'train'),
    ('syn_93_021', 'defense', 'flares signal military issue',                                            'Signal flare — military issue ordnance',                                  '93', 'train'),
    ('syn_93_022', 'defense', 'fuses for ammunition, mil-spec',                                          'Ammunition fuse — mil-spec component',                                    '93', 'train'),
    ('syn_93_023', 'defense', 'submachine gun mp5, mil-spec',                                            'MP5 submachine gun — mil-spec firearm',                                   '93', 'train'),
    ('syn_93_024', 'defense', 'pistol semi-automatic 9mm military',                                      'Semi-automatic 9mm pistol — military sidearm',                            '93', 'train'),
    ('syn_93_025', 'defense', 'revolver .38 special, mil-spec',                                          '.38 revolver — mil-spec firearm',                                         '93', 'train'),
    ('syn_93_026', 'defense', 'rifle stock wood, mil-spec part',                                         'Rifle wooden stock — mil-spec part',                                      '93', 'train'),
    ('syn_93_027', 'defense', 'firearm trigger group mil-spec',                                          'Firearm trigger group — mil-spec part',                                   '93', 'train'),
    ('syn_93_028', 'defense', 'bayonet sheath kydex, mil-spec',                                          'Kydex bayonet sheath — mil-spec accessory',                               '93', 'train'),
    ('syn_93_029', 'defense', 'rifle bipod tactical, mil-spec',                                          'Tactical rifle bipod — mil-spec accessory',                               '93', 'train'),
    ('syn_93_030', 'defense', 'silencer firearm suppressor mil-spec',                                    'Firearm suppressor silencer — mil-spec',                                  '93', 'train'),
    ('syn_93_031', 'defense', 'shotgun semi-auto 12-gauge, mil-spec',                                    'Semi-automatic 12-gauge shotgun — mil-spec firearm',                      '93', 'train'),
    ('syn_93_032', 'defense', 'recoil spring rifle mil-spec part',                                       'Rifle recoil spring — mil-spec part',                                     '93', 'train'),
    ('syn_93_033', 'defense', 'firing pin firearm mil-spec',                                             'Firearm firing pin — mil-spec part',                                      '93', 'train'),
    ('syn_93_034', 'defense', 'cartridge case 5.56mm brass',                                             'Brass 5.56mm cartridge case — ammunition component',                      '93', 'train'),
    ('syn_93_035', 'defense', 'ammunition belt linked 7.62mm',                                           'Linked 7.62mm ammunition belt',                                           '93', 'train'),
    ('syn_93_036', 'defense', 'percussion cap firearm mil-spec',                                         'Firearm percussion cap — mil-spec component',                             '93', 'train'),
    ('syn_93_037', 'defense', 'flare launcher military issue',                                           'Military flare launcher — signaling ordnance',                            '93', 'train'),
    ('syn_93_038', 'defense', 'mortar tube 60mm military',                                               '60mm mortar tube — military weapon',                                      '93', 'train'),
    ('syn_93_039', 'defense', 'recoil pad rifle mil-spec',                                               'Rifle recoil pad — mil-spec accessory',                                   '93', 'train'),
    ('syn_93_040', 'defense', 'gun safe military grade steel',                                           'Military-grade gun safe — steel firearm storage',                         '93', 'train'),
    ('syn_93_041', 'defense', 'firearm sight ironsight mil-spec',                                        'Iron sight — mil-spec firearm sighting',                                  '93', 'train'),
    ('syn_93_042', 'defense', 'rifle cleaning rod brass mil-spec',                                       'Brass cleaning rod — mil-spec firearm maintenance',                       '93', 'train'),
    ('syn_93_043', 'defense', 'live cartridges 5.56mm 1000-pack military',                               '5.56mm live cartridge bulk — military ammunition',                        '93', 'train'),
    ('syn_93_044', 'defense', 'gun belt tactical mil-spec',                                              'Tactical gun belt — mil-spec accessory',                                  '93', 'train'),
    ('syn_93_045', 'defense', 'magazine pouch tactical mil-spec',                                        'Tactical magazine pouch — mil-spec accessory',                            '93', 'train'),
    ('syn_93_046', 'defense', 'ammunition primer mil-spec',                                              'Ammunition primer — mil-spec component',                                  '93', 'train'),
    ('syn_93_047', 'defense', 'firearm lubricant mil-spec military',                                     'Military firearm lubricant — mil-spec maintenance',                       '93', 'train'),
    ('syn_93_048', 'defense', 'rifle case hard mil-spec ballistic',                                      'Ballistic hard rifle case — mil-spec',                                    '93', 'train'),
    ('syn_93_049', 'defense', 'pistol grips polymer mil-spec',                                           'Polymer pistol grip — mil-spec part',                                     '93', 'train'),
    ('syn_93_050', 'defense', 'tactical sling rifle mil-spec',                                           'Tactical rifle sling — mil-spec accessory',                               '93', 'train')
ON CONFLICT (shipment_id, category_name) DO NOTHING;


-- ── ch87 — vehicles other than railway ───────────────────────────────────────
INSERT INTO shipment_labels (shipment_id, category_name, cargo_text, commodity_text, hs_chapter, split) VALUES
    ('syn_87_001', 'automotive', 'passenger sedan vehicle, 2.0L, 4-door',                                'Passenger sedan car — 2.0L petrol 4-door automobile',                     '87', 'train'),
    ('syn_87_002', 'automotive', 'pickup truck commercial 2WD diesel',                                   'Pickup truck — commercial 2WD diesel vehicle',                            '87', 'train'),
    ('syn_87_003', 'automotive', 'suv 4WD diesel vehicle',                                               'SUV — 4WD diesel passenger vehicle',                                      '87', 'train'),
    ('syn_87_004', 'automotive', 'auto brake pads OEM replacement',                                      'Automobile brake pad — OEM spare part',                                   '87', 'train'),
    ('syn_87_005', 'automotive', 'engine block 2.0L gasoline OEM',                                       'Engine block — 2.0L gasoline OEM automotive part',                        '87', 'train'),
    ('syn_87_006', 'automotive', 'transmission gearbox 6-speed manual',                                  '6-speed manual transmission gearbox — automotive',                        '87', 'train'),
    ('syn_87_007', 'automotive', 'clutch assembly automotive OEM',                                       'Clutch assembly — OEM automotive part',                                   '87', 'train'),
    ('syn_87_008', 'automotive', 'tyres pneumatic 195/65R15 passenger',                                  'Pneumatic tyre — 195/65R15 passenger car',                                '87', 'train'),
    ('syn_87_009', 'automotive', 'tire light truck 245/75R16 OEM',                                       'Light-truck tire — 245/75R16 OEM',                                        '87', 'train'),
    ('syn_87_010', 'automotive', 'shock absorber rear pair, OEM',                                        'Rear shock absorber pair — OEM automotive',                               '87', 'train'),
    ('syn_87_011', 'automotive', 'alternator 12V 130A automotive',                                       'Automotive alternator — 12V 130A',                                        '87', 'train'),
    ('syn_87_012', 'automotive', 'starter motor 12V passenger car',                                      'Starter motor — 12V passenger car automotive',                            '87', 'train'),
    ('syn_87_013', 'automotive', 'radiator aluminum passenger car',                                      'Aluminum radiator — passenger car automotive',                            '87', 'train'),
    ('syn_87_014', 'automotive', 'wheel rim alloy 17-inch OEM',                                          '17-inch alloy wheel rim — OEM automotive',                                '87', 'train'),
    ('syn_87_015', 'automotive', 'headlight assembly LED OEM passenger car',                            'LED headlight assembly — OEM passenger car',                              '87', 'train'),
    ('syn_87_016', 'automotive', 'exhaust muffler stainless steel',                                      'Stainless steel exhaust muffler — automotive',                            '87', 'train'),
    ('syn_87_017', 'automotive', 'catalytic converter passenger car',                                    'Catalytic converter — passenger car emission part',                       '87', 'train'),
    ('syn_87_018', 'automotive', 'fuel pump 12V automotive OEM',                                         '12V automotive fuel pump — OEM',                                          '87', 'train'),
    ('syn_87_019', 'automotive', 'oxygen sensor automotive OEM',                                         'Oxygen sensor — OEM automotive',                                          '87', 'train'),
    ('syn_87_020', 'automotive', 'fuel injector OEM passenger car',                                      'Fuel injector — OEM passenger car automotive',                            '87', 'train'),
    ('syn_87_021', 'automotive', 'commercial vehicle truck 6-wheel diesel',                              'Commercial truck — 6-wheel diesel vehicle',                               '87', 'train'),
    ('syn_87_022', 'automotive', 'bus passenger 30-seat diesel',                                         'Passenger bus — 30-seat diesel commercial vehicle',                       '87', 'train'),
    ('syn_87_023', 'automotive', 'minivan 7-seat petrol vehicle',                                        'Minivan — 7-seat petrol passenger vehicle',                               '87', 'train'),
    ('syn_87_024', 'automotive', 'motorcycle 250cc 2-stroke vehicle',                                    'Motorcycle — 250cc 2-stroke road vehicle',                                '87', 'train'),
    ('syn_87_025', 'automotive', 'scooter electric 2-wheeler vehicle',                                   'Electric scooter — 2-wheeler road vehicle',                               '87', 'train'),
    ('syn_87_026', 'automotive', 'car battery lead-acid 12V passenger',                                  'Lead-acid 12V car battery — passenger automotive',                        '87', 'train'),
    ('syn_87_027', 'automotive', 'wiper blade OEM passenger car pair',                                   'Wiper blade — OEM passenger car automotive pair',                         '87', 'train'),
    ('syn_87_028', 'automotive', 'air filter OEM automotive engine',                                     'Engine air filter — OEM automotive',                                      '87', 'train'),
    ('syn_87_029', 'automotive', 'oil filter OEM passenger car',                                         'Oil filter — OEM passenger car automotive',                               '87', 'train'),
    ('syn_87_030', 'automotive', 'brake disc rotor automotive OEM',                                      'Brake disc rotor — OEM automotive',                                       '87', 'train'),
    ('syn_87_031', 'automotive', 'differential rear-axle automotive',                                    'Rear-axle differential — automotive driveline',                           '87', 'train'),
    ('syn_87_032', 'automotive', 'CV joint half-shaft OEM',                                              'CV joint half-shaft — OEM automotive driveline',                          '87', 'train'),
    ('syn_87_033', 'automotive', 'wheel bearing OEM automotive',                                         'Wheel bearing — OEM automotive',                                          '87', 'train'),
    ('syn_87_034', 'automotive', 'tyre tubes inner butyl rubber',                                        'Butyl rubber tyre inner tube — automotive',                               '87', 'train'),
    ('syn_87_035', 'automotive', 'cylinder head 2.0L OEM',                                               '2.0L cylinder head — OEM automotive engine',                              '87', 'train'),
    ('syn_87_036', 'automotive', 'piston ring set 2.0L OEM',                                             '2.0L piston ring set — OEM automotive engine',                            '87', 'train'),
    ('syn_87_037', 'automotive', 'timing belt OEM passenger car',                                        'Timing belt — OEM passenger car automotive',                              '87', 'train'),
    ('syn_87_038', 'automotive', 'spark plug iridium OEM',                                               'Iridium spark plug — OEM automotive',                                     '87', 'train'),
    ('syn_87_039', 'automotive', 'turbocharger 1.6L OEM diesel',                                         '1.6L diesel turbocharger — OEM automotive',                               '87', 'train'),
    ('syn_87_040', 'automotive', 'water pump engine OEM automotive',                                     'Engine water pump — OEM automotive',                                      '87', 'train'),
    ('syn_87_041', 'automotive', 'fuel tank steel passenger car',                                        'Steel fuel tank — passenger car automotive',                              '87', 'train'),
    ('syn_87_042', 'automotive', 'door panel passenger car OEM',                                         'Door panel — OEM passenger car automotive',                               '87', 'train'),
    ('syn_87_043', 'automotive', 'bumper front passenger car OEM',                                       'Front bumper — OEM passenger car automotive',                             '87', 'train'),
    ('syn_87_044', 'automotive', 'side mirror OEM passenger car',                                        'Side mirror — OEM passenger car automotive',                              '87', 'train'),
    ('syn_87_045', 'automotive', 'seat belt assembly passenger car OEM',                                 'Seat belt assembly — OEM passenger car automotive',                       '87', 'train'),
    ('syn_87_046', 'automotive', 'airbag module OEM driver-side',                                        'Driver-side airbag module — OEM automotive',                              '87', 'train'),
    ('syn_87_047', 'automotive', 'dashboard cluster OEM passenger car',                                  'Dashboard instrument cluster — OEM passenger car',                        '87', 'train'),
    ('syn_87_048', 'automotive', 'steering rack OEM automotive',                                         'Steering rack — OEM automotive',                                          '87', 'train'),
    ('syn_87_049', 'automotive', 'commercial pickup truck heavy-duty diesel',                            'Heavy-duty diesel pickup truck — commercial vehicle',                     '87', 'train'),
    ('syn_87_050', 'automotive', 'truck tyre 11R22.5 commercial',                                        'Commercial truck tyre — 11R22.5 freight',                                 '87', 'train')
ON CONFLICT (shipment_id, category_name) DO NOTHING;


-- ── ch88 — aircraft, spacecraft (machinery category) ─────────────────────────
INSERT INTO shipment_labels (shipment_id, category_name, cargo_text, commodity_text, hs_chapter, split) VALUES
    ('syn_88_001', 'machinery', 'commercial aircraft turbofan engine',                                   'Commercial aircraft turbofan engine — aviation',                          '88', 'train'),
    ('syn_88_002', 'machinery', 'turbojet engine aviation military',                                     'Turbojet engine — aviation propulsion',                                   '88', 'train'),
    ('syn_88_003', 'machinery', 'aircraft fuselage panel composite',                                     'Aircraft fuselage panel — composite aviation structure',                  '88', 'train'),
    ('syn_88_004', 'machinery', 'avionics flight management computer',                                   'Avionics flight management computer — aviation',                          '88', 'train'),
    ('syn_88_005', 'machinery', 'rotorcraft helicopter blade composite',                                 'Rotorcraft helicopter blade — composite aviation',                        '88', 'train'),
    ('syn_88_006', 'machinery', 'aircraft landing gear assembly',                                        'Aircraft landing gear assembly — aviation undercarriage',                 '88', 'train'),
    ('syn_88_007', 'machinery', 'spacecraft thruster solid-propellant',                                  'Spacecraft thruster — solid-propellant aerospace',                        '88', 'train'),
    ('syn_88_008', 'machinery', 'aircraft cockpit instrument panel',                                     'Cockpit instrument panel — aviation avionics',                            '88', 'train'),
    ('syn_88_009', 'machinery', 'aircraft propeller composite blade',                                    'Aircraft propeller blade — composite aviation',                           '88', 'train'),
    ('syn_88_010', 'machinery', 'turbofan fan blade titanium',                                           'Titanium turbofan fan blade — aviation',                                  '88', 'train'),
    ('syn_88_011', 'machinery', 'aircraft tire pneumatic aviation',                                      'Aircraft pneumatic tire — aviation undercarriage',                        '88', 'train'),
    ('syn_88_012', 'machinery', 'aircraft hydraulic pump aviation',                                      'Aircraft hydraulic pump — aviation system',                               '88', 'train'),
    ('syn_88_013', 'machinery', 'aircraft fuel pump aviation grade',                                     'Aircraft fuel pump — aviation grade',                                     '88', 'train'),
    ('syn_88_014', 'machinery', 'aircraft seat business class aviation',                                 'Aircraft passenger seat — aviation business class',                       '88', 'train'),
    ('syn_88_015', 'machinery', 'aircraft wing flap assembly composite',                                 'Aircraft wing flap assembly — composite aviation',                        '88', 'train'),
    ('syn_88_016', 'machinery', 'aircraft horizontal stabilizer aviation',                              'Horizontal stabilizer — aviation tail surface',                           '88', 'train'),
    ('syn_88_017', 'machinery', 'aircraft vertical stabilizer aviation',                                 'Vertical stabilizer — aviation tail surface',                             '88', 'train'),
    ('syn_88_018', 'machinery', 'aircraft cargo door aviation',                                          'Aircraft cargo door — aviation structure',                                '88', 'train'),
    ('syn_88_019', 'machinery', 'aircraft pilot ejection seat aviation',                                 'Pilot ejection seat — aviation safety',                                   '88', 'train'),
    ('syn_88_020', 'machinery', 'helicopter rotor head titanium',                                        'Helicopter rotor head — titanium aviation',                               '88', 'train'),
    ('syn_88_021', 'machinery', 'aircraft black box flight recorder',                                    'Flight data recorder black box — aviation avionics',                      '88', 'train'),
    ('syn_88_022', 'machinery', 'aircraft cockpit voice recorder',                                       'Cockpit voice recorder — aviation avionics',                              '88', 'train'),
    ('syn_88_023', 'machinery', 'aircraft transponder ADS-B aviation',                                   'ADS-B transponder — aviation avionics',                                   '88', 'train'),
    ('syn_88_024', 'machinery', 'spacecraft solar panel aerospace',                                      'Spacecraft solar panel — aerospace',                                      '88', 'train'),
    ('syn_88_025', 'machinery', 'satellite communications antenna aerospace',                            'Satellite communications antenna — aerospace',                            '88', 'train'),
    ('syn_88_026', 'machinery', 'aircraft brake disc carbon composite',                                  'Carbon-composite aircraft brake disc — aviation',                         '88', 'train'),
    ('syn_88_027', 'machinery', 'aircraft pitot tube aviation',                                          'Pitot tube — aviation airspeed sensor',                                   '88', 'train'),
    ('syn_88_028', 'machinery', 'rotorcraft tail rotor blade aviation',                                  'Tail rotor blade — rotorcraft aviation',                                  '88', 'train'),
    ('syn_88_029', 'machinery', 'aircraft windshield laminated aviation',                                'Laminated aircraft windshield — aviation',                                '88', 'train'),
    ('syn_88_030', 'machinery', 'aircraft galley oven aviation grade',                                   'Aircraft galley oven — aviation grade',                                   '88', 'train'),
    ('syn_88_031', 'machinery', 'aircraft lavatory module aviation',                                     'Aircraft lavatory module — aviation interior',                            '88', 'train'),
    ('syn_88_032', 'machinery', 'helicopter swashplate aviation',                                        'Helicopter swashplate — aviation rotor system',                           '88', 'train'),
    ('syn_88_033', 'machinery', 'aircraft engine mount aviation',                                        'Aircraft engine mount — aviation structure',                              '88', 'train'),
    ('syn_88_034', 'machinery', 'aircraft fire suppression bottle aviation',                             'Aircraft fire suppression bottle — aviation',                             '88', 'train'),
    ('syn_88_035', 'machinery', 'aircraft oxygen mask cabin aviation',                                   'Cabin oxygen mask — aviation safety',                                     '88', 'train'),
    ('syn_88_036', 'machinery', 'aircraft auxiliary power unit APU',                                     'APU auxiliary power unit — aviation',                                     '88', 'train'),
    ('syn_88_037', 'machinery', 'aircraft thrust reverser cascade',                                      'Thrust reverser cascade — aviation engine',                               '88', 'train'),
    ('syn_88_038', 'machinery', 'unmanned aerial vehicle UAV airframe',                                  'UAV airframe — unmanned aerial vehicle aviation',                         '88', 'train'),
    ('syn_88_039', 'machinery', 'aircraft wing spar composite aviation',                                 'Aircraft wing spar — composite aviation structure',                       '88', 'train'),
    ('syn_88_040', 'machinery', 'aircraft cabin pressurization valve',                                   'Cabin pressurization valve — aviation system',                            '88', 'train'),
    ('syn_88_041', 'machinery', 'aircraft de-icing boot aviation',                                       'Aircraft de-icing boot — aviation safety',                                '88', 'train'),
    ('syn_88_042', 'machinery', 'helicopter tail boom aviation composite',                               'Helicopter tail boom — composite aviation',                               '88', 'train'),
    ('syn_88_043', 'machinery', 'aircraft starter generator aviation',                                   'Aircraft starter generator — aviation',                                   '88', 'train'),
    ('syn_88_044', 'machinery', 'aircraft fuel control unit aviation',                                   'Fuel control unit — aviation engine',                                     '88', 'train'),
    ('syn_88_045', 'machinery', 'aircraft inertial reference unit IRU',                                  'Inertial reference unit — aviation avionics',                             '88', 'train'),
    ('syn_88_046', 'machinery', 'spacecraft attitude control thruster',                                  'Attitude control thruster — spacecraft',                                  '88', 'train'),
    ('syn_88_047', 'machinery', 'aircraft cargo container ULD aviation',                                 'ULD aircraft cargo container — aviation',                                 '88', 'train'),
    ('syn_88_048', 'machinery', 'aircraft slats leading edge aviation',                                  'Leading-edge slat — aviation aerodynamic',                                '88', 'train'),
    ('syn_88_049', 'machinery', 'aircraft rudder pedal assembly aviation',                               'Aircraft rudder pedal assembly — aviation',                               '88', 'train'),
    ('syn_88_050', 'machinery', 'aircraft flight control computer aviation',                             'Flight control computer — aviation avionics',                             '88', 'train')
ON CONFLICT (shipment_id, category_name) DO NOTHING;


-- ── ch28 — inorganic chemicals ───────────────────────────────────────────────
INSERT INTO shipment_labels (shipment_id, category_name, cargo_text, commodity_text, hs_chapter, split) VALUES
    ('syn_28_001', 'chemicals', 'sulphuric acid 98%, industrial bulk',                                   'Sulphuric acid — 98% industrial bulk inorganic chemical',                 '28', 'train'),
    ('syn_28_002', 'chemicals', 'hydrochloric acid 32% bulk industrial',                                 'Hydrochloric acid — 32% bulk inorganic chemical',                         '28', 'train'),
    ('syn_28_003', 'chemicals', 'sodium hydroxide flakes industrial',                                    'Sodium hydroxide flakes — industrial caustic soda inorganic',             '28', 'train'),
    ('syn_28_004', 'chemicals', 'caustic soda solution industrial bulk',                                 'Caustic soda solution — industrial bulk inorganic',                       '28', 'train'),
    ('syn_28_005', 'chemicals', 'nitric acid 65% industrial bulk',                                       'Nitric acid — 65% industrial bulk inorganic',                             '28', 'train'),
    ('syn_28_006', 'chemicals', 'ammonia anhydrous industrial bulk',                                     'Anhydrous ammonia — industrial bulk inorganic',                           '28', 'train'),
    ('syn_28_007', 'chemicals', 'chlorine gas industrial bulk',                                          'Chlorine gas — industrial bulk inorganic chemical',                       '28', 'train'),
    ('syn_28_008', 'chemicals', 'phosphoric acid food-grade',                                            'Phosphoric acid — food-grade inorganic chemical',                         '28', 'train'),
    ('syn_28_009', 'chemicals', 'hydrogen peroxide 35% industrial',                                      'Hydrogen peroxide — 35% industrial inorganic',                            '28', 'train'),
    ('syn_28_010', 'chemicals', 'sodium chloride pure industrial',                                       'Sodium chloride — pure industrial inorganic',                             '28', 'train'),
    ('syn_28_011', 'chemicals', 'potassium hydroxide industrial bulk',                                   'Potassium hydroxide — industrial bulk inorganic',                         '28', 'train'),
    ('syn_28_012', 'chemicals', 'calcium carbonate industrial bulk',                                     'Calcium carbonate — industrial bulk inorganic',                           '28', 'train'),
    ('syn_28_013', 'chemicals', 'sodium carbonate soda ash industrial',                                  'Sodium carbonate soda ash — industrial inorganic',                        '28', 'train'),
    ('syn_28_014', 'chemicals', 'magnesium oxide industrial grade',                                      'Magnesium oxide — industrial-grade inorganic',                            '28', 'train'),
    ('syn_28_015', 'chemicals', 'aluminum sulphate industrial water-treatment',                          'Aluminum sulphate — industrial inorganic water-treatment',                '28', 'train'),
    ('syn_28_016', 'chemicals', 'ferric chloride industrial water-treatment',                            'Ferric chloride — industrial inorganic water-treatment',                  '28', 'train'),
    ('syn_28_017', 'chemicals', 'zinc oxide industrial-grade powder',                                    'Zinc oxide powder — industrial-grade inorganic',                          '28', 'train'),
    ('syn_28_018', 'chemicals', 'titanium dioxide pigment industrial',                                   'Titanium dioxide pigment — industrial inorganic',                         '28', 'train'),
    ('syn_28_019', 'chemicals', 'silicon carbide industrial abrasive',                                   'Silicon carbide — industrial abrasive inorganic',                         '28', 'train'),
    ('syn_28_020', 'chemicals', 'potassium nitrate industrial-grade',                                    'Potassium nitrate — industrial-grade inorganic',                          '28', 'train'),
    ('syn_28_021', 'chemicals', 'sodium bicarbonate industrial bulk',                                    'Sodium bicarbonate — industrial bulk inorganic',                          '28', 'train'),
    ('syn_28_022', 'chemicals', 'iron oxide industrial pigment',                                         'Iron oxide pigment — industrial inorganic',                               '28', 'train'),
    ('syn_28_023', 'chemicals', 'lead oxide industrial bulk',                                            'Lead oxide — industrial bulk inorganic',                                  '28', 'train'),
    ('syn_28_024', 'chemicals', 'copper sulphate pentahydrate industrial',                               'Copper sulphate pentahydrate — industrial inorganic',                     '28', 'train'),
    ('syn_28_025', 'chemicals', 'silver nitrate industrial-grade',                                       'Silver nitrate — industrial-grade inorganic',                             '28', 'train'),
    ('syn_28_026', 'chemicals', 'mercury industrial bulk inorganic',                                     'Mercury — industrial bulk inorganic chemical',                            '28', 'train'),
    ('syn_28_027', 'chemicals', 'phosphorus pentoxide industrial',                                       'Phosphorus pentoxide — industrial inorganic',                             '28', 'train'),
    ('syn_28_028', 'chemicals', 'boric acid industrial-grade',                                           'Boric acid — industrial-grade inorganic',                                 '28', 'train'),
    ('syn_28_029', 'chemicals', 'sodium silicate water-glass industrial',                                'Sodium silicate water-glass — industrial inorganic',                      '28', 'train'),
    ('syn_28_030', 'chemicals', 'manganese dioxide industrial bulk',                                     'Manganese dioxide — industrial bulk inorganic',                           '28', 'train'),
    ('syn_28_031', 'chemicals', 'potassium chloride industrial bulk',                                    'Potassium chloride — industrial bulk inorganic',                          '28', 'train'),
    ('syn_28_032', 'chemicals', 'calcium chloride industrial-grade',                                     'Calcium chloride — industrial-grade inorganic',                           '28', 'train'),
    ('syn_28_033', 'chemicals', 'barium sulphate industrial-grade',                                      'Barium sulphate — industrial-grade inorganic',                            '28', 'train'),
    ('syn_28_034', 'chemicals', 'magnesium sulphate epsom industrial',                                   'Magnesium sulphate epsom — industrial inorganic',                         '28', 'train'),
    ('syn_28_035', 'chemicals', 'sodium sulphate industrial bulk',                                       'Sodium sulphate — industrial bulk inorganic',                             '28', 'train'),
    ('syn_28_036', 'chemicals', 'sulphur elemental industrial bulk',                                     'Elemental sulphur — industrial bulk inorganic',                           '28', 'train'),
    ('syn_28_037', 'chemicals', 'liquid argon industrial cryogenic',                                     'Liquid argon — industrial cryogenic inorganic',                           '28', 'train'),
    ('syn_28_038', 'chemicals', 'liquid nitrogen industrial cryogenic',                                  'Liquid nitrogen — industrial cryogenic inorganic',                        '28', 'train'),
    ('syn_28_039', 'chemicals', 'liquid oxygen industrial cryogenic',                                    'Liquid oxygen — industrial cryogenic inorganic',                          '28', 'train'),
    ('syn_28_040', 'chemicals', 'helium gas industrial-grade',                                           'Helium gas — industrial-grade inorganic',                                 '28', 'train'),
    ('syn_28_041', 'chemicals', 'fluorspar industrial inorganic',                                        'Fluorspar — industrial inorganic chemical',                               '28', 'train'),
    ('syn_28_042', 'chemicals', 'aluminum chloride industrial bulk',                                     'Aluminum chloride — industrial bulk inorganic',                           '28', 'train'),
    ('syn_28_043', 'chemicals', 'sodium thiosulphate industrial photographic',                           'Sodium thiosulphate — industrial inorganic photographic',                 '28', 'train'),
    ('syn_28_044', 'chemicals', 'cobalt oxide industrial pigment',                                       'Cobalt oxide pigment — industrial inorganic',                             '28', 'train'),
    ('syn_28_045', 'chemicals', 'nickel sulphate industrial plating',                                    'Nickel sulphate — industrial inorganic plating',                          '28', 'train'),
    ('syn_28_046', 'chemicals', 'sodium hypochlorite bleach industrial',                                 'Sodium hypochlorite bleach — industrial inorganic',                       '28', 'train'),
    ('syn_28_047', 'chemicals', 'potassium permanganate industrial',                                     'Potassium permanganate — industrial inorganic',                           '28', 'train'),
    ('syn_28_048', 'chemicals', 'calcium hypochlorite industrial',                                       'Calcium hypochlorite — industrial inorganic',                             '28', 'train'),
    ('syn_28_049', 'chemicals', 'sodium fluoride industrial-grade',                                      'Sodium fluoride — industrial-grade inorganic',                            '28', 'train'),
    ('syn_28_050', 'chemicals', 'lithium hydroxide industrial',                                          'Lithium hydroxide — industrial inorganic chemical',                       '28', 'train')
ON CONFLICT (shipment_id, category_name) DO NOTHING;
