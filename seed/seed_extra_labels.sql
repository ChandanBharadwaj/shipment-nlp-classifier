-- Extra labeled rows to improve centroid quality for categories that scored low.
-- Focuses on two improvements:
--   1. More varied language for electronics (mixed cargo context, abbreviations,
--      real shipping doc style) so the centroid covers a wider region of semantic space.
--   2. Similar improvements for other low-confidence categories.
-- Safe to re-run (ON CONFLICT DO NOTHING).

-- ── electronics (extra 30 rows — varied language and shipping doc style) ───────
INSERT INTO shipment_labels (shipment_id, category_name, cargo_text, commodity_text) VALUES
('el_e01','electronics','Electronic components mixed consignment',  'PCB semiconductor integrated circuits boards'),
('el_e02','electronics','Circuit boards and electronic hardware',   'printed circuit assemblies electronics components'),
('el_e03','electronics','Boards PCB electronic industrial mixed',   'semiconductor components microchips electronics'),
('el_e04','electronics','Electronic goods semiconductors boards',   'circuit boards chips electronic hardware'),
('el_e05','electronics','Mixed electronics PCB wafers',             'semiconductor wafer circuit electronic components'),
('el_e06','electronics','Electronic parts PCB assembly shipment',   'PCB assembly semiconductors components boards'),
('el_e07','electronics','Semiconductor chips electronic components','microchip transistor electronic circuit components'),
('el_e08','electronics','Computer hardware electronics PCB',        'PC hardware circuit board semiconductor chips'),
('el_e09','electronics','Electronic boards components cargo',       'PCB components semiconductor electronics mixed'),
('el_e10','electronics','Electronics shipment circuit boards',      'semiconductor PCB electronic assembly boards'),
('el_e11','electronics','Mobile phone components electronic parts', 'smartphone parts circuit board electronic chips'),
('el_e12','electronics','Laptop parts electronic components',       'motherboard RAM GPU semiconductor electronics'),
('el_e13','electronics','Electronic modules wireless boards',       'wireless module PCB Bluetooth semiconductor'),
('el_e14','electronics','Industrial electronics control boards',    'PLC control board industrial electronic semiconductor'),
('el_e15','electronics','Electronic sensors IoT components',        'sensor IoT circuit board electronic semiconductor'),
('el_e16','electronics','Power electronics inverter boards',        'inverter power electronics PCB semiconductor'),
('el_e17','electronics','Embedded systems modules boards',          'embedded module SOM PCB electronic semiconductor'),
('el_e18','electronics','Electronic display panels boards',         'display panel LCD OLED PCB semiconductor'),
('el_e19','electronics','Network electronics router boards',        'router network electronics PCB semiconductor chips'),
('el_e20','electronics','Electronic testing instruments boards',    'oscilloscope test PCB electronic semiconductor'),
('el_e21','electronics','Semiconductor wafers silicon chips',       'silicon wafer semiconductor chip electronic'),
('el_e22','electronics','Electronic components cargo assorted',     'transistor capacitor resistor PCB semiconductor'),
('el_e23','electronics','Computer chips semiconductor mixed',       'CPU GPU RAM semiconductor electronic chip'),
('el_e24','electronics','EV power electronics battery management',  'BMS battery management PCB semiconductor electronic'),
('el_e25','electronics','Electronic goods PCB boards cargo',        'PCB boards electronic components semiconductor mixed'),
('el_e26','electronics','Signal processing electronics boards',     'DSP FPGA PCB electronic semiconductor signal'),
('el_e27','electronics','Electronic relay control boards',          'relay control board PCB electronic semiconductor'),
('el_e28','electronics','High frequency electronics RF boards',     'RF microwave PCB electronic semiconductor board'),
('el_e29','electronics','Electronic assembly components parts',     'soldering PCB assembly semiconductor components'),
('el_e30','electronics','Wearable electronics components boards',   'wearable PCB sensor semiconductor electronic board')
ON CONFLICT DO NOTHING;

-- ── perishables (extra 20 rows — shipping doc style with cold chain context) ──
INSERT INTO shipment_labels (shipment_id, category_name, cargo_text, commodity_text) VALUES
('pe_e01','perishables','Frozen seafood mixed cold storage',       'frozen seafood cold chain reefer storage'),
('pe_e02','perishables','Chilled fresh produce airfreight',        'fresh produce perishable refrigerated airfreight'),
('pe_e03','perishables','Cold chain meat products frozen',         'frozen meat cold chain refrigerated storage'),
('pe_e04','perishables','Perishable food products reefer',         'perishable food reefer cold chain temperature'),
('pe_e05','perishables','Refrigerated cargo seafood products',     'seafood refrigerated cold chain perishable'),
('pe_e06','perishables','Temperature controlled live animals',     'live animals temperature controlled perishable'),
('pe_e07','perishables','Frozen fish products cold storage',       'fish frozen cold storage reefer seafood'),
('pe_e08','perishables','Dairy cold chain products chilled',       'dairy chilled cold chain refrigerated milk'),
('pe_e09','perishables','Perishable fresh vegetables airfreight',  'vegetables fresh perishable temperature controlled'),
('pe_e10','perishables','Chilled meat export frozen cold',        'meat export chilled frozen cold chain'),
('pe_e11','perishables','Seafood fresh frozen cold shipment',      'seafood cold chain frozen fresh reefer'),
('pe_e12','perishables','IQF frozen products cold chain export',   'IQF frozen cold chain export perishable'),
('pe_e13','perishables','Reefer container perishable goods',       'reefer perishable temperature controlled cargo'),
('pe_e14','perishables','Chilled products cold storage export',    'chilled cold storage refrigerated perishable'),
('pe_e15','perishables','Live seafood temperature controlled tank','live seafood temperature controlled tank perishable'),
('pe_e16','perishables','Cold chain fruit fresh export',           'fruit fresh cold chain export perishable'),
('pe_e17','perishables','Frozen poultry cold chain storage',       'poultry frozen cold chain refrigerated storage'),
('pe_e18','perishables','Perishable goods reefer refrigerated',    'perishable goods reefer refrigerated cold chain'),
('pe_e19','perishables','Temperature sensitive cargo cold',        'temperature sensitive cold chain perishable cargo'),
('pe_e20','perishables','Fresh produce cold chain transport',      'fresh produce cold chain refrigerated transport')
ON CONFLICT DO NOTHING;

-- ── food_beverages (extra 20 rows — packaged / ambient food, no cold chain) ───
INSERT INTO shipment_labels (shipment_id, category_name, cargo_text, commodity_text) VALUES
('fb_e01','food_beverages','Packaged food products assorted',       'packaged food beverage consumer goods'),
('fb_e02','food_beverages','Canned goods food beverages',          'canned food beverage packaged consumer'),
('fb_e03','food_beverages','Bottled beverages drinks export',      'bottled beverage drink export packaged'),
('fb_e04','food_beverages','Dry food products bulk packed',        'dry food grain flour bulk packaged'),
('fb_e05','food_beverages','Snack food products packaged goods',   'snack food packaged consumer goods'),
('fb_e06','food_beverages','Beverage drinks canned bottled',       'beverage canned bottled drink food'),
('fb_e07','food_beverages','Ambient food grocery products',        'ambient food grocery packaged consumer'),
('fb_e08','food_beverages','Food ingredients bulk powder',         'food ingredient bulk powder packaged'),
('fb_e09','food_beverages','Processed food products ambient',      'processed food ambient packaged grocery'),
('fb_e10','food_beverages','Consumer food goods packaged',         'consumer food packaged goods beverage'),
('fb_e11','food_beverages','Soft drinks carbonated bottled',       'soft drink carbonated bottled beverage'),
('fb_e12','food_beverages','Confectionery sweets packaged',        'confectionery sweets chocolate packaged food'),
('fb_e13','food_beverages','Cereal grains packaged food',          'cereal grain packaged food consumer'),
('fb_e14','food_beverages','Sauce condiment bottled food',         'sauce condiment bottled food packaged'),
('fb_e15','food_beverages','Food supplement health products',      'supplement health food packaged consumer'),
('fb_e16','food_beverages','Instant food products packaged',       'instant food noodle packaged consumer'),
('fb_e17','food_beverages','Tea coffee packaged beverages',        'tea coffee beverage packaged food'),
('fb_e18','food_beverages','Biscuit cookie bakery packaged',       'biscuit cookie bakery food packaged'),
('fb_e19','food_beverages','Sports nutrition drink packaged',      'sports drink nutrition packaged food beverage'),
('fb_e20','food_beverages','Cooking ingredient oil packaged',      'cooking oil ingredient food packaged')
ON CONFLICT DO NOTHING;

-- ── chemicals (extra 20 rows) ─────────────────────────────────────────────────
INSERT INTO shipment_labels (shipment_id, category_name, cargo_text, commodity_text) VALUES
('ch_e01','chemicals','Hazardous chemical goods industrial',    'hazmat chemical industrial MSDS dangerous'),
('ch_e02','chemicals','Industrial chemical solvent drums',      'solvent chemical industrial drum flammable'),
('ch_e03','chemicals','Chemical compounds industrial grade',    'chemical compound industrial grade hazmat'),
('ch_e04','chemicals','Dangerous goods chemical cargo',         'dangerous goods DG chemical hazmat UN'),
('ch_e05','chemicals','Industrial acid chemical corrosive',     'acid chemical corrosive industrial MSDS'),
('ch_e06','chemicals','Chemical reagent laboratory grade',      'reagent chemical laboratory analytical grade'),
('ch_e07','chemicals','Hazmat chemical flammable liquid',       'flammable liquid chemical hazmat UN number'),
('ch_e08','chemicals','Chemical oxidizer industrial drums',     'oxidizer chemical industrial drum hazmat'),
('ch_e09','chemicals','Solvent thinner paint chemical',         'paint thinner solvent chemical flammable'),
('ch_e10','chemicals','Industrial chemical cleaning agent',     'cleaning chemical industrial agent solvent'),
('ch_e11','chemicals','Toxic chemical substance drums',         'toxic substance chemical drum industrial'),
('ch_e12','chemicals','Chemical raw material industrial',       'chemical raw material industrial production'),
('ch_e13','chemicals','Adhesive chemical resin industrial',     'adhesive resin chemical industrial epoxy'),
('ch_e14','chemicals','Chemical gas compressed industrial',     'compressed gas chemical industrial cylinder'),
('ch_e15','chemicals','Chemical powder industrial bulk',        'chemical powder bulk industrial grade'),
('ch_e16','chemicals','Bleaching agent chlorine chemical',      'bleach chlorine chemical industrial oxidizer'),
('ch_e17','chemicals','Polymer resin chemical industrial',      'polymer resin chemical plastic industrial'),
('ch_e18','chemicals','Chemical fertilizer industrial grade',   'fertilizer chemical industrial nitrogen'),
('ch_e19','chemicals','Organic chemical compound industrial',   'organic chemical compound industrial solvent'),
('ch_e20','chemicals','Chemical mixture hazardous industrial',  'mixture chemical hazardous industrial MSDS')
ON CONFLICT DO NOTHING;

-- ── machinery (extra 20 rows) ─────────────────────────────────────────────────
INSERT INTO shipment_labels (shipment_id, category_name, cargo_text, commodity_text) VALUES
('ma_e01','machinery','Industrial machinery equipment heavy',    'industrial machinery heavy equipment plant'),
('ma_e02','machinery','Manufacturing equipment machine parts',   'manufacturing machine equipment parts industrial'),
('ma_e03','machinery','Heavy equipment plant machinery',         'heavy plant machinery industrial equipment'),
('ma_e04','machinery','Industrial pump compressor equipment',    'pump compressor industrial equipment machinery'),
('ma_e05','machinery','Machine tools industrial equipment',      'machine tool industrial equipment manufacturing'),
('ma_e06','machinery','Processing equipment industrial plant',   'processing equipment industrial plant machinery'),
('ma_e07','machinery','Factory equipment machinery parts',       'factory machinery parts industrial equipment'),
('ma_e08','machinery','Construction machinery heavy equipment',  'construction machinery heavy equipment plant'),
('ma_e09','machinery','Agricultural machinery farming equipment','agricultural machinery farming equipment parts'),
('ma_e10','machinery','Mining equipment machinery industrial',   'mining equipment machinery industrial heavy'),
('ma_e11','machinery','Packaging machinery equipment industrial','packaging machine equipment industrial plant'),
('ma_e12','machinery','Conveyor belt machinery equipment',       'conveyor belt machinery industrial equipment'),
('ma_e13','machinery','Generator turbine industrial equipment',  'generator turbine industrial power equipment'),
('ma_e14','machinery','Crane hoist industrial lifting machine',  'crane hoist lifting industrial machinery'),
('ma_e15','machinery','Hydraulic machinery industrial system',   'hydraulic system industrial machinery equipment'),
('ma_e16','machinery','Textile machinery equipment industrial',  'textile machinery equipment industrial plant'),
('ma_e17','machinery','Food machinery processing equipment',     'food processing machinery equipment industrial'),
('ma_e18','machinery','Welding equipment industrial machinery',  'welding equipment industrial machinery plant'),
('ma_e19','machinery','Printing machinery equipment industrial', 'printing machine industrial equipment plant'),
('ma_e20','machinery','Automation industrial robot equipment',   'automation robot industrial equipment machinery')
ON CONFLICT DO NOTHING;

-- ── metals (extra 20 rows) ────────────────────────────────────────────────────
INSERT INTO shipment_labels (shipment_id, category_name, cargo_text, commodity_text) VALUES
('me_e01','metals','Steel products metal cargo bulk',          'steel metal product bulk cargo'),
('me_e02','metals','Aluminium metal products bulk',            'aluminium metal product bulk industrial'),
('me_e03','metals','Metal products structural steel',          'structural steel metal product industrial'),
('me_e04','metals','Iron steel metal goods cargo',             'iron steel metal goods cargo bulk'),
('me_e05','metals','Metal alloy products industrial',          'alloy metal industrial product bulk'),
('me_e06','metals','Copper metal products bulk cargo',         'copper metal product bulk industrial'),
('me_e07','metals','Stainless steel metal products',           'stainless steel metal product industrial'),
('me_e08','metals','Metal scrap recycled cargo bulk',          'metal scrap recycled bulk cargo'),
('me_e09','metals','Steel coil sheet metal products',          'steel coil sheet metal product bulk'),
('me_e10','metals','Metal pipe tube structural products',      'pipe tube metal structural product'),
('me_e11','metals','Galvanized metal products zinc coated',    'galvanized metal product zinc industrial'),
('me_e12','metals','Metal ingot primary commodity bulk',       'ingot primary metal commodity bulk'),
('me_e13','metals','Steel bar rod metal products',             'steel bar rod metal product bulk'),
('me_e14','metals','Non-ferrous metal products cargo',         'non-ferrous metal product cargo bulk'),
('me_e15','metals','Metal fasteners hardware products',        'fastener hardware metal product industrial'),
('me_e16','metals','Cast metal products foundry goods',        'cast metal foundry product industrial'),
('me_e17','metals','Metal wire cable products industrial',     'wire cable metal product industrial'),
('me_e18','metals','Forged metal parts components',            'forged metal part component industrial'),
('me_e19','metals','Metal foil sheet thin gauge',              'foil sheet metal thin gauge product'),
('me_e20','metals','Precious metal refined product',          'precious metal refined product silver gold')
ON CONFLICT DO NOTHING;
