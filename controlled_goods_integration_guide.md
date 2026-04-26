# Controlled Goods Data Integration Guide

## 🎯 **Integration Strategy Overview**

Your controlled goods JSON contains **81,426 controlled item entries** from **92 regulations** (US, EU, AU, UK, etc.). This is valuable for enhancing your risk screening, but requires careful integration to avoid breaking your current system.

### **Key Insight**: 
This data is for **prohibited/controlled items** (what to flag as risky), not normal goods (what you need for positive anchors). It should **enhance compliance detection**, not fix classification consistency.

---

## 🔧 **Integration Option 1: Enhanced Hard_Negatives (Recommended)**

### **Current Problem**:
```json
"electronics": {
  "hard_negatives": ["battery", "lithium", "phone"]  // Limited patterns
}
```

### **Enhanced with Controlled Goods**:
```json
"electronics": {
  "hard_negatives": [
    "battery", "lithium", "phone",  // Current patterns
    
    // From controlled goods data:
    "enterprise resource planning software military",
    "software specially designed nuclear",
    "encryption software cryptographic",
    "radar jamming equipment",
    "night vision equipment military",
    "thermal imaging camera dual-use"
  ]
}
```

### **Implementation Steps**:

#### **Step 1: Extract Controlled Keywords by Category**
```python
def extract_controlled_patterns():
    controlled_goods = load_json("ControlledGoods_08212025 2.json")
    
    # Map controlled categories to shipment categories
    CATEGORY_MAPPING = {
        # Electronics-related controlled items
        "electronics": [
            "radar", "electronic", "computer", "software", "encryption", 
            "thermal imaging", "night vision", "semiconductor"
        ],
        
        # Chemical-related controlled items  
        "chemicals": [
            "chemical warfare", "toxic", "nerve agent", "radioactive",
            "uranium", "plutonium", "nuclear material"
        ],
        
        # Automotive-related controlled items
        "automotive": [
            "armored vehicle", "military vehicle", "tank",
            "missile guidance", "navigation military"
        ]
    }
    
    enhanced_patterns = {}
    
    for regulation in controlled_goods["Root"]:
        for good in regulation["Good"]:
            good_text = good["GoodText"].lower()
            keywords = [k.lower() for k in good.get("Keywords", [])]
            
            # Check which shipment category this controlled item relates to
            for category, patterns in CATEGORY_MAPPING.items():
                if any(pattern in good_text for pattern in patterns):
                    if category not in enhanced_patterns:
                        enhanced_patterns[category] = set()
                    
                    # Add controlled phrases
                    enhanced_patterns[category].add(good_text[:100])  # Truncate long descriptions
                    enhanced_patterns[category].update(keywords)
    
    return enhanced_patterns
```

#### **Step 2: Integrate into Risk Profile**
```python
def enhance_risk_profile(current_risk_profile, controlled_patterns):
    for category in current_risk_profile["categories"]:
        if category in controlled_patterns:
            # Add controlled patterns to existing hard_negatives
            existing = set(current_risk_profile["categories"][category]["hard_negatives"])
            controlled = controlled_patterns[category]
            
            # Combine and deduplicate
            enhanced = list(existing | controlled)
            current_risk_profile["categories"][category]["hard_negatives"] = enhanced
    
    return current_risk_profile
```

---

## 🔧 **Integration Option 2: HS Code Cross-Validation**

### **Purpose**: Validate if extracted HS codes match controlled items

### **Implementation**:
```python
def build_controlled_hs_lookup():
    """Extract all HS codes from controlled goods data"""
    controlled_goods = load_json("ControlledGoods_08212025 2.json")
    controlled_hs = {}
    
    for regulation in controlled_goods["Root"]:
        regulation_name = regulation["RegulationName"]
        
        for good in regulation["Good"]:
            for identification in good.get("Identifications", []):
                if identification["IdentificationTypeName"] == "HS Code":
                    hs_code = identification["IdentifierCode"]
                    
                    if hs_code not in controlled_hs:
                        controlled_hs[hs_code] = []
                    
                    controlled_hs[hs_code].append({
                        "regulation": regulation_name,
                        "description": good["GoodText"][:100],
                        "reference": good["Reference"]
                    })
    
    return controlled_hs

def check_hs_against_controlled(extracted_hs_codes, controlled_hs_lookup):
    """Check if any extracted HS codes match controlled items"""
    hits = []
    
    for hs_code in extracted_hs_codes:
        # Check exact match
        if hs_code in controlled_hs_lookup:
            hits.append({
                "hs_code": hs_code,
                "match_type": "exact",
                "controlled_items": controlled_hs_lookup[hs_code]
            })
        
        # Check chapter-level match (first 2-4 digits)
        for controlled_hs in controlled_hs_lookup.keys():
            if hs_code[:4] == controlled_hs[:4] and hs_code != controlled_hs:
                hits.append({
                    "hs_code": hs_code,
                    "match_type": "chapter",
                    "controlled_hs": controlled_hs,
                    "controlled_items": controlled_hs_lookup[controlled_hs]
                })
    
    return hits
```

### **Integration into Classification Pipeline**:
```python
def classify_with_controlled_validation(cargo_text, risk_profile):
    # Step 1: Normal classification
    classification_result = classify_shipment(cargo_text, risk_profile)
    
    # Step 2: Extract HS codes
    processed = preprocess(cargo_text)
    extracted_hs = processed.hs_codes
    
    # Step 3: Check against controlled goods
    controlled_hits = check_hs_against_controlled(extracted_hs, CONTROLLED_HS_LOOKUP)
    
    # Step 4: Enhance decision
    if controlled_hits:
        classification_result.compliance_flags.append("controlled_hs_match")
        classification_result.controlled_hits = controlled_hits
        
        # Escalate compliance status if exact match
        exact_matches = [h for h in controlled_hits if h["match_type"] == "exact"]
        if exact_matches:
            classification_result.compliance_status = "BLOCK"
    
    return classification_result
```

---

## 🔧 **Integration Option 3: Separate Controlled Goods Layer**

### **Architecture**: Run controlled goods screening parallel to category classification

```python
@dataclass
class EnhancedDecisionResult:
    # Classification dimension
    category: str
    classification_confidence: str
    classification_source: str
    
    # Controlled goods dimension  
    controlled_status: Literal["clear", "review", "block"]
    controlled_hits: list
    controlled_regulations: list
    
    # Combined compliance
    final_status: str
    
def screen_controlled_goods(cargo_text):
    """Dedicated controlled goods screening"""
    hits = []
    
    # Text-based screening
    for regulation in CONTROLLED_GOODS["Root"]:
        for good in regulation["Good"]:
            if text_matches_controlled_item(cargo_text, good):
                hits.append({
                    "regulation": regulation["RegulationName"],
                    "good_reference": good["ListGoodReference"],
                    "description": good["GoodText"],
                    "match_reason": "text_semantic"
                })
    
    # HS-based screening
    hs_hits = check_hs_against_controlled(extract_hs_codes(cargo_text))
    hits.extend(hs_hits)
    
    # Determine status
    if any(is_blocking_regulation(h) for h in hits):
        status = "block"
    elif hits:
        status = "review"
    else:
        status = "clear"
    
    return ControlledScreeningResult(
        status=status,
        hits=hits,
        regulations=list(set(h["regulation"] for h in hits))
    )

def enhanced_classification(cargo_text, risk_profile):
    # Run both layers
    category_result = classify_shipment(cargo_text, risk_profile)
    controlled_result = screen_controlled_goods(cargo_text)
    
    # Combine results
    final_status = determine_final_status(category_result, controlled_result)
    
    return EnhancedDecisionResult(
        category=category_result.category,
        classification_confidence=category_result.confidence,
        classification_source=category_result.source,
        controlled_status=controlled_result.status,
        controlled_hits=controlled_result.hits,
        controlled_regulations=controlled_result.regulations,
        final_status=final_status
    )
```

---

## ⚠️ **Critical Implementation Warnings**

### **1. False Positive Risk**
```python
# DANGEROUS - Too broad matching
def bad_matching(cargo_text, controlled_item):
    return controlled_item["Keywords"][0] in cargo_text  # Will flag everything!

# BETTER - Context-aware matching  
def good_matching(cargo_text, controlled_item):
    good_text = controlled_item["GoodText"]
    
    # Require multiple keyword matches
    keywords = controlled_item.get("Keywords", [])
    matches = sum(1 for k in keywords if k.lower() in cargo_text.lower())
    
    # Require threshold based on specificity
    if "nuclear" in good_text or "military" in good_text:
        return matches >= 2  # High-risk terms need confirmation
    else:
        return matches >= 3  # General terms need more evidence
```

### **2. Category Mapping Challenges**
```python
# Map controlled categories to shipment categories carefully
CONTROLLED_TO_SHIPMENT_CATEGORY = {
    # Clear mappings
    "Nuclear": None,  # No normal nuclear shipments expected
    "Munitions": None,  # No normal munitions expected
    "Dual Use Electronics": "electronics",
    "Chemical Weapons": "chemicals",
    
    # Ambiguous mappings (be careful)
    "Software": "electronics",  # But normal software exists!
    "Vehicles": "automotive",   # But normal vehicles exist!
}
```

### **3. Performance Considerations**
```python
# SLOW - Check every controlled item for every shipment
def slow_screening(cargo_text):
    for regulation in controlled_goods["Root"]:  # 92 regulations
        for good in regulation["Good"]:  # 81k items
            if matches(cargo_text, good):  # Expensive for each
                return "hit"

# FAST - Build indexed lookup tables
def fast_screening(cargo_text):
    # Pre-built keyword index
    keywords = extract_keywords(cargo_text)
    potential_hits = KEYWORD_INDEX.lookup(keywords)  # Fast lookup
    
    # Only check potential matches
    for hit in potential_hits:
        if detailed_match(cargo_text, hit):
            return hit
```

---

## 📊 **Recommended Implementation Plan**

### **Phase 1: Safe Integration (Immediate)**
1. **Extract obvious prohibited patterns** from controlled goods
2. **Add to existing hard_negatives** with conservative matching
3. **Test on current 99-shipment set** to ensure no new false positives
4. **Focus categories**: electronics, chemicals (high controlled goods overlap)

### **Phase 2: HS Validation (Short-term)** 
1. **Build controlled HS lookup table** from all regulations
2. **Add HS cross-validation** to existing HS-first classification
3. **Flag exact HS matches** for human review
4. **Monitor false positive rate**

### **Phase 3: Full Integration (Long-term)**
1. **Separate controlled goods screening layer**
2. **Multi-jurisdictional risk scoring**
3. **Advanced semantic matching** with LLM assistance
4. **Integration with external sanctions APIs**

---

## 🎯 **Key Success Metrics**

### **Positive Outcomes (Measure These)**
- **Increased controlled goods detection** without false positives
- **Better HS code validation** for high-risk items  
- **Multi-jurisdictional compliance coverage**

### **Risks to Monitor**
- **False positive rate increase** (normal goods flagged as controlled)
- **System performance degradation** (81k controlled items = large search space)
- **Classification accuracy regression** (don't break existing category accuracy)

---

## ✅ **Bottom Line Recommendation**

### **DO Use Controlled Goods Data For**:
✅ **Enhancing hard_negatives** with obvious prohibited patterns  
✅ **HS code cross-validation** against controlled items  
✅ **Building compliance screening layer** (separate from classification)  

### **DON'T Use Controlled Goods Data For**:
❌ **Fixing classification consistency** (it's prohibited items, not normal goods)  
❌ **Replacing positive anchors** (still need normal goods examples)  
❌ **Primary category classification** (will over-flag legitimate items)

**Priority Order**: 
1. **Fix positive anchor problem** (main classification bug)
2. **Enhance hard_negatives** with controlled goods (compliance improvement)  
3. **Add HS validation** (advanced feature)

**The controlled goods data is excellent for compliance, but your main bug (same commodity → different categories) still needs positive anchors for normal goods.**
