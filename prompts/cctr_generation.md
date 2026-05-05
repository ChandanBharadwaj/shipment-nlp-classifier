# CCTR generation prompt

Used by `scripts/llm_generate_cctr.py`. Two passes per chapter:
**generate** then **validate / refine**.

The script substitutes `{HS_CHAPTER}`, `{CHAPTER_TITLE}`,
`{CATEGORY_DISPLAY_NAME}`, `{TOP_USITC_DESCRIPTIONS}`,
`{ADJACENT_CHAPTERS}`, `{ALL_CATEGORY_DISPLAY_NAMES}`, and
`{PROPOSED_ROWS}` (pass 2 only) at runtime.

---

## Pass 1 — Generate

```
You are populating a customs-classification keyword registry for HS
chapter {HS_CHAPTER}: "{CHAPTER_TITLE}".
Business category: "{CATEGORY_DISPLAY_NAME}".

Sample HS items in this chapter (from USITC):
{TOP_USITC_DESCRIPTIONS}

Adjacent / commonly-confused chapters: {ADJACENT_CHAPTERS}

All categories in the system: {ALL_CATEGORY_DISPLAY_NAMES}

Produce a JSON response with three arrays. Be conservative — quality
over quantity. Don't invent words you don't recognise as real trade
vocabulary.

1. anchors — single words or short phrases that, if seen in a shipment
   description, are dispositive evidence for THIS chapter and not the
   adjacent ones. Trade names, common product nouns. Examples:
     ch09 (Coffee/Tea/Spices): "tea bags", "instant coffee", "cardamom"
     ch93 (Arms): "m16 rifle", "ak-47", "5.56mm"
   Aim for 3-8 anchors.

2. suppressors — words that, if seen, should PUSH classification AWAY
   from a specific other category. Each entry names which category to
   suppress. Examples:
     {"keyword": "military-grade", "suppress_category": "Toys & Games"}
       — phrase indicates real weapons, not toys
     {"keyword": "stuffed", "suppress_category": "Live Animals & Meat Products"}
       — "stuffed toy" is not livestock
   `suppress_category` MUST be a category display_name from the list
   above. Aim for 1-3 suppressors per chapter; many chapters won't need any.

3. modifiers — context phrases that re-route an ambiguous head_noun to
   a specific HS chapter. Examples:
     {"keyword": "li-ion", "head_noun": "battery", "target_chapter": "85"}
     {"keyword": "lead-acid", "head_noun": "battery", "target_chapter": "85"}
   `target_chapter` MUST be a 2-digit HS chapter ('01'..'97'). Aim for
   0-3 modifiers per chapter; most chapters don't have ambiguous head
   nouns.

Output strict JSON only, no commentary. Schema:

{
  "anchors":     [{"keyword": "..."}, ...],
  "suppressors": [{"keyword": "...", "suppress_category": "..."}, ...],
  "modifiers":   [{"keyword": "...", "head_noun": "...", "target_chapter": "NN"}, ...]
}
```

## Pass 2 — Validate / refine

```
You previously proposed these CCTR rows for HS chapter {HS_CHAPTER}
({CHAPTER_TITLE}, category "{CATEGORY_DISPLAY_NAME}"). Review each row.

Drop rows that are:
- Not real trade vocabulary (hallucinations)
- Too generic (would match in many unrelated chapters)
- Wrong category for the suppressor target
- Wrong target_chapter for the modifier

Keep only rows you are confident about.

Output strict JSON, same schema as input, with only the surviving rows.
Add nothing.

Proposed rows:
{PROPOSED_ROWS}
```
