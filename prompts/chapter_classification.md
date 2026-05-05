# Chapter classification prompt

This is the prompt used by `scripts/llm_classify_chapters.py` to derive
business categories from the 96 active HS chapters. Two passes:
**generate** then **validate**.

The script substitutes `{CHAPTERS_LIST}` and `{PROPOSED_GROUPING}` at runtime.
Keep the prompt in this file (committed) so the inputs to the LLM are
reproducible and reviewable.

---

## Pass 1 — Generate

```
You are designing a business-category taxonomy for a shipment
classification system. The system classifies free-text shipment
descriptions (e.g. "5 pallets of lithium-ion batteries for handheld
radios") against the standard 96-chapter Harmonised System (HS) taxonomy.

Your task: group the 96 HS chapters below into business categories.
Output a JSON object with one array of category groupings.

Rules:
- Number of categories is whatever fits the data. Do NOT pin to a fixed
  count. Use as few or as many as the chapter content warrants — somewhere
  between 12 and 30 typically works.
- Each chapter goes into exactly one category. Every chapter must appear
  in some category.
- Category names are NATURAL LANGUAGE — capitalised, may include spaces
  and punctuation ("&", commas). Examples: "Live Animals & Meat
  Products", "Electronics & Components", "Vegetables, Grains & Cereals".
- Group chapters by what they SELL TO operationally, not by HS section
  numbering. A category should feel like a sensible product line a
  shipping operator would recognise.
- Be conservative with overly-broad categories. "Manufactured Goods"
  spanning 30 chapters helps no one. "Industrial Machinery" + "Vehicles"
  + "Electronics" as separate categories is more useful.

Output strict JSON only, no commentary. Schema:

{
  "categories": [
    {
      "name": "<natural language category name>",
      "hs_chapters": ["01", "02", ...]
    },
    ...
  ]
}

Chapters:
{CHAPTERS_LIST}
```

## Pass 2 — Validate / refine

```
You previously proposed the following category grouping for the 96 HS
chapters. Review it critically.

Drop or merge categories that are too narrow (only 1 chapter). Split
categories that mix unrelated product lines. Move chapters between
categories if they're clearly mis-assigned.

Every chapter must still appear exactly once after your refinement.

Output strict JSON only, same schema as input.

Original chapters:
{CHAPTERS_LIST}

Proposed grouping:
{PROPOSED_GROUPING}
```
