"""
Text preprocessing for shipment descriptions.

Applied identically at centroid build time, inference time, AND when embedding
risk phrases for the compliance screen — so centroids, queries, and risk
vectors all share the same normalized distribution.

Scope (deliberately conservative):
    - NFKC Unicode normalization (collapse fullwidth/ligatures/compatibility)
    - lowercase
    - strip shipping boilerplate (pallets, container sizes, SKU codes, HS codes,
      "as per invoice", etc.)
    - normalize units of measure / quantity tokens (kg, pcs, ctn, plt, …)
    - expand a small set of common shipping-industry abbreviations
    - collapse runs of 3+ identical consecutive tokens
    - collapse whitespace

Not done:
    - number masking — batch sizes (`18650 cells`), units (`500g`), and HS-like
      codes often carry real signal. Stripping numbers hurt recall in a quick
      test.
    - stemming/lemmatization — handled instead at keyword-match time where it
      matters most (see classifier.keyword_score).

Idempotency invariant: normalize(normalize(x)) == normalize(x).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ── Boilerplate removal ───────────────────────────────────────────────────────
# Each entry is a compiled regex that gets substituted with a single space.
# Order matters only where patterns overlap; keep specific-before-general.
_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bas per invoice\b", re.IGNORECASE),
    re.compile(r"\bas per packing list\b", re.IGNORECASE),
    re.compile(r"\bfreight collect\b", re.IGNORECASE),
    re.compile(r"\bfreight prepaid\b", re.IGNORECASE),
    re.compile(r"\bsaid to contain\b", re.IGNORECASE),
    re.compile(r"\bstc\b", re.IGNORECASE),
    re.compile(r"\bhs\s*code\s*:?\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bhts\s*code\s*:?\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bsku\s*:?\s*[\w-]+\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*(?:x\s*)?\d*\s*ft\s*container\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*(?:pallets?|cartons?|boxes|bags|bales|drums|crates?)\b", re.IGNORECASE),
    re.compile(r"\bfcl\b", re.IGNORECASE),
    re.compile(r"\blcl\b", re.IGNORECASE),
    # VINs (17 chars, alphanumeric, excluding I/O/Q).
    re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b", re.IGNORECASE),
    # Invoice / reference / order numbers.
    re.compile(r"\binvoice\s*(?:no|number|#)?\.?\s*:?\s*[a-z0-9-]+\b", re.IGNORECASE),
    re.compile(r"\bref(?:erence)?\s*(?:no|#)?\.?\s*:?\s*[a-z0-9-]+\b", re.IGNORECASE),
    re.compile(r"\border\s*(?:no|#)?\.?\s*:?\s*[a-z0-9-]+\b", re.IGNORECASE),
    # Phone / fax numbers (tolerate +, spaces, dashes, parens).
    re.compile(r"\btel(?:ephone)?\.?\s*:?\s*[+\d][\d\s\-()]{7,}\b", re.IGNORECASE),
    re.compile(r"\bfax\.?\s*:?\s*[+\d][\d\s\-()]{7,}\b", re.IGNORECASE),
    # "Cargo 1:" / "Cargo 2:" concatenation prefixes from manifest joins.
    re.compile(r"\bcargo\s*\d+\s*:\s*", re.IGNORECASE),
    # ISO 6346 container numbers (4 letters + 7 digits).
    re.compile(r"\b[A-Z]{4}\d{7}\b", re.IGNORECASE),
    # Bill of lading numbers.
    re.compile(r"\bb\s*/?\s*l\s*(?:no)?\.?\s*:?\s*[a-z0-9-]+\b", re.IGNORECASE),
]

# ── Abbreviation expansion ────────────────────────────────────────────────────
# Small hand-curated list — the goal is to rescue obvious misses, not build a
# full synonym graph. Expanded form is what the embedding model was trained on.
#
# Includes shipping UoM / quantity tokens (pcs, kg, ctn, plt, …). Single-letter
# units (g, t, l) are NOT here because they would mangle prose; those live in
# `_UOM_DIGIT_GUARDED` and only match when prefixed by a digit.
_ABBREVIATIONS: dict[str, str] = {
    # Cargo handling
    "refr":       "refrigerated",
    "refrig":     "refrigerated",
    "frz":        "frozen",
    # Sectors / domains
    "elec":       "electronic",
    "electr":     "electronic",
    "pharma":     "pharmaceutical",
    "pharm":      "pharmaceutical",
    "choco":      "chocolate",
    # Manufacturing
    "mfg":        "manufactured",
    "mfd":        "manufactured",
    "mfr":        "manufacturer",
    "assy":       "assembly",
    "comp":       "component",
    "equip":      "equipment",
    "mach":       "machinery",
    "mat":        "material",
    "mats":       "materials",
    "chem":       "chemical",
    "chems":      "chemicals",
    "ind":        "industrial",
    "auto":       "automotive",
    "veh":        "vehicle",
    "agri":       "agriculture",
    "prod":       "product",
    # Materials shorthand
    "ss":         "stainless steel",
    "ms":         "mild steel",
    "alu":        "aluminum",
    "alum":       "aluminum",
    "galv":       "galvanized",
    # UoM / quantity tokens (multi-letter, safe everywhere)
    "pcs":        "pieces",
    "pc":         "pieces",
    "kgs":        "kilograms",
    "kg":         "kilograms",
    "mt":         "metric tons",
    "mts":        "metric tons",
    "lbs":        "pounds",
    "lb":         "pounds",
    "oz":         "ounces",
    "ltr":        "liters",
    "ltrs":       "liters",
    "ml":         "milliliters",
    "gal":        "gallons",
    "gals":       "gallons",
    "qty":        "quantity",
    "nos":        "numbers",
    "ea":         "each",
    "doz":        "dozen",
    "ctn":        "carton",
    "ctns":       "cartons",
    "plt":        "pallet",
    "plts":       "pallets",
    "pkg":        "package",
    "pkgs":       "packages",
    "nw":         "net weight",
    "gw":         "gross weight",
}

# Compile once: `\b(refr|refrig|…)\b` with case-insensitive matching.
# Sorted longest-first so multi-char keys win over their substrings.
_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Single-letter UoM that ONLY make sense when attached to a number, e.g. "500g",
# "2t", "5l". Plain "g"/"t"/"l" elsewhere is prose and must not be touched.
# Pattern: capture the digits, then the unit; rewrite as "<digits> <expansion>".
_UOM_DIGIT_GUARDED: dict[str, str] = {
    "g":  "grams",
    "t":  "metric tons",
    "l":  "liters",
}
_UOM_DIGIT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(" + "|".join(_UOM_DIGIT_GUARDED) + r")\b",
    re.IGNORECASE,
)

_WS_PATTERN = re.compile(r"\s+")


def _expand_uom_digit_guarded(match: re.Match) -> str:
    digits = match.group(1)
    unit = match.group(2).lower()
    return f"{digits} {_UOM_DIGIT_GUARDED[unit]}"


def _dedupe_consecutive(text: str) -> str:
    """
    Collapse runs of 3+ identical consecutive tokens to a single occurrence.

    Shipment descriptions often contain `pallets pallets pallets` style noise
    from concatenated manifests. A run of 2 may be intentional ("very very"),
    but 3+ is almost always boilerplate repetition.

    Idempotent: applying twice is a no-op (a single token never matches).
    """
    if not text:
        return text
    tokens = text.split(" ")
    if len(tokens) < 3:
        return text
    out: list[str] = []
    run_token: str | None = None
    run_len = 0
    for tok in tokens:
        if tok == run_token:
            run_len += 1
            if run_len <= 2:
                out.append(tok)
            # else: drop; we've already emitted two
        else:
            run_token = tok
            run_len = 1
            out.append(tok)
    return " ".join(out)


def normalize(text: str) -> str:
    """
    Normalize a single shipment text field.

    Deterministic and idempotent: normalize(normalize(x)) == normalize(x).
    """
    if not text:
        return ""

    # Unicode compatibility normalization (fullwidth → ascii, ligatures, etc.)
    s = unicodedata.normalize("NFKC", text)

    s = s.lower()

    # Boilerplate removal
    for pat in _BOILERPLATE_PATTERNS:
        s = pat.sub(" ", s)

    # Digit-guarded single-letter UoM (must run before whitespace collapse so
    # patterns like "500 g" still match after the digit).
    s = _UOM_DIGIT_PATTERN.sub(_expand_uom_digit_guarded, s)

    # Multi-letter abbreviation expansion
    s = _ABBREV_PATTERN.sub(lambda m: _ABBREVIATIONS[m.group(1).lower()], s)

    # Collapse whitespace
    s = _WS_PATTERN.sub(" ", s).strip()

    # Drop runs of 3+ identical tokens (after whitespace collapse so token
    # boundaries are clean)
    s = _dedupe_consecutive(s)

    return s


# ── Structured-signal extraction ──────────────────────────────────────────────
# Run BEFORE normalize() so the boilerplate strip doesn't destroy the signal.
# HS/HTS codes are 6- or 10-digit commodity codes; we accept 2–10 digits and
# downstream callers trim to the 2-digit chapter for category lookup.

# Locate a "HS code: ..." style anchor; used to identify the starting offset.
_HS_ANCHOR_RE = re.compile(
    r"\b(?:hs|hts|tariff)\s*(?:code|codes|no|number|#)?\.?\s*:?\s*",
    re.IGNORECASE,
)
# Digit runs that look like HS/HTS codes: 6, 8, or 10 digits (WCO HS-6, US
# HTS-8/10). 4-digit subheadings are too short to disambiguate from phone /
# invoice fragments; we start at 6 and require the anchor to be nearby.
_HS_DIGITS_RE = re.compile(r"\b(\d{6}|\d{8}|\d{10})\b")


@dataclass
class ExtractedSignals:
    """Structured signals pulled from raw text before normalization.

    ``hs_codes`` are raw digits (e.g. "30049090"). Callers derive the 2-digit
    chapter with ``code[:2]`` for HS→category lookup.
    """
    hs_codes: list[str] = field(default_factory=list)


def extract_signals(text: str) -> ExtractedSignals:
    """Pull HS/HTS codes out of raw text.

    Must run BEFORE ``normalize()`` — the boilerplate regex strips `hs code: 123`
    patterns to a single space, destroying the signal.

    Idempotent: extracting twice returns the same set of codes.
    """
    if not text:
        return ExtractedSignals()

    # Unicode-normalize so fullwidth digits (if any) become ASCII before regex.
    s = unicodedata.normalize("NFKC", text)

    codes: list[str] = []
    seen: set[str] = set()

    # Walk each HS/HTS/tariff anchor and grab digit runs that follow within a
    # generous window (handles "HS Code: 090961, 090619, and 090620").
    for anchor in _HS_ANCHOR_RE.finditer(s):
        window = s[anchor.end(): anchor.end() + 200]
        # Stop the window at sentence punctuation that would separate sections.
        split = re.search(r"[.;\n]", window)
        if split:
            window = window[: split.start()]
        for m in _HS_DIGITS_RE.finditer(window):
            code = m.group(1)
            if code not in seen:
                seen.add(code)
                codes.append(code)

    return ExtractedSignals(hs_codes=codes)


def build_query_text(cargo: str, commodity: str) -> str:
    """
    Combine cargo + commodity fields into the exact string that gets embedded.

    Plain concatenation after normalization. An earlier version used a
    "Cargo: X. Commodity: Y." structured prefix, but on MiniLM with cosine
    scoring that pushed all texts toward each other in embedding space (the
    shared literal tokens become common features), compressing the margin
    between categories. Plain concat avoids that.

    MUST be used identically at centroid-build time and inference time.
    """
    c = normalize(cargo or "")
    m = normalize(commodity or "")
    if c and m:
        return f"{c} {m}"
    return c or m
