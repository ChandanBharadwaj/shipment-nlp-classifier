"""
Text preprocessing for shipment descriptions.

Applied identically at centroid build time and inference time so centroids
and queries share the same normalized distribution.

Scope (deliberately conservative):
    - lowercase
    - strip shipping boilerplate (pallets, container sizes, SKU codes, HS codes,
      "as per invoice", etc.)
    - expand a small set of common shipping-industry abbreviations
    - collapse whitespace

Not done:
    - number masking — batch sizes (`18650 cells`), units (`500g`), and HS-like
      codes often carry real signal. Stripping numbers hurt recall in a quick
      test.
    - stemming/lemmatization — handled instead at keyword-match time where it
      matters most (see classifier.keyword_score).
"""

from __future__ import annotations

import os
import re

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
]

# ── Abbreviation expansion ────────────────────────────────────────────────────
# Small hand-curated list — the goal is to rescue obvious misses, not build a
# full synonym graph. Expanded form is what the embedding model was trained on.
_ABBREVIATIONS: dict[str, str] = {
    "refr":       "refrigerated",
    "refrig":     "refrigerated",
    "frz":        "frozen",
    "elec":       "electronic",
    "electr":     "electronic",
    "pharma":     "pharmaceutical",
    "pharm":      "pharmaceutical",
    "choco":      "chocolate",
    "mfg":        "manufactured",
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
    "mfd":        "manufactured",
    "ss":         "stainless steel",
    "ms":         "mild steel",
}

# Compile once: `\b(refr|refrig|…)\b` with case-insensitive matching.
_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_WS_PATTERN = re.compile(r"\s+")


def normalize(text: str) -> str:
    """
    Normalize a single shipment text field.

    Deterministic and idempotent: normalize(normalize(x)) == normalize(x).
    """
    if not text:
        return ""

    s = text.lower()

    # Boilerplate removal
    for pat in _BOILERPLATE_PATTERNS:
        s = pat.sub(" ", s)

    # Abbreviation expansion
    s = _ABBREV_PATTERN.sub(lambda m: _ABBREVIATIONS[m.group(1).lower()], s)

    # Collapse whitespace
    s = _WS_PATTERN.sub(" ", s).strip()
    return s


def _is_e5_model() -> bool:
    """True when the configured embedding model is an intfloat/e5* checkpoint."""
    name = os.environ.get("EMBEDDING_MODEL", "").lower()
    # Fallback to the default baked into classifier.MODEL_NAME — we can't import
    # classifier here (circular), but the env var is the authoritative override
    # at runtime and the branch's default is also an e5 checkpoint.
    if not name:
        try:
            from classifier import MODEL_NAME  # type: ignore
            name = MODEL_NAME.lower()
        except Exception:
            return False
    return name.startswith("intfloat/e5")


def build_query_text(cargo: str, commodity: str, role: str = "query") -> str:
    """
    Combine cargo + commodity fields into the exact string that gets embedded.

    Plain concatenation after normalization. An earlier version used a
    "Cargo: X. Commodity: Y." structured prefix, but on MiniLM with cosine
    scoring that pushed all texts toward each other in embedding space (the
    shared literal tokens become common features), compressing the margin
    between categories. Plain concat avoids that.

    MUST be used identically at centroid-build time and inference time.

    The `role` parameter only matters for asymmetric-trained models
    (intfloat/e5-*) that were fine-tuned to expect `"query: "` and
    `"passage: "` prefixes. For symmetric models (MiniLM, BGE) the role is
    ignored and the returned string is identical either way. Call sites:
        - centroid_builder.py -> role="passage"
        - classifier.predict / predict_batch -> role="query" (default)
        - fit_calibration.py -> role="query" (validation samples score as queries)
    """
    c = normalize(cargo or "")
    m = normalize(commodity or "")
    combined = f"{c} {m}" if (c and m) else (c or m)

    if _is_e5_model():
        prefix = "passage: " if role == "passage" else "query: "
        return prefix + combined
    return combined
