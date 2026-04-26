"""
HS code extraction regex — format-by-format pins.

Exercises `preprocess.extract_signals` against the anchor + digit patterns.
Each case fixes one surface format we want to keep supporting. New dotted
variants (H.S.CODE:, H.S.: dotted-digits) were added in the triage pass; the
plain HS / HTS / tariff / no-space cases are regression pins — must stay green.

Pure-function tests. No DB, no embeddings.
"""
from __future__ import annotations

import pytest

from preprocess import extract_signals


@pytest.mark.parametrize(
    "text,expected",
    [
        # Regression pins — existing formats must keep working.
        ("HS code# 870323", ["870323"]),
        ("HS CODE: 8421.99", ["842199"]),
        ("HS CODE:8426199000", ["8426199000"]),
        # New coverage — dotted anchor variants that `\bhs\b` failed to match.
        ("H.S.CODE: 24031900", ["24031900"]),
        ("H.S.: 09.01.11.90.00", ["0901119000"]),
        # Negative cases.
        ("random text no codes", []),
        ("tariff 1234", []),  # <6 digits, under the minimum
    ],
)
def test_hs_extraction_formats(text: str, expected: list[str]) -> None:
    signals = extract_signals(text)
    assert signals.hs_codes == expected


def test_multiple_codes_after_one_anchor() -> None:
    """Comma-separated codes after a single anchor should all be captured."""
    signals = extract_signals("HS Code: 090961, 090619, and 090620")
    assert signals.hs_codes == ["090961", "090619", "090620"]


def test_idempotent() -> None:
    """Extracting twice returns the same set."""
    text = "H.S.CODE: 24031900 and HS: 870323"
    assert extract_signals(text).hs_codes == extract_signals(text).hs_codes


def test_empty_input() -> None:
    assert extract_signals("").hs_codes == []


def test_hs_anchor_does_not_match_prefix_words() -> None:
    """Anchor must not trigger on words that start with 'hs' (HSBC, HSN)."""
    assert extract_signals("HSBC bank transfer 123456").hs_codes == []
