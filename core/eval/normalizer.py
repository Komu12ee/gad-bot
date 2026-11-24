"""Text, script, and digit normalization utilities for evaluation.

Handles:
- Unicode Normalization Form C (NFC)
- Devanagari numerals (०-९) to ASCII digits (0-9) and vice-versa
- Punctuation & zero-width characters (ZWNJ, ZWJ)
- Normalized tokenization for Devanagari and English
"""
from __future__ import annotations

import re
import unicodedata

# Devanagari numerals translation table
_DEV_TO_ARABIC = str.maketrans("०१२३४५६७८९", "0123456789")
_ARABIC_TO_DEV = str.maketrans("0123456789", "०१२३४५६७८९")

# Zero-width joiner / non-joiner and formatting characters
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
_WS_RE = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    """Normalize text to Unicode NFC and strip zero-width artifacts."""
    if not text:
        return ""
    # Strip zero-width characters first
    cleaned = _ZERO_WIDTH_RE.sub("", str(text))
    # Normalize to canonical composed form
    return unicodedata.normalize("NFC", cleaned).strip()


def normalize_digits(text: str, to_script: str = "arabic") -> str:
    """Normalize digits in text.
    
    Args:
        text: Input string
        to_script: 'arabic' (0-9, default) or 'devanagari' (०-९)
    """
    if not text:
        return ""
    if to_script == "devanagari":
        return text.translate(_ARABIC_TO_DEV)
    return text.translate(_DEV_TO_ARABIC)


def normalize_eval_text(text: str, to_digits: str = "arabic") -> str:
    """Comprehensive normalization for text comparison:
    1. Unicode NFC normalization
    2. Zero-width character stripping
    3. Digit script canonicalization
    4. Consecutive whitespace collapse
    """
    if not text:
        return ""
    norm = normalize_unicode(text)
    norm = normalize_digits(norm, to_script=to_digits)
    norm = _WS_RE.sub(" ", norm)
    return norm.strip()


def extract_numbers(text: str) -> list[str]:
    """Extract all numbers (integers, decimals, year ranges) normalized to ASCII digits."""
    if not text:
        return []
    normalized = normalize_eval_text(text, to_digits="arabic")
    # Matches integers, decimals, and codes containing digits
    return re.findall(r"\b\d+(?:\.\d+)?\b", normalized)

