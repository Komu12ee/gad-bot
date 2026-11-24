"""
Document Number Normalization and Candidate Extraction
Spec: Section 7.4
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Document number cue prefix regex
NUMBER_CUES = re.compile(
    r"\b(क्रमांक|क\.|क्र\.|संख्या|फा\.\s*क्र\.|f\.?\s*no\.?|order\s*no\.?|memo\s*no\.?|gr\s*no\.?|no\.?|file\s*no\.?)\b[:\s\-–.]*",
    re.IGNORECASE | re.UNICODE
)

# Number pattern following cues or standalone
DOC_NUMBER_PATTERN = re.compile(
    r"(?:\b(?:क्रमांक|क\.|क्र\.|संख्या|फा\.\s*क्र\.|f\.?\s*no\.?|order\s*no\.?|memo\s*no\.?|gr\s*no\.?|no\.?|file\s*no\.?)\b[:\s\-–.]*)"
    r"([A-Za-z0-9\u0900-\u097F][A-Za-z0-9\u0900-\u097F/\-\.\(\)\s]{3,40})",
    re.IGNORECASE | re.UNICODE
)

# Standalone patterns like F 1-1/2012/1-3 or ESTB-101(1)/7/2024-GAD-4
STANDALONE_PATTERN = re.compile(
    r"\b([A-Za-z\u0900-\u097F]{1,8}[-/ ]?\d{1,4}[/\-][^\s,;]{1,20}[/\-]\d{1,4}(?:[-/][A-Za-z0-9\u0900-\u097F]+)?)\b",
    re.IGNORECASE | re.UNICODE
)


PREFIX_STRIP_REGEX = re.compile(
    r"^(?:फा\.\s*क्र\.?|फा\.?|क्रमांक|क\.|क्र\.?|संख्या|no\.?|number|order\s*no\.?|memo\s*no\.?|file\s*no\.?|f\.?\s*no\.?)[\s:\-–.]*",
    re.IGNORECASE | re.UNICODE
)


def normalize_number(s: str) -> str:
    """Normalize document number string according to Section 7.4."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s).translate(DEV_DIGITS).strip()
    # Strip common leading prefixes
    s = PREFIX_STRIP_REGEX.sub("", s)
    # Also strip any internal prefix like 'no.' or 'क्र.'
    s = re.sub(r"(?:^|[\s/])(?:क्रमांक|क\.|क्र\.|संख्या|no\.?|number)[\s:\-–.]*", "", s, flags=re.I)
    s = re.sub(r"[\s]+", "", s)                  # drop spaces
    s = re.sub(r"[\\|]", "/", s)                 # unify separators
    s = s.replace("–", "-").replace("—", "-").rstrip(".,;:\"' ")
    # Transliterate leading Hindi file prefix 'एफ' -> 'F' for consistent linking
    if s.startswith("एफ"):
        s = "F" + s[2:]
    return s.upper()


def extract_number_candidates(line: str, line_no: int, text_context: str = "") -> List[Dict[str, Any]]:
    """Extract candidate document numbers from a single line."""
    candidates = []
    seen_norms = set()

    # Search for cued numbers
    for m in DOC_NUMBER_PATTERN.finditer(line):
        raw_val = m.group(1).strip()
        # Clean trailing noise words like 'प्रति', 'दिनांक', etc.
        raw_val = re.split(r"(?i)\b(दिनांक|dated|प्रति|विषय|subject)\b", raw_val)[0].strip()
        norm = normalize_number(raw_val)
        if len(norm) >= 4 and any(c.isdigit() for c in norm):
            if norm not in seen_norms:
                seen_norms.add(norm)
                candidates.append({
                    "raw": raw_val,
                    "normalized": norm,
                    "line_no": line_no,
                    "context": line.strip()[:150],
                })

    # Search for standalone complex patterns
    for m in STANDALONE_PATTERN.finditer(line):
        raw_val = m.group(1).strip()
        norm = normalize_number(raw_val)
        if len(norm) >= 5 and any(c.isdigit() for c in norm):
            if norm not in seen_norms:
                seen_norms.add(norm)
                candidates.append({
                    "raw": raw_val,
                    "normalized": norm,
                    "line_no": line_no,
                    "context": line.strip()[:150],
                })

    return candidates
