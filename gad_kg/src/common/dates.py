"""
Date Parsing and Normalization Utilities
Spec: Section 7.3
"""

import re
import unicodedata
from datetime import date
from typing import Optional, Dict, Any, List, Tuple

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

MONTHS = {
    # English
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    # Hindi
    "जनवरी": 1, "फरवरी": 2, "फ़रवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5,
    "जून": 6, "जुलाई": 7, "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10,
    "अक्तूबर": 10, "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12,
    # Marathi
    "जानेवारी": 1, "फेब्रुवारी": 2, "एप्रिल": 4, "मे": 5, "जुलै": 7,
    "ऑगस्ट": 8, "सप्टेंबर": 9, "ऑक्टोबर": 10, "नोव्हेंबर": 11, "डिसेंबर": 12,
}

DATE_NUMERIC = re.compile(r"\b(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{2,4})\b")
DATE_TEXT = re.compile(r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s+([A-Za-z\u0900-\u097F]+)[.,]?\s+(\d{4})\b")
DATE_TEXT_REVERSE = re.compile(r"\b([A-Za-z\u0900-\u097F]+)\s+(\d{1,2})\s*[.,]?\s*(\d{4})\b")
FIN_YEAR = re.compile(r"\b(\d{4})\s*[-–]\s*(\d{2,4})\b")


def parse_year(year_str: str) -> Tuple[int, bool]:
    """Parse 2-digit or 4-digit year. Returns (year, is_two_digit)."""
    y = int(year_str)
    if len(year_str) == 2 or y < 100:
        is_two_digit = True
        y = 1900 + y if y >= 50 else 2000 + y
    else:
        is_two_digit = False
    return y, is_two_digit


def validate_and_format_date(day: int, month: int, year: int) -> Optional[str]:
    """Validate date and return YYYY-MM-DD or None."""
    try:
        dt = date(year, month, day)
        # Indian government documents: valid range ~1950 to 2030
        if 1950 <= dt.year <= 2035:
            return dt.isoformat()
    except ValueError:
        pass
    return None


def extract_dates(text: str) -> List[Dict[str, Any]]:
    """Extract candidate dates from text with metadata and flags."""
    results = []
    seen_spans = set()

    # Pre-clean text for date matching (ASCII digits)
    search_text = text.translate(DEV_DIGITS)

    # 1. Textual dates: e.g. 15 August 2021 or 15 अगस्त 2021
    for m in DATE_TEXT.finditer(search_text):
        day_str, month_str, year_str = m.groups()
        month_key = month_str.lower().strip(".,;")
        if month_key in MONTHS:
            month = MONTHS[month_key]
            day = int(day_str)
            year, is_two_digit = parse_year(year_str)
            iso = validate_and_format_date(day, month, year)
            if iso:
                span = (m.start(), m.end())
                seen_spans.add(span)
                flags = []
                if is_two_digit:
                    flags.append("two_digit_year")
                results.append({
                    "raw": text[m.start():m.end()],
                    "iso": iso,
                    "span": span,
                    "flags": flags,
                })

    # Reverse textual dates: e.g. August 15, 2021
    for m in DATE_TEXT_REVERSE.finditer(search_text):
        month_str, day_str, year_str = m.groups()
        month_key = month_str.lower().strip(".,;")
        if month_key in MONTHS:
            month = MONTHS[month_key]
            day = int(day_str)
            year, is_two_digit = parse_year(year_str)
            iso = validate_and_format_date(day, month, year)
            if iso:
                span = (m.start(), m.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    flags = []
                    if is_two_digit:
                        flags.append("two_digit_year")
                    results.append({
                        "raw": text[m.start():m.end()],
                        "iso": iso,
                        "span": span,
                        "flags": flags,
                    })

    # 2. Numeric dates: e.g. 15-08-2021 or 15/08/2021 or 15.08.2021
    for m in DATE_NUMERIC.finditer(search_text):
        # Check span collision
        span = (m.start(), m.end())
        if any(s[0] <= span[0] and span[1] <= s[1] for s in seen_spans):
            continue

        p1_str, p2_str, p3_str = m.groups()
        p1, p2 = int(p1_str), int(p2_str)
        year, is_two_digit = parse_year(p3_str)

        flags = []
        if is_two_digit:
            flags.append("two_digit_year")

        # Indian order: Day-Month-Year
        day, month = p1, p2
        iso = validate_and_format_date(day, month, year)

        # Ambiguous order check: if day <= 12 and month <= 12
        if day <= 12 and month <= 12:
            flags.append("ambiguous_order")

        # If day-month fails, try month-day
        if not iso and p2 <= 31 and p1 <= 12:
            iso = validate_and_format_date(p2, p1, year)
            flags.append("month_first_order")

        if iso:
            seen_spans.add(span)
            results.append({
                "raw": text[m.start():m.end()],
                "iso": iso,
                "span": span,
                "flags": flags,
            })

    results.sort(key=lambda x: x["span"][0])
    return results
