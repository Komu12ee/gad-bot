"""
Normalization and Text Cleaning Utilities
Spec: Section 5 P1, Section 7.3 & Section 7.4
"""

import re
import unicodedata
from typing import Tuple, List, Set, Dict

# 1:1 mapping from Devanagari digits to ASCII digits
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Zero-width characters and soft hyphen
ZERO_WIDTH_CHARS = re.compile(r"[\u200B\u200C\u200D\uFEFF\u00AD]")

# OCR confusions only between digits or in date components
OCR_DIGIT_FIXES = [
    # Between digits
    (re.compile(r"(?<=\d)[Oo](?=\d)"), "0"),
    (re.compile(r"(?<=\d)[lI](?=\d)"), "1"),
    (re.compile(r"(?<=\d)[Ss](?=\d)"), "5"),
    (re.compile(r"(?<=\d)B(?=\d)"), "8"),
    (re.compile(r"(?<=\d)[Zz](?=\d)"), "2"),
    # Adjacent to date/number separators followed or preceded by digits
    (re.compile(r"(?<=\d)[Oo](?=[-\/.][0-9])"), "0"),
    (re.compile(r"(?<=[0-9][-\/.])[Oo](?=\d)"), "0"),
    (re.compile(r"(?<=\d)[lI](?=[-\/.][0-9])"), "1"),
    (re.compile(r"(?<=[0-9][-\/.])[lI](?=\d)"), "1"),
    (re.compile(r"(?<=\d)[Ss](?=[-\/.][0-9])"), "5"),
    (re.compile(r"(?<=[0-9][-\/.])[Ss](?=\d)"), "5"),
]


def strip_zero_width(text: str) -> str:
    """Remove zero-width characters and soft hyphens."""
    return ZERO_WIDTH_CHARS.sub("", text)


def fix_ocr_number_confusions(text: str) -> str:
    """Fix common OCR letter-for-digit confusions strictly inside numeric contexts."""
    for pattern, replacement in OCR_DIGIT_FIXES:
        text = pattern.sub(replacement, text)
    return text


def to_search_text(clean_text: str) -> str:
    """Map Devanagari digits to ASCII digits 1:1, strictly preserving length and offsets."""
    search_text = clean_text.translate(DEV_DIGITS)
    assert len(search_text) == len(clean_text), f"Length mismatch: clean {len(clean_text)} != search {len(search_text)}"
    return search_text


def remove_running_headers_footers(text: str) -> str:
    """Remove running headers, footers, and page numbers repeating across pages or ≥ 5 times.
    
    Pages are detected via `<!-- Page X -->` or `---`.
    """
    lines = text.split("\n")
    if len(lines) < 20:
        return text

    # Detect pages
    page_chunks = re.split(r"(<!--\s*[Pp]age\s*\d+\s*-->|^\s*---\s*$)", text, flags=re.MULTILINE)
    pages = [p for p in page_chunks if p.strip() and not re.match(r"^<!--\s*[Pp]age\s*\d+\s*-->|^\s*---\s*$", p.strip(), re.I)]
    n_pages = max(1, len(pages))

    line_counts: Dict[str, int] = {}
    page_presence: Dict[str, Set[int]] = {}

    for p_idx, page in enumerate(pages):
        for line in page.split("\n"):
            stripped = line.strip()
            if not stripped or len(stripped) >= 120:
                continue
            # Strip digits, punctuation, and markdown markup to get signature
            sig = re.sub(r"[\d\s\W]+", "", stripped)
            if len(sig) < 4:
                continue
            line_counts[sig] = line_counts.get(sig, 0) + 1
            if sig not in page_presence:
                page_presence[sig] = set()
            page_presence[sig].add(p_idx)

    # Header/footer candidates: repeat on >= 30% of pages (if >=3 pages) or >= 5 times
    repeating_sigs = set()
    for sig, count in line_counts.items():
        pages_seen = len(page_presence[sig])
        if n_pages >= 3 and (pages_seen / n_pages) >= 0.30:
            repeating_sigs.add(sig)
        elif count >= 5:
            repeating_sigs.add(sig)

    if not repeating_sigs:
        return text

    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) < 120:
            sig = re.sub(r"[\d\s\W]+", "", stripped)
            if sig in repeating_sigs:
                continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def validate_and_fix_tables(lines: List[str]) -> Tuple[List[str], List[str]]:
    """Validate markdown tables so each data row matches header column count.
    
    Pad with empty cells if truncated, flag broken_table if severely misaligned.
    """
    flags = []
    new_lines = []
    in_table = False
    expected_cols = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if not in_table:
                in_table = True
                expected_cols = len(cells)
                new_lines.append(line)
            else:
                # Check if it's separator row
                if all(re.match(r"^:?-+:?$", c) for c in cells if c):
                    new_lines.append(line)
                    continue
                if len(cells) == expected_cols:
                    new_lines.append(line)
                elif len(cells) < expected_cols:
                    # Pad missing columns
                    padded_cells = cells + [""] * (expected_cols - len(cells))
                    new_lines.append("| " + " | ".join(padded_cells) + " |")
                else:
                    # More cells than expected
                    new_lines.append(line)
                    if "broken_table" not in flags:
                        flags.append("broken_table")
        else:
            in_table = False
            expected_cols = 0
            new_lines.append(line)

    return new_lines, flags


def rejoin_broken_lines(lines: List[str]) -> List[str]:
    """Rejoin lines broken across soft linebreaks without ending sentence punctuation."""
    output = []
    sentence_endings = (".", "।", "?", "!", ":", ";", "---", "|", "#")

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Do not merge table rows, headings, lists, empty lines
        if (
            not stripped or
            stripped.startswith(("#", "|", "-", "*", ">")) or
            re.match(r"^\(?\d{1,3}[.)]\s+", stripped) or
            re.match(r"^\([a-zA-Zक-ह]\)\s+", stripped) or
            stripped.endswith(sentence_endings)
        ):
            output.append(line)
            i += 1
            continue

        # Lookahead to next line
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            next_stripped = next_line.strip()
            if (
                next_stripped and
                not next_stripped.startswith(("#", "|", "-", "*", ">", "<!--")) and
                not re.match(r"^\(?\d{1,3}[.)]\s+", next_stripped) and
                not re.match(r"^\([a-zA-Zक-ह]\)\s+", next_stripped)
            ):
                # Check hyphenation at end of line
                if stripped.endswith("-"):
                    merged = line.rstrip()[:-1] + next_stripped
                else:
                    merged = line.rstrip() + " " + next_stripped
                lines[i + 1] = merged
                i += 1
                continue

        output.append(line)
        i += 1

    return output


def clean_document_text(raw_text: str) -> Tuple[str, str, List[str]]:
    """Complete P1 cleaning pipeline for one document.
    
    Returns:
    - clean_text: NFC normalized, zero-width removed, OCR fixed in numbers, running headers removed
    - search_text: identical length to clean_text, with Devanagari digits mapped to ASCII
    - flags: list of warnings (e.g. broken_table)
    """
    # 1. Unicode NFC & Zero-width stripping
    text = unicodedata.normalize("NFC", raw_text)
    text = strip_zero_width(text)

    # Line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. OCR number/date confusions
    text = fix_ocr_number_confusions(text)

    # 3. Running headers and footers
    text = remove_running_headers_footers(text)

    # Lines processing: tables & line joining
    raw_lines = text.split("\n")
    fixed_table_lines, flags = validate_and_fix_tables(raw_lines)
    rejoined_lines = rejoin_broken_lines(fixed_table_lines)

    # Collapse multiple blank lines to at most 2
    clean_lines = []
    consecutive_blank = 0
    for l in rejoined_lines:
        # Collapse internal spaces
        collapsed = re.sub(r"[ \t]+", " ", l) if not l.strip().startswith("|") else l
        if not collapsed.strip():
            consecutive_blank += 1
            if consecutive_blank <= 2:
                clean_lines.append("")
        else:
            consecutive_blank = 0
            clean_lines.append(collapsed.strip())

    clean_text = "\n".join(clean_lines).strip() + "\n"

    # 4. Search text generation with 1:1 length guarantee
    search_text = to_search_text(clean_text)

    return clean_text, search_text, flags
