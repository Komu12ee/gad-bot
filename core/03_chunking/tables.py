"""Table parsing, 2D structure preservation, and table chunk builder.

Hard Rules:
- Never change, normalize-away, or drop numbers in the body text.
- Serialize rows as 'column: value' pairs, dropping empty/dash cells.
- Group up to 15 rows per chunk, repeating the table caption, fiscal unit, and column line.
- Never split inside a subtotal block; flag total/subtotal rows.
- Link tables that continue across pages (continued_from_previous_page).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


_TOTAL_ROW_REGEX = re.compile(
    r"(?i)\b(?:Total|Grand\s*Total|Subtotal|Sub-total|योग|कुल|महायोग|सकल\s*योग)\b",
    re.U
)

_FISCAL_UNIT_PATTERNS = [
    ("Rs. crore", re.compile(r"(?i)\b(?:in\s+crores?|cr\.?|करोड़(?:\s*रुपये)?)\b")),
    ("Rs. lakh", re.compile(r"(?i)\b(?:in\s+lakhs?|lac\.?|लाख(?:\s*रुपये)?)\b")),
    ("Rs. thousand", re.compile(r"(?i)\b(?:in\s+thousands?|हजार(?:\s*रुपये)?)\b")),
    ("Rupees", re.compile(r"(?i)\b(?:in\s+rupees?|रुपये|रु\.?)\b")),
]


def detect_fiscal_unit(text: str) -> Optional[str]:
    """Extract currency unit from table caption, text, or headers."""
    for label, pat in _FISCAL_UNIT_PATTERNS:
        if pat.search(text):
            return label
    return None


def is_total_row(row_cells: List[str]) -> bool:
    """Detect whether a row represents a subtotal, total, or grand total."""
    for cell in row_cells:
        if _TOTAL_ROW_REGEX.search(cell):
            return True
    return False


def parse_markdown_table(table_text: str) -> Tuple[List[str], List[List[str]]]:
    """Parse a Markdown table into (headers, rows)."""
    lines = [ln.strip() for ln in table_text.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return [], []

    # Strip outer pipes and split by pipe
    def split_row(line: str) -> List[str]:
        raw = line.strip()
        if raw.startswith("|"):
            raw = raw[1:]
        if raw.endswith("|"):
            raw = raw[:-1]
        return [c.strip() for c in raw.split("|")]

    headers = split_row(lines[0])
    
    # Check if second line is separator like |---|---|
    data_start = 1
    if len(lines) > 1 and re.match(r"^[\s|:-]+$", lines[1]):
        data_start = 2

    rows = []
    for ln in lines[data_start:]:
        cells = split_row(ln)
        # Pad or trim to match header length
        if len(cells) < len(headers):
            cells.extend([""] * (len(headers) - len(cells)))
        elif len(cells) > len(headers):
            cells = cells[:len(headers)]
        rows.append(cells)

    return headers, rows


def serialize_row(headers: List[str], row: List[str]) -> str:
    """Format row as 'Item — Col1: Val1; Col2: Val2' preserving exact numbers."""
    pairs = []
    item_label = ""
    
    # If first cell is non-numeric, use it as row anchor/label
    if row and not re.fullmatch(r"[\s(]*[-+]?[\d,]+(?:\.\d+)?\s*[)%]?\s*", row[0]):
        item_label = row[0].strip()
        start_idx = 1
    else:
        start_idx = 0

    for i in range(start_idx, len(row)):
        h = headers[i].strip() if i < len(headers) else f"Col {i+1}"
        v = row[i].strip()
        if not v or v in ("-", "--", "—", "N/A", "nil"):
            continue
        pairs.append(f"{h}: {v}")

    joined = "; ".join(pairs)
    if item_label and joined:
        return f"{item_label} — {joined}"
    elif item_label:
        return item_label
    return joined or " | ".join(c for c in row if c)


def chunk_table(
    table_id: str,
    table_text: str,
    page_no: int,
    caption: Optional[str] = None,
    fiscal_unit: Optional[str] = None,
    continued_from_previous_page: bool = False,
    max_rows_per_chunk: int = 15
) -> List[Dict[str, Any]]:
    """Slice a table into chunks adhering to subtotal blocks and row budgets."""
    headers, rows = parse_markdown_table(table_text)
    if not rows:
        return []

    unit = fiscal_unit or detect_fiscal_unit(table_text) or detect_fiscal_unit(caption or "") or "Not stated"
    cap_str = caption or "Data Table"
    
    # Header block rendered at the top of every chunk
    header_block_lines = [
        f"Table: {cap_str} | Unit: {unit}" + (" (Continued from previous page)" if continued_from_previous_page else ""),
        "Columns: " + " | ".join(h for h in headers if h)
    ]
    header_block = "\n".join(header_block_lines)

    chunks = []
    current_chunk_rows: List[str] = []
    start_row_idx = 1
    current_has_total = False
    
    i = 0
    while i < len(rows):
        r = rows[i]
        formatted = serialize_row(headers, r)
        r_is_total = is_total_row(r)
        
        # If adding this row exceeds max rows, but we are not in a subtotal block:
        if len(current_chunk_rows) >= max_rows_per_chunk:
            # Check if next row is a total row (do not split before a subtotal)
            if not r_is_total:
                # Flush current chunk
                body = header_block + "\n" + "\n".join(current_chunk_rows)
                chunks.append({
                    "table_id": table_id,
                    "table_caption": cap_str,
                    "fiscal_unit": unit,
                    "column_headers": headers,
                    "row_range": [start_row_idx, start_row_idx + len(current_chunk_rows) - 1],
                    "has_total_row": current_has_total,
                    "continued_from_previous_page": continued_from_previous_page if len(chunks) == 0 else True,
                    "body": body,
                    "pages": [page_no],
                })
                start_row_idx += len(current_chunk_rows)
                current_chunk_rows = []
                current_has_total = False

        if formatted:
            current_chunk_rows.append(formatted)
        if r_is_total:
            current_has_total = True
        i += 1

    if current_chunk_rows:
        body = header_block + "\n" + "\n".join(current_chunk_rows)
        chunks.append({
            "table_id": table_id,
            "table_caption": cap_str,
            "fiscal_unit": unit,
            "column_headers": headers,
            "row_range": [start_row_idx, start_row_idx + len(current_chunk_rows) - 1],
            "has_total_row": current_has_total,
            "continued_from_previous_page": continued_from_previous_page if len(chunks) == 0 else True,
            "body": body,
            "pages": [page_no],
        })

    return chunks
