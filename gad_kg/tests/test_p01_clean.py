import sys
from pathlib import Path

# Add gad_kg to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.common.normalize import (
    strip_zero_width,
    fix_ocr_number_confusions,
    to_search_text,
    remove_running_headers_footers,
    validate_and_fix_tables,
    clean_document_text
)


def test_zero_width_removal():
    raw = "छत्तीस\u200Bगढ़ \u200Cशा\u200Dसन\uFEFF \u00ADमंत्रालय"
    cleaned = strip_zero_width(raw)
    assert "\u200B" not in cleaned
    assert "\u200C" not in cleaned
    assert "\u200D" not in cleaned
    assert "\uFEFF" not in cleaned
    assert "\u00AD" not in cleaned
    assert cleaned == "छत्तीसगढ़ शासन मंत्रालय"


def test_ocr_digit_confusions_in_numbers():
    # Only between digits
    text = "Order No 1O2 dated 2l-05-2023 under section 4S6"
    fixed = fix_ocr_number_confusions(text)
    assert "102" in fixed
    assert "21-05-2023" in fixed
    assert "456" in fixed
    # Must NOT alter running words
    assert "Order" in fixed
    assert "section" in fixed


def test_devanagari_digit_search_length_invariant():
    hindi_text = "क्रमांक १२३४/५६ दिनांक ०१/०८/२०२४"
    search_text = to_search_text(hindi_text)
    assert len(search_text) == len(hindi_text)
    assert "1234/56" in search_text
    assert "01/08/2024" in search_text


def test_table_row_alignment():
    lines = [
        "| Col1 | Col2 | Col3 |",
        "|---|---|---|",
        "| val1 | val2 | val3 |",
        "| valA | valB |"  # Missing 3rd column
    ]
    fixed_lines, flags = validate_and_fix_tables(lines)
    assert len(fixed_lines) == 4
    assert fixed_lines[3].count("|") == 4  # Padded to 3 columns


def test_clean_document_text_end_to_end():
    raw = (
        "छत्तीसगढ़\u200B शासन\n"
        "सामान्य\r\nप्रशासन विभाग\r\n"
        "आदेश क्र. 1O5 दिनांक ०५/०६/२०२२\n"
    )
    clean, search, flags = clean_document_text(raw)
    assert "\u200B" not in clean
    assert "\r" not in clean
    assert "105" in clean
    assert len(clean) == len(search)
    assert "05/06/2022" in search
