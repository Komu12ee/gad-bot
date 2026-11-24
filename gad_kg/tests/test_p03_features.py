"""
Unit tests for Phase 3: Deterministic Feature Extraction
Spec: Section 5 P3, Section 7.1 - 7.5
"""

import sys
from pathlib import Path

# Add gad_kg to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.common.dates import extract_dates
from src.common.numbers import normalize_number, extract_number_candidates
from src.p03_features import extract_reference_block, extract_tables, extract_document_features


def test_date_extraction_hindi_and_english():
    text = "दिनांक 15 अगस्त 2021 को आदेश जारी किया गया। Notification dated 24/12/2023."
    dates = extract_dates(text)
    assert len(dates) == 2
    assert dates[0]["iso"] == "2021-08-15"
    assert dates[1]["iso"] == "2023-12-24"


def test_number_normalization():
    assert normalize_number("क्रमांक एफ 1-1/2012/1-3") == "F1-1/2012/1-3"
    assert normalize_number("Order No. ESTB-101(1)/7/2024-GAD-4") == "ESTB-101(1)/7/2024-GAD-4"
    assert normalize_number("फा. क्र. 20/2020") == "20/2020"


def test_reference_block_extraction():
    lines = [
        "छत्तीसगढ़ शासन सामान्य प्रशासन विभाग",
        "क्रमांक 102/2022",
        "संदर्भ :-",
        "1. आदेश क्रमांक 50/2020, दिनांक 10 जनवरी 2020",
        "2. अधिसूचना क्रमांक 80/2021, दिनांक 15 मार्च 2021",
        "",
        "विषय :- शासकीय सेवकों के स्थानांतरण बाबत।",
        "राज्य शासन द्वारा निर्णय लिया गया है...",
    ]
    text = "\n".join(lines)
    ref_block = extract_reference_block(lines)
    assert ref_block["found"] is True
    assert len(ref_block["items"]) == 2
    assert "50/2020" in ref_block["items"][0]
    assert "80/2021" in ref_block["items"][1]


def test_table_span_extraction():
    lines = [
        "तालिका 1: रिक्त पदों का विवरण",
        "| पद नाम | रिक्त संख्या |",
        "|---|---|",
        "| अवर सचिव | 5 |",
        "| उप सचिव | 2 |",
        "",
        "उपरोक्त पदों पर नियुक्ति प्रक्रियाधीन है।",
    ]
    tables = extract_tables(lines)
    assert len(tables) == 1
    t = tables[0]
    assert t["start_line"] == 1
    assert t["end_line"] == 4
    assert "रिक्त पदों का विवरण" in t["caption_above"]
    assert "उपरोक्त पदों पर" in t["paragraph_below"]
