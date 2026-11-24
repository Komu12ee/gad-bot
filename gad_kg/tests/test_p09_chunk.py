"""
Unit tests for Phase 9: Clause Segmentation & Chunking
Spec: Section 5 P9, Section 7.5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.p09_chunk import (
    segment_document_clauses,
    chunk_clauses,
    classify_section,
    estimate_tokens,
)


def test_classify_section():
    assert classify_section("संदर्भ:– सामान्य प्रशासन विभाग क्रमांक 123", "preamble") == "reference"
    assert classify_section("शासन निर्णय / आदेश:", "reference") == "operative"
    assert classify_section("परिशिष्ट - 1 (नियम 4 देखें)", "operative") == "annexure"
    assert classify_section("राज्यपाल के नाम से तथा आदेशानुसार", "operative") == "signature"


def test_segment_document_clauses():
    text = (
        "छत्तीसगढ़ शासन\n"
        "सामान्य प्रशासन विभाग\n\n"
        "संदर्भ:– आदेश क्र. 45/2020\n\n"
        "आदेश\n\n"
        "1. राज्य शासन एतदद्वारा निम्नलिखित निर्णय लेता है।\n\n"
        "(क) प्रथम श्रेणी अधिकारियों की पदोन्नति।\n\n"
        "2. यह आदेश तत्काल प्रभाव से लागू होगा।\n\n"
        "राज्यपाल के नाम से तथा आदेशानुसार"
    )
    clauses = segment_document_clauses("D_test01", "test01", text)
    assert len(clauses) >= 4

    # Check clause IDs
    assert clauses[0]["clause_id"] == "D_test01#C001"
    assert any(c["section_type"] == "reference" for c in clauses)
    assert any(c["section_type"] == "operative" for c in clauses)

    # Check parent clause pointer on sub-clause
    sub_clauses = [c for c in clauses if c.get("number") == "क"]
    if sub_clauses:
        assert sub_clauses[0]["parent_clause_id"] is not None


def test_chunk_clauses_token_limit():
    # Long text exceeding 600 tokens
    long_clause_text = "यह एक बहुत लंबा परिच्छेद है। " * 150  # ~600+ words
    clauses = [
        {
            "clause_id": "D_test02#C001",
            "doc_id": "D_test02",
            "file_id": "test02",
            "number": "1",
            "section_type": "operative",
            "text": long_clause_text,
            "char_start": 0,
            "char_end": len(long_clause_text),
            "parent_clause_id": None,
        }
    ]
    chunks = chunk_clauses(clauses, {"doc_id": "D_test02", "file_id": "test02"})
    assert len(chunks) > 1

    # Invariant: No chunk exceeds 700 tokens
    for ch in chunks:
        assert ch["token_count"] <= 700
