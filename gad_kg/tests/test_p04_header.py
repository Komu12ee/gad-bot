"""
Unit tests for Phase 4: Header Extraction and Sanity Checks
Spec: Section 4.3, Section 5 P4
"""

import sys
from pathlib import Path

# Add gad_kg to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.common.llm import verify_quote_substring
from src.p04_header import validate_header_record


def test_quote_substring_verification():
    source = "क्रमांक एफ 1-1/2012/1-3 नया रायपुर, दिनांक 19 मई, 2014 विषय:– भर्ती नियम।"
    assert verify_quote_substring("दिनांक 19 मई, 2014", source) is True
    assert verify_quote_substring("क्रमांक एफ 1-1/2012/1-3", source) is True
    # Fake identifier invented by LLM must fail
    assert verify_quote_substring("Order No. 999/2026/XYZ", source) is False


def test_header_date_sanity():
    doc_text = "क्रमांक 102 दिनांक 15-05-2020 विषय: अवकाश नियम"
    valid_record = {
        "file_id": "test_fid",
        "doc_type": "ORDER",
        "doc_number_raw": "102",
        "doc_number_norm": "102",
        "issue_date_raw": "15-05-2020",
        "issue_date": "2020-05-15",
        "subject": "अवकाश नियम",
        "subject_en": "Leave rules",
        "references": [],
        "evidence": {"doc_number": "क्रमांक 102", "issue_date": "15-05-2020"},
    }
    is_valid, reason = validate_header_record(valid_record, doc_text)
    assert is_valid is True

    # Future date test
    future_record = dict(valid_record)
    future_record["issue_date"] = "2099-01-01"
    is_valid, reason = validate_header_record(future_record, doc_text)
    assert is_valid is False
    assert "future" in reason.lower()


def test_reference_date_ordering_sanity():
    doc_text = "आदेश दिनांक 10-01-2020 संदर्भ: आदेश दिनांक 15-05-2022"
    invalid_ref_record = {
        "file_id": "test_fid2",
        "doc_type": "ORDER",
        "doc_number_raw": "10",
        "doc_number_norm": "10",
        "issue_date_raw": "10-01-2020",
        "issue_date": "2020-01-10",
        "subject": "विषय",
        "subject_en": "Subject",
        "references": [
            {"idx": 1, "raw": "संदर्भ 1", "number_raw": "99", "date": "2022-05-15"}
        ],
        "evidence": {"doc_number": "10", "issue_date": "10-01-2020"},
    }
    is_valid, reason = validate_header_record(invalid_ref_record, doc_text)
    # A reference dated after the document's own issue date triggers warning/flag
    assert "ref_date_after_issue" in reason
