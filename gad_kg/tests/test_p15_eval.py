"""
Unit tests for Phase 15: Evaluation Harness
Spec: Section 5 P15, Section 10
"""

import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.p15_eval import run_evaluation


def test_eval_harness_mock(tmp_path):
    mock_gold = tmp_path / "mock_gold.json"
    mock_report = tmp_path / "eval_report.md"

    sample_questions = [
        {
            "id": "T01",
            "question": "पदोन्नति संबंधी सामान्य नियम क्या हैं?",
            "language": "hi",
            "intent": "CURRENT_RULE",
            "as_of": "2026-11-19",
            "category": "promotion",
            "expected_answerable": True,
        },
        {
            "id": "T02",
            "question": "मंगल ग्रह पर भूमि आवंटन के क्या नियम हैं?",
            "language": "hi",
            "intent": "LOOKUP",
            "as_of": "2026-11-19",
            "category": "unanswerable",
            "expected_answerable": False,
        }
    ]
    with open(mock_gold, "w", encoding="utf-8") as f:
        json.dump(sample_questions, f, ensure_ascii=False)

    res = run_evaluation(gold_path=str(mock_gold), report_out_path=str(mock_report))
    assert res["total_questions"] == 2
    assert os.path.exists(mock_report)

    with open(mock_report, "r", encoding="utf-8") as f:
        report_text = f.read()

    assert "Final Evaluation Report" in report_text
    assert "Overall System Performance" in report_text
    assert "Error Taxonomy Breakdown" in report_text
    assert "Ablations" in report_text
