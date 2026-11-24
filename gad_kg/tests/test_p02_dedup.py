"""
Unit tests for Phase 2: Deduplication
Spec: Section 5 P2
"""

import sys
from pathlib import Path

# Add gad_kg to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from datasketch import MinHash
from src.p02_dedup import (
    get_5word_shingles,
    compute_minhash,
    exact_jaccard,
    cluster_documents,
)


def test_5word_shingles():
    text = "The General Administration Department of Chhattisgarh issues orders"
    shingles = get_5word_shingles(text)
    assert len(shingles) == 4
    assert shingles[0] == "The General Administration Department of"
    assert shingles[1] == "General Administration Department of Chhattisgarh"


def test_exact_duplicates():
    docs = [
        {"file_id": "f1", "clean_text": "This is order number 123 issued by GAD Raipur.", "ocr_quality": 0.85},
        {"file_id": "f2", "clean_text": "This is order number 123 issued by GAD Raipur.", "ocr_quality": 0.95},
    ]
    clusters = cluster_documents(docs, threshold=0.90)
    assert len(clusters) == 1
    c = clusters[0]
    assert c["canonical_file_id"] == "f2"  # Higher OCR quality wins
    assert set(c["file_ids"]) == {"f1", "f2"}


def test_near_duplicates_above_threshold():
    # Long text with minor variation
    base = "छत्तीसगढ़ शासन सामान्य प्रशासन विभाग मंत्रालय महानदी भवन नया रायपुर आदेश क्रमांक " + " ".join([f"शब्द{i}" for i in range(100)])
    var1 = base + " दिनांक 2020-01-01"
    var2 = base + " दिनांक 2020-01-02"

    docs = [
        {"file_id": "f_hi_1", "clean_text": var1, "ocr_quality": 0.90},
        {"file_id": "f_hi_2", "clean_text": var2, "ocr_quality": 0.92},
    ]
    clusters = cluster_documents(docs, threshold=0.90)
    assert len(clusters) == 1
    assert clusters[0]["canonical_file_id"] == "f_hi_2"


def test_distinct_documents_separate_clusters():
    docs = [
        {"file_id": "doc_a", "clean_text": "First completely distinct notification regarding transfer of collectors.", "ocr_quality": 0.9},
        {"file_id": "doc_b", "clean_text": "Second completely separate circular regarding budget allocation rules.", "ocr_quality": 0.9},
    ]
    clusters = cluster_documents(docs, threshold=0.90)
    assert len(clusters) == 2
    for c in clusters:
        assert len(c["file_ids"]) == 1
