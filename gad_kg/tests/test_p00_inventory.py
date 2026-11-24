import sys
from pathlib import Path

# Add gad_kg to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.p00_inventory import compute_ocr_quality


def test_empty_text():
    score, lang, dev_ratio, flags = compute_ocr_quality("")
    assert score == 0.0
    assert "empty_file" in flags
    assert "needs_reocr" in flags


def test_clean_english_text():
    text = (
        "GOVERNMENT OF CHHATTISGARH\n"
        "GENERAL ADMINISTRATION DEPARTMENT\n"
        "Notification regarding appointment of officers and transfer guidelines for 2024.\n"
        "The Governor is pleased to order the transfer of following Indian Administrative Service officers."
    )
    score, lang, dev_ratio, flags = compute_ocr_quality(text)
    assert lang == "en"
    assert dev_ratio < 0.1
    assert score >= 0.95
    assert "needs_reocr" not in flags


def test_clean_hindi_text():
    text = (
        "छत्तीसगढ़ शासन\n"
        "सामान्य प्रशासन विभाग\n"
        "मंत्रालय, महानदी भवन, नवा रायपुर अटल नगर\n"
        "आदेश: राज्य प्रशासनिक सेवा के अधिकारियों की नवीन पदस्थापना की जाती है।"
    )
    score, lang, dev_ratio, flags = compute_ocr_quality(text)
    assert lang == "hi"
    assert dev_ratio >= 0.8
    assert score >= 0.95
    assert "needs_reocr" not in flags


def test_corrupted_characters():
    text = "छत्तीसगढ़ शासन \ufffd\ufffd\ufffd\x00\x01\x02\ufffd\ufffd garbled \ufffd\ufffd\ufffd text \ufffd\ufffd"
    score, lang, dev_ratio, flags = compute_ocr_quality(text)
    assert "corrupted_chars" in flags
    assert score < 0.85


def test_isolated_single_characters():
    text = "क ा र ् य ा ल य अ ध ी क ् ष क स ा म ा न ् य प ् र श ा स न"
    score, lang, dev_ratio, flags = compute_ocr_quality(text)
    assert "high_isolated_chars" in flags
    assert score < 0.85


def test_invalid_matras():
    # Matras appearing at word beginnings or repeated
    text = "ासामान्य ोप्रशासन ूूविभाग ीीमंत्रालय"
    score, lang, dev_ratio, flags = compute_ocr_quality(text)
    assert "invalid_matras" in flags
    assert score < 0.85
