"""
Unit tests for Phase 10: Table Parsing & Number Formatting
Spec: Section 5 P10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import sqlite3
from src.p10_tables import (
    parse_numeric_value,
    extract_tables_from_markdown,
    init_table_database,
    save_table_to_db,
)


def test_parse_numeric_value_devanagari_and_indian_format():
    # Devanagari numerals
    val, unit = parse_numeric_value("१,२३,४५६")
    assert val == 123456.0
    assert unit is None

    # Currency format with INR
    val2, unit2 = parse_numeric_value("₹ 75,000/-")
    assert val2 == 75000.0
    assert unit2 == "INR"

    # Percentage
    val3, unit3 = parse_numeric_value("10.5%")
    assert val3 == 10.5
    assert unit3 == "%"

    # Non-numeric text
    val4, unit4 = parse_numeric_value("रायपुर")
    assert val4 is None


def test_extract_tables_from_markdown():
    md = (
        "विषय: वेतन पुनरीक्षण\n\n"
        "विवरण निम्नानुसार है:-\n\n"
        "| क्र. | पद नाम | वेतनमान | पद संख्या |\n"
        "|---|---|---|---|\n"
        "| 1 | उप सचिव | 78,800 | 5 |\n"
        "| 2 | अवर सचिव | 67,700 | 12 |\n"
        "| 3 | अनुभाग अधिकारी | 56,100 | 25 |\n\n"
        "उक्त पदों पर भर्ती नियम लागू होंगे।"
    )
    tables = extract_tables_from_markdown(md)
    assert len(tables) == 1

    tbl = tables[0]
    assert tbl["row_count"] == 3
    assert tbl["headers"] == ["क्र.", "पद नाम", "वेतनमान", "पद संख्या"]
    assert len(tbl["rows"]) == 3

    # Check numeric cell parsing in row 0
    row0 = tbl["rows"][0]
    assert row0["वेतनमान"]["num"] == 78800.0
    assert row0["पद संख्या"]["num"] == 5.0


def test_table_database_integration(tmp_path):
    db_file = str(tmp_path / "test_tables.db")
    conn = init_table_database(db_file)

    meta = {
        "title_en": "Pay Scale Table",
        "title_hi": "वेतनमान विवरण",
        "table_kind": "pay_scale",
        "columns": [
            {"name_raw": "पद नाम", "name_en": "post_name", "type": "text", "unit": None},
            {"name_raw": "वेतनमान", "name_en": "pay_scale", "type": "amount", "unit": "INR"},
        ],
    }
    parsed = {
        "headers": ["पद नाम", "वेतनमान"],
        "row_count": 2,
        "rows": [
            {"पद नाम": {"raw": "उप सचिव", "num": None, "unit": None}, "वेतनमान": {"raw": "78,800", "num": 78800.0, "unit": "INR"}},
            {"पद नाम": {"raw": "अवर सचिव", "num": None, "unit": None}, "वेतनमान": {"raw": "67,700", "num": 67700.0, "unit": "INR"}},
        ],
    }

    save_table_to_db(conn, "D_test#T01", "D_test", meta, parsed)

    cur = conn.cursor()
    cur.execute("SELECT title_en, n_rows FROM table_catalog WHERE table_id = ?", ("D_test#T01",))
    cat_row = cur.fetchone()
    assert cat_row == ("Pay Scale Table", 2)

    cur.execute("SELECT col_name_en, value_num FROM table_cells WHERE col_name_en = 'pay_scale' ORDER BY row_idx")
    cell_rows = cur.fetchall()
    assert len(cell_rows) == 2
    assert cell_rows[0][1] == 78800.0
    assert cell_rows[1][1] == 67700.0
    conn.close()


def test_verify_table_acceptance_synthetic(tmp_path):
    from src.p10_tables import verify_table_acceptance
    db_file = str(tmp_path / "acceptance_test.db")
    conn = init_table_database(db_file)

    meta = {
        "title_en": "Sample Table",
        "title_hi": "नमूना तालिका",
        "table_kind": "pay_scale",
        "columns": [
            {"name_raw": "पद", "name_en": "post", "type": "text", "unit": None},
            {"name_raw": "वेतन", "name_en": "salary", "type": "amount", "unit": "INR"},
        ],
    }
    parsed = {
        "headers": ["पद", "वेतन"],
        "row_count": 3,
        "rows": [
            {"पद": {"raw": "अधिकारी 1", "num": None, "unit": None}, "वेतन": {"raw": "50,000", "num": 50000.0, "unit": "INR"}},
            {"पद": {"raw": "अधिकारी 2", "num": None, "unit": None}, "वेतन": {"raw": "६०,०००", "num": 60000.0, "unit": "INR"}},
            {"पद": {"raw": "अधिकारी 3", "num": None, "unit": None}, "वेतन": {"raw": "₹ 70,000/-", "num": 70000.0, "unit": "INR"}},
        ],
    }
    save_table_to_db(conn, "D_acc#T01", "D_acc", meta, parsed)
    conn.close()

    res = verify_table_acceptance(db_file, "", sample_size=1)
    assert res["tables_checked"] == 1
    assert res["total_numeric_cells"] == 3
    assert res["matched_numeric_cells"] == 3
    assert res["numeric_cell_accuracy"] == 1.0
    assert res["passed"] is True


    conn.close()
