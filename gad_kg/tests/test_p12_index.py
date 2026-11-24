"""
Unit tests for Phase 12: Contextual Prefix & Lexical FTS5 Search
Spec: Section 5 P12
"""

import sys
import sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.p12_index import (
    build_contextual_prefix,
    init_fts5_index,
    index_chunks,
    search_lexical_fts5,
)


def test_build_contextual_prefix():
    header = {
        "doc_type": "ORDER",
        "doc_number_raw": "F 1-1/2020/1-3",
        "issue_date": "2020-05-15",
        "subject": "शासकीय सेवकों के स्थानांतरण नीति",
    }
    r9_meta = {
        "covers": "transfer guidelines and exception conditions",
        "keywords_en": ["transfer", "posting"],
        "keywords_hi": ["स्थानांतरण", "पदस्थापना"],
    }
    prefix = build_contextual_prefix(header, r9_meta)
    assert "[ORDER F 1-1/2020/1-3 dt. 2020-05-15 | शासकीय सेवकों के स्थानांतरण नीति]" in prefix
    assert "transfer guidelines" in prefix
    assert "Keywords:" in prefix


def test_fts5_indexing_and_search(tmp_path):
    db_file = str(tmp_path / "test_fts5.db")
    conn = init_fts5_index(db_file)

    chunks = [
        {
            "chunk_id": "D_01#K001",
            "doc_id": "D_01",
            "file_id": "fid_01",
            "text": "सामान्य प्रशासन विभाग द्वारा शासकीय सेवकों के लिए नई स्थानांतरण नीति लागू की गई है।",
        },
        {
            "chunk_id": "D_02#K001",
            "doc_id": "D_02",
            "file_id": "fid_02",
            "text": "पेंशन पुनरीक्षण और पारिवारिक पेंशन के संबंध में स्पष्टीकरण।",
        },
    ]

    headers_map = {
        "fid_01": {
            "doc_type": "CIRCULAR",
            "doc_number_raw": "F 12-3/2021",
            "issue_date": "2021-06-01",
            "subject": "स्थानांतरण नीति",
        },
        "fid_02": {
            "doc_type": "ORDER",
            "doc_number_raw": "F 5-1/2019",
            "issue_date": "2019-11-20",
            "subject": "पेंशन नियम",
        },
    }

    index_chunks(conn, chunks, headers_map)

    # Search for स्थानांतरण (transfer)
    res_transfer = search_lexical_fts5(conn, "स्थानांतरण", limit=5)
    assert len(res_transfer) == 1
    assert res_transfer[0]["chunk_id"] == "D_01#K001"
    assert "स्थानांतरण नीति" in res_transfer[0]["prefix"]

    # Search for पेंशन (pension)
    res_pension = search_lexical_fts5(conn, "पेंशन", limit=5)
    assert len(res_pension) == 1
    assert res_pension[0]["chunk_id"] == "D_02#K001"

    conn.close()
