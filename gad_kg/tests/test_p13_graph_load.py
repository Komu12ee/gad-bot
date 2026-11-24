"""
Unit tests for Phase 13: Graph Loader
Spec: Section 5 P13, Section 8
"""

import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import sqlite3
import networkx as nx

from src.p13_graph_load import (
    init_graph_db,
    build_cypher_export,
    load_graph,
    verify_graph_acceptance,
)


def test_init_graph_db(tmp_path):
    db_file = str(tmp_path / "test_graph.db")
    conn = init_graph_db(db_file)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    assert "nodes" in tables
    assert "edges" in tables
    conn.close()


def test_build_cypher_export(tmp_path):
    cypher_file = str(tmp_path / "test_graph.cypher")
    docs = [{"doc_id": "D1", "doc_type": "ORDER", "doc_number_raw": "101", "doc_number_norm": "101", "issue_date": "2020-01-01", "subject": "Test", "language": "hi", "file_id": "f1"}]
    stubs = [{"stub_id": "S1", "number_raw": "50", "number_norm": "50", "date": "2019-01-01"}]
    clauses = [{"clause_id": "D1-C01", "doc_id": "D1", "clause_number": "1", "section_type": "operative", "text": "Clause 1 text", "parent_clause_id": None}]
    tables = [{"table_id": "D1#T01", "doc_id": "D1", "title_en": "Table 1", "title_hi": "तालिका 1", "table_kind": "pay_scale"}]
    edges = [{"source_doc_id": "D1", "target_doc_id": "S1", "relation_type": "AMENDS", "tier": "A", "scope": "whole", "effective_date": "2020-01-01", "quote": "संशोधन"}]

    build_cypher_export(docs, stubs, clauses, tables, edges, cypher_file)
    assert os.path.exists(cypher_file)
    with open(cypher_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "CREATE CONSTRAINT doc_id" in content
    assert "MERGE (d:Document {doc_id: row.doc_id})" in content
    assert "MERGE (s:Stub {stub_id: row.stub_id})" in content
    assert "MERGE (c:Clause {clause_id: row.clause_id})" in content
    assert "MERGE (t:Table {table_id: row.table_id})" in content


def test_load_graph_integration(tmp_path):
    # Create temporary mock jsonl files
    h_file = tmp_path / "header.jsonl"
    s_file = tmp_path / "stubs.jsonl"
    c_file = tmp_path / "clauses.jsonl"
    t_file = tmp_path / "tables.jsonl"
    e_file = tmp_path / "edges.jsonl"
    db_file = tmp_path / "graph.db"
    cypher_file = tmp_path / "graph.cypher"

    with open(h_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"doc_id": "D1", "doc_type": "ORDER", "doc_number_norm": "100/2020", "issue_date": "2020-01-01", "subject": "Test Order"}) + "\n")
    with open(s_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"stub_id": "S1", "number_norm": "50/2018", "date": "2018-01-01"}) + "\n")
    with open(c_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"clause_id": "D1-C01", "doc_id": "D1", "clause_number": "1", "section_type": "operative", "text": "Clause 1 text"}) + "\n")
    with open(t_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"table_id": "D1#T01", "doc_id": "D1", "title_en": "T1"}) + "\n")
    with open(e_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"source_doc_id": "D1", "target_doc_id": "S1", "relation_type": "AMENDS", "tier": "A", "scope": "whole"}) + "\n")

    conn, G, stats = load_graph(
        header_path=str(h_file),
        stubs_path=str(s_file),
        clauses_path=str(c_file),
        tables_path=str(t_file),
        edges_path=str(e_file),
        graph_db_path=str(db_file),
        cypher_out_path=str(cypher_file),
    )

    assert stats["document_nodes"] == 1
    assert stats["stub_nodes"] == 1
    assert stats["clause_nodes"] == 1
    assert stats["table_nodes"] == 1
    assert stats["legal_edges"] == 1
    assert G.has_edge("D1", "S1")
    assert G.has_edge("D1", "D1-C01")
    assert G.has_edge("D1", "D1#T01")

    # Test acceptance check on mock
    acc = verify_graph_acceptance(conn, G, header_path=str(h_file), sample_size=1)
    assert acc["passed"] is True
    assert acc["match_rate"] == 1.0
    conn.close()
