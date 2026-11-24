"""
Unit & Integration tests for Phase 14: Query Service & Answering Engine
Spec: Section 5 P14, Section 9
"""

import sys
import os
import json
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.p14_query_service import QueryEngine, app, AskRequest
import src.p14_query_service as p14_mod


@pytest.fixture
def test_engine(tmp_path):
    h_file = tmp_path / "header.jsonl"
    e_file = tmp_path / "edges.jsonl"
    trace_file = tmp_path / "trace.jsonl"

    # Seed docs
    docs = [
        {"doc_id": "D1", "doc_number_raw": "F 1-1/2020", "doc_number_norm": "1-1/2020", "issue_date": "2020-01-01", "subject": "सेवा नियम 2020"},
        {"doc_id": "D2", "doc_number_raw": "F 1-2/2022", "doc_number_norm": "1-2/2022", "issue_date": "2022-06-01", "subject": "सेवा नियम संशोधन"},
    ]
    with open(h_file, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

    edges = [
        {"edge_id": "E1", "relation": "SUPERSEDES", "source_doc_id": "D2", "target_doc_id": "D1", "scope": "whole", "effective_date": "2022-06-01", "tier": "Tier A"}
    ]
    with open(e_file, "w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e) + "\n")

    engine = QueryEngine(
        header_path=str(h_file),
        edges_path=str(e_file),
        trace_log_path=str(trace_file),
    )
    return engine


def test_lookup_document(test_engine):
    doc = test_engine.lookup_document("1-1/2020")
    assert doc is not None
    assert doc["doc_id"] == "D1"
    assert doc["issue_date"] == "2020-01-01"

    doc_missing = test_engine.lookup_document("NONEXISTENT_999")
    assert doc_missing is None


def test_deterministic_status_and_timeline(test_engine):
    # As of 2021: D1 is IN_FORCE
    st_2021 = test_engine.get_status("D1", as_of="2021-01-01")
    assert st_2021["status"] == "IN_FORCE"

    # As of 2023: D1 is SUPERSEDED by D2
    st_2023 = test_engine.get_status("D1", as_of="2023-01-01")
    assert st_2023["status"] == "SUPERSEDED"
    assert st_2023["evidence"]["replacing_doc_id"] == "D2"

    # Timeline includes both ISSUED and SUPERSEDED
    timeline = test_engine.get_timeline("D1")
    assert len(timeline) == 2
    assert timeline[0]["event"] == "ISSUED"
    assert "SUPERSEDES by D2" in timeline[1]["event"]


def test_expand_via_graph(test_engine):
    expanded = test_engine.expand_via_graph(["D1"])
    assert "D2" in expanded

    expanded_from_d2 = test_engine.expand_via_graph(["D2"])
    assert "D1" in expanded_from_d2


def test_fastapi_endpoints(test_engine):
    p14_mod.ENGINE = test_engine
    client = TestClient(app)

    # 1. /doc/D1
    resp = client.get("/doc/D1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["document"]["doc_id"] == "D1"

    # 2. /status
    resp_st = client.get("/status?doc=D1&asof=2023-01-01")
    assert resp_st.status_code == 200
    assert resp_st.json()["status"] == "SUPERSEDED"

    # 3. /timeline/D1
    resp_tl = client.get("/timeline/D1")
    assert resp_tl.status_code == 200
    assert len(resp_tl.json()["timeline"]) == 2

    # 4. /in_force
    resp_inf = client.get("/in_force?asof=2021-01-01")
    assert resp_inf.status_code == 200
    assert resp_inf.json()["count"] >= 1
