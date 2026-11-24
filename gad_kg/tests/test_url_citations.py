import importlib.util
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ADAPTER_PATH = REPO_ROOT / "core" / "05_webui" / "gad_kg_adapter.py"

spec = importlib.util.spec_from_file_location("gad_kg_adapter", str(ADAPTER_PATH))
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

linkify_document_citations = adapter.linkify_document_citations
handle_gad_query = adapter.handle_gad_query
route_query_intent = adapter.route_query_intent


def test_linkify_citations_converts_doc_id():
    raw_text = "Here is the SAS list [D_63dfd3acb3182dc5#K024] and order [D_614d54c918cd9c13#K001]."
    linkified = linkify_document_citations(raw_text)

    # Assert raw ID is not present bare
    assert "[D_63dfd3acb3182dc5#K024]" not in linkified
    assert "https://gad.cg.gov.in/statecirculars/Criculars/Scan-file-2000-2002/year2026/202600089_002.pdf" in linkified
    assert "https://gad.cg.gov.in/statecirculars/Criculars/Scan-file-2000-2002/year2022/202600375_001.pdf" in linkified
    assert "### 📑 प्रमाणिक स्त्रोत दस्तावेज़" in linkified


def test_anupam_toppo_officer_lookup():
    assert route_query_intent("who is the anupam asish toppo") == "OFFICER"
    res = handle_gad_query("who is the anupam asish toppo")
    assert res["success"] is True
    ans = res["answer"]
    assert "Anupam Ashish Toppo" in ans
    assert "https://gad.cg.gov.in/statecirculars/Criculars/Scan-file-2000-2002/year2026/202600089_002.pdf" in ans
    assert "[D_" not in ans


def test_mayank_agrawal_officer_lookup():
    assert route_query_intent("who is mayank agrawal?") == "OFFICER"
    res = handle_gad_query("who is mayank agrawal?")
    assert res["success"] is True
    ans = res["answer"]
    assert "Mayank Agrawal" in ans
    assert "DFO" in ans or "वन अधिकारी" in ans
    assert "Amit Agrawal" not in ans


def test_router_document_queries():
    # Both queries from the user's screenshot must route to DOCUMENT
    assert route_query_intent("Can you tell me about the leave policy?") == "DOCUMENT"
    assert route_query_intent("Can you tell me about the latest transfer order letter?") == "DOCUMENT"

    res = handle_gad_query("Can you tell me about the leave policy?")
    assert res["success"] is True
    # Must NOT produce the false officer directory lookup message
    assert "अधिकारी खोज परिणाम" not in res["answer"]
    assert "Can you tell me about the leave policy" not in res["answer"]
