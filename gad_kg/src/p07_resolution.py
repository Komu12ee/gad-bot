"""
Phase 7: Reference Resolution
Spec: Section 4.4, Section 5 P7, Section 8.1
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from rapidfuzz import fuzz

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.io import read_jsonl, write_jsonl, ensure_dir


def build_doc_lookup_indices(headers: List[Dict[str, Any]]) -> Tuple[Dict[Tuple[str, str], List[str]], Dict[str, List[str]]]:
    """Index canonical documents by (number_norm, date) and number_norm alone."""
    num_date_index = {}
    num_only_index = {}

    for h in headers:
        doc_id = h.get("doc_id", f"D_{h['file_id']}")
        num_norm = h.get("doc_number_norm")
        issue_date = h.get("issue_date")

        if num_norm:
            num_only_index.setdefault(num_norm, []).append(doc_id)
            if issue_date:
                num_date_index.setdefault((num_norm, issue_date), []).append(doc_id)

    return num_date_index, num_only_index


def resolve_relation_target(
    rel: Dict[str, Any],
    num_date_index: Dict[Tuple[str, str], List[str]],
    num_only_index: Dict[str, List[str]],
    all_headers_by_id: Dict[str, Dict[str, Any]],
    stub_counter: int,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], int]:
    """Resolve relation target to canonical doc_id, Instrument, or Stub.
    
    Returns (edge_record, new_stub_or_none, updated_stub_counter)
    """
    t_kind = rel.get("target_kind", "DOCUMENT")
    t_num_norm = rel.get("target_number_norm")
    t_num_raw = rel.get("target_number_raw")
    t_date = rel.get("target_date")

    target_doc_id = None
    resolution_method = None
    stub_record = None

    if t_kind == "ACT_RULE":
        inst_name = t_num_raw or rel.get("target_subject_raw") or "ACT_RULE"
        clean_inst = "".join(c if c.isalnum() else "_" for c in inst_name)[:40].strip("_")
        target_doc_id = f"INST_{clean_inst}"
        resolution_method = "instrument_rule"

    elif t_kind == "COURT_ORDER":
        order_name = t_num_raw or "COURT_ORDER"
        clean_order = "".join(c if c.isalnum() else "_" for c in order_name)[:40].strip("_")
        target_doc_id = f"COURT_{clean_order}"
        resolution_method = "court_order"

    else:
        # DOCUMENT target resolution hierarchy:
        # 1. Exact match on (number_norm, date)
        if t_num_norm and t_date and (t_num_norm, t_date) in num_date_index:
            candidates = num_date_index[(t_num_norm, t_date)]
            target_doc_id = candidates[0]
            resolution_method = "exact_number_and_date"

        # 2. Exact match on number_norm alone
        elif t_num_norm and t_num_norm in num_only_index:
            candidates = num_only_index[t_num_norm]
            target_doc_id = candidates[0]
            resolution_method = "exact_number"

        # 3. Fuzzy match on number_raw
        elif t_num_raw:
            best_match = None
            best_score = 0.0
            for cand_num, c_doc_ids in num_only_index.items():
                score = fuzz.ratio(t_num_raw.lower(), cand_num.lower())
                if score > best_score and score >= 85:
                    best_score = score
                    best_match = c_doc_ids[0]
            if best_match:
                target_doc_id = best_match
                resolution_method = "fuzzy_number"

        # 4. If unresolved: create a Stub document node
        if not target_doc_id:
            stub_counter += 1
            stub_id = f"STUB_{stub_counter:05d}"
            target_doc_id = stub_id
            resolution_method = "stub_created"
            stub_record = {
                "stub_id": stub_id,
                "number_raw": t_num_raw,
                "number_norm": t_num_norm,
                "date": t_date,
                "subject_raw": rel.get("target_subject_raw"),
                "source_file_id": rel.get("source_file_id"),
            }

    edge_record = {
        "edge_id": f"E_{rel['relation_id']}",
        "relation_id": rel["relation_id"],
        "source_doc_id": rel["source_doc_id"],
        "target_doc_id": target_doc_id,
        "relation": rel["relation"],
        "target_kind": t_kind,
        "scope": rel.get("scope", "whole"),
        "scope_text": rel.get("scope_text"),
        "quote": rel.get("quote"),
        "effective_date": t_date,
        "tier": rel.get("tier", "Tier A"),
        "confidence": rel.get("confidence", 1.0),
        "resolution_method": resolution_method,
    }

    return edge_record, stub_record, stub_counter


def resolve_references_pipeline(
    relations_path: str = "gad_kg/data/work/relations_verified.jsonl",
    header_path: str = "gad_kg/data/work/header.jsonl",
    edges_output_path: str = "gad_kg/data/work/edges.jsonl",
    stubs_output_path: str = "gad_kg/data/work/stubs.jsonl",
) -> Dict[str, Any]:
    ensure_dir(Path(edges_output_path).parent)
    ensure_dir(Path(stubs_output_path).parent)

    headers = list(read_jsonl(Path(header_path))) if os.path.exists(header_path) else []
    headers_by_id = {h.get("doc_id", f"D_{h['file_id']}"): h for h in headers}
    num_date_index, num_only_index = build_doc_lookup_indices(headers)

    relations = list(read_jsonl(Path(relations_path))) if os.path.exists(relations_path) else []

    edges = []
    stubs = []
    stub_counter = 0

    for rel in relations:
        edge, stub, stub_counter = resolve_relation_target(
            rel=rel,
            num_date_index=num_date_index,
            num_only_index=num_only_index,
            all_headers_by_id=headers_by_id,
            stub_counter=stub_counter,
        )
        edges.append(edge)
        if stub:
            stubs.append(stub)

    write_jsonl(Path(edges_output_path), edges)
    write_jsonl(Path(stubs_output_path), stubs)

    return {
        "edges_count": len(edges),
        "stubs_count": len(stubs),
        "edges_path": edges_output_path,
        "stubs_path": stubs_output_path,
    }


if __name__ == "__main__":
    res = resolve_references_pipeline()
    print("\n" + "=" * 60)
    print("PHASE 7 REFERENCE RESOLUTION RESULTS")
    print("=" * 60)
    print(f"Total Edges Created:       {res['edges_count']}")
    print(f"Total Stubs Created:       {res['stubs_count']}")
    print(f"Edges Output:              {res['edges_path']}")
    print(f"Stubs Output:              {res['stubs_path']}")
    print("=" * 60)
