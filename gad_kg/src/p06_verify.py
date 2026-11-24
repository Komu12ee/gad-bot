"""
Phase 6: Verification & Tier Assignment
Spec: Section 4.4, Section 5 P6, Section 6 Prompt R11
"""

import os
import sys
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.io import read_jsonl, write_jsonl, ensure_dir
from src.common.llm import call_llm
from schemas.p05_relation_schemas import R11_SCHEMA


def load_prompt_template(name: str) -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / f"{name}.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def verify_relation_code_rules(
    rel: Dict[str, Any],
    source_header: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """Step 1: Deterministic code-level checks."""
    s_doc_type = source_header.get("doc_type", "")
    s_issue_date = source_header.get("issue_date")
    t_date = rel.get("target_date")
    t_kind = rel.get("target_kind", "DOCUMENT")
    rtype = rel.get("relation", "")

    # 1. Circular cannot supersede an Act
    if s_doc_type == "CIRCULAR" and t_kind == "ACT_RULE" and rtype in ("SUPERSEDES", "RESCINDS"):
        return False, "circular_cannot_supersede_act"

    # 2. Corrigendum cannot change whole document
    if s_doc_type == "CORRIGENDUM" and rel.get("scope") == "whole":
        rel["scope"] = "partial"  # Auto-correct scope for corrigendum

    # 3. Order cannot amend/supersede document issued later
    if s_issue_date and t_date and t_date > s_issue_date:
        if rtype in ("SUPERSEDES", "AMENDS", "RESCINDS", "EXTENDS", "CORRECTS"):
            return False, f"date_inversion: target {t_date} > source {s_issue_date}"

    # 4. Self-reference check
    s_num = source_header.get("doc_number_norm")
    t_num = rel.get("target_number_norm")
    if s_num and t_num and s_num == t_num:
        return False, "self_reference"

    return True, None


def verify_relation_llm(
    rel: Dict[str, Any],
    source_header: Dict[str, Any],
    clean_text: str,
    r11_tmpl: str,
) -> Dict[str, Any]:
    """Step 2: LLM verification on ambiguous cases (confidence < 0.8)."""
    doc_label = f"Document {source_header.get('doc_number_norm', rel['source_file_id'])} dated {source_header.get('issue_date', 'Unknown')}"
    rel_in_words = f"{rel['relation']} (changes/replaces/refers to)"
    target_desc = f"{rel.get('target_number_raw') or rel.get('target_number_norm') or 'document'} dated {rel.get('target_date') or rel.get('target_date_raw') or 'unknown'}"

    quote = rel.get("quote", "")
    prompt = (
        r11_tmpl.replace("{{this_doc_label}}", doc_label)
        .replace("{{relation_in_words}}", rel_in_words)
        .replace("{{target_description}}", target_desc)
        .replace("{{scope}}", rel.get("scope", "unknown"))
        .replace("{{quote}}", quote)
        .replace("{{prev_sentence}}", "")
        .replace("{{next_sentence}}", "")
    )

    try:
        res = call_llm(prompt, schema=R11_SCHEMA, schema_name="r11_verify", temperature=0.0)
        return res
    except Exception as e:
        return {"verdict": "UNCLEAR", "reason": str(e)}


def assign_tier(
    rel: Dict[str, Any],
    code_passed: bool,
    reject_reason: Optional[str],
    llm_verdict: Optional[str],
) -> Tuple[str, Optional[str]]:
    """Step 3: Tier assignment (Tier A, Tier B, Tier C)."""
    if not code_passed:
        return "Tier C", reject_reason

    conf = rel.get("confidence", 0.0)
    rtype = rel.get("relation", "")

    if llm_verdict == "NOT_SUPPORTED":
        return "Tier C", "llm_not_supported"

    # Tier A criteria
    if conf >= 0.9 and code_passed and (llm_verdict in (None, "SUPPORTED")):
        if rtype in ("SUPERSEDES", "AMENDS", "RESCINDS", "CORRECTS", "EXTENDS", "CLARIFIES", "IMPLEMENTS", "REFERS_TO"):
            return "Tier A", None

    # Tier B criteria
    if conf >= 0.6 and (llm_verdict in (None, "SUPPORTED", "UNCLEAR")):
        return "Tier B", None

    return "Tier C", "low_confidence_or_unresolved"


def verify_relations_pipeline(
    relations_raw_path: str = "gad_kg/data/work/relations_raw.jsonl",
    header_path: str = "gad_kg/data/work/header.jsonl",
    clean_dir: str = "gad_kg/data/clean",
    output_path: str = "gad_kg/data/work/relations_verified.jsonl",
    review_csv_path: str = "gad_kg/data/work/review/p6_relations_review.csv",
) -> Dict[str, Any]:
    ensure_dir(Path(output_path).parent)
    ensure_dir(Path(review_csv_path).parent)

    # Load headers for lookup
    headers = {}
    if os.path.exists(header_path):
        for h in read_jsonl(Path(header_path)):
            headers[h["file_id"]] = h

    r11_tmpl = load_prompt_template("R11")

    verified_records = []
    review_records = []

    tier_counts = {"Tier A": 0, "Tier B": 0, "Tier C": 0}

    if os.path.exists(relations_raw_path):
        raw_relations = list(read_jsonl(Path(relations_raw_path)))
    else:
        raw_relations = []

    for rel in raw_relations:
        fid = rel.get("source_file_id")
        header = headers.get(fid, {})

        # 1. Code checks
        code_passed, reject_reason = verify_relation_code_rules(rel, header)

        # 2. LLM check if needed
        llm_verdict = None
        llm_reason = None
        if code_passed and rel.get("confidence", 1.0) < 0.8:
            clean_path = os.path.join(clean_dir, f"{fid}.md")
            clean_text = ""
            if os.path.exists(clean_path):
                with open(clean_path, "r", encoding="utf-8", errors="replace") as f:
                    clean_text = f.read()
            llm_res = verify_relation_llm(rel, header, clean_text, r11_tmpl)
            llm_verdict = llm_res.get("verdict")
            llm_reason = llm_res.get("reason")

        # 3. Tier assignment
        tier, tier_reason = assign_tier(rel, code_passed, reject_reason, llm_verdict)
        tier_counts[tier] += 1

        rel["tier"] = tier
        rel["llm_verification"] = {"verdict": llm_verdict, "reason": llm_reason} if llm_verdict else None
        rel["verification_notes"] = tier_reason

        verified_records.append(rel)
        if tier == "Tier C":
            review_records.append({
                "relation_id": rel.get("relation_id"),
                "source_file_id": fid,
                "relation": rel.get("relation"),
                "quote": rel.get("quote"),
                "target_number_raw": rel.get("target_number_raw"),
                "target_date": rel.get("target_date"),
                "reason": tier_reason,
            })

    write_jsonl(Path(output_path), verified_records)

    # Write review CSV if any
    if review_records:
        keys = ["relation_id", "source_file_id", "relation", "quote", "target_number_raw", "target_date", "reason"]
        with open(review_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(review_records)

    return {
        "total": len(raw_relations),
        "tier_counts": tier_counts,
        "verified_path": output_path,
        "review_path": review_csv_path,
    }


if __name__ == "__main__":
    res = verify_relations_pipeline()
    print("\n" + "=" * 60)
    print("PHASE 6 VERIFICATION & TIER ASSIGNMENT RESULTS")
    print("=" * 60)
    print(f"Total Relations Processed: {res['total']}")
    print(f"Tier Breakdown:            {res['tier_counts']}")
    print(f"Verified Output:           {res['verified_path']}")
    print(f"Human Review CSV:          {res['review_path']}")
    print("=" * 60)
