"""
Phase 5: LLM Relation Extraction (on candidate sentences)
Spec: Section 4.4, Section 5 P5, Section 6 Prompt R10
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.io import read_jsonl, write_jsonl, ensure_dir
from src.common.dates import extract_dates
from src.common.numbers import normalize_number
from src.common.llm import call_llm, verify_quote_substring
from schemas.p05_relation_schemas import R10_SCHEMA


def load_prompt_template(name: str) -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / f"{name}.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_ref_map_string(ref_items: List[Dict[str, Any]]) -> str:
    if not ref_items:
        return "None"
    lines = []
    for ref in ref_items:
        idx = ref.get("idx", "?")
        num = ref.get("number_norm") or ref.get("number_raw") or "Unknown"
        d = ref.get("date") or ref.get("date_raw") or "Unknown"
        subj = ref.get("subject_raw") or ""
        lines.append(f"({idx}) {num} dated {d} {subj}".strip())
    return "\n".join(lines)


def vote_relations(
    votes_by_run: List[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Combine votes from 3 runs (temp 0.0, 0.4, 0.4).
    Matches relations by target_kind and relation type.
    """
    if not votes_by_run:
        return []

    # Flatten all relations across runs with their run index
    all_extracted = []
    for run_idx, rel_list in enumerate(votes_by_run):
        temp = 0.0 if run_idx == 0 else 0.4
        for rel in rel_list:
            all_extracted.append((run_idx, temp, rel))

    if not all_extracted:
        return []

    # Group by (relation, target_number_raw or target_ref_idx)
    groups = {}
    for run_idx, temp, rel in all_extracted:
        rtype = rel.get("relation", "REFERS_TO")
        if rtype == "NONE":
            continue
        key = (rtype, rel.get("target_ref_idx"), rel.get("target_number_raw"))
        if key not in groups:
            groups[key] = []
        groups[key].append({"run_idx": run_idx, "temp": temp, "rel": rel})

    resolved = []
    for key, group_items in groups.items():
        # Representative relation
        rep = group_items[0]["rel"]
        votes = [{"relation": item["rel"].get("relation"), "temp": item["temp"]} for item in group_items]
        count = len(group_items)
        if count >= 3:
            confidence = 1.0
        elif count == 2:
            confidence = 0.67
        else:
            confidence = 0.33

        resolved.append({
            "quote": rep.get("quote"),
            "relation": rep.get("relation"),
            "target_kind": rep.get("target_kind", "DOCUMENT"),
            "target_ref_idx": rep.get("target_ref_idx"),
            "target_number_raw": rep.get("target_number_raw"),
            "target_date_raw": rep.get("target_date_raw"),
            "target_subject_raw": rep.get("target_subject_raw"),
            "scope": rep.get("scope", "unknown"),
            "scope_text": rep.get("scope_text"),
            "votes": votes,
            "confidence": confidence,
        })

    return resolved


def extract_relations_for_doc(
    fid: str,
    doc_header: Dict[str, Any],
    feats: Dict[str, Any],
    clean_text: str,
    r10_tmpl: str,
) -> List[Dict[str, Any]]:
    """Run R10 on cue sentences of a document with 3-vote consensus."""
    all_cues = feats.get("cue_sentences") or feats.get("relation_cues", [])
    ref_items = doc_header.get("references", [])
    ref_map_str = build_ref_map_string(ref_items)

    doc_label = f"Document {doc_header.get('doc_number_norm', fid)} dated {doc_header.get('issue_date', 'Unknown')}"
    clean_lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    doc_relations = []
    targeted_ref_indices = set()

    # Filter to substantive legal cues (skip purely mechanical reference block headers)
    legal_cue_sentences = []
    for c in all_cues:
        m = c.get("matched_cues", [])
        # If it has any substantive cue or is a sentence with length >= 15
        if not m or any(k != "REFERS_TO_BLOCK_START" for k in m):
            legal_cue_sentences.append(c)

    for cue_idx, cue_item in enumerate(legal_cue_sentences[:12], 1):
        sentence = cue_item.get("sentence", "").strip()
        if not sentence or len(sentence) < 10:
            continue

        prev_sentence = cue_item.get("prev_sentence", "")
        next_sentence = cue_item.get("next_sentence", "")
        # Find local context if not already present
        if not prev_sentence or not next_sentence:
            for l_idx, line in enumerate(clean_lines):
                if sentence[:40] in line:
                    if l_idx > 0 and not prev_sentence:
                        prev_sentence = clean_lines[l_idx - 1]
                    if l_idx + 1 < len(clean_lines) and not next_sentence:
                        next_sentence = clean_lines[l_idx + 1]
                    break

        prompt = (
            r10_tmpl.replace("{{this_doc_label}}", doc_label)
            .replace("{{ref_map}}", ref_map_str)
            .replace("{{prev_sentence}}", prev_sentence[:200])
            .replace("{{sentence}}", sentence)
            .replace("{{next_sentence}}", next_sentence[:200])
        )

        # 3 calls: 1 greedy (temp=0.0, seed=0), 2 sampled (temp=0.4, seed=1, seed=2)
        run_outputs = []
        seeds = [0, 1, 2]
        temps = [0.0, 0.4, 0.4]
        for t, s in zip(temps, seeds):
            try:
                res = call_llm(
                    prompt,
                    schema=R10_SCHEMA,
                    schema_name="r10_relations",
                    temperature=t,
                    max_tokens=600,
                    seed=s,
                )
                run_outputs.append(res.get("relations", []))
            except Exception as e:
                run_outputs.append([])

        agreed_relations = vote_relations(run_outputs)

        for rel_idx, rel in enumerate(agreed_relations, 1):
            quote = rel.get("quote")
            # Rule 4: quote verification
            if quote and not verify_quote_substring(quote, clean_text):
                continue  # Reject unverified quote

            # Normalization
            t_num_raw = rel.get("target_number_raw")
            t_num_norm = normalize_number(t_num_raw) if t_num_raw else None
            t_date_raw = rel.get("target_date_raw")
            t_date_iso = None
            if t_date_raw:
                extracted_d = extract_dates(t_date_raw)
                if extracted_d:
                    t_date_iso = extracted_d[0]["iso"]

            # If target_ref_idx present, resolve against references
            t_ref_idx = rel.get("target_ref_idx")
            if t_ref_idx and 1 <= t_ref_idx <= len(ref_items):
                targeted_ref_indices.add(t_ref_idx)
                ref_obj = ref_items[t_ref_idx - 1]
                if not t_num_norm and ref_obj.get("number_norm"):
                    t_num_norm = ref_obj["number_norm"]
                if not t_num_raw and ref_obj.get("number_raw"):
                    t_num_raw = ref_obj["number_raw"]
                if not t_date_iso and ref_obj.get("date"):
                    t_date_iso = ref_obj["date"]
                if not t_date_raw and ref_obj.get("date_raw"):
                    t_date_raw = ref_obj["date_raw"]

            record = {
                "relation_id": f"REL_{fid}_{cue_idx}_{rel_idx}",
                "source_doc_id": doc_header.get("doc_id", f"D_{fid}"),
                "source_file_id": fid,
                "cue_sentence": sentence,
                "quote": quote,
                "relation": rel.get("relation"),
                "target_kind": rel.get("target_kind", "DOCUMENT"),
                "target_ref_idx": t_ref_idx,
                "target_number_raw": t_num_raw,
                "target_number_norm": t_num_norm,
                "target_date_raw": t_date_raw,
                "target_date": t_date_iso,
                "target_subject_raw": rel.get("target_subject_raw"),
                "scope": rel.get("scope", "unknown"),
                "scope_text": rel.get("scope_text"),
                "votes": rel.get("votes", []),
                "confidence": rel.get("confidence", 1.0),
            }
            doc_relations.append(record)

    # Spec Section 5 P5: Reference-block item with no cue anywhere defaults to REFERS_TO (no LLM needed)
    for r_idx, ref_obj in enumerate(ref_items, 1):
        if r_idx not in targeted_ref_indices:
            ref_num_raw = ref_obj.get("number_raw")
            ref_num_norm = ref_obj.get("number_norm")
            ref_date = ref_obj.get("date")
            ref_raw_text = ref_obj.get("raw", "")
            # Only generate default REFERS_TO if it has an identifiable number or date
            if ref_num_norm or ref_num_raw or ref_date:
                doc_relations.append({
                    "relation_id": f"REL_{fid}_REF_{r_idx}",
                    "source_doc_id": doc_header.get("doc_id", f"D_{fid}"),
                    "source_file_id": fid,
                    "cue_sentence": ref_raw_text[:200],
                    "quote": ref_raw_text[:200],
                    "relation": "REFERS_TO",
                    "target_kind": "DOCUMENT",
                    "target_ref_idx": r_idx,
                    "target_number_raw": ref_num_raw,
                    "target_number_norm": ref_num_norm,
                    "target_date_raw": ref_obj.get("date_raw"),
                    "target_date": ref_date,
                    "target_subject_raw": ref_obj.get("subject_raw"),
                    "scope": "unknown",
                    "scope_text": None,
                    "votes": [{"relation": "REFERS_TO", "temp": 0.0}],
                    "confidence": 1.0,
                })

    return doc_relations


def run_p05_pilot(
    pilot_path: str = "gad_kg/data/work/pilot_files.json",
    header_path: str = "gad_kg/data/work/header.jsonl",
    features_path: str = "gad_kg/data/work/features.jsonl",
    clean_dir: str = "gad_kg/data/clean",
    output_path: str = "gad_kg/data/work/relations_raw.jsonl",
    rejects_path: str = "gad_kg/data/work/rejects/p5_relations.jsonl",
    max_workers: int = 3,
) -> Dict[str, Any]:
    """Execute Phase 5 relation extraction on the pilot set."""
    ensure_dir(Path(output_path).parent)
    ensure_dir(Path(rejects_path).parent)

    pilot_data = json.load(open(pilot_path, encoding="utf-8"))
    pilot_fids = {item["file_id"] for item in pilot_data}

    # Load headers
    headers_map = {}
    if os.path.exists(header_path):
        for h in read_jsonl(Path(header_path)):
            if h["file_id"] in pilot_fids:
                headers_map[h["file_id"]] = h

    # Load features
    features_map = {}
    if os.path.exists(features_path):
        for f in read_jsonl(Path(features_path)):
            if f["file_id"] in pilot_fids:
                features_map[f["file_id"]] = f

    r10_tmpl = load_prompt_template("R10")

    # Resumable tracking
    done_fids = set()
    existing_relations = []
    if os.path.exists(output_path):
        for rel in read_jsonl(Path(output_path)):
            done_fids.add(rel["source_file_id"])
            existing_relations.append(rel)

    fids_to_process = [fid for fid in headers_map if fid not in done_fids]
    print(f"Phase 5 Pilot: {len(headers_map)} pilot docs in header, {len(done_fids)} already processed, {len(fids_to_process)} remaining.")

    new_relations = []
    reject_records = []
    completed = 0

    def process_one_doc(fid: str):
        try:
            h = headers_map[fid]
            feats = features_map.get(fid, {})
            clean_path = os.path.join(clean_dir, f"{fid}.md")
            clean_text = ""
            if os.path.exists(clean_path):
                with open(clean_path, "r", encoding="utf-8", errors="replace") as f:
                    clean_text = f.read()

            rels = extract_relations_for_doc(
                fid=fid,
                doc_header=h,
                feats=feats,
                clean_text=clean_text,
                r10_tmpl=r10_tmpl,
            )
            return rels, None
        except Exception as e:
            return [], {"file_id": fid, "reason": str(e)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_fid = {executor.submit(process_one_doc, fid): fid for fid in fids_to_process}
        for fut in as_completed(future_to_fid):
            fid = future_to_fid[fut]
            completed += 1
            rels, err = fut.result()
            if err:
                reject_records.append(err)
                write_jsonl(Path(rejects_path), [err], append=True)
            else:
                if rels:
                    new_relations.extend(rels)
                    write_jsonl(Path(output_path), rels, append=True)

            print(f"  [P5 Progress] {completed}/{len(fids_to_process)} docs processed (New Relations: {len(new_relations)})", flush=True)

    all_relations = existing_relations + new_relations
    relation_types = Counter(r.get("relation") for r in all_relations)

    return {
        "pilot_docs_evaluated": len(headers_map),
        "total_relations_extracted": len(all_relations),
        "relation_breakdown": dict(relation_types),
        "output_path": output_path,
        "sample": random.sample(all_relations, min(10, len(all_relations))) if all_relations else [],
    }


if __name__ == "__main__":
    res = run_p05_pilot()
    print("\n" + "=" * 60)
    print("PHASE 5 PILOT SET RESULTS: RELATION EXTRACTION")
    print("=" * 60)
    print(f"Evaluated Documents:      {res['pilot_docs_evaluated']}")
    print(f"Total Relations Extracted:{res['total_relations_extracted']}")
    print("Relation Breakdown:")
    for rtype, count in res["relation_breakdown"].items():
        print(f"  {rtype}: {count}")
    print("=" * 60)
    print("\n10 Sample Relations from Pilot:")
    for i, rel in enumerate(res["sample"], 1):
        print(f"\n--- Relation {i}: {rel['relation_id']} ---")
        print(f"  Type: {rel['relation']} | Target: {rel.get('target_number_raw') or rel.get('target_number_norm')} ({rel.get('target_date')})")
        print(f"  Confidence: {rel.get('confidence')} | Quote: {rel.get('quote')}")
