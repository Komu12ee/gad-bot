"""
Phase 3: Deterministic Feature Extraction (No LLM)
Spec: Section 5 P3, Section 7.1 - 7.5

Extracts candidate features for each canonical document:
  - number_candidates (with in_top, in_bottom, in_reference_block)
  - date_candidates (with in_top, in_bottom, in_reference_block)
  - reference_block (items, start/end lines)
  - cue_sentences (relation cues + context)
  - effective_sentences (effective-date cues + context)
  - tables (spans, captions, paragraphs below)
  - clause_markers (line numbers, pattern types)
Outputs:
  - work/features.jsonl
"""

import os
import sys
import re
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.io import read_jsonl, write_jsonl, ensure_dir
from src.common.dates import extract_dates
from src.common.numbers import extract_number_candidates, normalize_number

# Reference cues
REF_START_PATTERNS = re.compile(
    r"^\s*(?:[-*#>\s]*)\s*(संदर्भ|सन्दर्भ|वाचा|पढ़ें|read\s*:|reference\s*:?|references\s*:?|ref\s*[:.]|refer\s*:?)",
    re.IGNORECASE | re.UNICODE
)
REF_END_PATTERNS = re.compile(
    r"^\s*(?:[-*#>\s]*)\s*(प्रस्तावना|preamble|विषय\s*[:\-]|subject\s*[:\-]|शासन\s*निर्णय|government\s*resolution)",
    re.IGNORECASE | re.UNICODE
)

ITEM_SPLIT_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?(?:(\d{1,3})[.)]|\((\d{1,3})\)|([क-ह])[.)]|\(([क-ह])\))\s*",
    re.UNICODE
)

CLAUSE_PATTERNS = [
    ("NUM_DOT", re.compile(r"^\s*(\d{1,3})\s*[.)]\s+")),
    ("NUM_PAREN", re.compile(r"^\s*\((\d{1,3})\)\s+")),
    ("ALPHA_PAREN", re.compile(r"^\s*\(([a-z])\)\s+")),
    ("ROMAN_PAREN", re.compile(r"^\s*\(((?:i{1,3}|iv|v|vi{0,3}|ix|x))\)\s+")),
    ("DEV_DOT", re.compile(r"^\s*([क-ह])\s*[.)]\s+")),
    ("DEV_PAREN", re.compile(r"^\s*\(([क-ह])\)\s+")),
    ("NAMED_CLAUSE", re.compile(r"^\s*(परिच्छेद|कलम|अनुच्छेद|धारा|para(?:graph)?|clause|section|rule)\s+(\d+)", re.I)),
    ("MD_HEADING", re.compile(r"^\s*#{1,4}\s+")),
]


def load_lexicons(lexicons_dir: str = "gad_kg/lexicons") -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Load relation cues and effective cues from YAML."""
    rel_path = os.path.join(lexicons_dir, "relation_cues.yaml")
    eff_path = os.path.join(lexicons_dir, "effective_cues.yaml")

    rel_cues = {}
    if os.path.exists(rel_path):
        with open(rel_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            for k, langs in raw.items():
                if isinstance(langs, dict):
                    cues = []
                    for lang, phrases in langs.items():
                        cues.extend(phrases)
                    rel_cues[k] = cues

    eff_cues = {}
    if os.path.exists(eff_path):
        with open(eff_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            for k, langs in raw.items():
                if isinstance(langs, dict):
                    cues = []
                    for lang, phrases in langs.items():
                        cues.extend(phrases)
                    eff_cues[k] = cues

    return rel_cues, eff_cues


def extract_reference_block(lines: List[str]) -> Dict[str, Any]:
    """Detect reference block and extract individual numbered items."""
    ref_start_line = -1
    ref_end_line = -1

    for idx, line in enumerate(lines[:60]):  # Reference blocks appear near document start
        if REF_START_PATTERNS.search(line):
            ref_start_line = idx
            break

    if ref_start_line == -1:
        return {"found": False, "start_line": -1, "end_line": -1, "raw_text": "", "items": []}

    # Find end line
    blank_lines = 0
    for idx in range(ref_start_line + 1, min(len(lines), ref_start_line + 40)):
        line = lines[idx]
        if REF_END_PATTERNS.search(line):
            ref_end_line = idx
            break
        if not line.strip():
            blank_lines += 1
            if blank_lines >= 2 and idx > ref_start_line + 2:
                ref_end_line = idx
                break
        else:
            blank_lines = 0

    if ref_end_line == -1:
        ref_end_line = min(len(lines), ref_start_line + 15)

    ref_lines = lines[ref_start_line:ref_end_line]
    raw_text = "\n".join(ref_lines)

    # Split into numbered items
    items = []
    current_item = []
    for line in ref_lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\s*(?:[-*]\s*)?(?:\d{1,3}[.)]|\(\d{1,3}\)|[क-ह][.)]|\([क-ह]\))\s*", stripped):
            if current_item:
                items.append(" ".join(current_item))
                current_item = []
            current_item.append(stripped)
        else:
            if current_item:
                current_item.append(stripped)
            elif stripped:
                current_item.append(stripped)

    if current_item:
        items.append(" ".join(current_item))

    return {
        "found": True,
        "start_line": ref_start_line,
        "end_line": ref_end_line,
        "raw_text": raw_text,
        "items": items,
    }


def extract_tables(lines: List[str]) -> List[Dict[str, Any]]:
    """Extract markdown table spans, nearest caption above, and paragraph below."""
    tables = []
    in_table = False
    start_line = -1
    table_lines = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        is_table_row = stripped.startswith("|") and stripped.endswith("|")

        if is_table_row:
            if not in_table:
                in_table = True
                start_line = idx
                table_lines = [line]
            else:
                table_lines.append(line)
        else:
            if in_table:
                in_table = False
                end_line = idx - 1

                # Caption above: look back up to 3 non-empty lines
                caption = ""
                for back_idx in range(start_line - 1, max(-1, start_line - 4), -1):
                    cand = lines[back_idx].strip()
                    if cand and not cand.startswith("|"):
                        caption = cand
                        break

                # Paragraph below: first non-empty line after table
                paragraph_below = ""
                for fwd_idx in range(end_line + 1, min(len(lines), end_line + 5)):
                    cand = lines[fwd_idx].strip()
                    if cand and not cand.startswith("|"):
                        paragraph_below = cand
                        break

                tables.append({
                    "start_line": start_line,
                    "end_line": end_line,
                    "row_count": len(table_lines),
                    "caption_above": caption,
                    "paragraph_below": paragraph_below,
                    "sample_header": table_lines[0] if table_lines else "",
                })

    if in_table:
        end_line = len(lines) - 1
        caption = ""
        for back_idx in range(start_line - 1, max(-1, start_line - 4), -1):
            cand = lines[back_idx].strip()
            if cand:
                caption = cand
                break
        tables.append({
            "start_line": start_line,
            "end_line": end_line,
            "row_count": len(table_lines),
            "caption_above": caption,
            "paragraph_below": "",
            "sample_header": table_lines[0] if table_lines else "",
        })

    return tables


def split_into_sentences(text: str) -> List[Tuple[str, int, int]]:
    """Split text into sentences with char start and end offsets."""
    sentences = []
    # Sentence delimiters: ।, ., ?, !, \n\n
    splits = re.finditer(r"([^।?!.\n]+[।?!.]?|\n{2,})", text)
    for m in splits:
        s = m.group(0).strip()
        if len(s) >= 5:
            sentences.append((s, m.start(), m.end()))
    return sentences


def extract_document_features(
    clean_text: str,
    search_text: str,
    rel_cues: Dict[str, List[str]],
    eff_cues: Dict[str, List[str]],
) -> Dict[str, Any]:
    lines = clean_text.splitlines()
    search_lines = search_text.splitlines()
    total_lines = len(lines)

    # 1. Reference block
    ref_block = extract_reference_block(lines)
    ref_start = ref_block["start_line"]
    ref_end = ref_block["end_line"]

    # 2. Number candidates
    number_candidates = []
    seen_norm_numbers = set()
    for idx, s_line in enumerate(search_lines):
        cands = extract_number_candidates(s_line, line_no=idx)
        in_top = (idx <= 25)
        in_bottom = (idx >= max(0, total_lines - 25))
        in_ref = (ref_block["found"] and ref_start <= idx <= ref_end)

        for c in cands:
            key = (c["normalized"], idx)
            if key not in seen_norm_numbers:
                seen_norm_numbers.add(key)
                number_candidates.append({
                    "raw": c["raw"],
                    "normalized": c["normalized"],
                    "line_no": idx,
                    "in_top": in_top,
                    "in_bottom": in_bottom,
                    "in_reference_block": in_ref,
                    "context": c["context"],
                })

    # 3. Date candidates
    date_candidates = []
    seen_dates = set()
    for idx, s_line in enumerate(search_lines):
        extracted = extract_dates(s_line)
        in_top = (idx <= 25)
        in_bottom = (idx >= max(0, total_lines - 25))
        in_ref = (ref_block["found"] and ref_start <= idx <= ref_end)

        for d in extracted:
            key = (d["iso"], idx)
            if key not in seen_dates:
                seen_dates.add(key)
                date_candidates.append({
                    "raw": d["raw"],
                    "iso": d["iso"],
                    "line_no": idx,
                    "in_top": in_top,
                    "in_bottom": in_bottom,
                    "in_reference_block": in_ref,
                    "flags": d["flags"],
                    "context": s_line[:150],
                })

    # 4. Tables
    tables = extract_tables(lines)

    # 5. Clause markers
    clause_markers = []
    for idx, line in enumerate(lines):
        for pattern_name, pat in CLAUSE_PATTERNS:
            m = pat.match(line)
            if m:
                clause_markers.append({
                    "line_no": idx,
                    "pattern": pattern_name,
                    "match": m.group(0).strip(),
                })
                break

    # 6. Cue sentences & effective sentences
    sentences = split_into_sentences(clean_text)
    cue_sentences = []
    effective_sentences = []

    for s_idx, (sent_text, start_char, end_char) in enumerate(sentences):
        sent_lower = sent_text.lower()
        prev_s = sentences[s_idx - 1][0] if s_idx > 0 else ""
        next_s = sentences[s_idx + 1][0] if s_idx + 1 < len(sentences) else ""

        # Relation cues matching
        matched_rel = []
        for rel_type, phrases in rel_cues.items():
            for p in phrases:
                if p.lower() in sent_lower:
                    matched_rel.append(rel_type)
                    break

        if matched_rel:
            cue_sentences.append({
                "sentence": sent_text,
                "prev_sentence": prev_s,
                "next_sentence": next_s,
                "matched_cues": matched_rel,
                "char_start": start_char,
                "char_end": end_char,
            })

        # Effective date cues matching
        matched_eff = []
        for eff_type, phrases in eff_cues.items():
            for p in phrases:
                if p.lower() in sent_lower:
                    matched_eff.append(eff_type)
                    break

        if matched_eff:
            effective_sentences.append({
                "sentence": sent_text,
                "prev_sentence": prev_s,
                "next_sentence": next_s,
                "matched_cues": matched_eff,
                "char_start": start_char,
                "char_end": end_char,
            })

    return {
        "number_candidates": number_candidates,
        "date_candidates": date_candidates,
        "reference_block": ref_block,
        "cue_sentences": cue_sentences,
        "effective_sentences": effective_sentences,
        "tables": tables,
        "clause_markers": clause_markers,
    }


def run_p03_features(
    clean_dir: str = "gad_kg/data/clean",
    dedup_path: str = "gad_kg/data/work/dedup.jsonl",
    output_path: str = "gad_kg/data/work/features.jsonl",
    lexicons_dir: str = "gad_kg/lexicons",
) -> Dict[str, Any]:
    ensure_dir(Path(output_path).parent)
    rel_cues, eff_cues = load_lexicons(lexicons_dir)

    # Read canonical files only
    canonical_files = []
    if os.path.exists(dedup_path):
        for rec in read_jsonl(Path(dedup_path)):
            if not rec.get("is_duplicate", False):
                canonical_files.append(rec["canonical_file_id"])

    print(f"[*] Processing features for {len(canonical_files)} canonical documents...")
    records = []
    docs_with_numbers = 0
    docs_with_dates = 0
    docs_with_ref_blocks = 0
    docs_with_tables = 0
    docs_with_cues = 0

    for idx, fid in enumerate(canonical_files):
        clean_file = os.path.join(clean_dir, f"{fid}.md")
        search_file = os.path.join(clean_dir, f"{fid}.search.txt")

        if not os.path.exists(clean_file) or not os.path.exists(search_file):
            continue

        with open(clean_file, "r", encoding="utf-8", errors="replace") as f:
            clean_text = f.read()
        with open(search_file, "r", encoding="utf-8", errors="replace") as f:
            search_text = f.read()

        feats = extract_document_features(clean_text, search_text, rel_cues, eff_cues)
        feats["file_id"] = fid
        feats["doc_id"] = f"D_{fid}"

        if feats["number_candidates"]:
            docs_with_numbers += 1
        if feats["date_candidates"]:
            docs_with_dates += 1
        if feats["reference_block"]["found"]:
            docs_with_ref_blocks += 1
        if feats["tables"]:
            docs_with_tables += 1
        if feats["cue_sentences"]:
            docs_with_cues += 1

        records.append(feats)

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(canonical_files):
            print(f"    Progress: {idx + 1}/{len(canonical_files)} completed...")

    write_jsonl(Path(output_path), records)
    print(f"[*] Successfully saved {len(records)} feature records to {output_path}")

    total = max(1, len(records))
    return {
        "total_canonical": total,
        "docs_with_numbers": docs_with_numbers,
        "number_recall_rate": docs_with_numbers / total,
        "docs_with_dates": docs_with_dates,
        "date_recall_rate": docs_with_dates / total,
        "docs_with_ref_blocks": docs_with_ref_blocks,
        "docs_with_tables": docs_with_tables,
        "docs_with_cues": docs_with_cues,
    }


if __name__ == "__main__":
    results = run_p03_features()
    print("\n" + "=" * 60)
    print("PHASE 3 ACCEPTANCE RESULTS: DETERMINISTIC FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Total Canonical Documents:    {results['total_canonical']}")
    print(f"Docs with Number Candidates:  {results['docs_with_numbers']} ({results['number_recall_rate']*100:.2f}%)")
    print(f"Docs with Date Candidates:    {results['docs_with_dates']} ({results['date_recall_rate']*100:.2f}%)")
    print(f"Docs with Reference Blocks:   {results['docs_with_ref_blocks']} ({results['docs_with_ref_blocks']/results['total_canonical']*100:.2f}%)")
    print(f"Docs with Tables:             {results['docs_with_tables']}")
    print(f"Docs with Relation Cues:      {results['docs_with_cues']}")
    print("=" * 60)
