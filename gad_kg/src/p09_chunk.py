"""
Phase 9: Clause Segmentation & Chunking
Spec: Section 4.2, Section 5 P9, Section 7.5
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.io import read_jsonl, write_jsonl, ensure_dir


CLAUSE_PATTERNS = [
    (r"^\s*(\d{1,3})\s*[.)]\s+", "numbered"),                     # 1.  2)
    (r"^\s*\((\d{1,3})\)\s+", "paren_numbered"),                  # (1)
    (r"^\s*\(([a-z])\)\s+", "lettered"),                         # (a)
    (r"^\s*\(((?:i{1,3}|iv|v|vi{0,3}|ix|x))\)\s+", "roman"),     # (i) (ii) (iv)
    (r"^\s*([क-ह])\s*[.)]\s+", "hindi_lettered"),                 # क)  ख.
    (r"^\s*\(([क-ह])\)\s+", "paren_hindi_lettered"),              # (क)
    (r"^\s*(परिच्छेद|कलम|अनुच्छेद|धारा|Para(?:graph)?|Clause|Section|Rule)\s+(\d+)", "named_para"),
    (r"^\s*(#{1,4})\s+(.+)", "heading"),                          # markdown headings
]


def classify_section(line: str, current_section: str) -> str:
    """Identify document sections: preamble, reference, operative, annexure, signature."""
    lower = line.strip().lower()

    if re.search(r"^\s*(संदर्भ|reference)[\s:–-]", line, re.I):
        return "reference"

    if re.search(r"(शासन\s+निर्णय|अधिसूचना|notification|\bआदेश\b|\border\b)", line, re.I):
        if not re.search(r"राज्यपाल के नाम से|आदेशानुसार", line, re.I):
            return "operative"

    if re.search(r"^\s*(परिशिष्ट|अनुबंध|प्रपत्र|annexure|schedule|appendix)[\s:–-]", line, re.I):
        return "annexure"

    if re.search(r"(राज्यपाल के नाम से|हस्ताक्षर|आदेशानुसार|उप सचिव|संयुक्त सचिव|अवर सचिव|secretary to govt)", line, re.I):
        return "signature"

    return current_section


def estimate_tokens(text: str) -> int:
    """Rough token estimation for English/Devanagari text."""
    # Words in Hindi / English average ~1.3 to 1.8 tokens
    words = text.split()
    return int(len(words) * 1.4)


def split_sentences(text: str) -> List[str]:
    """Split text on sentence boundaries (Devanagari danda | or English period .)."""
    # Danda, double danda, or period followed by whitespace
    sentences = re.split(r"(?<=[।!?.\n])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def segment_document_clauses(
    doc_id: str,
    file_id: str,
    clean_text: str,
) -> List[Dict[str, Any]]:
    """Segment clean document text into clauses with section labels and parent hierarchy."""
    lines = clean_text.splitlines(keepends=True)
    clauses = []
    current_clause_lines = []
    current_section = "preamble"
    clause_idx = 0
    current_parent_clause_id = None
    current_clause_kind = None

    line_offset = 0
    clause_start_char = 0
    clause_number = None

    for line in lines:
        stripped = line.strip()
        new_section = classify_section(stripped, current_section)
        is_new_section = new_section != current_section

        # Check clause patterns
        is_new_clause = False
        matched_number = None
        matched_kind = None

        if stripped:
            for pat, kind in CLAUSE_PATTERNS:
                m = re.match(pat, stripped, flags=re.I)
                if m:
                    is_new_clause = True
                    matched_kind = kind
                    matched_number = m.group(1) if len(m.groups()) >= 1 else None
                    break

        if (is_new_clause or is_new_section) and current_clause_lines:
            clause_text = "".join(current_clause_lines).strip()
            if clause_text:
                clause_idx += 1
                c_id = f"{doc_id}#C{clause_idx:03d}"
                is_sub = current_clause_kind in ("lettered", "roman", "paren_hindi_lettered", "hindi_lettered")
                parent_id = current_parent_clause_id if is_sub else None
                clauses.append({
                    "clause_id": c_id,
                    "doc_id": doc_id,
                    "file_id": file_id,
                    "number": clause_number or str(clause_idx),
                    "section_type": current_section,
                    "text": clause_text,
                    "char_start": clause_start_char,
                    "char_end": clause_start_char + len("".join(current_clause_lines)),
                    "parent_clause_id": parent_id,
                })
                if current_clause_kind in ("numbered", "paren_numbered", "named_para"):
                    current_parent_clause_id = c_id

            current_clause_lines = []
            clause_start_char = line_offset

        if is_new_section:
            current_section = new_section

        if is_new_clause:
            clause_number = matched_number
            current_clause_kind = matched_kind
        elif is_new_section:
            current_clause_kind = None

        current_clause_lines.append(line)
        line_offset += len(line)

    # Flush final clause
    if current_clause_lines:
        clause_text = "".join(current_clause_lines).strip()
        if clause_text:
            clause_idx += 1
            c_id = f"{doc_id}#C{clause_idx:03d}"
            is_sub = current_clause_kind in ("lettered", "roman", "paren_hindi_lettered", "hindi_lettered")
            clauses.append({
                "clause_id": c_id,
                "doc_id": doc_id,
                "file_id": file_id,
                "number": clause_number or str(clause_idx),
                "section_type": current_section,
                "text": clause_text,
                "char_start": clause_start_char,
                "char_end": clause_start_char + len("".join(current_clause_lines)),
                "parent_clause_id": current_parent_clause_id if is_sub else None,
            })

    return clauses


def enforce_token_ceiling(
    text: str,
    max_tokens: int = 500,
    overlap_words: int = 20,
) -> List[str]:
    """Guarantees strictly <= 700 tokens per chunk under all conditions (Spec Section 5 P9)."""
    if estimate_tokens(text) <= max_tokens:
        return [text]

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    table_lines = [l for l in lines if l.startswith("|")]

    # If any single row in table_lines has > max_tokens, it cannot be kept as a single row
    has_giant_rows = any(estimate_tokens(l) > max_tokens for l in table_lines)

    # If it is predominantly a markdown table with normal rows
    if len(table_lines) >= 3 and len(table_lines) >= len(lines) * 0.5 and not has_giant_rows:
        header = table_lines[0]
        sep = table_lines[1] if re.match(r"^\|[\s\-:|]+\|$", table_lines[1]) else "|---|---|"
        data_rows = table_lines[2:] if sep == table_lines[1] else table_lines[1:]
        sub_tbls = []
        curr_rows = []
        for r in data_rows:
            cand = f"{header}\n{sep}\n" + "\n".join(curr_rows + [r])
            if estimate_tokens(cand) > max_tokens and curr_rows:
                sub_tbls.append(f"{header}\n{sep}\n" + "\n".join(curr_rows))
                curr_rows = [r]
            else:
                curr_rows.append(r)
        if curr_rows:
            sub_tbls.append(f"{header}\n{sep}\n" + "\n".join(curr_rows))
        if sub_tbls:
            return sub_tbls

    # Universal word-level slicing with overlap: strictly <= max_tokens
    words = text.split()
    step = int(max_tokens / 1.5)  # ~330 words (~460 tokens)
    word_chunks = []
    for i in range(0, len(words), max(1, step - overlap_words)):
        w_sub = words[i:i + step]
        if w_sub:
            word_chunks.append(" ".join(w_sub))
    return word_chunks or [text]


def chunk_clauses(
    clauses: List[Dict[str, Any]],
    doc_header: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Group short clauses (< 350 tokens) and split long clauses (> 600 tokens) with 1-sentence overlap.
    
    Acceptance invariant: Strictly <= 700 tokens per chunk.
    """
    raw_chunks = []
    chunk_idx = 0
    buffer_clauses = []
    buffer_text = ""

    doc_id = doc_header.get("doc_id", "D_UNKNOWN")
    file_id = doc_header.get("file_id", "UNKNOWN")

    def flush_buffer(clause_list):
        nonlocal chunk_idx
        if not clause_list:
            return
        combined_text = "\n\n".join(c["text"] for c in clause_list)
        chunk_idx += 1
        raw_chunks.append({
            "chunk_id": f"{doc_id}#K{chunk_idx:03d}",
            "doc_id": doc_id,
            "file_id": file_id,
            "clause_ids": [c["clause_id"] for c in clause_list],
            "primary_clause_id": clause_list[0]["clause_id"],
            "section_type": clause_list[0]["section_type"],
            "text": combined_text,
            "char_start": clause_list[0]["char_start"],
            "char_end": clause_list[-1]["char_end"],
            "token_count": estimate_tokens(combined_text),
        })

    for clause in clauses:
        c_text = clause["text"]
        c_tokens = estimate_tokens(c_text)

        if c_tokens > 500:
            if buffer_clauses:
                flush_buffer(buffer_clauses)
                buffer_clauses = []
                buffer_text = ""

            sub_texts = enforce_token_ceiling(c_text, max_tokens=500)
            for sub_t in sub_texts:
                chunk_idx += 1
                raw_chunks.append({
                    "chunk_id": f"{doc_id}#K{chunk_idx:03d}",
                    "doc_id": doc_id,
                    "file_id": file_id,
                    "clause_ids": [clause["clause_id"]],
                    "primary_clause_id": clause["clause_id"],
                    "section_type": clause["section_type"],
                    "text": sub_t,
                    "char_start": clause["char_start"],
                    "char_end": clause["char_end"],
                    "token_count": estimate_tokens(sub_t),
                })
        else:
            cand_tokens = estimate_tokens(buffer_text + "\n\n" + c_text) if buffer_text else c_tokens
            if cand_tokens <= 350:
                buffer_clauses.append(clause)
                buffer_text = buffer_text + "\n\n" + c_text if buffer_text else c_text
            else:
                flush_buffer(buffer_clauses)
                buffer_clauses = [clause]
                buffer_text = c_text

    if buffer_clauses:
        flush_buffer(buffer_clauses)

    # Final pass: enforce invariant strictly <= 700 tokens on all output chunks
    final_chunks = []
    final_idx = 0
    for rch in raw_chunks:
        if rch["token_count"] > 600:
            words = rch["text"].split()
            step = 300  # ~420 tokens
            overlap = 25
            for i in range(0, len(words), max(1, step - overlap)):
                w_sub = words[i:i + step]
                if w_sub:
                    sub_text = " ".join(w_sub)
                    final_idx += 1
                    final_chunks.append({
                        "chunk_id": f"{doc_id}#K{final_idx:03d}",
                        "doc_id": doc_id,
                        "file_id": file_id,
                        "clause_ids": rch["clause_ids"],
                        "primary_clause_id": rch["primary_clause_id"],
                        "section_type": rch["section_type"],
                        "text": sub_text,
                        "char_start": rch["char_start"],
                        "char_end": rch["char_end"],
                        "token_count": estimate_tokens(sub_text),
                    })
        else:
            final_idx += 1
            rch["chunk_id"] = f"{doc_id}#K{final_idx:03d}"
            final_chunks.append(rch)

    return final_chunks


def process_document_clauses_and_chunks(
    fid: str,
    doc_header: Dict[str, Any],
    clean_dir: str = "gad_kg/data/clean",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    clean_path = os.path.join(clean_dir, f"{fid}.md")
    if not os.path.exists(clean_path):
        return [], []

    with open(clean_path, "r", encoding="utf-8", errors="replace") as f:
        clean_text = f.read()

    doc_id = doc_header.get("doc_id", f"D_{fid}")
    clauses = segment_document_clauses(doc_id, fid, clean_text)
    chunks = chunk_clauses(clauses, doc_header)
    return clauses, chunks


def run_p09_pipeline(
    header_path: str = "gad_kg/data/work/header.jsonl",
    clean_dir: str = "gad_kg/data/clean",
    clauses_output_path: str = "gad_kg/data/work/clauses.jsonl",
    chunks_output_path: str = "gad_kg/data/work/chunks.jsonl",
) -> Dict[str, Any]:
    ensure_dir(Path(clauses_output_path).parent)
    ensure_dir(Path(chunks_output_path).parent)

    headers = list(read_jsonl(Path(header_path))) if os.path.exists(header_path) else []
    all_clauses = []
    all_chunks = []
    max_tokens_seen = 0

    for h in headers:
        fid = h["file_id"]
        c_list, k_list = process_document_clauses_and_chunks(fid, h, clean_dir)
        all_clauses.extend(c_list)
        all_chunks.extend(k_list)
        for k in k_list:
            if k["token_count"] > max_tokens_seen:
                max_tokens_seen = k["token_count"]

    write_jsonl(Path(clauses_output_path), all_clauses)
    write_jsonl(Path(chunks_output_path), all_chunks)

    return {
        "docs_processed": len(headers),
        "total_clauses": len(all_clauses),
        "total_chunks": len(all_chunks),
        "max_tokens_per_chunk": max_tokens_seen,
        "clauses_path": clauses_output_path,
        "chunks_path": chunks_output_path,
    }


if __name__ == "__main__":
    res = run_p09_pipeline()
    print("\n" + "=" * 60)
    print("PHASE 9 CLAUSE SEGMENTATION & CHUNKING RESULTS")
    print("=" * 60)
    print(f"Total Evaluated Documents: {res['docs_processed']}")
    print(f"Total Clauses Extracted:   {res['total_clauses']}")
    print(f"Total Chunks Created:      {res['total_chunks']}")
    print(f"Max Tokens per Chunk:      {res['max_tokens_per_chunk']} (Spec Target: <= 700)")
    print(f"Clauses Output Path:       {res['clauses_path']}")
    print(f"Chunks Output Path:        {res['chunks_path']}")
    print("=" * 60)
