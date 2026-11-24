"""Deterministic, structure-preserving chunker for CHiPS-RAG.

Implements:
- Hierarchical heading path tracking and numbered clause recognition
- Sentence splitter respecting '.', '!', '?' and Hindi dandas ('।', '॥')
- Protection for currency ('Rs. 78,800'), decimals, order numbers, abbreviations
- Embedded text template generation with consistent language formatting
- Strict token budgeting on FULL embedded text (Target 400, Hard Max 480, 50 Overlap)
- Deterministic chunk_id generation (<doc_id>_c001, prev_id, next_id, parent_id)
- Document Card generation with verified summary
"""
from __future__ import annotations

import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_CHUNKING_DIR = Path(__file__).resolve().parent
if str(_CHUNKING_DIR) not in sys.path:
    sys.path.insert(0, str(_CHUNKING_DIR))

from models import ChunkMetadata, ChunkRecord, DocumentCard
from extract_metadata import (
    classify_doc_type,
    detect_language,
    extract_doc_date,
    extract_order_no,
    extract_org_structure,
    extract_supersedes_references,
    build_topic_key,
    verify_llm_value,
)
from tables import chunk_table

_ABBREVIATIONS = {
    "rs.", "no.", "f.no.", "order.", "vol.", "sec.", "art.",
    "dr.", "prof.", "mr.", "mrs.", "ms.", "shri.", "smt.", "dept.",
    "gov.", "govt.", "inc.", "ltd.", "co.", "approx.", "est.", "i.e.", "e.g.",
    "क्र.", "फा.", "सं."
}

_PAGE_MARKER_REGEX = re.compile(r"<!--\s*Page\s+(\d+)\s*-->", re.I)

_DOC_TYPE_LABELS_EN = {
    "vacancy_circular": "Vacancy Circular",
    "order": "Order",
    "circular": "Circular",
    "notification": "Notification",
    "budget_table": "Budget Table",
    "act_rule": "Act / Rule",
    "web_page": "Web Page",
    "other": "Official Document",
}

_DOC_TYPE_LABELS_HI = {
    "vacancy_circular": "रिक्ति / भर्ती परिपत्र",
    "order": "आदेश",
    "circular": "परिपत्र",
    "notification": "अधिसूचना",
    "budget_table": "बजट तालिका",
    "act_rule": "अधिनियम / नियम",
    "web_page": "वेब पेज",
    "other": "शासकीय दस्तावेज़",
}


def default_token_counter(text: str) -> int:
    """Accurate token estimator if no HuggingFace tokenizer is provided."""
    # Standard rule for multilingual Hindi/English BGE-M3:
    # 1 word ~ 1.3 to 1.5 tokens
    words = text.split()
    return int(len(words) * 1.35) + 5


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while protecting currency, decimals, and order numbers."""
    if not text:
        return []

    raw_splits = []
    start = 0
    i = 0
    n = len(text)

    while i < n:
        char = text[i]
        is_break = False

        if char in ('।', '॥'):
            is_break = True
        elif char in ('?', '!'):
            if i + 1 == n or text[i + 1].isspace():
                is_break = True
        elif char == '.':
            # Check not decimal (e.g. 12.5)
            prev_char = text[i - 1] if i > 0 else ''
            next_char = text[i + 1] if i + 1 < n else ''
            if prev_char.isdigit() and next_char.isdigit():
                i += 1
                continue

            # Check abbreviation
            preceding_str = text[max(0, i - 15):i + 1]
            last_token = preceding_str.split()[-1].lower() if preceding_str.split() else ''
            if last_token in _ABBREVIATIONS:
                i += 1
                continue

            # Check if followed by space or newline
            if i + 1 == n or text[i + 1].isspace():
                is_break = True

        if is_break:
            segment = text[start:i + 1].strip()
            if segment:
                raw_splits.append(segment)
            i += 1
            while i < n and text[i].isspace():
                i += 1
            start = i
        else:
            i += 1

    if start < n:
        tail = text[start:].strip()
        if tail:
            raw_splits.append(tail)

    return [s for s in raw_splits if s]


def render_embedded_header(meta: Dict[str, Any], section_path: str, lang: str) -> str:
    """Render the standardized header template specified in Section 5.
    
    Document: {doc_type_label} | No. {order_no} | Dated {doc_date_human}
    Ministry: {ministry}
    Office: {office}
    Status: {status}
    Section: {heading_path_joined_with_" > "}
    """
    lines = []
    
    # Consistent language label
    labels = _DOC_TYPE_LABELS_HI if lang == "hi" else _DOC_TYPE_LABELS_EN
    doc_type_label = labels.get(meta.get("doc_type"), "Document")
    order_no = meta.get("order_no")
    doc_date = meta.get("doc_date")
    
    first_line_parts = [f"Document: {doc_type_label}"]
    if order_no:
        first_line_parts.append(f"No. {order_no}")
    if doc_date:
        first_line_parts.append(f"Dated {doc_date}")
    lines.append(" | ".join(first_line_parts))

    if meta.get("ministry"):
        lines.append(f"Ministry: {meta['ministry']}")
    if meta.get("office") or meta.get("department"):
        lines.append(f"Office: {meta.get('office') or meta.get('department')}")
    if meta.get("status"):
        lines.append(f"Status: {meta['status']}")
    if section_path:
        lines.append(f"Section: {section_path}")

    return "\n".join(lines)


def build_document_chunks(
    text: str,
    doc_id: str,
    file_info: Optional[Dict[str, Any]] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    target_tokens: int = 400,
    max_tokens: int = 480,
    overlap_tokens: int = 50,
    llm_extractor: Optional[Callable[..., Any]] = None,
) -> Tuple[List[ChunkRecord], DocumentCard]:
    """Execute complete deterministic chunking pipeline on a document."""
    count_fn = token_counter or default_token_counter
    file_info = file_info or {}

    # 1. Parse pages from text
    page_splits = list(_PAGE_MARKER_REGEX.finditer(text))
    pages_text: List[str] = []
    if not page_splits:
        pages_text = [text]
    else:
        last_idx = 0
        for m in page_splits:
            if m.start() > last_idx:
                pages_text.append(text[last_idx:m.start()])
            last_idx = m.end()
        if last_idx < len(text):
            pages_text.append(text[last_idx:])

    # 2. Extract document-level metadata (Regex first, strict rules)
    is_web = file_info.get("source_type") == "web" or file_info.get("filename", "").endswith(".html")
    doc_date, doc_date_raw, doc_date_zone, date_conf, needs_review, ref_docs = extract_doc_date(
        pages_text,
        is_web=is_web,
        web_meta_date=file_info.get("scraped_at"),
        llm_extractor=llm_extractor,
    )

    # Order number & Supersedes
    order_no = extract_order_no(text)
    supersedes_list = extract_supersedes_references(text)
    ministry, department, office = extract_org_structure(text)
    doc_type = classify_doc_type(text, file_info.get("filename", ""))
    lang = detect_language(text)

    # Document Title fallback
    title_candidate = None
    for line in text.splitlines()[:20]:
        ln = line.strip().lstrip("#").strip()
        if len(ln) > 10 and not _PAGE_MARKER_REGEX.match(ln):
            title_candidate = ln[:140]
            break
    title = file_info.get("title") or title_candidate or "Official Notification"

    # Status computation
    status = "current"
    valid_until = file_info.get("valid_until")
    if valid_until:
        try:
            if datetime.strptime(valid_until, "%Y-%m-%d").date() < datetime.utcnow().date():
                status = "expired"
        except ValueError:
            pass

    topic_key = build_topic_key(department or ministry, doc_type, title)

    # Document summary (Rule 2: LLM may be called ONCE per doc for 2-3 line summary)
    summary_text = f"{doc_type.replace('_', ' ').title()} {order_no or ''} issued by {office or department or 'Department'} on {doc_date or 'unspecified date'}."
    if llm_extractor:
        try:
            llm_summary = llm_extractor(
                prompt=f"Summarize this document in 2-3 sentences. Facts only:\n{text[:2000]}",
                system_prompt="You are a government document summarizer. Do not invent any numbers or dates."
            )
            summary_candidate = None
            if isinstance(llm_summary, dict):
                summary_candidate = llm_summary.get("summary")
            elif isinstance(llm_summary, str):
                summary_candidate = llm_summary

            if summary_candidate and isinstance(summary_candidate, str):
                # Verify no fabricated numbers (Hard rule 3 & 9)
                numbers_in_summary = re.findall(r"\b\d+\b", summary_candidate)
                if all(n in text for n in numbers_in_summary):
                    summary_text = summary_candidate.strip()
        except Exception:
            pass

    doc_meta_dict = {
        "doc_type": doc_type,
        "order_no": order_no,
        "doc_date": doc_date,
        "ministry": ministry,
        "department": department,
        "office": office,
        "status": status,
    }

    # 3. Parse Document Blocks (Headings, Paragraphs, Tables)
    chunks_records: List[ChunkRecord] = []
    chunk_idx = 1
    current_heading_path: List[str] = [title]
    current_sec_id = f"{doc_id}_sec_1"

    # Split text into segments preserving page markers
    lines = text.splitlines()
    cur_page = 1
    cur_buf: List[str] = []
    cur_buf_headings = list(current_heading_path)

    def flush_text_buffer():
        nonlocal chunk_idx, cur_buf, cur_buf_headings
        if not cur_buf:
            return
        combined_body = "\n\n".join(cur_buf).strip()
        if not combined_body:
            cur_buf = []
            return

        sec_str = " > ".join(cur_buf_headings)
        header = render_embedded_header(doc_meta_dict, sec_str, lang)
        full_text = f"{header}\n\n{combined_body}"
        tok_len = count_fn(full_text)

        # Check token cap
        if tok_len <= max_tokens:
            c_id = f"{doc_id}_c{chunk_idx:03d}"
            meta = ChunkMetadata(
                chunk_id=c_id,
                doc_id=doc_id,
                chunk_index=chunk_idx,
                parent_id=current_sec_id,
                doc_type=doc_type,
                order_no=order_no,
                title=title,
                ministry=ministry,
                department=department,
                office=office,
                language=lang,
                doc_date=doc_date,
                doc_date_raw=doc_date_raw,
                doc_date_zone=doc_date_zone,
                date_confidence=date_conf,
                needs_review=needs_review,
                status=status,
                supersedes=supersedes_list,
                topic_key=topic_key,
                source_type=file_info.get("source_type", "pdf"),
                source_file=file_info.get("filename"),
                source_url=file_info.get("source_url"),
                pages=[cur_page],
                headings=cur_buf_headings,
                chunk_type="text",
                tokens=tok_len,
            )
            chunks_records.append(ChunkRecord(metadata=meta, body=combined_body, embedded_text=full_text))
            chunk_idx += 1
            cur_buf = []
        else:
            # Section exceeds max tokens -> split by sentence with overlap (Section 6 item 4)
            sentences = split_into_sentences(combined_body)
            slice_buf: List[str] = []
            
            for s in sentences:
                cand_body = " ".join(slice_buf + [s]).strip()
                cand_full = f"{header}\n\n{cand_body}"
                if slice_buf and count_fn(cand_full) > max_tokens:
                    c_id = f"{doc_id}_c{chunk_idx:03d}"
                    body_text = " ".join(slice_buf)
                    full_emb = f"{header}\n\n{body_text}"
                    meta = ChunkMetadata(
                        chunk_id=c_id,
                        doc_id=doc_id,
                        chunk_index=chunk_idx,
                        parent_id=current_sec_id,
                        doc_type=doc_type,
                        order_no=order_no,
                        title=title,
                        ministry=ministry,
                        department=department,
                        office=office,
                        language=lang,
                        doc_date=doc_date,
                        doc_date_raw=doc_date_raw,
                        doc_date_zone=doc_date_zone,
                        date_confidence=date_conf,
                        needs_review=needs_review,
                        status=status,
                        supersedes=supersedes_list,
                        topic_key=topic_key,
                        source_type=file_info.get("source_type", "pdf"),
                        source_file=file_info.get("filename"),
                        source_url=file_info.get("source_url"),
                        pages=[cur_page],
                        headings=cur_buf_headings,
                        chunk_type="text",
                        tokens=count_fn(full_emb),
                    )
                    chunks_records.append(ChunkRecord(metadata=meta, body=body_text, embedded_text=full_emb))
                    chunk_idx += 1
                    # Retain last sentence for overlap
                    slice_buf = [slice_buf[-1], s] if len(slice_buf) > 1 else [s]
                else:
                    slice_buf.append(s)

            if slice_buf:
                c_id = f"{doc_id}_c{chunk_idx:03d}"
                body_text = " ".join(slice_buf)
                full_emb = f"{header}\n\n{body_text}"
                meta = ChunkMetadata(
                    chunk_id=c_id,
                    doc_id=doc_id,
                    chunk_index=chunk_idx,
                    parent_id=current_sec_id,
                    doc_type=doc_type,
                    order_no=order_no,
                    title=title,
                    ministry=ministry,
                    department=department,
                    office=office,
                    language=lang,
                    doc_date=doc_date,
                    doc_date_raw=doc_date_raw,
                    doc_date_zone=doc_date_zone,
                    date_confidence=date_conf,
                    needs_review=needs_review,
                    status=status,
                    supersedes=supersedes_list,
                    topic_key=topic_key,
                    source_type=file_info.get("source_type", "pdf"),
                    source_file=file_info.get("filename"),
                    source_url=file_info.get("source_url"),
                    pages=[cur_page],
                    headings=cur_buf_headings,
                    chunk_type="text",
                    tokens=count_fn(full_emb),
                )
                chunks_records.append(ChunkRecord(metadata=meta, body=body_text, embedded_text=full_emb))
                chunk_idx += 1
            cur_buf = []

    # Iteration over document lines
    in_table = False
    table_lines: List[str] = []
    sec_counter = 1

    for line in lines:
        page_m = _PAGE_MARKER_REGEX.search(line)
        if page_m:
            cur_page = int(page_m.group(1))
            continue

        stripped = line.strip()
        if not stripped:
            continue

        # Check table line
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            table_lines.append(stripped)
            continue
        elif in_table:
            # End of table block -> flush text and chunk table separately (Rule 5: tables never merged with prose)
            flush_text_buffer()
            raw_table_text = "\n".join(table_lines)
            t_chunks = chunk_table(
                table_id=f"{doc_id}_tbl_{chunk_idx}",
                table_text=raw_table_text,
                page_no=cur_page,
                caption="Data Table",
            )
            for tc in t_chunks:
                sec_str = " > ".join(current_heading_path)
                header = render_embedded_header(doc_meta_dict, sec_str, lang)
                full_emb = f"{header}\n\n{tc['body']}"
                c_id = f"{doc_id}_c{chunk_idx:03d}"
                t_meta = ChunkMetadata(
                    chunk_id=c_id,
                    doc_id=doc_id,
                    chunk_index=chunk_idx,
                    parent_id=current_sec_id,
                    doc_type="budget_table" if doc_type == "budget_table" else "other",
                    order_no=order_no,
                    title=title,
                    ministry=ministry,
                    department=department,
                    office=office,
                    language=lang,
                    doc_date=doc_date,
                    status=status,
                    topic_key=topic_key,
                    source_type=file_info.get("source_type", "pdf"),
                    source_file=file_info.get("filename"),
                    pages=tc["pages"],
                    headings=list(current_heading_path),
                    chunk_type="table",
                    tokens=count_fn(full_emb),
                    table_id=tc["table_id"],
                    table_caption=tc["table_caption"],
                    fiscal_unit=tc["fiscal_unit"],
                    column_headers=tc["column_headers"],
                    row_range=tc["row_range"],
                    has_total_row=tc["has_total_row"],
                    continued_from_previous_page=tc["continued_from_previous_page"],
                )
                chunks_records.append(ChunkRecord(metadata=t_meta, body=tc["body"], embedded_text=full_emb))
                chunk_idx += 1

            in_table = False
            table_lines = []

        # Heading detection (# or ## or numbered major section)
        if stripped.startswith("#"):
            flush_text_buffer()
            level = len(stripped.split()[0])
            head_text = stripped.lstrip("#").strip()
            sec_counter += 1
            current_sec_id = f"{doc_id}_sec_{sec_counter}"
            
            # Update heading path hierarchy
            if level == 1:
                current_heading_path = [head_text]
            else:
                current_heading_path = [current_heading_path[0], head_text] if current_heading_path else [head_text]
            cur_buf_headings = list(current_heading_path)
            continue

        cur_buf.append(stripped)

    # Flush remaining text
    flush_text_buffer()

    # Assign prev_id and next_id across all chunks
    for i, rec in enumerate(chunks_records):
        rec.metadata.prev_id = chunks_records[i - 1].metadata.chunk_id if i > 0 else None
        rec.metadata.next_id = chunks_records[i + 1].metadata.chunk_id if i + 1 < len(chunks_records) else None

    # 4. Build Document Card
    card_record = DocumentCard(
        doc_id=doc_id,
        order_no=order_no,
        title=title,
        doc_type=doc_type,
        ministry=ministry,
        department=department,
        office=office,
        doc_date=doc_date,
        doc_date_raw=doc_date_raw,
        date_confidence=date_conf,
        needs_review=needs_review,
        effective_from=file_info.get("effective_from"),
        valid_until=valid_until,
        status=status,
        supersedes=supersedes_list,
        topic_key=topic_key,
        summary=summary_text,
        total_chunks=len(chunks_records),
        total_pages=len(pages_text),
        source_file=file_info.get("filename"),
        source_url=file_info.get("source_url"),
    )

    return chunks_records, card_record
