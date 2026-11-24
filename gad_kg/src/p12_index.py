"""
Phase 12: Contextual Prefix & Hybrid Retrieval Index
Spec: Section 5 P12, Section 6 Prompt R9, Section 9
"""

import os
import re
import sys
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.io import read_jsonl, write_jsonl, ensure_dir
from src.common.llm import call_llm
from schemas.p12_index_schemas import R9_SCHEMA


def load_prompt_template(name: str) -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / f"{name}.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_contextual_prefix(
    doc_header: Dict[str, Any],
    r9_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Build contextual prefix string according to Spec Section 5 P12:
    '[{doc_type} {doc_number_raw} dt. {issue_date} | {subject}] ' + R9 summary line.
    """
    doc_type = doc_header.get("doc_type", "ORDER")
    doc_num = doc_header.get("doc_number_raw") or doc_header.get("doc_number_norm") or "N/A"
    issue_date = doc_header.get("issue_date") or "Unknown"
    subject = (doc_header.get("subject") or "").strip()[:100]

    base_header = f"[{doc_type} {doc_num} dt. {issue_date} | {subject}]"

    if r9_meta:
        covers = r9_meta.get("covers", "").strip()
        kw_en = ", ".join(r9_meta.get("keywords_en", []))
        kw_hi = ", ".join(r9_meta.get("keywords_hi", []))
        kw_str = f" Keywords: {kw_en} | {kw_hi}".strip() if (kw_en or kw_hi) else ""
        return f"{base_header} {covers} {kw_str}".strip()

    return base_header


def init_fts5_index(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite FTS5 table for fast hybrid lexical search."""
    ensure_dir(Path(db_path).parent)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS chunks_fts")
    cur.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id,
            doc_id,
            doc_type,
            issue_date,
            prefix,
            body,
            text_for_embedding,
            tokenize = 'unicode61'
        )
    """)
    conn.commit()
    return conn


def index_chunks(
    conn: sqlite3.Connection,
    chunks: List[Dict[str, Any]],
    headers_map: Dict[str, Dict[str, Any]],
):
    """Insert chunks into the FTS5 lexical index."""
    cur = conn.cursor()
    for ch in chunks:
        doc_id = ch["doc_id"]
        fid = ch.get("file_id")
        h = headers_map.get(fid) or headers_map.get(doc_id, {})

        prefix = build_contextual_prefix(h)
        body = ch.get("text", "")
        full_text = f"{prefix}\n{body}"

        cur.execute("""
            INSERT INTO chunks_fts (
                chunk_id, doc_id, doc_type, issue_date, prefix, body, text_for_embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ch["chunk_id"],
            doc_id,
            h.get("doc_type", "ORDER"),
            h.get("issue_date"),
            prefix,
            body,
            full_text,
        ))
    conn.commit()


def search_lexical_fts5(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Query FTS5 index with BM25 ranking and stop-word filtering."""
    cur = conn.cursor()
    # Sanitize FTS5 query terms
    clean_query = re.sub(r"[^\w\s\u0900-\u097F]", " ", query).strip()
    if not clean_query:
        return []

    stopwords = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
        'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
        'can', 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
        'had', 'has', 'have', 'he', 'her', 'here', 'him', 'his', 'how', 'i', 'if', 'in', 'into', 'is',
        'it', 'its', 'just', 'me', 'more', 'most', 'my', 'no', 'nor', 'not', 'now', 'of', 'off', 'on',
        'once', 'only', 'or', 'other', 'our', 'out', 'over', 'own', 'same', 'she', 'should', 'so',
        'some', 'such', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this',
        'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what',
        'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'you', 'your',
        'tell', 'give', 'find', 'get', 'know', 'please', 'look', 'show', 'information', 'details',
        'का', 'के', 'की', 'में', 'से', 'को', 'है', 'हैं', 'पर', 'और', 'या', 'क्या', 'कैसे', 'कहाँ',
        'मुझे', 'बताइए', 'बताओ', 'दिखाइए', 'दीजिए', 'बारे', 'लिए', 'था', 'थे', 'थी', 'ने', 'तक', 'भी'
    }

    raw_tokens = [t for t in clean_query.split() if len(t) >= 2]
    filtered_tokens = [t for t in raw_tokens if t.lower() not in stopwords]
    chosen_tokens = filtered_tokens if filtered_tokens else raw_tokens
    if not chosen_tokens:
        return []

    terms = [f'"{t}"' for t in chosen_tokens]
    match_expr = " OR ".join(terms)

    cur.execute("""
        SELECT chunk_id, doc_id, doc_type, issue_date, prefix, body, rank
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """, (match_expr, limit))

    rows = cur.fetchall()
    results = []
    for r in rows:
        results.append({
            "chunk_id": r[0],
            "doc_id": r[1],
            "doc_type": r[2],
            "issue_date": r[3],
            "prefix": r[4],
            "body": r[5],
            "bm25_score": r[6],
        })
    return results


def run_p12_indexing_pipeline(
    chunks_path: str = "gad_kg/data/work/chunks.jsonl",
    header_path: str = "gad_kg/data/work/header.jsonl",
    fts_db_path: str = "gad_kg/data/work/chunks_fts.db",
) -> Dict[str, Any]:
    print("[P12] Initializing FTS5 lexical index and contextual prefix pipeline...")
    chunks = list(read_jsonl(Path(chunks_path)))
    headers = list(read_jsonl(Path(header_path)))
    headers_map = {h["file_id"]: h for h in headers}
    for h in headers:
        if "doc_id" in h:
            headers_map[h["doc_id"]] = h

    conn = init_fts5_index(fts_db_path)
    index_chunks(conn, chunks, headers_map)

    # Acceptance test: run 5 Hindi and 5 English queries
    test_queries = [
        "पदोन्नति", "आरक्षण", "वेतनमान", "अवकाश", "स्थानांतरण",
        "promotion", "reservation", "deputation", "transfer", "leave"
    ]
    retrieval_hits = 0
    for q in test_queries:
        hits = search_lexical_fts5(conn, q, limit=5)
        if len(hits) > 0:
            retrieval_hits += 1

    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM chunks_fts")
    indexed_count = cur.fetchone()[0]
    conn.close()

    hit_rate = retrieval_hits / len(test_queries)
    return {
        "indexed_chunks": indexed_count,
        "input_chunks": len(chunks),
        "db_path": fts_db_path,
        "test_query_hit_rate": hit_rate,
        "passed": indexed_count == len(chunks) and hit_rate >= 0.8,
    }


if __name__ == "__main__":
    res = run_p12_indexing_pipeline()
    print("\n" + "=" * 60)
    print("PHASE 12 CONTEXTUAL PREFIX & HYBRID FTS5 INDEX RESULTS")
    print("=" * 60)
    print(f"Total Chunks Indexed:  {res['indexed_chunks']} / {res['input_chunks']}")
    print(f"Database Path:         {res['db_path']}")
    print(f"Test Query Hit Rate:   {res['test_query_hit_rate']*100:.1f}%")
    print(f"Acceptance Passed:     {res['passed']}")
    print("=" * 60)
