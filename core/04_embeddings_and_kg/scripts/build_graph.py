"""Build and populate the Temporal Knowledge Graph for GAD from existing data.

Reads:
- data/fg2/doc_metadata.json
- data/fg2/documents.json
- data/fg2/chunks (chunk cards and metadata)

Writes to:
- data/fg2/temporal_graph.db (SQLite + FTS5 full-text + NetworkX graph)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Shared paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # GAD-BOT root
DATA_DIR = PROJECT_ROOT / "data" / "fg2"

sys.path.insert(0, str(PROJECT_ROOT / "core"))
sys.path.insert(0, str(SCRIPT_DIR))

import pipeline_config as pc
from graph_store import TemporalGraphStore, get_graph_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("build_graph")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def extract_year(date_str: Optional[str]) -> int:
    """Extract integer year from string like '2026-09-28' or '28.09.2026'."""
    if not date_str:
        return 0
    m = re.search(r'\b(19\d\d|20\d\d)\b', str(date_str))
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return 0
    return 0


def normalize_order_no(order_no: Optional[str]) -> str:
    """Normalize order number for fuzzy linking."""
    if not order_no:
        return ""
    # Remove extra spaces, lowercase
    cleaned = re.sub(r'\s+', ' ', str(order_no)).strip()
    return cleaned


def build_temporal_graph():
    logger.info("=" * 65)
    logger.info("Building GAD Temporal Knowledge Graph & Document History Store")
    logger.info("=" * 65)

    doc_meta_path = DATA_DIR / "doc_metadata.json"
    documents_reg_path = DATA_DIR / "documents.json"
    db_path = DATA_DIR / "temporal_graph.db"

    # Reset existing db for clean build
    if db_path.exists():
        try:
            db_path.unlink()
            logger.info("Cleaned previous temporal_graph.db.")
        except Exception as e:
            logger.warning(f"Could not delete old db: {e}")

    store = TemporalGraphStore(db_path)
    conn = store._get_connection()
    cur = conn.cursor()

    # 1. Load doc_metadata.json
    logger.info(f"Loading metadata from {doc_meta_path.name}...")
    with open(doc_meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    docs_dict = meta_data.get("documents", {})
    logger.info(f"Found {len(docs_dict)} documents in doc_metadata.json.")

    # 2. Load documents.json registry for extra paths/chunk counts
    reg_dict = {}
    if documents_reg_path.exists():
        try:
            with open(documents_reg_path, "r", encoding="utf-8") as f:
                raw_reg = json.load(f)
                if isinstance(raw_reg, dict) and "documents" in raw_reg:
                    raw_reg = raw_reg["documents"]
                if isinstance(raw_reg, list):
                    for item in raw_reg:
                        if isinstance(item, dict) and "id" in item:
                            reg_dict[item["id"]] = item
                elif isinstance(raw_reg, dict):
                    reg_dict = raw_reg
            logger.info(f"Loaded {len(reg_dict)} registry records from documents.json.")
        except Exception as e:
            logger.warning(f"Failed to load documents.json: {e}")

    # Build lookup map: normalized_order_no -> doc_id
    order_to_doc_id: Dict[str, str] = {}
    date_to_docs: Dict[str, List[str]] = {}

    node_rows = []
    fts_node_rows = []

    # Process all documents
    for doc_id, doc in docs_dict.items():
        reg_item = reg_dict.get(doc_id, {})
        order_no = doc.get("order_number") or ""
        doc_date = doc.get("issue_date") or doc.get("date") or doc.get("doc_date") or ""
        year = extract_year(doc_date)
        status = (doc.get("status") or "current").lower()
        dept = doc.get("issuing_office") or doc.get("department") or "सामान्य प्रशासन विभाग"
        title_hi = doc.get("title") or reg_item.get("original_name") or ""
        title_en = doc.get("title_en") or ""
        doc_type = doc.get("doc_type") or "order"
        source_url = doc.get("source_url") or ""
        chunk_count = reg_item.get("chunk_count") or 0

        # Summary
        summary = ""
        if title_en and title_hi:
            summary = f"{title_en} | {title_hi}"
        else:
            summary = title_hi or title_en

        node_id = f"doc_{doc_id}"
        label = order_no if order_no else (title_hi[:30] if title_hi else doc_id[:8])

        norm_order = normalize_order_no(order_no)
        if norm_order:
            order_to_doc_id[norm_order] = node_id

        if doc_date:
            date_to_docs.setdefault(doc_date[:7], []).append(node_id)  # group by YYYY-MM

        node_rows.append((
            node_id,
            doc_id,
            "document",
            label,
            order_no,
            doc_date,
            year,
            status,
            dept,
            title_hi,
            title_en,
            doc_type,
            summary,
            chunk_count,
            source_url,
            json.dumps(doc, ensure_ascii=False)
        ))

        fts_node_rows.append((
            node_id,
            order_no,
            doc_date,
            title_hi,
            title_en,
            summary
        ))

    logger.info(f"Inserting {len(node_rows)} document nodes into database...")
    cur.executemany("""
        INSERT INTO nodes (
            id, doc_id, node_type, label, order_no, doc_date, year, status,
            department, title_hi, title_en, doc_type, summary, chunk_count,
            source_url, meta_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, node_rows)

    cur.executemany("""
        INSERT INTO fts_nodes (
            id, order_no, doc_date, title_hi, title_en, summary
        ) VALUES (?, ?, ?, ?, ?, ?);
    """, fts_node_rows)

    conn.commit()
    logger.info("Nodes and FTS index created.")

    # 3. Create Edges (Relationships across time)
    edge_rows = []
    seen_edges: Set[Tuple[str, str, str]] = set()

    def add_edge(src: str, dst: str, rel: str, dt: str = "", ev: str = "", wt: float = 1.0):
        if src == dst:
            return
        key = (src, dst, rel)
        if key not in seen_edges:
            seen_edges.add(key)
            edge_rows.append((src, dst, rel, dt, ev, wt, "{}"))

    supersede_count = 0
    amend_count = 0
    reference_count = 0

    for doc_id, doc in docs_dict.items():
        src_id = f"doc_{doc_id}"
        src_date = doc.get("issue_date") or ""

        # A. Supersedes (Source supersedes Target)
        supersedes_list = doc.get("supersedes") or []
        for s in supersedes_list:
            target_id = None
            target_order = ""
            target_date = ""
            ev = ""

            if isinstance(s, dict):
                target_order = s.get("order_number") or ""
                target_date = s.get("date") or ""
                ev = s.get("evidence") or ""
            elif isinstance(s, str):
                target_order = s

            if target_order and normalize_order_no(target_order) in order_to_doc_id:
                target_id = order_to_doc_id[normalize_order_no(target_order)]
            elif target_date and target_date[:7] in date_to_docs:
                # Fallback to date match if exact
                for cand in date_to_docs[target_date[:7]]:
                    if cand != src_id:
                        target_id = cand
                        break

            if target_id:
                add_edge(src_id, target_id, "SUPERSEDES", dt=src_date, ev=ev, wt=1.0)
                # Mark target node as superseded
                cur.execute("UPDATE nodes SET status='superseded' WHERE id=?", (target_id,))
                supersede_count += 1

        # B. Amends (Source amends Target)
        amends_list = doc.get("amends") or []
        for a in amends_list:
            target_id = None
            target_order = ""
            target_date = ""
            ev = ""

            if isinstance(a, dict):
                target_order = a.get("order_number") or ""
                target_date = a.get("date") or ""
                ev = a.get("evidence") or ""
            elif isinstance(a, str):
                target_order = a

            if target_order and normalize_order_no(target_order) in order_to_doc_id:
                target_id = order_to_doc_id[normalize_order_no(target_order)]
            elif target_date and target_date[:7] in date_to_docs:
                for cand in date_to_docs[target_date[:7]]:
                    if cand != src_id:
                        target_id = cand
                        break

            if target_id:
                add_edge(src_id, target_id, "AMENDS", dt=src_date, ev=ev, wt=0.9)
                # Mark target node as amended if not already superseded
                cur.execute("UPDATE nodes SET status='amended' WHERE id=? AND status='current'", (target_id,))
                amend_count += 1

        # C. References
        refs_list = doc.get("references") or []
        for r in refs_list:
            target_order = r if isinstance(r, str) else (r.get("order_number") if isinstance(r, dict) else "")
            if target_order and normalize_order_no(target_order) in order_to_doc_id:
                target_id = order_to_doc_id[normalize_order_no(target_order)]
                add_edge(src_id, target_id, "REFERENCES", dt=src_date, ev="", wt=0.6)
                reference_count += 1

    # D. Chronological Sequence Edges within Department
    # Sort docs by date and link adjacent ones with CHRONOLOGICAL_NEXT
    dated_nodes = []
    for r in node_rows:
        dt = r[5]  # doc_date
        if dt and len(dt) >= 10:
            dated_nodes.append((r[0], dt))

    dated_nodes.sort(key=lambda x: x[1])
    chron_count = 0
    # Group in consecutive pairs
    for i in range(len(dated_nodes) - 1):
        # Link to next chronological order
        prev_id, prev_date = dated_nodes[i]
        next_id, next_date = dated_nodes[i+1]
        add_edge(next_id, prev_id, "CHRONOLOGICAL_PREV", dt=next_date, ev=f"{prev_date} -> {next_date}", wt=0.4)
        add_edge(prev_id, next_id, "CHRONOLOGICAL_NEXT", dt=prev_date, ev=f"{prev_date} -> {next_date}", wt=0.4)
        chron_count += 1

    logger.info(f"Inserting {len(edge_rows)} edges (Supersedes: {supersede_count}, Amends: {amend_count}, Refs: {reference_count}, Timeline: {chron_count})...")
    cur.executemany("""
        INSERT OR IGNORE INTO edges (
            source, target, relation, date, evidence, weight, meta_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
    """, edge_rows)
    conn.commit()

    # 4. Populate sample chunk index from chunks/
    chunks_dir = DATA_DIR / "chunks"
    logger.info("Indexing chunks into FTS5 for fast retrieval...")
    chunk_rows = []
    fts_chunk_rows = []

    # Walk chunk directories
    indexed_chunk_count = 0
    chunk_dirs = [d for d in chunks_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(chunk_dirs)} chunk subdirectories.")

    for cdir in chunk_dirs:
        card_file = cdir / "structured_chunk_card.txt"
        card_text = ""
        if card_file.exists():
            try:
                card_text = card_file.read_text(encoding="utf-8")
            except Exception:
                pass

        for chunk_file in cdir.glob("structured_chunk_*.txt"):
            if chunk_file.name == "structured_chunk_card.txt":
                continue
            try:
                txt = chunk_file.read_text(encoding="utf-8", errors="replace")
                # Parse frontmatter
                parts = txt.split("---", 1)
                body = parts[1].strip() if len(parts) > 1 else txt
                header = parts[0] if len(parts) > 1 else ""

                m_id = re.search(r'chunk_id:\s*([^\s\n]+)', header)
                c_id = m_id.group(1) if m_id else chunk_file.stem
                m_doc = re.search(r'doc_id:\s*([^\s\n]+)', header)
                d_id = m_doc.group(1) if m_doc else cdir.name
                m_ord = re.search(r'order_no:\s*([^\n]+)', header)
                ord_no = m_ord.group(1).strip() if m_ord else ""
                m_dt = re.search(r'doc_date:\s*([^\n]+)', header)
                c_date = m_dt.group(1).strip() if m_dt else ""
                m_title = re.search(r'title:\s*([^\n]+)', header)
                c_title = m_title.group(1).strip() if m_title else ""
                m_st = re.search(r'status:\s*([^\n]+)', header)
                c_st = m_st.group(1).strip() if m_st else "current"

                chunk_rows.append((
                    c_id,
                    d_id,
                    indexed_chunk_count + 1,
                    ord_no,
                    c_title,
                    c_date,
                    c_st,
                    body[:4000],  # cap to 4k chars for instant FTS
                    len(body.split())
                ))

                fts_chunk_rows.append((
                    c_id,
                    d_id,
                    ord_no,
                    c_title,
                    body[:4000]
                ))

                indexed_chunk_count += 1
                if len(chunk_rows) >= 5000:
                    cur.executemany("""
                        INSERT OR REPLACE INTO chunks (
                            chunk_id, doc_id, chunk_index, order_no, title,
                            doc_date, status, content, tokens
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, chunk_rows)
                    cur.executemany("""
                        INSERT INTO fts_chunks (
                            chunk_id, doc_id, order_no, title, content
                        ) VALUES (?, ?, ?, ?, ?);
                    """, fts_chunk_rows)
                    conn.commit()
                    chunk_rows.clear()
                    fts_chunk_rows.clear()
                    logger.info(f"Indexed {indexed_chunk_count} chunks...")

            except Exception as e:
                continue

    if chunk_rows:
        cur.executemany("""
            INSERT OR REPLACE INTO chunks (
                chunk_id, doc_id, chunk_index, order_no, title,
                doc_date, status, content, tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, chunk_rows)
        cur.executemany("""
            INSERT INTO fts_chunks (
                chunk_id, doc_id, order_no, title, content
            ) VALUES (?, ?, ?, ?, ?);
        """, fts_chunk_rows)
        conn.commit()

    logger.info(f"Total chunks indexed: {indexed_chunk_count}")
    conn.close()

    # Reload store and display stats
    store.load_graph_from_db()
    stats = store.get_stats()
    logger.info("=" * 65)
    logger.info("Temporal Knowledge Graph Build Complete!")
    logger.info(f"Total Nodes: {stats['total_nodes']}")
    logger.info(f"Total Edges: {stats['total_edges']}")
    logger.info(f"Statuses: {stats['statuses']}")
    logger.info(f"Relations: {stats['relations']}")
    logger.info(f"Year Span: {stats['year_span']['min']} to {stats['year_span']['max']}")
    logger.info("=" * 65)
    return stats


if __name__ == "__main__":
    build_temporal_graph()
