"""Temporal Knowledge Graph & Document History Store for GAD.

Stores documents, orders, chunks, and their temporal/chronological relationships:
- SUPERSEDES / SUPERSEDED_BY
- AMENDS / AMENDED_BY
- REFERENCES / REFERENCED_BY
- CHRONOLOGICAL_NEXT / CHRONOLOGICAL_PREV
- BELONGS_TO_TOPIC

Supports:
1. Fast graph traversal (predecessors, successors, full lineage across time).
2. Status resolution (determining if an order is currently active, amended, or superseded).
3. Visual subgraph extraction for interactive graphical representation in the UI.
4. FTS5 full-text search over documents and chunks.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

# Ensure OpenMP stability on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class DocumentNode:
    id: str
    doc_id: str
    order_no: str
    doc_date: str
    year: int
    status: str  # current, amended, superseded, unknown
    department: str
    title_hi: str
    title_en: str
    doc_type: str
    summary: str
    chunk_count: int
    source_url: str


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str  # SUPERSEDES, AMENDS, REFERENCES, CHRONOLOGICAL_NEXT, BELONGS_TO
    date: str
    evidence: str
    weight: float


class TemporalGraphStore:
    """Manages SQLite storage and an in-memory NetworkX directed graph."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            # Default to data/fg2/temporal_graph.db
            root_dir = Path(__file__).resolve().parents[3]
            db_path = root_dir / "data" / "fg2" / "temporal_graph.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph = nx.DiGraph()
        self._init_sqlite()
        self.load_graph_from_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self):
        """Create tables for nodes, edges, chunks, and FTS full-text index."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    node_type TEXT,
                    label TEXT,
                    order_no TEXT,
                    doc_date TEXT,
                    year INTEGER,
                    status TEXT,
                    department TEXT,
                    title_hi TEXT,
                    title_en TEXT,
                    doc_type TEXT,
                    summary TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    source_url TEXT,
                    meta_json TEXT
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_doc_id ON nodes(doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_order_no ON nodes(order_no);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_date ON nodes(doc_date);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_year ON nodes(year);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    target TEXT,
                    relation TEXT,
                    date TEXT,
                    evidence TEXT,
                    weight REAL DEFAULT 1.0,
                    meta_json TEXT,
                    UNIQUE(source, target, relation)
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    chunk_index INTEGER,
                    order_no TEXT,
                    title TEXT,
                    doc_date TEXT,
                    status TEXT,
                    content TEXT,
                    tokens INTEGER DEFAULT 0
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);")

            # FTS5 tables for instant full text search over docs and chunks
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_nodes USING fts5(
                    id UNINDEXED,
                    order_no,
                    doc_date,
                    title_hi,
                    title_en,
                    summary,
                    tokenize = 'unicode61'
                );
            """)

            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                    chunk_id UNINDEXED,
                    doc_id UNINDEXED,
                    order_no,
                    title,
                    content,
                    tokenize = 'unicode61'
                );
            """)
            conn.commit()

    def load_graph_from_db(self):
        """Populate in-memory NetworkX DiGraph from SQLite."""
        self.graph.clear()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM nodes")
            nodes = cur.fetchall()
            for r in nodes:
                r_keys = set(r.keys())
                self.graph.add_node(
                    r["id"],
                    doc_id=r["doc_id"],
                    doc_dir=(r["doc_dir"] if "doc_dir" in r_keys else "") or "",
                    node_type=r["node_type"],
                    label=r["label"],
                    order_no=r["order_no"] or "",
                    doc_date=r["doc_date"] or "",
                    year=r["year"] or 0,
                    status=r["status"] or "unknown",
                    department=r["department"] or "",
                    title_hi=r["title_hi"] or "",
                    title_en=r["title_en"] or "",
                    summary=r["summary"] or "",
                    category=(r["category"] if "category" in r_keys else "") or "",
                    category_id=(r["category_id"] if "category_id" in r_keys else "") or "",
                    chunk_count=r["chunk_count"] or 0,
                    source_url=r["source_url"] or ""
                )

            cur.execute("SELECT * FROM edges")
            edges = cur.fetchall()
            for e in edges:
                self.graph.add_edge(
                    e["source"],
                    e["target"],
                    relation=e["relation"],
                    date=e["date"] or "",
                    evidence=e["evidence"] or "",
                    weight=e["weight"] or 1.0
                )
        logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def _get_id_to_dir_map(self) -> Dict[str, str]:
        if not hasattr(self, '_id_to_dir'):
            self._id_to_dir = {}
            doc_reg_path = self.db_path.parent / "documents.json"
            if doc_reg_path.exists():
                try:
                    with open(doc_reg_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        docs = data.get("documents", []) if isinstance(data, dict) else data
                        for d in docs:
                            if isinstance(d, dict) and "id" in d and "doc_dir" in d:
                                self._id_to_dir[d["id"]] = d["doc_dir"]
                except Exception as e:
                    logger.warning(f"Could not load documents.json for OCR mapping: {e}")
        return self._id_to_dir

    def get_document_full_ocr(self, node_id_or_doc_id: str) -> str:
        """Read the complete OCRed document markdown from data/fg2/stage2/<doc_dir>/structured.md"""
        doc_id = str(node_id_or_doc_id).replace("doc_", "").strip()
        stage2_dir = self.db_path.parent / "stage2"
        # Direct check
        direct_path = stage2_dir / doc_id / "structured.md"
        if direct_path.exists():
            try:
                return direct_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                return f"(Error reading OCR text: {e})"
        
        # Mapping via documents.json
        id_map = self._get_id_to_dir_map()
        doc_dir = id_map.get(doc_id)
        if doc_dir:
            mapped_path = stage2_dir / doc_dir / "structured.md"
            if mapped_path.exists():
                try:
                    return mapped_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    return f"(Error reading OCR text: {e})"
        return ""

    def traverse_history(self, node_id_or_doc_id: str) -> Dict[str, Any]:
        """Traverse the document's history across time (predecessors and successors).
        
        Returns:
            - root_orders: original ancestor documents that were modified/superseded
            - current_node: current selected document
            - successors: subsequent amendments or superseding documents
            - timeline_chain: sorted chronological sequence of related orders
        """
        node_id = self._resolve_node_id(node_id_or_doc_id)
        if not node_id or node_id not in self.graph:
            return {"error": f"Node '{node_id_or_doc_id}' not found in temporal graph."}

        target_data = dict(self.graph.nodes[node_id])
        target_data["full_ocr_text"] = self.get_document_full_ocr(node_id)

        semantic_prev_rels = {
            "SUPERSEDES", "AMENDS", "CONTINUES", "REFERENCES",
            "DUPLICATE_OF", "CITES_RULEBOOK", "SAME_FILE_SERIES",
            "LATEST_IN_SERIES", "RELATED_POLICY_TOPIC", "CHRONOLOGICAL_PREV",
        }
        semantic_next_rels = {
            "SUPERSEDES", "AMENDS", "CONTINUES", "REFERENCES",
            "DUPLICATE_OF", "CITES_RULEBOOK", "SAME_FILE_SERIES",
            "LATEST_IN_SERIES", "RELATED_POLICY_TOPIC", "CHRONOLOGICAL_NEXT",
        }

        # 1. Predecessors (Ancestors in history - what came before)
        # Edges like A -[SUPERSEDES/AMENDS]-> B mean A is newer, B is older.
        predecessors = []
        for src, dst, data in self.graph.out_edges(node_id, data=True):
            rel = data.get("relation", "")
            if rel in semantic_prev_rels:
                if dst in self.graph:
                    p_data = dict(self.graph.nodes[dst])
                    p_data["node_id"] = dst
                    p_data["relation_to_target"] = rel
                    p_data["evidence"] = data.get("evidence", "")
                    p_data["weight"] = data.get("weight", 0.5)
                    predecessors.append(p_data)
        predecessors.sort(key=lambda x: (x.get("relation_to_target") == "CHRONOLOGICAL_PREV", -float(x.get("weight") or 0)))

        # 2. Successors (Descendants in history - what came after / what superseded this)
        successors = []
        for src, dst, data in self.graph.in_edges(node_id, data=True):
            rel = data.get("relation", "")
            if rel in semantic_next_rels:
                if src in self.graph:
                    s_data = dict(self.graph.nodes[src])
                    s_data["node_id"] = src
                    s_data["relation_from_successor"] = rel
                    s_data["evidence"] = data.get("evidence", "")
                    s_data["weight"] = data.get("weight", 0.5)
                    successors.append(s_data)
        successors.sort(key=lambda x: (x.get("relation_from_successor") == "CHRONOLOGICAL_NEXT", -float(x.get("weight") or 0)))

        # 3. Build chronological lineage chain
        all_related_ids = {node_id}
        # Gather 2-hop neighborhood in temporal edges
        for p in predecessors:
            all_related_ids.add(p["node_id"])
        for s in successors:
            all_related_ids.add(s["node_id"])

        timeline = []
        for nid in all_related_ids:
            nd = dict(self.graph.nodes[nid])
            nd["node_id"] = nid
            nd["is_selected"] = (nid == node_id)
            timeline.append(nd)

        # Sort chronologically by doc_date / year
        timeline.sort(key=lambda x: (x.get("doc_date") or "0000-00-00", x.get("year") or 0))

        # Determine effective current standing order in this cluster
        standing_order = None
        for item in reversed(timeline):
            if item.get("status") == "current":
                standing_order = item
                break
        if not standing_order and timeline:
            standing_order = timeline[-1]

        return {
            "node_id": node_id,
            "target": target_data,
            "standing_order": standing_order,
            "predecessors": predecessors,
            "successors": successors,
            "timeline_chain": timeline,
            "total_lineage_count": len(timeline)
        }

    def _resolve_node_id(self, key: str) -> Optional[str]:
        """Resolves doc_id, doc_dir, order_no, or exact node_id to graph node key."""
        if not key:
            return None
        clean_k = str(key).strip()
        if clean_k.startswith("D_") and "#" in clean_k:
            clean_k = clean_k.split("#", 1)[0][2:]
        elif clean_k.startswith("D_") and len(clean_k) >= 18:
            clean_k = clean_k[2:]
        if clean_k in self.graph:
            return clean_k
        if f"doc_{clean_k}" in self.graph:
            return f"doc_{clean_k}"
        # Check by doc_id, doc_dir, or order_no
        for nid, data in self.graph.nodes(data=True):
            if data.get("doc_id") == clean_k or data.get("doc_dir") == clean_k:
                return nid
            if data.get("order_no") and data.get("order_no").strip().lower() == clean_k.lower():
                return nid
        return None

    def search_nodes(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Full-text search over nodes using SQLite FTS5."""
        clean_q = re.sub(r'[^\w\s\u0900-\u097F]', ' ', query).strip()
        if not clean_q:
            return []

        tokens = clean_q.split()
        match_expr = " AND ".join([f'"{t}"*' for t in tokens])

        with self._get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute("""
                    SELECT n.*, rank
                    FROM fts_nodes f
                    JOIN nodes n ON f.id = n.id
                    WHERE fts_nodes MATCH ?
                    ORDER BY rank
                    LIMIT ?;
                """, (match_expr, limit))
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                # Fallback to LIKE if syntax issue
                like_term = f"%{clean_q}%"
                cur.execute("""
                    SELECT * FROM nodes
                    WHERE title_hi LIKE ? OR title_en LIKE ? OR order_no LIKE ? OR summary LIKE ?
                    LIMIT ?;
                """, (like_term, like_term, like_term, like_term, limit))
                rows = cur.fetchall()

            return [dict(r) for r in rows]

    def semantic_search(self, query: str, limit: int = 50) -> Dict[str, Any]:
        """Intelligent semantic, category-routed, and keyword/number/date search across documents.
        
        Handles:
        1. Exact/prefix Order Number and Date matching (e.g. '801', '14-1/2007', '2024')
        2. Semantic Domain / Category routing (e.g. 'leave-related', 'holiday', 'अनुकंपा', 'transfer', 'pension', etc.)
        3. Full-Text BM25 search across nodes and chunks
        """
        clean_q = query.strip()
        if not clean_q:
            return {"query": "", "matched_category": None, "matched_node_ids": [], "matched_hub_ids": [], "results": []}

        q_lower = clean_q.lower()

        CATEGORY_KEYWORDS = {
            'leave': [
                'leave', 'holiday', 'holidays', 'vacation', 'casual leave', 'maternity', 'paternity', 
                'earned leave', 'medical leave', 'अवकाश', 'छुट्टी', 'छुट्टियां', 'त्योहार', 'कार्यदिवस', 
                'ऐच्छिक अवकाश', 'सार्वजनिक अवकाश'
            ],
            'compassionate': [
                'compassionate', 'anukampa', 'dependent', 'anukampa niyukti', 'deceased', 
                'अनुकंपा', 'अनुकम्पा', 'आश्रित', 'मृतक शासकीय', 'अनुकंपा नियुक्ति'
            ],
            'transfer': [
                'transfer', 'posting', 'deputation', 'placement', 'relocation', 
                'स्थानांतरण', 'पदस्थापना', 'प्रतिनियुक्ति', 'तबादला', 'प्रभार'
            ],
            'recruitment': [
                'recruitment', 'exam', 'qualification', 'eligibility', 'vacancy', 'service rules', 
                'bharti', 'appointment', 'direct recruitment', 'भर्ती', 'पात्रता', 'परीक्षा', 
                'सेवा नियम', 'सीधी भर्ती', 'चयन'
            ],
            'promotion': [
                'promotion', 'seniority', 'dpc', 'grading', 'gradation', 
                'पदोन्नति', 'वरिष्ठता', 'पदोन्नत', 'डीपीसी', 'पदोन्नति सूची'
            ],
            'finance': [
                'budget', 'finance', 'audit', 'expenditure', 'sanction', 'financial', 'grant', 
                'बजट', 'वित्त', 'लेखा', 'स्वीकृति', 'आवंटन', 'व्यय', 'वित्तीय'
            ],
            'pension': [
                'pension', 'pay', 'allowance', 'salary', 'da', 'gpf', 'retirement', 'gratuity', 'pay scale', 
                'वेतन', 'पेंशन', 'भत्ता', 'महंगाई भत्ता', 'सेवानिवृत्ति', 'वेतनमान', 'ग्रेच्युटी'
            ],
            'vigilance': [
                'vigilance', 'discipline', 'inquiry', 'suspension', 'conduct', 'penalty', 'charge sheet', 
                'जांच', 'सतर्कता', 'निलंबन', 'अनुशासन', 'दण्ड', 'विभागीय जांच', 'आचरण'
            ],
            'reservation': [
                'reservation', 'roster', 'sc', 'st', 'obc', 'ews', 'divyang', 'handicapped', 'quota', 
                'आरक्षण', 'रोस्टर', 'दिव्यांग', 'अनुसूचित जाति', 'अनुसूचित जनजाति', 'अन्य पिछड़ा वर्ग'
            ],
            'schemes': [
                'scheme', 'donation', 'award', 'yojana', 'voluntary donation', 
                'योजना', 'अनुदान', 'स्वेच्छानुदान', 'पुरस्कार', 'कल्याण निधि'
            ],
            'elections': [
                'election', 'poll', 'code of conduct', 'observer', 'voting', 'duty', 
                'निर्वाचन', 'चुनाव', 'आचार संहिता', 'पर्यवेक्षक', 'मतदान'
            ],
            'cadre': [
                'cadre', 'structure', 'secretariat', 'departmental setup', 'reorganization', 
                'संवर्ग', 'ढांचा', 'पुनर्गठन', 'मंत्रालय', 'कार्य आवंटन'
            ],
            'training': [
                'training', 'capacity', 'workshop', 'academy', 'orientation', 
                'प्रशिक्षण', 'कार्यशाला', 'दक्षता', 'प्रशासनिक अकादमी'
            ],
            'rti': [
                'rti', 'information', 'appeal', 'grievance', 'right to information', 'appellate', 
                'सूचना का अधिकार', 'अपील', 'जन शिकायत', 'सूचना अधिकारी'
            ]
        }

        # 1. Detect category match
        matched_category_id = None
        for cat_id, syns in CATEGORY_KEYWORDS.items():
            for syn in syns:
                if syn in q_lower:
                    matched_category_id = cat_id
                    break
            if matched_category_id:
                break

        results_map = {}
        matched_hub_ids = set()
        matched_category_info = None

        with self._get_connection() as conn:
            cur = conn.cursor()

            # If category matched, get category info
            if matched_category_id:
                cur.execute("""
                    SELECT category, count(*) FROM nodes WHERE category_id = ? GROUP BY category;
                """, (matched_category_id,))
                cat_row = cur.fetchone()
                cat_name = cat_row["category"] if cat_row else matched_category_id
                cat_count = cat_row[1] if cat_row else 0
                matched_category_info = {
                    "id": matched_category_id,
                    "name": cat_name,
                    "hub_id": f"hub_{matched_category_id}",
                    "count": cat_count
                }
                matched_hub_ids.add(f"hub_{matched_category_id}")

                # Fetch top representative circulars from that category
                cur.execute("""
                    SELECT id, doc_id, label, order_no, doc_date, year, status, title_hi, title_en, summary, category, category_id
                    FROM nodes
                    WHERE category_id = ?
                    ORDER BY year DESC, doc_date DESC
                    LIMIT 30;
                """, (matched_category_id,))
                for row in cur.fetchall():
                    nid = row["id"]
                    results_map[nid] = {
                        "id": nid,
                        "order_no": row["order_no"] or row["label"] or "Order N/A",
                        "doc_date": row["doc_date"] or "",
                        "year": row["year"] or 0,
                        "status": row["status"] or "current",
                        "title_hi": row["title_hi"] or "",
                        "title_en": row["title_en"] or "",
                        "summary": row["summary"] or "",
                        "category": row["category"] or "",
                        "category_id": row["category_id"] or "",
                        "match_type": "category_domain",
                        "score": 40.0
                    }

            # 2. Exact or Prefix Order Number / Date match
            has_digits = bool(re.search(r'\d', clean_q))
            if has_digits or len(clean_q) >= 3:
                like_q = f"%{clean_q}%"
                cur.execute("""
                    SELECT id, doc_id, label, order_no, doc_date, year, status, title_hi, title_en, summary, category, category_id
                    FROM nodes
                    WHERE order_no LIKE ? OR doc_date LIKE ? OR id LIKE ?
                    LIMIT 20;
                """, (like_q, like_q, like_q))
                for row in cur.fetchall():
                    nid = row["id"]
                    score = 100.0 if clean_q.lower() in (row["order_no"] or "").lower() else 80.0
                    results_map[nid] = {
                        "id": nid,
                        "order_no": row["order_no"] or row["label"] or "Order N/A",
                        "doc_date": row["doc_date"] or "",
                        "year": row["year"] or 0,
                        "status": row["status"] or "current",
                        "title_hi": row["title_hi"] or "",
                        "title_en": row["title_en"] or "",
                        "summary": row["summary"] or "",
                        "category": row["category"] or "",
                        "category_id": row["category_id"] or "",
                        "match_type": "exact_order_date",
                        "score": score
                    }
                    if row["category_id"]:
                        matched_hub_ids.add(f"hub_{row['category_id']}")

            # 3. FTS5 Keyword / Semantic search on nodes and chunks
            tokens = re.findall(r'[\w\u0900-\u097F]+', clean_q)
            stop_words = {"the", "a", "an", "of", "and", "in", "to", "for", "is", "on", "का", "के", "की", "में", "से", "को", "पर", "द्वारा", "है", "संबंध", "संबंधी"}
            search_tokens = [t for t in tokens if len(t) > 1 and t.lower() not in stop_words]
            if search_tokens:
                fts_expr = " OR ".join([f'"{t}"*' for t in search_tokens[:6]])
                try:
                    cur.execute("""
                        SELECT n.id, n.doc_id, n.label, n.order_no, n.doc_date, n.year, n.status, 
                               n.title_hi, n.title_en, n.summary, n.category, n.category_id, rank
                        FROM fts_nodes f
                        JOIN nodes n ON f.id = n.id
                        WHERE fts_nodes MATCH ?
                        ORDER BY rank
                        LIMIT 25;
                    """, (fts_expr,))
                    for row in cur.fetchall():
                        nid = row["id"]
                        rank_score = max(5.0, 50.0 - abs(float(row["rank"] or 0)))
                        if nid in results_map:
                            results_map[nid]["score"] += rank_score
                            results_map[nid]["match_type"] = "exact_and_semantic"
                        else:
                            results_map[nid] = {
                                "id": nid,
                                "order_no": row["order_no"] or row["label"] or "Order N/A",
                                "doc_date": row["doc_date"] or "",
                                "year": row["year"] or 0,
                                "status": row["status"] or "current",
                                "title_hi": row["title_hi"] or "",
                                "title_en": row["title_en"] or "",
                                "summary": row["summary"] or "",
                                "category": row["category"] or "",
                                "category_id": row["category_id"] or "",
                                "match_type": "semantic_title",
                                "score": rank_score
                            }
                        if row["category_id"]:
                            matched_hub_ids.add(f"hub_{row['category_id']}")
                except Exception as e:
                    logger.debug(f"FTS search error: {e}")

                # Also search fts_chunks for deep text matches if fewer than 15 results
                if len(results_map) < 15:
                    try:
                        cur.execute("""
                            SELECT c.doc_id, min(rank) as best_rank
                            FROM fts_chunks f
                            JOIN chunks c ON f.rowid = c.rowid
                            WHERE fts_chunks MATCH ?
                            GROUP BY c.doc_id
                            ORDER BY best_rank
                            LIMIT 15;
                        """, (fts_expr,))
                        for row in cur.fetchall():
                            doc_hex = row["doc_id"]
                            cur.execute(
                                "SELECT * FROM nodes WHERE doc_dir = ? OR id = ? OR doc_id = ? LIMIT 1;",
                                (doc_hex, f"doc_{doc_hex}", doc_hex),
                            )
                            n_row = cur.fetchone()
                            if n_row:
                                real_nid = n_row["id"]
                                if real_nid not in results_map:
                                    results_map[real_nid] = {
                                        "id": real_nid,
                                        "order_no": n_row["order_no"] or n_row["label"] or "Order N/A",
                                        "doc_date": n_row["doc_date"] or "",
                                        "year": n_row["year"] or 0,
                                        "status": n_row["status"] or "current",
                                        "title_hi": n_row["title_hi"] or "",
                                        "title_en": n_row["title_en"] or "",
                                        "summary": n_row["summary"] or "",
                                        "category": n_row["category"] or "",
                                        "category_id": n_row["category_id"] or "",
                                        "match_type": "chunk_content",
                                        "score": 25.0
                                    }
                                    if n_row["category_id"]:
                                        matched_hub_ids.add(f"hub_{n_row['category_id']}")
                    except Exception as e:
                        logger.debug(f"FTS chunks search error: {e}")

            # 4. Check for Officer Match from officers table
            officer_match = None
            try:
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='officers';")
                if cur.fetchone():
                    from gad_kg.src.officer_store import OfficerStore
                    ostore = OfficerStore(self.db_path)
                    off_hits = ostore.search_officers(clean_q, limit=3)
                    if off_hits:
                        officer_match = off_hits[0]
                        for notif in officer_match.get("notifications", []):
                            doc_nid = f"doc_{notif['doc_id']}"
                            if doc_nid not in results_map:
                                cur.execute("SELECT * FROM nodes WHERE id = ?;", (doc_nid,))
                                n_row = cur.fetchone()
                                if n_row:
                                    results_map[doc_nid] = {
                                        "id": doc_nid,
                                        "order_no": n_row["order_no"] or n_row["label"] or notif.get("order_no", "Circular"),
                                        "doc_date": n_row["doc_date"] or notif.get("order_date", ""),
                                        "year": n_row["year"] or 2024,
                                        "status": n_row["status"] or "current",
                                        "title_hi": n_row["title_hi"] or notif.get("subject", ""),
                                        "title_en": f"Official Notification referencing {officer_match['name_en']}",
                                        "summary": notif.get("excerpt", ""),
                                        "category": n_row["category"] or "स्थानांतरण व पदस्थापना (Transfer & Postings)",
                                        "category_id": n_row["category_id"] or "transfer",
                                        "match_type": "officer_order",
                                        "score": 250.0
                                    }
                                    if n_row["category_id"]:
                                        matched_hub_ids.add(f"hub_{n_row['category_id']}")
            except Exception as e:
                logger.debug(f"Officer search error in graph_store: {e}")

        # Sort all matched results by score descending
        sorted_results = sorted(results_map.values(), key=lambda x: -x["score"])[:limit]
        matched_node_ids = [r["id"] for r in sorted_results]

        return {
            "query": query,
            "matched_category": matched_category_info,
            "officer_match": officer_match,
            "matched_node_ids": matched_node_ids,
            "matched_hub_ids": list(matched_hub_ids),
            "results": sorted_results,
            "total_matches": len(matched_node_ids)
        }

    def get_subgraph_for_visualization(
        self,
        center_node_id: Optional[str] = None,
        query: Optional[str] = None,
        year_filter: Optional[int] = None,
        max_nodes: int = 100
    ) -> Dict[str, Any]:
        """Generate formatted nodes & edges for Cytoscape.js or Vis.js visual rendering."""
        selected_nodes = set()

        if center_node_id:
            nid = self._resolve_node_id(center_node_id)
            if nid and nid in self.graph:
                selected_nodes.add(nid)
                # Add 1st and 2nd degree neighbors
                for neighbor in nx.single_source_shortest_path_length(self.graph.to_undirected(), nid, cutoff=2):
                    selected_nodes.add(neighbor)
                    if len(selected_nodes) >= max_nodes:
                        break

        if query and not selected_nodes:
            matches = self.search_nodes(query, limit=max_nodes // 2)
            for m in matches:
                selected_nodes.add(m["id"])
                # Add connected neighbors
                for src, dst in list(self.graph.edges(m["id"])):
                    selected_nodes.add(dst)
                for src, dst in list(self.graph.in_edges(m["id"])):
                    selected_nodes.add(src)
                if len(selected_nodes) >= max_nodes:
                    break

        if not selected_nodes:
            # Default: Pick most connected orders and recent active orders
            # Degree centrality / in-degree for amendments & supersessions
            nodes_by_degree = sorted(
                self.graph.nodes(),
                key=lambda n: self.graph.degree(n),
                reverse=True
            )
            for n in nodes_by_degree[:max_nodes]:
                selected_nodes.add(n)

        # Apply year filter if requested
        if year_filter:
            filtered = set()
            for nid in selected_nodes:
                y = self.graph.nodes[nid].get("year", 0)
                if y == year_filter or y == 0:
                    filtered.add(nid)
            selected_nodes = filtered or selected_nodes

        # Build Cytoscape/Vis.js format
        vis_nodes = []
        vis_edges = []
        node_id_set = set(selected_nodes)

        # Color palette for statuses
        status_colors = {
            "current": "#10b981",     # Emerald Green
            "amended": "#f59e0b",     # Amber Yellow
            "superseded": "#ef4444",  # Crimson Red
            "unknown": "#6b7280"      # Slate Grey
        }

        for nid in node_id_set:
            nd = self.graph.nodes[nid]
            status = nd.get("status", "unknown")
            color = status_colors.get(status, "#6b7280")
            year = nd.get("year", 0)
            date_str = nd.get("doc_date", "Date N/A")
            label = nd.get("order_no") or nd.get("title_hi") or nid
            if len(label) > 28:
                label = label[:26] + "…"

            vis_nodes.append({
                "id": nid,
                "label": label,
                "title": f"<b>{nd.get('order_no', 'No Order No')}</b><br/>Date: {date_str}<br/>Status: {status.upper()}<br/>{nd.get('title_hi', '')}",
                "color": color,
                "shape": "dot",
                "size": 22 if status == "current" else (18 if status == "amended" else 14),
                "status": status,
                "date": date_str,
                "year": year,
                "order_no": nd.get("order_no", ""),
                "full_title": nd.get("title_hi") or nd.get("title_en", ""),
                "summary": nd.get("summary", "")
            })

        for src, dst, data in self.graph.edges(data=True):
            if src in node_id_set and dst in node_id_set:
                rel = data.get("relation", "RELATES_TO")
                edge_color = "#ef4444" if rel == "SUPERSEDES" else ("#f59e0b" if rel == "AMENDS" else "#94a3b8")
                vis_edges.append({
                    "from": src,
                    "to": dst,
                    "label": rel,
                    "arrows": "to",
                    "color": {"color": edge_color, "highlight": "#2563eb"},
                    "dashes": rel in ("REFERENCES", "CHRONOLOGICAL_NEXT"),
                    "title": f"{rel}: {data.get('evidence', '')}"
                })

        return {
            "nodes": vis_nodes,
            "edges": vis_edges,
            "count": len(vis_nodes),
            "total_in_graph": self.graph.number_of_nodes()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return high-level statistics of the temporal graph."""
        statuses = {"current": 0, "amended": 0, "superseded": 0, "unknown": 0}
        relations = {}
        years = {}

        for _, d in self.graph.nodes(data=True):
            st = d.get("status", "unknown")
            statuses[st] = statuses.get(st, 0) + 1
            y = d.get("year", 0)
            if y:
                years[y] = years.get(y, 0) + 1

        for _, _, d in self.graph.edges(data=True):
            r = d.get("relation", "OTHER")
            relations[r] = relations.get(r, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "statuses": statuses,
            "relations": relations,
            "year_span": {
                "min": min(years.keys()) if years else None,
                "max": max(years.keys()) if years else None,
                "by_year": dict(sorted(years.items()))
            }
        }

    def get_categories(self) -> List[Dict[str, Any]]:
        """Return the high-level governance domain categories with statistics."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT category_id, category, status, COUNT(*) as cnt
                FROM nodes
                WHERE category_id IS NOT NULL
                GROUP BY category_id, category, status
            """)
            rows = cur.fetchall()
            cats_dict = {}
            for r in rows:
                cid = r["category_id"]
                if cid not in cats_dict:
                    cats_dict[cid] = {
                        "id": cid,
                        "name": r["category"],
                        "total": 0,
                        "current": 0,
                        "amended": 0,
                        "superseded": 0
                    }
                cnt = r["cnt"]
                cats_dict[cid]["total"] += cnt
                st = (r["status"] or "current").lower()
                if st in cats_dict[cid]:
                    cats_dict[cid][st] += cnt

            return sorted(cats_dict.values(), key=lambda x: -x["total"])

    def get_hierarchical_graph(
        self,
        expanded_categories: Optional[List[str]] = None,
        max_children_per_cat: int = 120
    ) -> Dict[str, Any]:
        """Generate a static, hierarchical macro-to-micro graph layout (Hub-and-Spoke).

        - Macro Level: High-level category hub nodes positioned on a static circle.
        - Micro Level: When a category is clicked/expanded, its document nodes radiate
          outwards on static concentric spokes around the category hub (radial cluster).
        - Nodes have fixed coordinates (fixed: {x: true, y: true}) and physics: false,
          guaranteeing zero continuous movement and zero CPU overhead.
        """
        import math
        categories = self.get_categories()
        expanded_set = set(expanded_categories or [])
        if "all" in expanded_set:
            expanded_set = {c["id"] for c in categories}

        vis_nodes = []
        vis_edges = []
        R_macro = 920  # Radius of macro hub circle

        hub_colors = [
            "#0d9488", "#0284c7", "#6366f1", "#8b5cf6", "#d97706",
            "#059669", "#dc2626", "#ea580c", "#475569", "#0891b2",
            "#14b8a6", "#ec4899", "#84cc16", "#e11d48", "#64748b"
        ]

        status_colors = {
            "current": "#10b981",    # Emerald Green
            "amended": "#f59e0b",    # Amber Yellow
            "superseded": "#ef4444"  # Crimson Red
        }

        # 1. Place the 10 Category Hubs at static circular coordinates
        num_cats = len(categories)
        for k, cat in enumerate(categories):
            cid = cat["id"]
            angle = (2 * math.pi * k / num_cats) - (math.pi / 2)
            hx = round(R_macro * math.cos(angle))
            hy = round(R_macro * math.sin(angle))
            is_expanded = cid in expanded_set
            color = hub_colors[k % len(hub_colors)]

            vis_nodes.append({
                "id": f"hub_{cid}",
                "label": f"{cat['name']}\n[{cat['total']} Orders]",
                "x": hx,
                "y": hy,
                "fixed": {"x": True, "y": True},
                "physics": False,
                "shape": "dot",
                "size": 38,
                "color": {
                    "background": color,
                    "border": "#ffffff" if is_expanded else color,
                    "highlight": {"background": "#38bdf8", "border": "#ffffff"}
                },
                "borderWidth": 3 if is_expanded else 1,
                "font": {"color": "#ffffff", "size": 13, "bold": True},
                "title": f"<b>{cat['name']}</b><br/>Total Orders: {cat['total']}<br/>Active: {cat['current']}<br/>Amended: {cat['amended']}<br/>Superseded: {cat['superseded']}<br/><i>(Click to expand / collapse)</i>",
                "is_hub": True,
                "category_id": cid,
                "is_expanded": is_expanded,
                "count": cat["total"]
            })

            # Connect consecutive hubs with a subtle macro ring edge
            next_k = (k + 1) % num_cats
            next_cid = categories[next_k]["id"]
            vis_edges.append({
                "from": f"hub_{cid}",
                "to": f"hub_{next_cid}",
                "color": {"color": "rgba(51, 65, 85, 0.4)"},
                "width": 1,
                "dashes": True,
                "physics": False
            })

            # 2. If this category is expanded, load child document nodes and arrange them radially
            if is_expanded:
                with self._get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT id, doc_id, label, order_no, doc_date, year, status, title_hi, title_en, summary, chunk_count
                        FROM nodes
                        WHERE category_id = ?
                        ORDER BY year DESC, doc_date DESC
                        LIMIT ?;
                    """, (cid, max_children_per_cat))
                    children = cur.fetchall()

                num_children = len(children)
                child_ids = set()

                for j, ch in enumerate(children):
                    child_id = ch["id"]
                    child_ids.add(child_id)
                    theta = (2 * math.pi * j / num_children)
                    # Concentric staggered radii (140, 210, 280) so children don't collide
                    spoke_r = 150 + (j % 3) * 65
                    cx = hx + round(spoke_r * math.cos(theta))
                    cy = hy + round(spoke_r * math.sin(theta))

                    st = (ch["status"] or "current").lower()
                    c_color = status_colors.get(st, "#64748b")
                    c_label = ch["order_no"] or ch["title_hi"] or ch["doc_id"]
                    if len(c_label) > 22:
                        c_label = c_label[:20] + "…"

                    vis_nodes.append({
                        "id": child_id,
                        "label": c_label,
                        "x": cx,
                        "y": cy,
                        "fixed": {"x": True, "y": True},
                        "physics": False,
                        "shape": "dot",
                        "size": 15 if st == "current" else (12 if st == "amended" else 10),
                        "color": {
                            "background": c_color,
                            "border": "#ffffff",
                            "highlight": {"background": "#60a5fa", "border": "#ffffff"}
                        },
                        "borderWidth": 1.5,
                        "font": {"color": "#cbd5e1", "size": 10},
                        "title": f"<b>{ch['order_no'] or 'Order N/A'}</b><br/>Date: {ch['doc_date']}<br/>Status: {st.upper()}<br/>{ch['title_hi'] or ''}",
                        "is_hub": False,
                        "parent_hub": f"hub_{cid}",
                        "category_id": cid,
                        "order_no": ch["order_no"] or "",
                        "doc_date": ch["doc_date"] or "",
                        "year": ch["year"] or 0,
                        "status": st,
                        "title_hi": ch["title_hi"] or "",
                        "summary": ch["summary"] or "",
                        "chunk_count": ch["chunk_count"] or 1
                    })

                    # Spoke edge from category hub to document
                    vis_edges.append({
                        "from": f"hub_{cid}",
                        "to": child_id,
                        "color": {"color": "rgba(100, 116, 139, 0.35)"},
                        "width": 1,
                        "dashes": True,
                        "physics": False
                    })

                # Internal relationships between child nodes (SUPERSEDES, AMENDS, REFERENCES)
                if child_ids:
                    with self._get_connection() as conn:
                        cur = conn.cursor()
                        placeholders = ",".join(["?"] * len(child_ids))
                        cur.execute(f"""
                            SELECT source, target, relation, evidence
                            FROM edges
                            WHERE source IN ({placeholders}) AND target IN ({placeholders});
                        """, list(child_ids) + list(child_ids))
                        for er in cur.fetchall():
                            rel = er["relation"]
                            edge_color = "#ef4444" if rel == "SUPERSEDES" else ("#f59e0b" if rel == "AMENDS" else "#64748b")
                            vis_edges.append({
                                "from": er["source"],
                                "to": er["target"],
                                "label": rel,
                                "arrows": "to",
                                "color": {"color": edge_color, "highlight": "#38bdf8"},
                                "width": 2 if rel in ("SUPERSEDES", "AMENDS") else 1,
                                "dashes": rel in ("REFERENCES", "CHRONOLOGICAL_NEXT"),
                                "title": f"{rel}: {er['evidence']}",
                                "physics": False
                            })

        return {
            "nodes": vis_nodes,
            "edges": vis_edges,
            "count": len(vis_nodes),
            "total_in_graph": self.graph.number_of_nodes(),
            "categories": categories,
            "expanded_categories": list(expanded_set)
        }


# Singleton accessor
_STORE_INSTANCE: Optional[TemporalGraphStore] = None

def get_graph_store(db_path: Optional[Path] = None) -> TemporalGraphStore:
    global _STORE_INSTANCE
    if _STORE_INSTANCE is None:
        _STORE_INSTANCE = TemporalGraphStore(db_path)
    return _STORE_INSTANCE
