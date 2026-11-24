"""Retriever Experimentation Framework for GAD.

Provides modular retriever implementations for experimentation:
1. LexicalFtsRetriever: Fast BM25 full-text search over chunks using SQLite FTS5.
2. TemporalGraphRetriever: Knowledge graph-augmented retriever that performs
   temporal traversal over document history (identifies supersessions, amendments,
   lineage, and latest active standing rules).
3. HybridTemporalRetriever: Combines lexical matching, graph traversal, and
   temporal validity scoring.
4. RetrieverExperimentHarness: Benchmarking and comparing different retrieval
   strategies on real government queries.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

# Ensure OpenMP stability on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "fg2"

from graph_store import TemporalGraphStore, get_graph_store

logger = logging.getLogger("retrievers")


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    order_no: str
    title: str
    doc_date: str
    status: str
    content: str
    score: float
    retrieval_source: str  # 'fts', 'graph_expansion', 'vector'
    effective_status: str = "current"  # current, superseded, amended
    superseded_by: Optional[str] = None
    amended_by: Optional[str] = None
    lineage_summary: str = ""
    history_chain: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RetrievalResponse:
    query: str
    retriever_name: str
    latency_ms: float
    total_candidates: int
    results: List[RetrievedChunk]
    timeline_trace: List[Dict[str, Any]]
    warnings: List[str]


class BaseRetriever:
    """Abstract interface for all retriever experiments."""

    def __init__(self, name: str):
        self.name = name

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResponse:
        raise NotImplementedError


class LexicalFtsRetriever(BaseRetriever):
    """High-speed BM25 full-text search over indexed chunks."""

    def __init__(self, graph_store: Optional[TemporalGraphStore] = None):
        super().__init__("lexical_fts")
        self.store = graph_store or get_graph_store()

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResponse:
        start_t = time.time()
        clean_q = re.sub(r'[^\w\s\u0900-\u097F]', ' ', query).strip()
        tokens = [t for t in clean_q.split() if len(t) > 1]
        
        results: List[RetrievedChunk] = []
        if not tokens:
            return RetrievalResponse(query, self.name, 0.0, 0, [], [], [])

        # Build BM25 match query
        match_expr = " OR ".join([f'"{t}"*' for t in tokens])

        with self.store._get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute("""
                    SELECT c.chunk_id, c.doc_id, c.order_no, c.title, c.doc_date, c.status, c.content,
                           bm25(fts_chunks) AS score
                    FROM fts_chunks f
                    JOIN chunks c ON f.chunk_id = c.chunk_id
                    WHERE fts_chunks MATCH ?
                    ORDER BY score
                    LIMIT ?;
                """, (match_expr, top_k))
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                rows = []

            for r in rows:
                score = abs(float(r["score"])) if r["score"] else 1.0
                results.append(RetrievedChunk(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    order_no=r["order_no"] or "",
                    title=r["title"] or "",
                    doc_date=r["doc_date"] or "",
                    status=r["status"] or "current",
                    content=r["content"] or "",
                    score=score,
                    retrieval_source="fts"
                ))

        latency = (time.time() - start_t) * 1000.0
        return RetrievalResponse(
            query=query,
            retriever_name=self.name,
            latency_ms=round(latency, 2),
            total_candidates=len(results),
            results=results,
            timeline_trace=[],
            warnings=[]
        )


class TemporalGraphRetriever(BaseRetriever):
    """Knowledge Graph-Augmented Temporal Retriever.
    
    1. Retrieves initial candidate chunks via lexical/semantic match.
    2. Maps chunks to document nodes in the Temporal Knowledge Graph.
    3. Traverses temporal history edges:
       - Identifies if a document is superseded by a newer order.
       - Finds amendments and chronological successor orders.
    4. Automatically resolves the active standing order and appends
       its latest chunks if the retrieved passage is superseded.
    5. Returns an enriched result set with timeline lineage and warnings.
    """

    def __init__(self, graph_store: Optional[TemporalGraphStore] = None):
        super().__init__("temporal_graph_retriever")
        self.store = graph_store or get_graph_store()
        self.fts_retriever = LexicalFtsRetriever(self.store)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResponse:
        start_t = time.time()
        
        # 1. Candidate Retrieval (get 2x top_k to allow graph filtering & reranking)
        base_resp = self.fts_retriever.retrieve(query, top_k=top_k * 2)
        candidates = base_resp.results

        # 2. Also search directly in node titles / order numbers
        node_matches = self.store.search_nodes(query, limit=top_k)
        seen_doc_ids = {c.doc_id for c in candidates}

        # If direct order or title matched in graph, pull its primary chunks
        with self.store._get_connection() as conn:
            cur = conn.cursor()
            for nm in node_matches:
                d_id = nm["doc_id"]
                if d_id not in seen_doc_ids:
                    cur.execute("""
                        SELECT chunk_id, doc_id, order_no, title, doc_date, status, content
                        FROM chunks
                        WHERE doc_id = ?
                        ORDER BY chunk_index ASC
                        LIMIT 2;
                    """, (d_id,))
                    for cr in cur.fetchall():
                        candidates.append(RetrievedChunk(
                            chunk_id=cr["chunk_id"],
                            doc_id=cr["doc_id"],
                            order_no=cr["order_no"] or nm.get("order_no", ""),
                            title=cr["title"] or nm.get("title_hi", ""),
                            doc_date=cr["doc_date"] or nm.get("doc_date", ""),
                            status=cr["status"] or nm.get("status", "current"),
                            content=cr["content"] or "",
                            score=0.5,  # high priority graph match
                            retrieval_source="graph_expansion"
                        ))
                        seen_doc_ids.add(d_id)

        # 3. Temporal Graph Traversal & History Resolution
        enriched_results: List[RetrievedChunk] = []
        timeline_events: List[Dict[str, Any]] = []
        warnings: List[str] = []

        is_asking_historical = bool(re.search(
            r'\b(old|previous|earlier|history|पहले|पूर्व|इतिहास|संशोधन से पहले)\b',
            query,
            re.IGNORECASE
        ))

        for c in candidates:
            # Graph lookup for this doc_id
            history = self.store.traverse_history(c.doc_id)
            if "error" in history:
                enriched_results.append(c)
                continue

            target_info = history.get("target", {})
            standing_order = history.get("standing_order", {})
            timeline_chain = history.get("timeline_chain", [])

            # Check supersession / amendments
            is_superseded = False
            superseded_by_order = None
            is_amended = False
            amended_by_order = None

            for succ in history.get("successors", []):
                rel = succ.get("relation_from_successor")
                if rel == "SUPERSEDES":
                    is_superseded = True
                    superseded_by_order = succ.get("order_no") or succ.get("doc_date")
                elif rel == "AMENDS":
                    is_amended = True
                    amended_by_order = succ.get("order_no") or succ.get("doc_date")

            c.effective_status = "superseded" if is_superseded else ("amended" if is_amended else target_info.get("status", "current"))
            c.superseded_by = superseded_by_order
            c.amended_by = amended_by_order
            c.history_chain = timeline_chain

            # Construct clear lineage summary
            if timeline_chain:
                lineage_items = [
                    f"{item.get('doc_date') or item.get('year', 'N/A')} ({item.get('status', '').upper()})"
                    for item in timeline_chain[:4]
                ]
                c.lineage_summary = " -> ".join(lineage_items)

            # Record timeline event for UI display
            timeline_events.append({
                "doc_id": c.doc_id,
                "order_no": c.order_no,
                "date": c.doc_date,
                "status": c.effective_status,
                "lineage": timeline_chain
            })

            # Temporal Scoring Adjustment
            if is_superseded and not is_asking_historical:
                c.score *= 0.4  # Penalize obsolete order
                warn_msg = f"Order '{c.order_no or c.doc_date}' is SUPERSEDED by '{superseded_by_order}'."
                if warn_msg not in warnings:
                    warnings.append(warn_msg)

                # Pull the standing active order chunks if missing from results!
                if standing_order and standing_order.get("doc_id") and standing_order["doc_id"] not in seen_doc_ids:
                    so_doc_id = standing_order["doc_id"]
                    with self.store._get_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("""
                            SELECT chunk_id, doc_id, order_no, title, doc_date, status, content
                            FROM chunks WHERE doc_id = ? LIMIT 1;
                        """, (so_doc_id,))
                        so_row = cur.fetchone()
                        if so_row:
                            active_chunk = RetrievedChunk(
                                chunk_id=so_row["chunk_id"],
                                doc_id=so_row["doc_id"],
                                order_no=so_row["order_no"] or standing_order.get("order_no", ""),
                                title=so_row["title"] or standing_order.get("title_hi", ""),
                                doc_date=so_row["doc_date"] or standing_order.get("doc_date", ""),
                                status="current",
                                content=so_row["content"] or "",
                                score=1.2,  # Boost active replacement!
                                retrieval_source="graph_active_replacement",
                                effective_status="current",
                                lineage_summary=f"Latest active order superseding {c.order_no}"
                            )
                            enriched_results.append(active_chunk)
                            seen_doc_ids.add(so_doc_id)
            elif c.effective_status == "current":
                c.score *= 1.3  # Boost verified active orders

            enriched_results.append(c)

        # Sort by adjusted score descending
        enriched_results.sort(key=lambda x: x.score, reverse=True)
        final_results = enriched_results[:top_k]

        latency = (time.time() - start_t) * 1000.0
        return RetrievalResponse(
            query=query,
            retriever_name=self.name,
            latency_ms=round(latency, 2),
            total_candidates=len(final_results),
            results=final_results,
            timeline_trace=timeline_events[:top_k],
            warnings=warnings
        )


class RetrieverExperimentHarness:
    """Runs comparative experiments across retrievers."""

    def __init__(self, graph_store: Optional[TemporalGraphStore] = None):
        self.store = graph_store or get_graph_store()
        self.fts_retriever = LexicalFtsRetriever(self.store)
        self.graph_retriever = TemporalGraphRetriever(self.store)

    def run_experiment(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Runs both retrievers on the same query and compares results."""
        fts_res = self.fts_retriever.retrieve(query, top_k=top_k)
        graph_res = self.graph_retriever.retrieve(query, top_k=top_k)

        # Comparative analysis
        fts_doc_ids = [c.doc_id for c in fts_res.results]
        graph_doc_ids = [c.doc_id for c in graph_res.results]

        superseded_caught = [c for c in graph_res.results if c.effective_status == "superseded"]
        active_standing = [c for c in graph_res.results if c.effective_status == "current"]

        return {
            "query": query,
            "lexical_fts": {
                "latency_ms": fts_res.latency_ms,
                "result_count": len(fts_res.results),
                "top_orders": [c.order_no or c.doc_date for c in fts_res.results],
                "results": [asdict(c) for c in fts_res.results]
            },
            "temporal_graph": {
                "latency_ms": graph_res.latency_ms,
                "result_count": len(graph_res.results),
                "top_orders": [c.order_no or c.doc_date for c in graph_res.results],
                "superseded_detected_count": len(superseded_caught),
                "active_standing_count": len(active_standing),
                "warnings": graph_res.warnings,
                "timeline_trace": graph_res.timeline_trace,
                "results": [asdict(c) for c in graph_res.results]
            },
            "insights": {
                "graph_divergence": len(set(graph_doc_ids) - set(fts_doc_ids)),
                "temporal_warnings": graph_res.warnings,
                "recommended_standing_order": active_standing[0].order_no if active_standing else None
            }
        }
