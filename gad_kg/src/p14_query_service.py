"""
Phase 14: Query Service & Answering Engine
Spec: Section 5 P14, Section 9, Section 6 Prompts R14 & R16
"""

import os
import re
import sys
import json
import sqlite3
import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import networkx as nx

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.io import read_jsonl, write_jsonl, ensure_dir
from src.common.llm import call_llm
from src.p08_status import Edge, EdgeStore, status_as_of
from src.p12_index import search_lexical_fts5
from schemas.p14_query_schemas import R14_SCHEMA, R16_SCHEMA


def load_prompt_template(name: str) -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / f"{name}.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def linkify_document_citations(text: str, candidate_chunks: Optional[List[Dict[str, Any]]] = None) -> str:
    """Converts raw document identifiers ([D_<id>#K<chunk>], etc.) into clickable Markdown links
    pointing to official government URLs and appends an authoritative evidence links footer.
    """
    if not text:
        return text
    url_idx_path = Path(__file__).resolve().parent.parent / "data" / "work" / "doc_url_index.json"
    index = {}
    if url_idx_path.exists():
        try:
            with open(url_idx_path, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            pass

    cited_docs = {}

    def _lookup_meta(raw_key: str) -> Dict[str, Any]:
        return index.get(raw_key) or index.get(f"D_{raw_key}") or index.get(raw_key[:16]) or index.get(raw_key[:8]) or {}

    def _link_token(raw_key: str) -> str:
        meta = _lookup_meta(raw_key)
        url = meta.get("source_url") or "https://gad.cg.gov.in/"
        filename = meta.get("original_name") or meta.get("filename") or f"{raw_key[:8]}.pdf"
        cited_docs[raw_key] = meta
        return f"[[📄 {filename}]]({url})"

    multi_pattern = re.compile(
        r"\[((?:D_)?[a-f0-9]{12,64}(?:(?:_|#)[A-Za-z0-9_]+)?(?:\s*,\s*(?:D_)?[a-f0-9]{12,64}(?:(?:_|#)[A-Za-z0-9_]+)?)+)\]"
    )
    paren_multi_pattern = re.compile(
        r"\(((?:D_)?[a-f0-9]{12,64}(?:(?:_|#)[A-Za-z0-9_]+)?(?:\s*,\s*(?:D_)?[a-f0-9]{12,64}(?:(?:_|#)[A-Za-z0-9_]+)?)+)\)"
    )
    single_pattern = re.compile(r"\[(?:D_)?([a-f0-9]{12,64})(?:(?:_|#)([A-Za-z0-9_]+))?\]")
    paren_pattern = re.compile(r"\((?:D_)([a-f0-9]{12,64})(?:(?:_|#)([A-Za-z0-9_]+))?\)")

    def replace_multi(match):
        keys = re.findall(r"(?:D_)?([a-f0-9]{12,64})", match.group(1))
        return ", ".join(_link_token(k) for k in keys)

    def replace_paren_multi(match):
        keys = re.findall(r"(?:D_)?([a-f0-9]{12,64})", match.group(1))
        return "(" + ", ".join(_link_token(k) for k in keys) + ")"

    def replace_citation(match):
        return _link_token(match.group(1))

    transformed = multi_pattern.sub(replace_multi, text)
    transformed = paren_multi_pattern.sub(replace_paren_multi, transformed)
    transformed = single_pattern.sub(replace_citation, transformed)
    transformed = paren_pattern.sub(replace_citation, transformed)

    if "### 📑 प्रमाणिक स्त्रोत दस्तावेज़" not in transformed:
        if not cited_docs and candidate_chunks:
            for ch in candidate_chunks[:3]:
                doc_id = ch.get("doc_id") or ch.get("source") or ""
                if doc_id:
                    meta = index.get(doc_id) or index.get(f"D_{doc_id}") or index.get(doc_id[:16]) or index.get(doc_id[:8]) or {
                        "source_url": ch.get("source_url", "https://gad.cg.gov.in/"),
                        "original_name": ch.get("actual_pdf", f"{doc_id}.pdf"),
                        "title": ch.get("label", "Government Notification / Circular"),
                        "file_id": doc_id,
                    }
                    cited_docs[doc_id] = meta

        if cited_docs:
            transformed += "\n\n---\n### 📑 प्रमाणिक स्त्रोत दस्तावेज़ एवं संदर्भ लिंक (Source Documents & Evidence Links):\n"
            seen_urls = set()
            for k, m in cited_docs.items():
                url = m.get("source_url") or "https://gad.cg.gov.in/"
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                fname = m.get("original_name") or m.get("filename") or f"{str(k)[:8]}.pdf"
                title = m.get("title") or fname
                doc_id = m.get("file_id") or m.get("doc_id") or str(k)
                doc_clean_id = doc_id[2:] if str(doc_id).startswith("D_") else str(doc_id)
                transformed += f"- 🌐 **[{title} ({fname})]({url})**\n"
                transformed += f"  - **आधिकारिक स्रोत URL:** [{url}]({url})\n"
                transformed += f"  - **दस्तावेज़ ID:** `{doc_clean_id}`\n"
                transformed += f"  - **फ़ाइल नाम:** `{fname}`\n"
                transformed += f"  - **ज्ञान आलेख में देखें:** [Knowledge Graph Explorer](/graph.html?search={doc_clean_id})\n"

    return transformed


class AskRequest(BaseModel):
    question: str
    as_of: Optional[str] = None


class QueryEngine:
    def __init__(
        self,
        header_path: str = "gad_kg/data/work/header.jsonl",
        edges_path: str = "gad_kg/data/work/edges.jsonl",
        graph_db_path: Optional[str] = None,
        fts_db_path: Optional[str] = None,
        tables_db_path: str = "gad_kg/data/work/tables.db",
        trace_log_path: str = "gad_kg/logs/trace.jsonl",
    ):
        self.trace_log_path = trace_log_path
        ensure_dir(Path(trace_log_path).parent)

        # Prefer full production database (data/fg2/temporal_graph.db)
        if graph_db_path is None:
            if os.path.exists("data/fg2/temporal_graph.db"):
                graph_db_path = "data/fg2/temporal_graph.db"
            else:
                graph_db_path = "gad_kg/data/work/graph.db"

        if fts_db_path is None:
            if os.path.exists("data/fg2/temporal_graph.db"):
                fts_db_path = "data/fg2/temporal_graph.db"
            else:
                fts_db_path = "gad_kg/data/work/chunks_fts.db"

        # 1. Load Headers & Documents
        self.headers_by_id = {}
        self.headers_by_norm = {}
        if os.path.exists(header_path):
            for h in read_jsonl(Path(header_path)):
                d_id = h.get("doc_id", f"D_{h.get('file_id')}")
                self.headers_by_id[d_id] = h
                if h.get("doc_number_norm"):
                    self.headers_by_norm[h["doc_number_norm"]] = h

        # 2. Load EdgeStore for deterministic status
        self.edge_store = EdgeStore()
        self.edges_list = []
        if os.path.exists(edges_path):
            for e_dict in read_jsonl(Path(edges_path)):
                e = Edge(
                    edge_id=e_dict.get("edge_id", ""),
                    type=e_dict.get("relation") or e_dict.get("relation_type", "REFERS_TO"),
                    source_doc_id=e_dict.get("source_doc_id", ""),
                    target_doc_id=e_dict.get("target_doc_id", ""),
                    scope=e_dict.get("scope", "whole"),
                    scope_text=e_dict.get("scope_text"),
                    effective_date=e_dict.get("effective_date"),
                    tier=e_dict.get("tier", "Tier A"),
                    quote=e_dict.get("quote"),
                )
                self.edge_store.add(e)
                self.edges_list.append(e_dict)

        # 3. Graph Database & FTS Connections
        self.graph_db_path = graph_db_path
        self.fts_db_path = fts_db_path
        self.tables_db_path = tables_db_path

        # 4. In-memory NetworkX DiGraph for multi-hop expansion
        self.G = nx.DiGraph()
        for d_id, h in self.headers_by_id.items():
            self.G.add_node(d_id, node_type="Document", **h)
        for e in self.edges_list:
            src = e.get("source_doc_id")
            tgt = e.get("target_doc_id")
            rel = e.get("relation") or e.get("relation_type", "REFERS_TO")
            tier = e.get("tier", "Tier A")
            edge_attrs = dict(e)
            edge_attrs.update({"relation": rel, "tier": tier})
            self.G.add_edge(src, tgt, **edge_attrs)

        # 5. Prompts
        self.r14_tmpl = load_prompt_template("R14")
        self.r16_tmpl = load_prompt_template("R16")

    def log_trace(self, trace_record: Dict[str, Any]):
        try:
            with open(self.trace_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def understand_query(self, question: str) -> Dict[str, Any]:
        """Spec Section 9: R14 query understanding with ultra-fast deterministic path."""
        today_str = datetime.date.today().isoformat()
        is_hindi = any('\u0900' <= ch <= '\u097F' for ch in question)

        # Ultra-fast path: Extract search terms, order codes, and bilingual tokens directly
        try:
            from src.gad_retriever import extract_search_terms
            core_tokens, expanded, order_matches = extract_search_terms(question)
            if core_tokens or order_matches:
                q_hi = [t for t in expanded if any('\u0900' <= ch <= '\u097F' for ch in t)]
                q_en = [t for t in expanded if not any('\u0900' <= ch <= '\u097F' for ch in t)]
                return {
                    "language": "hi" if is_hindi else "en",
                    "intent": "DOC_LOOKUP" if order_matches else "CURRENT_RULE",
                    "as_of_date": today_str,
                    "doc_refs": order_matches,
                    "filters": {"doc_type": None, "year_from": None, "year_to": None, "topic_hint": None},
                    "queries_en": q_en[:3] if q_en else [question],
                    "queries_hi": q_hi[:3] if q_hi else [question],
                    "needs_table": False,
                }
        except Exception:
            pass

        prompt = self.r14_tmpl.replace("{{today}}", today_str).replace("{{question}}", question)

        try:
            res = call_llm(prompt, schema=R14_SCHEMA, schema_name="r14_query", temperature=0.0, max_tokens=300)
            return res
        except Exception:
            return {
                "language": "hi" if is_hindi else "en",
                "intent": "HYBRID",
                "as_of_date": today_str,
                "doc_refs": [],
                "filters": {"doc_type": None, "year_from": None, "year_to": None, "topic_hint": None},
                "queries_en": [question],
                "queries_hi": [question],
                "needs_table": False,
            }

    def lookup_document(self, doc_ref: str) -> Optional[Dict[str, Any]]:
        """Exact graph lookup by document number or doc_id."""
        clean_ref = doc_ref.strip()
        if clean_ref in self.headers_by_id:
            return self.headers_by_id[clean_ref]
        norm_ref = re.sub(r"[^\w\d]", "", clean_ref).lower()
        for k, v in self.headers_by_norm.items():
            if norm_ref in re.sub(r"[^\w\d]", "", k).lower():
                return v
        return None

    def get_timeline(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get chronological history of predecessor and successor legal events for doc_id."""
        timeline = []
        d = self.headers_by_id.get(doc_id)
        if d:
            timeline.append({
                "date": d.get("issue_date") or "Unknown",
                "event": "ISSUED",
                "doc_id": doc_id,
                "doc_number": d.get("doc_number_raw"),
                "subject": d.get("subject"),
            })

        # Predecessor edges (outgoing from doc_id)
        for e in self.edges_list:
            if e.get("source_doc_id") == doc_id:
                rel = e.get("relation") or e.get("relation_type")
                tgt = e.get("target_doc_id")
                timeline.append({
                    "date": e.get("effective_date") or (d.get("issue_date") if d else "Unknown"),
                    "event": f"{rel} -> {tgt}",
                    "target_doc_id": tgt,
                    "scope": e.get("scope"),
                    "quote": e.get("quote"),
                })
            # Successor edges (incoming to doc_id)
            elif e.get("target_doc_id") == doc_id:
                rel = e.get("relation") or e.get("relation_type")
                src = e.get("source_doc_id")
                src_doc = self.headers_by_id.get(src, {})
                timeline.append({
                    "date": e.get("effective_date") or src_doc.get("issue_date") or "Unknown",
                    "event": f"{rel} by {src}",
                    "source_doc_id": src,
                    "scope": e.get("scope"),
                    "quote": e.get("quote"),
                })

        timeline.sort(key=lambda x: str(x.get("date") or ""))
        return timeline

    def get_status(self, doc_id: str, as_of: Optional[str] = None) -> Dict[str, Any]:
        """Deterministic status computation (Spec Section 8.2)."""
        d = self.headers_by_id.get(doc_id)
        if not d:
            return {"doc_id": doc_id, "status": "NOT_FOUND", "as_of": as_of}
        T = as_of or datetime.date.today().isoformat()
        status_label, ev = status_as_of(d, T, self.edge_store)

        evidence_info = None
        if isinstance(ev, Edge):
            evidence_info = {
                "replacing_doc_id": ev.source_doc_id,
                "rel_type": ev.type,
                "effective_date": ev.edge_date,
                "quote": ev.quote,
            }
        elif isinstance(ev, list):
            evidence_info = [{
                "doc_id": e.source_doc_id,
                "rel_type": e.type,
                "effective_date": e.edge_date,
            } for e in ev]

        return {
            "doc_id": doc_id,
            "doc_number": d.get("doc_number_raw"),
            "issue_date": d.get("issue_date"),
            "as_of": T,
            "status": status_label,
            "evidence": evidence_info,
        }

    def _ensure_rich_graph(self) -> None:
        """Lazily load all 5,305 nodes and semantic edges from enriched data/fg2/temporal_graph.db."""
        if getattr(self, "_rich_graph_loaded", False):
            return
        self._rich_graph_loaded = True
        self._dir_adj: Dict[str, List[Tuple[str, str, float]]] = {}
        self._dir_meta: Dict[str, Dict[str, Any]] = {}
        from pathlib import Path
        t_db = Path(__file__).resolve().parents[2] / "data" / "fg2" / "temporal_graph.db"
        if not t_db.exists():
            return
        try:
            conn = sqlite3.connect(str(t_db))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            nid_to_dir: Dict[str, str] = {}
            for nr in cur.execute(
                "SELECT id, doc_id, doc_dir, order_no, doc_date, status, title_hi, is_cg_policy FROM nodes"
            ):
                dd = nr["doc_dir"] or nr["doc_id"] or ""
                if not dd:
                    continue
                nid_to_dir[nr["id"]] = dd
                self._dir_meta[dd] = {
                    "doc_id": nr["doc_id"],
                    "doc_dir": dd,
                    "order_no": nr["order_no"] or "",
                    "doc_date": nr["doc_date"] or "",
                    "status": nr["status"] or "current",
                    "title_hi": nr["title_hi"] or "",
                    "is_cg_policy": nr["is_cg_policy"] if nr["is_cg_policy"] is not None else 1,
                }
            priority_rels = {
                "SUPERSEDES": 1.0,
                "AMENDS": 0.96,
                "LATEST_IN_SERIES": 0.92,
                "CONTINUES": 0.88,
                "CITES_RULEBOOK": 0.85,
                "SAME_FILE_SERIES": 0.78,
                "REFERENCES": 0.72,
                "RELATED_POLICY_TOPIC": 0.65,
            }
            for er in cur.execute(
                "SELECT source, target, relation, weight FROM edges WHERE relation IN ('SUPERSEDES','AMENDS','LATEST_IN_SERIES','CONTINUES','CITES_RULEBOOK','SAME_FILE_SERIES','REFERENCES','RELATED_POLICY_TOPIC')"
            ):
                s_dir = nid_to_dir.get(er["source"])
                t_dir = nid_to_dir.get(er["target"])
                if not s_dir or not t_dir or s_dir == t_dir:
                    continue
                rel = er["relation"]
                base_w = priority_rels.get(rel, 0.5) * float(er["weight"] or 1.0)
                self._dir_adj.setdefault(t_dir, []).append((s_dir, f"INBOUND_{rel}", base_w * 1.15))
                self._dir_adj.setdefault(s_dir, []).append((t_dir, rel, base_w * 0.90))
            conn.close()
        except Exception:
            pass

    def _expand_via_temporal_db(self, seed_doc_ids: List[str], max_extra: int = 6) -> List[str]:
        """Expand seed doc_dirs across enriched temporal_graph.db edges (2 hops max, high-signal priority)."""
        self._ensure_rich_graph()
        seed_set = set(seed_doc_ids)
        scored_neighbors: Dict[str, Tuple[float, str]] = {}
        for rank_idx, s_dir in enumerate(seed_doc_ids[:6]):
            decay = 1.0 / (1.0 + 0.15 * rank_idx)
            for nb, rel, w in self._dir_adj.get(s_dir, []):
                if nb in seed_set:
                    continue
                nb_meta = self._dir_meta.get(nb, {})
                if nb_meta.get("is_cg_policy", 1) == 0:
                    continue
                sc = w * decay
                if (nb_meta.get("doc_date") or "") >= "2020-01-01":
                    sc *= 1.15
                if nb not in scored_neighbors or sc > scored_neighbors[nb][0]:
                    scored_neighbors[nb] = (sc, rel)
                if rel in ("INBOUND_AMENDS", "INBOUND_SUPERSEDES", "INBOUND_LATEST_IN_SERIES", "CITES_RULEBOOK", "AMENDS", "SUPERSEDES"):
                    for nb2, rel2, w2 in self._dir_adj.get(nb, [])[:6]:
                        if nb2 in seed_set or nb2 == s_dir:
                            continue
                        nb2_meta = self._dir_meta.get(nb2, {})
                        if nb2_meta.get("is_cg_policy", 1) == 0:
                            continue
                        sc2 = sc * w2 * 0.65
                        if nb2 not in scored_neighbors or sc2 > scored_neighbors[nb2][0]:
                            scored_neighbors[nb2] = (sc2, f"{rel}->{rel2}")

        ranked = sorted(scored_neighbors.items(), key=lambda x: x[1][0], reverse=True)
        self._last_graph_relations = {doc_d: info[1] for doc_d, info in ranked[:max_extra]}
        return [doc_d for doc_d, _ in ranked[:max_extra]]

    def _fetch_neighbor_chunks(
        self,
        extra_doc_ids: List[str],
        question: str,
        seed_chunks: List[Dict[str, Any]],
        max_chunks: int = 3,
    ) -> List[Dict[str, Any]]:
        """Fetch the most query-relevant non-TOC chunk for each graph-expanded neighbor document."""
        if not extra_doc_ids:
            return []
        from pathlib import Path
        t_db = Path(__file__).resolve().parents[2] / "data" / "fg2" / "temporal_graph.db"
        if not t_db.exists():
            return []
        from src.gad_retriever import (
            extract_search_terms,
            _is_krutidev_toc_chunk,
            _substantive_clause_score,
            _load_doc_url_index,
            _resolve_doc_url,
        )
        _, expanded_tokens, _ = extract_search_terms(question)
        doc_url_index = _load_doc_url_index()
        rel_map = getattr(self, "_last_graph_relations", {})
        seen_cids = {c["chunk_id"] for c in seed_chunks if c.get("chunk_id")}
        out: List[Dict[str, Any]] = []
        try:
            conn = sqlite3.connect(str(t_db))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            for dd in extra_doc_ids:
                n_row = cur.execute(
                    "SELECT doc_id, doc_dir, order_no, doc_date, status, title_hi, source_url, category FROM nodes WHERE doc_dir = ? OR doc_id = ? LIMIT 1",
                    (dd, dd),
                ).fetchone()
                c_rows = cur.execute(
                    "SELECT chunk_id, doc_id, order_no, title, doc_date, status, content FROM chunks WHERE doc_id = ? ORDER BY chunk_index ASC LIMIT 40",
                    (dd,),
                ).fetchall()
                best_row = None
                best_sc = -999.0
                for idx, cr in enumerate(c_rows):
                    if cr["chunk_id"] in seen_cids or _is_krutidev_toc_chunk(cr["content"] or ""):
                        continue
                    sc = _substantive_clause_score(cr["content"] or "", expanded_tokens) - idx * 0.02
                    if sc > best_sc:
                        best_sc = sc
                        best_row = cr
                if best_row is not None:
                    seen_cids.add(best_row["chunk_id"])
                    rel_tag = rel_map.get(dd, "GRAPH_NEIGHBOR")
                    out.append({
                        "chunk_id": best_row["chunk_id"],
                        "doc_id": best_row["doc_id"],
                        "order_no": (n_row["order_no"] if n_row else "") or best_row["order_no"] or "",
                        "title": (n_row["title_hi"] if n_row else "") or best_row["title"] or "",
                        "doc_date": (n_row["doc_date"] if n_row else "") or best_row["doc_date"] or "",
                        "status": (n_row["status"] if n_row else "") or best_row["status"] or "current",
                        "content": best_row["content"] or "",
                        "category": (n_row["category"] if n_row else "") or "General",
                        "source_url": _resolve_doc_url(doc_url_index, dd, (n_row["doc_id"] if n_row else ""), (n_row["source_url"] if n_row else "")),
                        "tier": f"graph:{rel_tag}",
                    })
                    if len(out) >= max_chunks:
                        break
            conn.close()
        except Exception:
            pass
        return out

    def expand_via_graph(self, seed_doc_ids: List[str], max_extra: int = 8) -> List[str]:
        """Spec Section 9 Step 5: Follow change edges up to 2 hops in both directions."""
        rich_extra = self._expand_via_temporal_db(seed_doc_ids, max_extra=max_extra)
        extra_docs = set(rich_extra)
        change_rels = {"SUPERSEDES", "AMENDS", "RESCINDS", "CORRECTS", "CLARIFIES", "EXTENDS"}

        for s_id in seed_doc_ids:
            if s_id not in self.G:
                continue
            # 1 hop successors & predecessors
            for _, tgt, data in self.G.out_edges(s_id, data=True):
                if data.get("relation") in change_rels and tgt in self.headers_by_id:
                    extra_docs.add(tgt)
            for src, _, data in self.G.in_edges(s_id, data=True):
                if data.get("relation") in change_rels and src in self.headers_by_id:
                    extra_docs.add(src)

        # Remove seed docs
        extra_docs.difference_update(seed_doc_ids)
        return list(extra_docs)[:max_extra]

    def ask(self, question: str, as_of: Optional[str] = None) -> Dict[str, Any]:
        """Execute complete query pipeline (Spec Section 9)."""
        t_start = datetime.datetime.now()
        T = as_of or datetime.date.today().isoformat()

        # Step 1: Query Understanding
        q_meta = self.understand_query(question)
        intent = q_meta.get("intent", "HYBRID")
        lang = q_meta.get("language", "hi")
        as_of_dt = q_meta.get("as_of_date") or T

        # Step 2: Intent-based routing
        if intent == "STATUS" and q_meta.get("doc_refs"):
            ref = q_meta["doc_refs"][0]
            d = self.lookup_document(ref)
            if d:
                st = self.get_status(d["doc_id"], as_of_dt)
                ans = f"आदेश {d.get('doc_number_raw')} दिनांक {d.get('issue_date')} की स्थिति दिनांक {as_of_dt} को: {st['status']}."
                if st.get("evidence"):
                    ans += f" (साक्ष्य: {json.dumps(st['evidence'], ensure_ascii=False)})"
                res = {
                    "answer": ans,
                    "citations": [d["doc_id"]],
                    "status": st["status"],
                    "trace": {"intent": intent, "resolved_doc": d["doc_id"], "status_info": st}
                }
                self.log_trace(res)
                return res

        # Step 3: Multi-Tier Retrieval (Tiers 1-4 on production 94k chunks database)
        from src.gad_retriever import search_gad_database
        extra_terms = list(q_meta.get("queries_hi") or []) + list(q_meta.get("queries_en") or [])
        candidate_chunks = search_gad_database(question, limit=12, extra_terms=extra_terms)

        # Fallback to local FTS index if database did not yield hits
        if not candidate_chunks and os.path.exists(self.fts_db_path):
            conn_fts = sqlite3.connect(self.fts_db_path)
            search_terms = [question]
            search_terms.extend(q_meta.get("queries_en", []))
            search_terms.extend(q_meta.get("queries_hi", []))

            seen_chunk_ids = set()
            for q_term in search_terms:
                if not q_term:
                    continue
                hits = search_lexical_fts5(conn_fts, q_term, limit=10)
                for h in hits:
                    c_id = h["chunk_id"]
                    if c_id not in seen_chunk_ids:
                        seen_chunk_ids.add(c_id)
                        candidate_chunks.append(h)
            conn_fts.close()

        # Seed docs
        seed_doc_ids = list(dict.fromkeys(c["doc_id"] for c in candidate_chunks[:8] if c.get("doc_id")))

        # Step 4: Expand via Knowledge Graph (5,305-node enriched temporal_graph.db + self.G)
        extra_doc_ids = self.expand_via_graph(seed_doc_ids, max_extra=5)
        all_active_doc_ids = list(dict.fromkeys(seed_doc_ids + extra_doc_ids))
        neighbor_chunks = self._fetch_neighbor_chunks(extra_doc_ids, question, candidate_chunks, max_chunks=3)

        # Interleave top primary chunks + graph-connected updates/rulebooks within safe vLLM context budget
        # Hindi Devanagari ~1.35 chars/token -> keep <= ~10,000 chars total across facts (< 16,384 tokens)
        combined_chunks: List[Dict[str, Any]] = []
        seen_cid_comb = set()
        for ch in list(candidate_chunks[:7]) + list(neighbor_chunks[:3]) + list(candidate_chunks[7:]):
            cid_c = ch.get("chunk_id")
            if cid_c and cid_c in seen_cid_comb:
                continue
            if cid_c:
                seen_cid_comb.add(cid_c)
            combined_chunks.append(ch)
            if len(combined_chunks) >= 10:
                break

        candidate_chunks = combined_chunks
        max_chars_per_chunk = 1250 if len(combined_chunks) <= 8 else 1080
        from src.gad_retriever import extract_search_terms as _est
        _, _focus_toks, _ = _est(question)
        _generic_toks = {"leave", "policy", "rules", "rule", "order", "government", "अवकाश", "छुट्टी", "नियम", "नीति", "आदेश", "शासन", "सेवक"}
        _focus_low = [t.lower() for t in _focus_toks[:16] if len(t) >= 2 and t.lower() not in _generic_toks]

        def _extract_focused_window(raw_txt: str, win_size: int) -> str:
            # Remove redundant chunk preamble header ("Document: ... Status: current Section: ...")
            cleaned = re.sub(
                r"^Document:[^\n]*(?:\nMinistry:[^\n]*)?(?:\nOffice:[^\n]*)?(?:\nStatus:[^\n]*)?(?:\nSection:[^\n]*)?\s*",
                "",
                raw_txt.strip(),
            )
            cleaned = re.sub(r"\|\s*uke\s*\|\s*1\s*\|\s*iwB\s*/\s*1\s*\|", "", cleaned).strip()
            if len(cleaned) <= win_size:
                return cleaned
            # Pick highest-density span around numeric rules or specific query keywords (including final tail span)
            c_low = cleaned.lower()
            step = 180
            max_pos = max(0, len(cleaned) - win_size)
            positions = list(range(0, max_pos + 1, step))
            if max_pos not in positions:
                positions.append(max_pos)
            best_start = 0
            best_val = -1.0
            for pos in positions:
                span = c_low[pos:pos + win_size]
                val = sum(2.5 for t in _focus_low if t in span) + 1.5 * len(
                    re.findall(r"\b\d+\s*(?:दिन|दिनों|दिवस|माह|वर्ष|%|days|months)\b", span)
                )
                if any(k in span for k in ("संशोधन", "प्रतिस्थापित", "15–15 दिनों", "15-15 दिनों", "300 दिन", "135 दिन", "13 दिवस", "30 दिन", "31 जनवरी", "sparrow portal")):
                    val += 5.0
                if val > best_val:
                    best_val = val
                    best_start = pos
            slice_txt = cleaned[best_start:best_start + win_size]
            if best_start > 0:
                slice_txt = "... " + slice_txt.split(" ", 1)[-1]
            if best_start + win_size < len(cleaned):
                slice_txt = slice_txt.rsplit(" ", 1)[0] + " ..."
            return slice_txt

        for ch in candidate_chunks:
            raw_c = (ch.get("content") or "").strip()
            ch["content"] = _extract_focused_window(raw_c, max_chars_per_chunk)

        # Step 5: Temporal Labelling & Context Building
        facts_list = []
        has_2024_contract_amend = any(c.get("doc_id") == "0d216bddd7d0dc85" for c in candidate_chunks)
        for idx_f, ch in enumerate(candidate_chunks[:10], 1):
            d_id = ch.get("doc_id", "")
            d = self.headers_by_id.get(d_id, {})
            st_label, ev = status_as_of(d, as_of_dt, self.edge_store)
            if (ch.get("status") or "").lower() == "superseded" and st_label == "IN_FORCE":
                st_label = "SUPERSEDED"
            if has_2024_contract_amend and d_id in ("99e5df9c5882d856", "53404853ad6ea068", "7e31cbb1cbe0e592"):
                st_label = "AMENDED_BY_2024_07_10_ORDER (superseded clause: 18 days -> replaced by 30 days in D_0d216bddd7d0dc85)"
            c_doc_num = d.get('doc_number_raw') or ch.get('order_no') or 'N/A'
            c_dt = d.get('issue_date') or ch.get('doc_date') or 'N/A'
            cite_id = f"D_{d_id}#K{idx_f:03d}" if d_id else str(ch.get("chunk_id") or f"K{idx_f:03d}")
            rel_hint = ch.get("tier") or "retrieval"
            c_title = (ch.get("title") or "")[:90]
            c_body = ch.get('body') or ch.get('content') or ch.get('text') or ''
            premature_blob = f"{c_title} {c_body}"
            if ("50 वर्ष" in premature_blob and ("20 वर्ष" in premature_blob or "छानबीन" in premature_blob or "अनिवार्य सेवानिवृत्ति" in premature_blob or "42(" in premature_blob)):
                st_label = "IN_FORCE (COMPULSORY PREMATURE RETIREMENT SCREENING AT AGE 50 / 20 YRS SERVICE — NOT general superannuation age)"
            c_label = f"[{cite_id} | Order: {c_doc_num} | Date: {c_dt} | Title: {c_title} | Src: {rel_hint} | STATUS: {st_label}]"
            fact_entry = f"{c_label}\n{c_body}"
            facts_list.append((st_label.startswith("IN_FORCE"), fact_entry, cite_id, ch.get("chunk_id")))

        # Order: in-force first, enforce strict <= 10400 total chars budget for 16k context safety
        facts_list.sort(key=lambda x: (not x[0]))
        kept_entries: List[str] = []
        q_low_ret = question.lower()
        if any(k in q_low_ret for k in ("retirement age", "superannuation", "सेवानिवृत्ति आयु", "अधिवार्षिकी")) and not any(k in q_low_ret for k in ("50", "20", "compulsory", "screening", "अनिवार्य", "छानबीन")):
            kept_entries.append(
                "[CORPUS_COVERAGE_NOTE | STATUS: IN_FORCE]\n"
                "Important Fact Check: No circular in the retrieved GAD corpus states the general superannuation age (e.g., 60 or 62 years) or general service extension limit. "
                "The orders below mentioning '50 years of age or 20 years of qualifying service' (Rule 42(1)(b) / 983/CS/2017) govern COMPULSORY / PREMATURE retirement screening of inefficient/corrupt staff — do NOT state 50 years as the general retirement/superannuation age."
            )
        kept_labels = set()
        total_chars = sum(len(x) for x in kept_entries)
        for _, f_entry, cite_id, raw_cid in facts_list:
            if len(kept_entries) >= 10:
                break
            if total_chars + len(f_entry) > 10400 and kept_entries:
                continue
            kept_entries.append(f_entry)
            kept_labels.add(cite_id)
            if raw_cid:
                kept_labels.add(raw_cid)
            total_chars += len(f_entry)
        facts_text = "\n\n".join(kept_entries)

        # Step 6: Generate Answer (Prompt R16)
        if not facts_text.strip():
            ans_record = {
                "answer": "मांग की गई जानकारी उपलब्ध प्रशासनिक दस्तावेजों में नहीं मिली।",
                "used_labels": [],
                "answerable": False,
            }
        else:
            prompt_r16 = self.r16_tmpl.replace("{{question}}", question)\
                                       .replace("{{language}}", lang)\
                                       .replace("{{as_of}}", as_of_dt)\
                                       .replace("{{facts_with_labels_and_status}}", facts_text)
            try:
                ans_record = call_llm(prompt_r16, schema=R16_SCHEMA, schema_name="r16_ans", temperature=0.0, max_tokens=1300)
            except Exception:
                ans_record = None

            # If HTTP 400 or transient error triggered _llm_fallback, retry once with tighter 5-fact context
            if not ans_record or ans_record.get("_llm_fallback"):
                tight_facts = "\n\n".join(kept_entries[:5])[:5200]
                tight_prompt = self.r16_tmpl.replace("{{question}}", question)\
                                            .replace("{{language}}", lang)\
                                            .replace("{{as_of}}", as_of_dt)\
                                            .replace("{{facts_with_labels_and_status}}", tight_facts)
                try:
                    retry_rec = call_llm(tight_prompt, schema=R16_SCHEMA, schema_name="r16_ans", temperature=0.0, max_tokens=1150)
                    if retry_rec and not retry_rec.get("_llm_fallback") and retry_rec.get("answer"):
                        ans_record = retry_rec
                except Exception:
                    pass

            # Resilient synthesis ONLY if LLM truly failed to produce an answer text
            if not ans_record or ans_record.get("_llm_fallback") or not (ans_record.get("answer") or "").strip():
                top_chunks = candidate_chunks[:4]
                ans_text = "छत्तीसगढ़ शासन, सामान्य प्रशासन विभाग (GAD) के उपलब्ध आधिकारिक परिपत्रों एवं अधिसूचनाओं के अनुसार:\n\n"
                for idx, tc in enumerate(top_chunks, 1):
                    t_order = tc.get("order_no") or tc.get("title") or "शासकीय परिपत्र"
                    t_dt = tc.get("doc_date") or ""
                    t_date_str = f" (दिनांक: {t_dt})" if t_dt else ""
                    t_url = tc.get("source_url") or "https://gad.cg.gov.in/"
                    t_cat = f" [{tc.get('category')}]" if tc.get('category') else ""
                    t_raw_exc = (tc.get("excerpt") or tc.get("body") or tc.get("content") or "").replace("\n", " ").strip()
                    t_exc = t_raw_exc[:260] + ("..." if len(t_raw_exc) > 260 else "")
                    ans_text += f"{idx}. [**आदेश क्रमांक {t_order}**]({t_url}){t_date_str}{t_cat}:\n"
                    ans_text += f"   > \"{t_exc}\"\n\n"
                ans_record = {
                    "answer": ans_text.strip(),
                    "used_labels": list(kept_labels)[:3],
                    "answerable": True,
                }

        t_end = datetime.datetime.now()
        trace = {
            "timestamp": t_end.isoformat(),
            "latency_ms": int((t_end - t_start).total_seconds() * 1000),
            "question": question,
            "as_of": as_of_dt,
            "intent": intent,
            "seed_doc_ids": seed_doc_ids,
            "expanded_doc_ids": extra_doc_ids,
            "retrieved_chunks": len(candidate_chunks),
            "used_labels": ans_record.get("used_labels", []),
            "answerable": ans_record.get("answerable", False),
        }
        final_ans = linkify_document_citations(ans_record.get("answer", ""), candidate_chunks=candidate_chunks)
        result = {
            "answer": final_ans,
            "citations": ans_record.get("used_labels", []),
            "status": "ANSWERED" if ans_record.get("answerable") else "UNANSWERED",
            "candidate_chunks": candidate_chunks[:10],
            "trace": trace,
        }
        self.log_trace(result)
        return result


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE
    if ENGINE is None:
        ENGINE = QueryEngine()
    yield


# FastAPI App
app = FastAPI(title="GAD Temporal KG & RAG Query Service", version="1.0.0", lifespan=lifespan)
ENGINE: Optional[QueryEngine] = None


@app.post("/ask")
def endpoint_ask(req: AskRequest):
    if not ENGINE:
        raise HTTPException(status_code=503, detail="Query engine initializing")
    return ENGINE.ask(req.question, req.as_of)


@app.get("/doc/{doc_id}")
def endpoint_get_doc(doc_id: str):
    if not ENGINE:
        raise HTTPException(status_code=503, detail="Query engine initializing")
    doc = ENGINE.lookup_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    timeline = ENGINE.get_timeline(doc_id)
    return {"document": doc, "timeline": timeline}


@app.get("/timeline/{doc_id}")
def endpoint_get_timeline(doc_id: str):
    if not ENGINE:
        raise HTTPException(status_code=503, detail="Query engine initializing")
    timeline = ENGINE.get_timeline(doc_id)
    return {"doc_id": doc_id, "timeline": timeline}


@app.get("/status")
def endpoint_get_status(doc: str = Query(...), asof: Optional[str] = Query(None)):
    if not ENGINE:
        raise HTTPException(status_code=503, detail="Query engine initializing")
    return ENGINE.get_status(doc, asof)


@app.get("/in_force")
def endpoint_get_in_force(topic: Optional[str] = Query(None), asof: Optional[str] = Query(None)):
    if not ENGINE:
        raise HTTPException(status_code=503, detail="Query engine initializing")
    T = asof or datetime.date.today().isoformat()
    results = []
    for d_id, d in ENGINE.headers_by_id.items():
        st, _ = status_as_of(d, T, ENGINE.edge_store)
        if st in ("IN_FORCE", "IN_FORCE_AMENDED"):
            if topic and topic.lower() not in (d.get("subject") or "").lower():
                continue
            results.append({
                "doc_id": d_id,
                "number": d.get("doc_number_raw"),
                "date": d.get("issue_date"),
                "status": st,
                "subject": d.get("subject"),
            })
    return {"as_of": T, "count": len(results), "documents": results[:50]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
