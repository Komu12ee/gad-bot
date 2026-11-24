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
    pattern = re.compile(r"\[(?:D_)?([a-f0-9]{8,64})(?:#([A-Za-z0-9_]+))?\]")

    def replace_citation(match):
        raw_key = match.group(1)
        meta = index.get(raw_key) or index.get(f"D_{raw_key}") or index.get(raw_key[:16]) or index.get(raw_key[:8]) or {}
        url = meta.get("source_url") or "https://gad.cg.gov.in/"
        filename = meta.get("original_name") or meta.get("filename") or f"{raw_key[:8]}.pdf"
        cited_docs[raw_key] = meta
        return f"[[📄 {filename}]]({url})"

    transformed = pattern.sub(replace_citation, text)

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

    def expand_via_graph(self, seed_doc_ids: List[str], max_extra: int = 8) -> List[str]:
        """Spec Section 9 Step 5: Follow change edges up to 2 hops in both directions."""
        extra_docs = set()
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
        candidate_chunks = search_gad_database(question, limit=12)

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
        seed_doc_ids = list({c["doc_id"] for c in candidate_chunks[:8]})

        # Step 4: Expand via Graph (fixes interrelated documents)
        extra_doc_ids = self.expand_via_graph(seed_doc_ids, max_extra=8)
        all_active_doc_ids = list(set(seed_doc_ids + extra_doc_ids))

        # Step 5: Temporal Labelling & Context Building
        facts_list = []
        for ch in candidate_chunks[:15]:
            d_id = ch.get("doc_id", "")
            d = self.headers_by_id.get(d_id, {})
            st_label, ev = status_as_of(d, as_of_dt, self.edge_store)
            c_doc_num = d.get('doc_number_raw') or ch.get('order_no') or 'N/A'
            c_dt = d.get('issue_date') or ch.get('doc_date') or 'N/A'
            c_label = f"[{ch.get('chunk_id')} | {c_doc_num} | {c_dt} | STATUS: {st_label}]"
            c_body = ch.get('body') or ch.get('content') or ch.get('text') or ''
            fact_entry = f"{c_label}\n{c_body}"
            facts_list.append((st_label == "IN_FORCE", fact_entry, ch.get("chunk_id")))

        # Order: in-force first
        facts_list.sort(key=lambda x: (not x[0]))
        facts_text = "\n\n".join([f[1] for f in facts_list[:12]])

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
                ans_record = call_llm(prompt_r16, schema=R16_SCHEMA, schema_name="r16_ans", temperature=0.0, max_tokens=500)
            except Exception:
                ans_record = None

            # Rule 4: Verify that used_labels actually exist in the retrieved facts
            valid_labels = {f[2] for f in facts_list if f[2]}
            llm_labels = set(ans_record.get("used_labels", [])) if ans_record else set()
            is_hallucinated = bool(llm_labels and not llm_labels.intersection(valid_labels))

            # Resilient synthesis if LLM returned None, empty, hallucinated labels, or unanswerable when good circulars exist
            if not ans_record or not ans_record.get("answer") or is_hallucinated or (not ans_record.get("answerable") and candidate_chunks):
                top_chunks = candidate_chunks[:4]
                ans_text = "छत्तीसगढ़ शासन, सामान्य प्रशासन विभाग (GAD) के उपलब्ध आधिकारिक परिपत्रों एवं अधिसूचनाओं के अनुसार:\n\n"
                for idx, tc in enumerate(top_chunks, 1):
                    t_order = tc.get("order_no") or tc.get("title") or "शासकीय परिपत्र"
                    t_dt = tc.get("doc_date") or ""
                    t_date_str = f" (दिनांक: {t_dt})" if t_dt else ""
                    t_url = tc.get("source_url") or "https://gad.cg.gov.in/"
                    t_cat = f" [{tc.get('category')}]" if tc.get('category') else ""
                    t_exc = tc.get("excerpt") or (tc.get("body") or "")[:200]
                    ans_text += f"{idx}. [**आदेश क्रमांक {t_order}**]({t_url}){t_date_str}{t_cat}:\n"
                    ans_text += f"   > \"{t_exc}\"\n\n"
                ans_record = {
                    "answer": ans_text.strip(),
                    "used_labels": [f[2] for f in facts_list[:3] if f[2]],
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
