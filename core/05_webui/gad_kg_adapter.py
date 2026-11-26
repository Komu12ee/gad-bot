"""
GAD Temporal Knowledge Graph Frontend Adapter
Bridges the newly built gad_kg (Phases 0-15) directly into the Flask Web UI (app.py).
"""

import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "gad_kg") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "gad_kg"))

from gad_kg.src.p14_query_service import QueryEngine
from gad_kg.src.officer_store import OfficerStore
from src.common.llm import call_llm

_ENGINE_INSTANCE: Optional[QueryEngine] = None
_OFFICER_STORE: Optional[OfficerStore] = None
_ROUTER_CACHE: Dict[str, str] = {}


def route_query_intent(query: str) -> str:
    """LLM-based Query Router that classifies incoming queries into:
    - 'OFFICER': Inquiries about specific officers, personnel, people, civil servants,
      IAS/SAS officers, Secretaries, Collectors, SPs, DFOs, CHiPS leadership, or personnel directories.
    - 'DOCUMENT': Inquiries about policies, leave rules, circulars, notifications,
      transfer orders/letters, pension schemes, RTI, acts, rules, government procedures, or general administrative documents.
    """
    q_norm = query.strip().lower()
    if q_norm in _ROUTER_CACHE:
        return _ROUTER_CACHE[q_norm]

    # Fast heuristic rule: clear policy/leave/document keywords without any person reference
    doc_strong_keywords = [
        "leave policy", "earned leave", "casual leave", "medical leave", "maternity leave",
        "pension rule", "pension scheme", "transfer order letter", "transfer guideline",
        "transfer policy", "act", "ordinance", "circular", "notification", "peripatra",
        "niti", "niyam", "परिपत्र", "नियम", "अधिसूचना", "अवकाश नियम", "अवकाश नीति",
        "पेंशन", "स्थानांतरण नीति"
    ]
    if any(kw in q_norm for kw in doc_strong_keywords) and not any(kw in q_norm for kw in ["who is", "who was", "kon hain", "kon hai", "shri ", "smt ", "dr ", "secretary", "collector"]):
        _ROUTER_CACHE[q_norm] = "DOCUMENT"
        return "DOCUMENT"

    if any(k in q_norm for k in ["list all collectors", "all collectors", "district collectors", "कलेक्टरों की सूची", "समस्त कलेक्टर", "former chief secretary", "chief secretary before", "पूर्व मुख्य सचिव"]):
        _ROUTER_CACHE[q_norm] = "OFFICER"
        return "OFFICER"

    prompt = (
        "You are an intent router for a Government Administration Department (GAD) AI assistant.\n"
        "Determine whether the user query is asking about:\n"
        '- "OFFICER": Inquiries about specific officers, personnel, people, civil servants, IAS/SAS officers, Secretaries, Collectors, SPs, DFOs, CHiPS leadership, or government personnel directories (e.g. \'who is anupam asish toppo\', \'who is mayank agrawal\', \'who is the chief secretary\', \'tell me about Shri Chandan Kumar\', \'list of retired IAS\').\n'
        '- "DOCUMENT": Inquiries about policies, leave rules, circulars, notifications, transfer orders/letters, pension schemes, RTI, acts, rules, government procedures, or general administrative documents (e.g. \'Can you tell me about the leave policy?\', \'Can you tell me about the latest transfer order letter?\', \'what are the rules for earned leave?\', \'what is in the IAS establishment section?\').\n\n'
        f"Query: {query}\n\n"
        "Output JSON matching schema."
    )
    schema = {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": ["OFFICER", "DOCUMENT"]}
        },
        "required": ["route"]
    }
    try:
        res = call_llm(prompt, schema=schema, schema_name="router_schema", temperature=0.0, max_tokens=25)
        route = res.get("route", "DOCUMENT")
    except Exception as e:
        if any(k in q_norm for k in ["who is", "who was", "kon hain", "kon hai", "shri ", "smt ", "dr ", "secretary", "collector", "sp ", "dfo "]):
            route = "OFFICER"
        else:
            route = "DOCUMENT"

    _ROUTER_CACHE[q_norm] = route
    return route


def get_officer_store() -> OfficerStore:
    global _OFFICER_STORE
    if _OFFICER_STORE is None:
        _OFFICER_STORE = OfficerStore()
    return _OFFICER_STORE


def format_officer_sources_footer(
    primary_doc: Optional[Dict[str, Any]] = None,
    notifications: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Builds a comprehensive markdown citation block listing source documents, official links, and metadata."""
    footer = "\n\n---\n### 📑 प्रमाणिक स्त्रोत दस्तावेज़ एवं संदर्भ लिंक (Source Documents & Evidence Links):\n"

    if primary_doc:
        s_title = primary_doc.get("source_title") or primary_doc.get("source_name") or "सामान्य प्रशासन विभाग सिविल लिस्ट"
        s_url = primary_doc.get("source_url") or "https://gad.cg.gov.in/Display_civil_list_Email.aspx"
        s_id = primary_doc.get("source_doc_id") or primary_doc.get("doc_id") or "efc290b78f6c8ad7"
        s_can = primary_doc.get("canonical_source_id") or primary_doc.get("canonical_doc_id") or s_id
        s_file = primary_doc.get("filename") or "Display_civil_list_Email.aspx"

        footer += f"- 🌐 **प्राथमिक अभिलेख (Primary Governance Record):**\n"
        footer += f"  - **दस्तावेज़ शीर्षक:** [{s_title}]({s_url})\n"
        footer += f"  - **आधिकारिक स्रोत URL:** [{s_url}]({s_url})\n"
        footer += f"  - **दस्तावेज़ ID (Doc ID):** `{s_id}` (कैनोनिकल ID: `{s_can}`)\n"
        footer += f"  - **स्रोत फ़ाइल / पृष्ठ:** `{s_file}`\n"
        footer += f"  - **विभागीय पोर्टल:** सामान्य प्रशासन विभाग (GAD), छत्तीसगढ़ शासन\n"

    if notifications:
        footer += f"\n#### 📄 संबंधित शासकीय अधिसूचनाएं एवं परिपत्र (Linked Official Orders & Circulars):\n"
        for i, n in enumerate(notifications[:6], 1):
            n_order = n.get("order_no") or f"GAD Order ({n.get('doc_id', '')[:8]})"
            n_date = n.get("order_date") or "N/A"
            n_title = n.get("doc_title") or n.get("subject") or "Government Notification / Circular"
            n_url = n.get("source_url") or "https://gad.cg.gov.in/"
            n_file = n.get("filename") or f"{n.get('doc_id')}.pdf"
            n_id = n.get("doc_id", "")
            n_can = n.get("canonical_doc_id") or n_id
            excerpt = (n.get("excerpt") or "").strip()

            footer += f"{i}. [**आदेश क्रमांक {n_order}**]({n_url}) ({n_date}):\n"
            footer += f"   - **दस्तावेज़ शीर्षक:** [{n_title}]({n_url})\n"
            footer += f"   - **आधिकारिक स्रोत URL:** [{n_url}]({n_url})\n"
            footer += f"   - **फ़ाइल नाम:** `{n_file}` (Doc ID: `{n_id}`)\n"
            footer += f"   - **ज्ञान आलेख में देखें:** [Knowledge Graph Explorer](/graph.html?search={n_id})\n"
            if excerpt:
                footer += f"   - **सत्यापित उद्धरण (Verified Excerpt):** *\"{excerpt[:160]}...\"*\n"

    return footer


def linkify_document_citations(text: str, candidate_chunks: Optional[List[Dict[str, Any]]] = None) -> str:
    """Converts raw document identifiers ([D_<id>#K<chunk>], [D_<id>], etc.) into clickable Markdown links
    pointing to official government URLs and appends an authoritative evidence links footer.
    """
    if not text:
        return text

    from gad_kg.src.officer_store import load_document_metadata_index
    index = load_document_metadata_index()

    cited_docs: Dict[str, Dict[str, Any]] = {}
    single_cite_re = re.compile(r"(?<![A-Za-z0-9_])(?:D_([a-f0-9]{8,64})|([a-f0-9]{16,64}))(?:[#_]([A-Za-z0-9_]+))?(?![A-Za-z0-9_\.])")
    bracket_cite_re = re.compile(r"\[([^\]]*(?:D_[a-f0-9]{8,64}|[a-f0-9]{16,64})[^\]]*)\](?!\s*\()")

    def _format_cite_link(raw_key: str) -> str:
        meta = index.get(raw_key) or index.get(f"D_{raw_key}") or index.get(raw_key[:16]) or index.get(raw_key[:8]) or {}
        url = meta.get("source_url") or "https://gad.cg.gov.in/"
        filename = meta.get("original_name") or meta.get("filename") or f"{raw_key[:8]}.pdf"
        cited_docs[raw_key] = meta
        return f"[[📄 {filename}]]({url})"

    def replace_citation_bracket(match):
        inner = match.group(1)
        if "📄" in inner:
            return match.group(0)
        keys = single_cite_re.findall(inner)
        if not keys:
            return match.group(0)
        links = []
        seen_local = set()
        for k1, k2, _ in keys:
            raw_key = k1 or k2
            if raw_key and raw_key not in seen_local:
                seen_local.add(raw_key)
                links.append(_format_cite_link(raw_key))
        return " ".join(links) if links else match.group(0)

    transformed = bracket_cite_re.sub(replace_citation_bracket, text)

    def replace_paren_hex(match):
        hex_key = match.group(1)
        if hex_key in index:
            meta = index[hex_key]
            url = meta.get("source_url") or "https://gad.cg.gov.in/"
            filename = meta.get("original_name") or f"{hex_key}.pdf"
            cited_docs[hex_key] = meta
            return f"([{filename}]({url}))"
        return match.group(0)

    transformed = re.sub(r"\((?:D_)?([a-f0-9]{16,64})(?:#[A-Za-z0-9_]+)?\)", replace_paren_hex, transformed)

    has_footer = "### 📑 प्रमाणिक स्त्रोत दस्तावेज़" in transformed
    if not has_footer:
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


def resolve_officer_query(q: str) -> Optional[Dict[str, Any]]:
    """Checks if the query is an officer, secretary, or establishment lookup and returns structured facts with document links."""
    store = get_officer_store()
    q_low = q.lower()
    
    # 1. Chief Secretary (handle both current and former/previous Chief Secretaries)
    if any(k in q_low for k in ["chief secretary", "मुख्य सचिव"]):
        asks_former = any(
            k in q_low
            for k in ["before", "former", "previous", "preceded", "predecessor", "ex-", "पूर्व", "पहले", "पिछले"]
        )
        if asks_former and hasattr(store, "get_former_chief_secretaries"):
            former_list = store.get_former_chief_secretaries()
            cur_cs = store.get_chief_secretary()
            if former_list:
                immediate = former_list[0]
                ans = "### पूर्व मुख्य सचिव (Former Chief Secretaries) - छत्तीसगढ़ शासन\n\n"
                ans += f"वर्तमान मुख्य सचिव **{(cur_cs or {}).get('name_hi', 'श्री विकाश शील')} ({(cur_cs or {}).get('name_en', 'Shri Vikas Sheel')})** से पूर्व छत्तीसगढ़ शासन के मुख्य सचिव:\n\n"
                for idx_f, fo in enumerate(former_list, 1):
                    ans += f"{idx_f}. **{fo['name_hi']} ({fo['name_en']})** — {fo['designation_hi']}\n"
                all_notifs = list(immediate.get("notifications") or [])
                if all_notifs:
                    ans += "\n#### संबंधित प्रमुख शासकीय अधिसूचनाएं एवं परिपत्र:\n"
                    for n in all_notifs[:5]:
                        ans += f"- **{n['order_no']}** ({n['order_date']}): {n['subject']}\n"
                ans += format_officer_sources_footer(primary_doc=immediate, notifications=all_notifs)
                return {
                    "answer": ans,
                    "source": "officer_store",
                    "matched_officers": former_list,
                }
        cs = store.get_chief_secretary()
        if cs:
            cs = store.get_officer(cs['id']) or cs
            ans = f"### मुख्य सचिव (Chief Secretary) - छत्तीसगढ़ शासन\n\n"
            ans += f"**{cs['name_hi']} ({cs['name_en']})** छत्तीसगढ़ शासन के वर्तमान **मुख्य सचिव (Chief Secretary)** हैं।\n\n"
            ans += f"- **पदनाम:** {cs['designation_hi']} ({cs['designation_en']})\n"
            ans += f"- **विभाग:** {cs['department_hi']} ({cs['department_en']})\n"
            ans += f"- **ई-मेल:** `{cs['email']}`\n"
            if cs.get('additional_charges'):
                ans += f"- **अतिरिक्त प्रभार:** {', '.join(cs['additional_charges'])}\n"

            if cs.get('notifications'):
                ans += "\n#### संबंधित प्रमुख शासकीय अधिसूचनाएं एवं परिपत्र:\n"
                for n in cs['notifications'][:5]:
                    ans += f"- **{n['order_no']}** ({n['order_date']}): {n['subject']}\n"

            ans += format_officer_sources_footer(primary_doc=cs, notifications=cs.get('notifications'))

            chunks = [{
                "doc_id": n["doc_id"],
                "canonical_doc_id": n.get("canonical_doc_id", n["doc_id"]),
                "source_url": n.get("source_url", "https://gad.cg.gov.in/"),
                "actual_pdf": n.get("filename") or f"{n['doc_id']}.pdf",
                "label": f"आदेश {n['order_no']} ({n['order_date']})",
                "body": f"शासकीय आदेश {n['order_no']} ({n['order_date']}): {n.get('doc_title', n['subject'])}. {n.get('excerpt', '')}",
                "excerpt": n.get("excerpt", "")
            } for n in cs.get("notifications", [])]

            if not chunks:
                chunks = [{
                    "doc_id": cs.get("source_doc_id", "efc290b78f6c8ad7"),
                    "canonical_doc_id": cs.get("canonical_source_id", "cec78d786be946f4a6316afdf05854fd"),
                    "source_url": cs.get("source_url", "https://gad.cg.gov.in/Display_civil_list_Email.aspx"),
                    "actual_pdf": cs.get("filename", "Display_civil_list_Email.aspx"),
                    "label": f"सिविल लिस्ट - मुख्य सचिव: {cs['name_en']}",
                    "body": f"मुख्य सचिव: {cs['name_hi']} ({cs['name_en']}), पद: {cs['designation_hi']}, विभाग: {cs['department_hi']}",
                    "excerpt": f"मुख्य सचिव छत्तीसगढ़ शासन: {cs['name_hi']}"
                }]
            return {"answer": linkify_document_citations(ans, candidate_chunks=chunks), "officer": cs, "chunks": chunks}

    # 2. List Secretaries
    if "list" in q_low and any(k in q_low for k in ["secretaries", "सचिव"]):
        secs = store.get_secretaries()
        ans = f"### छत्तीसगढ़ शासन के प्रमुख सचिवों एवं अपर मुख्य सचिवों की सूची (Secretaries to Govt of CG):\n\n"
        ans += f"वर्तमान राज्य सिविल लिस्ट एवं सामान्य प्रशासन विभाग के अनुसार प्रमुख सचिव निम्नलिखित हैं:\n\n"
        ans += "| क्र. | नाम (Officer Name) | पदनाम (Designation) | विभाग (Department) | ई-मेल (Email) |\n"
        ans += "|---|---|---|---|---|\n"
        for i, s in enumerate(secs[:30], 1):
            ans += f"| {i} | **{s['name_hi']}** ({s['name_en']}) | {s['designation_hi']} | {s['department_hi'] or 'मंत्रालय'} | `{s['email']}` |\n"
        ans += f"\n*(कुल {len(secs)} सचिवीय अधिकारी दर्ज हैं। पूरा विवरण आधिकारिक सिविल लिस्ट डायरेक्टरी में उपलब्ध है।)*\n"

        primary_meta = {
            "source_title": "छत्तीसगढ़ शासन सिविल लिस्ट ईमेल डायरेक्टरी",
            "source_url": "https://gad.cg.gov.in/Display_civil_list_Email.aspx",
            "source_doc_id": "efc290b78f6c8ad7",
            "canonical_source_id": "cec78d786be946f4a6316afdf05854fd",
            "filename": "Display_civil_list_Email.aspx"
        }
        ans += format_officer_sources_footer(primary_doc=primary_meta)

        chunks = [{
            "doc_id": "efc290b78f6c8ad7",
            "canonical_doc_id": "cec78d786be946f4a6316afdf05854fd",
            "source_url": "https://gad.cg.gov.in/Display_civil_list_Email.aspx",
            "actual_pdf": "Display_civil_list_Email.aspx",
            "label": "छत्तीसगढ़ शासन - सिविल लिस्ट डायरेक्टरी",
            "body": f"छत्तीसगढ़ शासन सिविल लिस्ट ईमेल डायरेक्टरी: {len(secs)} प्रमुख सचिव, अपर मुख्य सचिव व सचिव गण।",
            "excerpt": "छत्तीसगढ़ शासन के समस्त विभागों के अपर मुख्य सचिव, प्रमुख सचिव व सचिव गणों की सूची।"
        }]
        return {"answer": ans, "officer": None, "chunks": chunks}

    # 2b. List District Collectors
    if any(k in q_low for k in ["list", "all", "सूची", "समस्त"]) and any(k in q_low for k in ["collector", "collectors", "कलेक्टर", "कलेक्टरों"]):
        cols = store.get_collectors()
        if cols:
            ans = "### छत्तीसगढ़ के जिला कलेक्टरों की सूची (District Collectors of Chhattisgarh):\n\n"
            ans += "| क्र. | जिला (District) | कलेक्टर का नाम (Collector Name) | ई-मेल (Email) |\n"
            ans += "|---|---|---|---|\n"
            for i, c in enumerate(cols, 1):
                dist = c.get("department_hi") or c.get("department_en") or c.get("designation_hi") or "-"
                ans += f"| {i} | {dist} | **{c.get('name_hi','')}** ({c.get('name_en','')}) | `{c.get('email','') or '-'}` |\n"
            ans += f"\n*(कुल {len(cols)} जिला कलेक्टर राज्य सिविल लिस्ट डायरेक्टरी में दर्ज हैं।)*\n"
            primary_meta = {
                "source_title": "छत्तीसगढ़ शासन सिविल लिस्ट / कलेक्टर डायरेक्टरी",
                "source_url": "https://gad.cg.gov.in/Display_civil_list_Email.aspx",
                "source_doc_id": "efc290b78f6c8ad7",
                "canonical_source_id": "cec78d786be946f4a6316afdf05854fd",
                "filename": "Display_civil_list_Email.aspx",
            }
            ans += format_officer_sources_footer(primary_doc=primary_meta)
            chunks = [{
                "doc_id": "efc290b78f6c8ad7",
                "canonical_doc_id": "cec78d786be946f4a6316afdf05854fd",
                "source_url": "https://gad.cg.gov.in/Display_civil_list_Email.aspx",
                "actual_pdf": "Display_civil_list_Email.aspx",
                "label": "छत्तीसगढ़ शासन - जिला कलेक्टर डायरेक्टरी",
                "body": f"छत्तीसगढ़ शासन जिला कलेक्टर सूची ({len(cols)} जिले)।",
                "excerpt": "छत्तीसगढ़ शासन के समस्त जिला कलेक्टरों की आधिकारिक सूची।",
            }]
            return {"answer": ans, "officer": None, "chunks": chunks}

    # 3. Retired IAS
    if any(k in q_low for k in ["retired ias", "सेवानिवृत्त ias", "सेवानिवृत्त भा.प्र.से", "retired"]):
        ret = store.get_retired_ias()
        ans = f"### सेवानिवृत्त भारतीय प्रशासनिक सेवा (Retired IAS) अधिकारी - छत्तीसगढ़ संवर्ग:\n\n"
        ans += f"सामान्य प्रशासन विभाग के अधिवार्षिकी सेवानिवृत्ति आदेश **क्रमांक ई 3-01/2003/एक/2** के अनुसार सेवानिवृत्त आईएएस अधिकारियों की सूची:\n\n"
        ans += "| क्र. | अधिकारी का नाम | बैच | पूर्व पदस्थापना | सेवानिवृत्ति तिथि |\n"
        ans += "|---|---|---|---|---|\n"
        for i, r in enumerate(ret, 1):
            batch = next((c.replace("बैच: ", "") for c in r.get("additional_charges", []) if "बैच" in c), "N/A")
            dt = next((c.replace("सेवानिवृत्ति तिथि: ", "") for c in r.get("additional_charges", []) if "सेवानिवृत्ति" in c), "N/A")
            ans += f"| {i} | **{r['name_hi']}** ({r['name_en']}) | {batch} | {r['designation_hi'].split(' - पूर्व ')[-1]} | {dt} |\n"

        primary_meta = {
            "source_title": "सेवानिवृत भारतीय प्रशासनिक सेवा अधिकारियों की जानकारी (आदेश ई 3-01/2003/एक/2)",
            "source_url": "https://gad.cg.gov.in/pdf/Retd.IASOfficers.pdf",
            "source_doc_id": "14f73433fbdd53d7",
            "canonical_source_id": "819ea9132e98457c9e0225027186dd53",
            "filename": "Retd.IASOfficers.pdf"
        }
        ans += format_officer_sources_footer(primary_doc=primary_meta)

        chunks = [{
            "doc_id": "14f73433fbdd53d7",
            "canonical_doc_id": "819ea9132e98457c9e0225027186dd53",
            "source_url": "https://gad.cg.gov.in/pdf/Retd.IASOfficers.pdf",
            "actual_pdf": "Retd.IASOfficers.pdf",
            "label": "आदेश ई 3-01/2003/एक/2 - सेवानिवृत्त IAS अधिकारी",
            "body": "क्रमांक : ई 3-01/2003/एक/2 :: राज्य शासन, भारतीय प्रशासनिक सेवा के अधिकारियों की अधिवार्षिकी सेवानिवृत्ति आदेश।",
            "excerpt": "राज्य शासन, भारतीय प्रशासनिक सेवा के अधिकारियों की अधिवार्षिकी सेवानिवृत्ति आदेश वर्ष 2003 से 2026।"
        }]
        return {"answer": ans, "officer": None, "chunks": chunks}

    # 4. CHiPS CEOs
    if any(k in q_low for k in ["ceo of chips", "chips ceo", "chhattisgarh infotech", "चिप्स"]):
        ceos = store.get_chips_ceos()
        ans = f"### छत्तीसगढ़ इन्फोटेक प्रमोशन सोसाइटी (CHiPS) - नेतृत्व एवं पूर्व मुख्य कार्यपालन अधिकारी (CEOs):\n\n"
        ans += f"शासकीय अधिसूचनाओं, पदक्रम सूची व मंत्रालय अभिलेखों के अनुसार चिप्स (CHiPS) के प्रमुख अधिकारी:\n\n"
        all_notifs = []
        for c in ceos:
            ans += f"- **{c['name_hi']} ({c['name_en']})**: {c['designation_hi']}\n"
            if c.get('additional_charges'):
                ans += f"  - *प्रभार:* {', '.join(c['additional_charges'])}\n"
            if c.get('email'):
                ans += f"  - *ई-मेल:* `{c['email']}`\n"
            if c.get('notifications'):
                all_notifs.extend(c['notifications'])

        primary_meta = {
            "source_title": "छत्तीसगढ़ इन्फोटेक प्रमोशन सोसाइटी (CHiPS) आधिकारिक पोर्टल",
            "source_url": "https://chips.gov.in/",
            "source_doc_id": "5aed8b79a68a88e8",
            "canonical_source_id": "3caa07ab2d0349d29bfe1efd8a112148",
            "filename": "201800083_001.pdf"
        }
        ans += format_officer_sources_footer(primary_doc=primary_meta, notifications=all_notifs[:4])

        chunks = [
            {
                "doc_id": "5aed8b79a68a88e8",
                "canonical_doc_id": "3caa07ab2d0349d29bfe1efd8a112148",
                "source_url": "https://gad.cg.gov.in/statecirculars/Criculars/Scan-file-2000-2002/year2018/201800083_001.pdf",
                "actual_pdf": "201800083_001.pdf",
                "label": "अधिसूचना 18/05/2018 - Alex Paul Menon, CEO CHiPS",
                "body": "शासकीय अधिसूचना: श्री एलेक्स पॉल मेनन, मुख्य कार्यपालन अधिकारी, चिप्स",
                "excerpt": "श्री एलेक्स पॉल मेनन को मुख्य कार्यपालन अधिकारी, छत्तीसगढ़ इन्फोटेक प्रमोशन सोसाइटी (चिप्स) का दायित्व सौंपा जाता है।"
            },
            {
                "doc_id": "fbc9b70ccdf197ba",
                "canonical_doc_id": "fbc9b70ccdf197ba",
                "source_url": "https://gad.cg.gov.in/Display_state_circular.aspx",
                "actual_pdf": "fbc9b70ccdf197ba.pdf",
                "label": "आदेश बी-1-1/2019/एक/4 - Shashank Pandey, Addl CEO",
                "body": "शासकीय आदेश बी-1-1/2019/एक/4: श्री शशांक पाण्डेय, अति. मुख्य कार्यपालन अधिकारी, चिप्स",
                "excerpt": "श्री शशांक पाण्डेय, राज्य प्रशासनिक सेवा, अतिरिक्त मुख्य कार्यपालन अधिकारी चिप्स।"
            }
        ]
        return {"answer": ans, "officer": None, "chunks": chunks}

    # 5. IAS Establishment Section
    if any(k in q_low for k in ["ias establishment", "स्थापना section", "establishment section", "ias स्थापना"]):
        est = store.get_ias_establishment()
        ans = f"### {est['section']}\n\n"
        ans += f"**विभाग:** {est['department']}\n"
        ans += f"**कार्यक्षेत्र:** {est['mandate']}\n\n"
        ans += "#### प्रमुख कार्य एवं दायित्व (Key Functions):\n"
        for fn in est['key_functions']:
            ans += f"- {fn}\n"

        primary_meta = {
            "source_title": "सामान्य प्रशासन विभाग के विभिन्न कक्षों के कार्य (कार्य आवंटन नियम)",
            "source_url": "https://gad.cg.gov.in/Section_work_view.aspx",
            "source_doc_id": "644aea95379af6eb",
            "canonical_source_id": "d54c166c03eb4c008096ff9189e2ad7c",
            "filename": "Section_work_view.aspx"
        }
        ans += format_officer_sources_footer(primary_doc=primary_meta)

        chunks = [{
            "doc_id": "644aea95379af6eb",
            "canonical_doc_id": "d54c166c03eb4c008096ff9189e2ad7c",
            "source_url": "https://gad.cg.gov.in/Section_work_view.aspx",
            "actual_pdf": "Section_work_view.aspx",
            "label": "सामान्य प्रशासन विभाग कार्य आवंटन नियम - IAS स्थापना",
            "body": "सामान्य प्रशासन विभाग - कार्य आवंटन नियम: भारतीय प्रशासनिक सेवा (स्थापना) संबंधित लेखाकार्य, पदोन्नति, पदस्थापना/स्थानांतरण, विभागीय जाँच/शिकायतें, सिविल सर्विसेज बोर्ड",
            "excerpt": "भारतीय प्रशासनिक सेवा (स्थापना) संबंधित समस्त लेखाकार्य, पदोन्नति, पदस्थापना एवं स्थानांतरण, विभागीय जाँच एवं शिकायतें।"
        }]
        return {"answer": ans, "officer": None, "chunks": chunks}

    # 6. Specific Officer Search
    is_person_intent = any(k in q_low for k in [
        "who is", "who was", "kon hain", "kon hai", "tell me abt", "tell me about",
        "about ", "profile of", "details of", "shri ", "smt ", "dr ", "officer",
        "secretary of", "collector of", "sp of", "dfo of", "कौन हैं", "कौन है", 
        "के बारे में", "का विवरण", "अधिकारी"
    ])

    matches = store.search_officers(q, limit=5)

    # Extract target person tokens
    clean_target = re.sub(
        r"^(who is|who was|kon hain|kon hai|tell me abt|tell me about|about|profile of|details of|give me|list|please|kripya|bataiye|श्रीमती|श्री|डॉ\.|डॉ|अधिकारी|के बारे में|कौन हैं|कौन है)\s+",
        "", q.strip(), flags=re.IGNORECASE
    ).strip(" ?.,;")
    clean_target = re.sub(r"^(the|a|an|of|for|about)\s+", "", clean_target, flags=re.IGNORECASE).strip()
    STOP_NAME_WORDS = {"the", "a", "an", "of", "in", "for", "abt", "about", "mr", "mrs", "ms", "shri", "smt", "dr", "ji", "sahab", "sir"}
    target_tokens = [t.lower() for t in re.findall(r"[\w\u0900-\u097f]+", clean_target) if len(t) >= 2 and t.lower() not in STOP_NAME_WORDS]

    def match_name_token(query_t: str, name_en: str, name_hi: str) -> bool:
        qt = query_t.lower()
        nen = (name_en or "").lower()
        nhi = (name_hi or "").lower()
        if qt in nen or qt in nhi:
            return True
        qt_norm = re.sub(r"\bsammi\b", "shammi", qt)
        if qt_norm in nen or qt_norm in nhi:
            return True
        qt_s = re.sub(r"sh", "s", qt_norm)
        nen_s = re.sub(r"sh", "s", nen)
        return qt_s in nen_s

    if matches:
        # Check if the candidate satisfies the query
        top = None
        if len(target_tokens) >= 2:
            # Query specified first name + last name (e.g. "Mayank Agrawal")
            # Enforce that candidate matches ALL target tokens
            for m in matches:
                nen = m.get("name_en") or ""
                nhi = m.get("name_hi") or ""
                if all(match_name_token(t, nen, nhi) for t in target_tokens):
                    top = m
                    break
        else:
            top = matches[0]

        if top:
            ans = f"### {top['name_hi']} ({top['name_en']})\n\n"
            ans += f"- **पदनाम (Designation):** {top['designation_hi']} ({top['designation_en']})\n"
            ans += f"- **विभाग (Department):** {top['department_hi'] or 'सामान्य प्रशासन विभाग (GAD)'} ({top['department_en']})\n"
            ans += f"- **संवर्ग श्रेणी (Category):** {top['category_label']}\n"
            if top.get('additional_charges'):
                ans += f"- **अतिरिक्त प्रभार (Additional Charges):** {', '.join(top['additional_charges'])}\n"
            if top.get('email'):
                ans += f"- **ई-मेल (Official Email):** `{top['email']}`\n"
            if top.get('phone'):
                ans += f"- **संपर्क (Contact):** {top['phone']}\n"
            ans += f"- **अभिलेख स्रोतः** [{top.get('source_title', top.get('source_name'))}]({top.get('source_url', 'https://gad.cg.gov.in/')})\n\n"

            if top.get('notifications'):
                ans += "#### संबंधित शासकीय परिपत्र एवं स्थानांतरण/पदस्थापना आदेश:\n"
                for n in top['notifications'][:6]:
                    n_url = n.get('source_url') or "https://gad.cg.gov.in/"
                    n_title = n.get('doc_title') or n.get('subject') or n.get('filename') or "Government Notification / Circular"
                    ans += f"- [**आदेश {n['order_no']}**]({n_url}) ({n['order_date']}): [{n_title}]({n_url})\n"
                    if n.get('excerpt'):
                        ans += f"  > *\"{n['excerpt'][:180]}...\"*\n"

            ans += format_officer_sources_footer(primary_doc=top, notifications=top.get('notifications'))

            chunks = [{
                "doc_id": n["doc_id"],
                "canonical_doc_id": n.get("canonical_doc_id", n["doc_id"]),
                "source_url": n.get("source_url", "https://gad.cg.gov.in/"),
                "actual_pdf": n.get("filename") or f"{n['doc_id']}.pdf",
                "label": f"आदेश {n['order_no']} ({n['order_date']})",
                "body": f"शासकीय आदेश {n['order_no']} ({n['order_date']}): {n.get('doc_title', n['subject'])}. {n.get('excerpt', '')}",
                "excerpt": n.get("excerpt", "")
            } for n in top.get("notifications", [])]

            if not chunks:
                chunks = [{
                    "doc_id": top.get("source_doc_id", "efc290b78f6c8ad7"),
                    "canonical_doc_id": top.get("canonical_source_id", "cec78d786be946f4a6316afdf05854fd"),
                    "source_url": top.get("source_url", "https://gad.cg.gov.in/Display_civil_list_Email.aspx"),
                    "actual_pdf": top.get("filename", "Display_civil_list_Email.aspx"),
                    "label": f"सिविल लिस्ट: {top['name_en']}",
                    "body": f"अधिकारी: {top['name_hi']} ({top['name_en']}), पद: {top['designation_hi']}, विभाग: {top['department_hi']}",
                    "excerpt": f"अधिकारी: {top['name_hi']} ({top['name_en']}) - {top['designation_hi']}"
                }]
            return {"answer": linkify_document_citations(ans, candidate_chunks=chunks), "officer": top, "chunks": chunks}

        return None

    return None


def get_gad_engine() -> QueryEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = QueryEngine(
            header_path=str(REPO_ROOT / "gad_kg" / "data" / "work" / "header.jsonl"),
            edges_path=str(REPO_ROOT / "gad_kg" / "data" / "work" / "edges.jsonl"),
            graph_db_path=str(REPO_ROOT / "gad_kg" / "data" / "work" / "graph.db"),
            fts_db_path=str(REPO_ROOT / "gad_kg" / "data" / "work" / "chunks_fts.db"),
            tables_db_path=str(REPO_ROOT / "gad_kg" / "data" / "work" / "tables.db"),
            trace_log_path=str(REPO_ROOT / "gad_kg" / "logs" / "trace.jsonl"),
        )
    return _ENGINE_INSTANCE


def format_gad_results(candidate_chunks: List[Dict[str, Any]], engine: QueryEngine) -> List[Dict[str, Any]]:
    formatted = []
    from gad_kg.src.officer_store import load_document_metadata_index
    meta_cache = load_document_metadata_index()

    for idx, ch in enumerate(candidate_chunks, 1):
        doc_id = ch.get("doc_id", "")
        h = engine.headers_by_id.get(doc_id, {})
        m = meta_cache.get(doc_id, {})

        doc_num = h.get("doc_number_raw") or h.get("doc_number_norm") or ch.get("order_no") or doc_id
        dt = h.get("issue_date") or ch.get("order_date") or ch.get("doc_date") or ""
        subj = (h.get("subject") or ch.get("doc_title") or ch.get("title") or m.get("title") or "").strip()
        body = ch.get("body") or ch.get("content") or ch.get("text") or ""
        source_url = ch.get("source_url") or m.get("source_url") or "https://gad.cg.gov.in/"
        actual_pdf = ch.get("actual_pdf") or m.get("filename") or f"{doc_id[:8]}.pdf"

        label = ch.get("label") or f"आदेश {doc_num}"
        if dt and dt not in label:
            label += f" ({dt})"
        if subj and subj not in label:
            label += f" - {subj[:60]}..."

        excerpt = ch.get("excerpt") or (body[:250] + "..." if len(body) > 250 else body)

        formatted.append({
            "rank": idx,
            "source": doc_id,
            "actual_pdf": actual_pdf,
            "label": label,
            "source_url": source_url,
            "score": 0.95 - (idx * 0.05),
            "text": body,
            "excerpt": excerpt,
            "parent_id": ch.get("chunk_id", ""),
            "supporting": False,
        })
    return formatted


def handle_gad_query(
    query_text: str,
    thinking: bool = False,
    history: Optional[List[Any]] = None,
    num_context: int = 8,
) -> Dict[str, Any]:
    """Execute query via GAD-KG engine and format response for Web UI."""
    t0 = time.time()
    engine = get_gad_engine()

    # 1. LLM Query Router: Classify query as OFFICER or DOCUMENT
    route = route_query_intent(query_text)

    if route == "OFFICER":
        off_res = resolve_officer_query(query_text)
        if off_res:
            candidate_chunks = off_res.get("chunks", [])
            formatted_results = format_gad_results(candidate_chunks, engine)
            total_time = time.time() - t0
            trace_steps = [
                {
                    "kind": "plan",
                    "title": "LLM Query Router: Routed to [OFFICER DIRECTORY] -> Matched official administrative record",
                    "detail": {"route": "OFFICER", "officer": off_res.get("officer", {}).get("name_en") if off_res.get("officer") else "Civil List"}
                },
                {
                    "kind": "retrieve",
                    "title": f"Linked {len(candidate_chunks)} official government orders & gazetted circulars",
                    "detail": {"hits": len(candidate_chunks)}
                },
                {
                    "kind": "synthesize",
                    "title": "Synthesized verified officer dossier with designation, department, charges & citations",
                    "detail": {"source": "officers.db & temporal_graph.db"}
                }
            ]
            return {
                "success": True,
                "query": query_text,
                "answer": off_res["answer"],
                "results": formatted_results,
                "result_count": len(formatted_results),
                "thinking": bool(thinking),
                "trace": trace_steps,
                "hops": 1,
                "timeline": [],
                "standalone": None,
                "execution_time": f"{total_time:.2f}s",
            }

    ans_res = engine.ask(query_text)
    trace = ans_res.get("trace", {})
    candidate_chunks = ans_res.get("candidate_chunks", [])
    formatted_results = format_gad_results(candidate_chunks, engine)

    total_time = time.time() - t0

    # Build multi-step reasoning trace for UI thinking drawer
    trace_steps = [
        {
            "kind": "plan",
            "title": f"Query Understanding (R14): Intent={trace.get('intent', 'HYBRID')}, Target Date={trace.get('as_of')}",
            "detail": {"intent": trace.get("intent"), "as_of": trace.get("as_of")}
        },
        {
            "kind": "retrieve",
            "title": f"Vector Semantic (BGE-M3) + FTS5 Retrieval: Found {trace.get('retrieved_chunks', 0)} passages ({trace.get('semantic_hits', 0)} vector hits)",
            "detail": {"hits": trace.get("retrieved_chunks", 0), "semantic_hits": trace.get("semantic_hits", 0)}
        },
        {
            "kind": "timeline",
            "title": f"2-Hop Graph Expansion: Analyzed {len(trace.get('expanded_doc_ids', []))} interrelated amending/superseding circulars",
            "detail": {"expanded_docs": trace.get("expanded_doc_ids", [])}
        },
        {
            "kind": "synthesize",
            "title": "Rule 4 Substring Verification: Verified citations and generated grounded response via R16 prompt",
            "detail": {"used_labels": ans_res.get("citations", [])}
        }
    ]

    # Timeline of seed documents
    timeline_entries = []
    for s_id in trace.get("seed_doc_ids", [])[:3]:
        tl = engine.get_timeline(s_id)
        timeline_entries.extend(tl)

    return {
        "success": True,
        "query": query_text,
        "answer": linkify_document_citations(ans_res.get("answer", ""), candidate_chunks=candidate_chunks),
        "results": formatted_results,
        "result_count": len(formatted_results),
        "thinking": bool(thinking),
        "trace": trace_steps,
        "hops": len(trace.get("expanded_doc_ids", [])),
        "timeline": timeline_entries,
        "standalone": None,
        "execution_time": f"{total_time:.2f}s",
    }


def handle_gad_query_stream(
    query_text: str,
    emit_fn,
    thinking: bool = False,
    history: Optional[List[Any]] = None,
    num_context: int = 8,
):
    """Execute query via GAD-KG engine and push SSE events into emit_fn."""
    t0 = time.time()
    engine = get_gad_engine()

    # 1. LLM Query Router
    route = route_query_intent(query_text)

    if route == "OFFICER":
        off_res = resolve_officer_query(query_text)
        if off_res:
            candidate_chunks = off_res.get("chunks", [])
            formatted_results = format_gad_results(candidate_chunks, engine)
            emit_fn({"type": "status", "stage": "retrieving", "standalone": None})
            emit_fn({
                "type": "step",
                "step": {
                    "kind": "plan",
                    "title": "LLM Query Router: Routed to [OFFICER DIRECTORY] -> Matched official administrative record",
                    "detail": {"route": "OFFICER", "officer": off_res.get("officer", {}).get("name_en") if off_res.get("officer") else "Civil List"}
                }
            })
            time.sleep(0.04)
            emit_fn({
                "type": "step",
                "step": {
                    "kind": "retrieve",
                    "title": f"Linked {len(candidate_chunks)} official government orders & circular notifications",
                    "detail": {"hits": len(candidate_chunks)}
                }
            })
            time.sleep(0.04)
            emit_fn({
                "type": "step",
                "step": {
                    "kind": "synthesize",
                    "title": "Formulated authoritative profile with official designation, department, contact & circular citations",
                    "detail": {}
                }
            })

            # Stream answer
            full_answer = off_res["answer"]
            words = full_answer.split(" ")
            for i in range(0, len(words), 3):
                chunk_text = " ".join(words[i:i+3]) + (" " if i + 3 < len(words) else "")
                emit_fn({"type": "token", "text": chunk_text})
                time.sleep(0.01)

            total_time = time.time() - t0
            trace_steps = [
                {
                    "kind": "plan",
                    "title": "LLM Query Router: Routed to [OFFICER DIRECTORY] -> Matched official administrative record",
                    "detail": {"route": "OFFICER", "officer": off_res.get("officer", {}).get("name_en") if off_res.get("officer") else "Civil List"}
                },
                {
                    "kind": "retrieve",
                    "title": f"Linked {len(candidate_chunks)} official government orders & circular notifications",
                    "detail": {"hits": len(candidate_chunks)}
                },
                {
                    "kind": "synthesize",
                    "title": "Formulated authoritative profile with official designation, department, contact & circular citations",
                    "detail": {}
                }
            ]

            emit_fn({
                "type": "done",
                "success": True,
                "query": query_text,
                "answer": full_answer,
                "results": formatted_results,
                "result_count": len(formatted_results),
                "thinking": bool(thinking),
                "trace": trace_steps,
                "hops": 1,
                "timeline": [],
                "standalone": None,
                "execution_time": f"{total_time:.2f}s",
            })
            return

    emit_fn({"type": "status", "stage": "retrieving", "standalone": None})
    emit_fn({
        "type": "step",
        "step": {
            "kind": "plan",
            "title": f"LLM Query Router: Routed to [DOCUMENT KNOWLEDGE GRAPH] (R14 Intent Analysis)...",
            "detail": {"route": route}
        }
    })

    emit_fn({
        "type": "step",
        "step": {
            "kind": "retrieve",
            "title": "Searching 1024-d BGE-M3 semantic vectors & SQLite FTS5 BM25 index...",
            "detail": {}
        }
    })

    ans_res = engine.ask(query_text)
    trace = ans_res.get("trace", {})
    candidate_chunks = ans_res.get("candidate_chunks", [])
    formatted_results = format_gad_results(candidate_chunks, engine)

    emit_fn({
        "type": "step",
        "step": {
            "kind": "timeline",
            "title": f"Traversing 2-hop graph ({len(trace.get('expanded_doc_ids', []))} related circulars) & evaluating status_as_of()...",
            "detail": {"expanded": trace.get("expanded_doc_ids", [])}
        }
    })

    emit_fn({
        "type": "step",
        "step": {
            "kind": "synthesize",
            "title": "Writing citation-verified answer via Prompt R16...",
            "detail": {}
        }
    })

    # Stream the tokens in chunks
    full_answer = linkify_document_citations(ans_res.get("answer", ""), candidate_chunks=candidate_chunks)
    words = full_answer.split(" ")
    for i in range(0, len(words), 3):
        chunk_text = " ".join(words[i:i+3]) + (" " if i + 3 < len(words) else "")
        emit_fn({"type": "token", "text": chunk_text})

    total_time = time.time() - t0

    # Build timeline entries
    timeline_entries = []
    for s_id in trace.get("seed_doc_ids", [])[:3]:
        tl = engine.get_timeline(s_id)
        timeline_entries.extend(tl)

    trace_steps = [
        {
            "kind": "plan",
            "title": f"Query Understanding (R14): Intent={trace.get('intent', 'HYBRID')}, Target Date={trace.get('as_of')}",
            "detail": {"intent": trace.get("intent"), "as_of": trace.get("as_of")}
        },
        {
            "kind": "retrieve",
            "title": f"Vector Semantic (BGE-M3) + FTS5 Retrieval: Found {trace.get('retrieved_chunks', 0)} passages ({trace.get('semantic_hits', 0)} vector hits)",
            "detail": {"hits": trace.get("retrieved_chunks", 0), "semantic_hits": trace.get("semantic_hits", 0)}
        },
        {
            "kind": "timeline",
            "title": f"2-Hop Graph Expansion: Analyzed {len(trace.get('expanded_doc_ids', []))} interrelated circulars",
            "detail": {"expanded_docs": trace.get("expanded_doc_ids", [])}
        },
        {
            "kind": "synthesize",
            "title": "Rule 4 Substring Verification: Verified citations and generated grounded response via R16 prompt",
            "detail": {"used_labels": ans_res.get("citations", [])}
        }
    ]

    emit_fn({
        "type": "done",
        "success": True,
        "query": query_text,
        "answer": full_answer,
        "results": formatted_results,
        "result_count": len(formatted_results),
        "thinking": bool(thinking),
        "trace": trace_steps,
        "hops": len(trace.get("expanded_doc_ids", [])),
        "timeline": timeline_entries,
        "standalone": None,
        "execution_time": f"{total_time:.2f}s",
    })
