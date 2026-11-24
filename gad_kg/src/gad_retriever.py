"""
High-Precision Multi-Tier Retrieval Engine for Chhattisgarh GAD Documents.
Searches the production database (data/fg2/temporal_graph.db) across 5,305 documents
and 94,493 FTS chunks using a 4-tier retrieval strategy:
  Tier 1: Exact Order Code / Notification Number match
  Tier 2: Direct GAD Category match (15 primary administrative domains)
  Tier 3: Document Node Title & Category FTS5 match (fts_nodes)
  Tier 4: BM25 Ranked Semantic/Lexical Keyword Match with stop-word filtering
All retrieved candidates are enriched with official government portal URLs, actual PDF
filenames, clean subject titles, and temporal status.
"""

import os
import re
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DB_PATH = REPO_ROOT / "data" / "fg2" / "temporal_graph.db"
URL_INDEX_PATH = REPO_ROOT / "gad_kg" / "data" / "work" / "doc_url_index.json"

_DOC_URL_INDEX: Optional[Dict[str, Any]] = None


def get_doc_url_index() -> Dict[str, Any]:
    global _DOC_URL_INDEX
    if _DOC_URL_INDEX is None:
        if URL_INDEX_PATH.exists():
            try:
                with open(URL_INDEX_PATH, "r", encoding="utf-8") as f:
                    _DOC_URL_INDEX = json.load(f)
            except Exception:
                _DOC_URL_INDEX = {}
        else:
            _DOC_URL_INDEX = {}
    return _DOC_URL_INDEX


STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren',
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'can', 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself',
    'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most',
    'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our',
    'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than',
    'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this',
    'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what',
    'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'tell', 'give', 'find', 'get', 'know', 'please', 'look', 'show',
    'information', 'detail', 'details', 'latest', 'recent',
    'का', 'के', 'की', 'में', 'से', 'को', 'है', 'हैं', 'पर', 'और', 'या', 'क्या', 'कैसे', 'कहाँ',
    'कौन', 'किस', 'किसने', 'किसको', 'मुझे', 'बताइए', 'बताओ', 'दिखाइए', 'दीजिए', 'बारे', 'लिए',
    'था', 'थे', 'थी', 'हो', 'होता', 'होती', 'होते', 'गया', 'गए', 'गई', 'कर', 'करता', 'करते', 'करती',
    'ने', 'तक', 'भी', 'तो', 'ही', 'व', 'एवं', 'तथा', 'अथवा', 'सकता', 'सकते', 'सकती', 'कृपया',
    'जानकारी', 'विवरण', 'नवीनतम', 'वर्तमान', 'अधिसूचनाएं'
}

SYNONYM_MAP = {
    'leave': ['अवकाश', 'leave'],
    'अवकाश': ['leave', 'अवकाश'],
    'policy': ['नीति', 'नियम', 'policy'],
    'नीति': ['policy', 'नीति'],
    'transfer': ['स्थानांतरण', 'पदस्थापना', 'transfer'],
    'स्थानांतरण': ['transfer', 'स्थानांतरण', 'पदस्थापना'],
    'पदस्थापना': ['स्थानांतरण', 'transfer', 'पदस्थापना'],
    'recruitment': ['भर्ती', 'recruitment', 'रिक्ति'],
    'vacancy': ['रिक्ति', 'भर्ती', 'vacancy'],
    'vacancies': ['रिक्ति', 'भर्ती', 'vacancy'],
    'job': ['भर्ती', 'नौकरी', 'पद', 'job'],
    'notices': ['परिपत्र', 'सूचना', 'notices', 'circular'],
    'भर्ती': ['recruitment', 'vacancy', 'भर्ती'],
    'रिक्ति': ['vacancy', 'रिक्ति', 'भर्ती'],
    'compassionate': ['अनुकंपा', 'compassionate'],
    'appointment': ['नियुक्ति', 'appointment'],
    'अनुकंपा': ['compassionate', 'अनुकंपा'],
    'नियुक्ति': ['appointment', 'नियुक्ति'],
    'pension': ['पेंशन', 'सेवानिवृत्ति', 'pension'],
    'पेंशन': ['pension', 'पेंशन'],
    'da': ['महंगाई भत्ता', 'dearness allowance', 'da'],
    'महंगाई': ['महंगाई भत्ता', 'da', 'dearness allowance'],
    'salary': ['वेतन', 'वेतनमान', 'salary'],
    'वेतन': ['salary', 'वेतन', 'वेतनमान'],
    'deputation': ['प्रतिनियुक्ति', 'deputation'],
    'प्रतिनियुक्ति': ['deputation', 'प्रतिनियुक्ति'],
    'order': ['आदेश', 'order'],
    'आदेश': ['order', 'आदेश'],
    'letter': ['पत्र', 'परिपत्र', 'letter'],
    'circular': ['परिपत्र', 'circular'],
    'परिपत्र': ['circular', 'परिपत्र'],
    'rules': ['नियम', 'rules'],
    'नियम': ['rules', 'नियम'],
    'promotion': ['पदोन्नति', 'promotion'],
    'पदोन्नति': ['promotion', 'पदोन्नति'],
    'seniority': ['वरिष्ठता', 'seniority'],
    'वरिष्ठता': ['seniority', 'वरिष्ठता'],
    'reservation': ['आरक्षण', 'reservation'],
    'आरक्षण': ['reservation', 'आरक्षण'],
}

CATEGORY_MAP = {
    "अवकाश व कार्यदिवस (Leave & Holidays)": ["leave", "holiday", "holidays", "अवकाश", "छुट्टी", "maternity", "casual leave", "earned leave", "child care"],
    "स्थानांतरण व पदस्थापना (Transfer & Postings)": ["transfer", "posting", "postings", "स्थानांतरण", "पदस्थापना", "तबादला"],
    "अनुकंपा नियुक्ति (Compassionate Appointment)": ["compassionate", "अनुकंपा", "आश्रित"],
    "भर्ती व सेवा नियम (Recruitment & Rules)": ["recruitment", "vacancy", "vacancies", "job", "jobs", "भर्ती", "रिक्ति", "सेवा नियम"],
    "वेतन, पेंशन व भत्ते (Pay, Pension & Allowances)": ["pension", "salary", "pay", "da", "dearness", "allowance", "भत्ता", "पेंशन", "वेतन", "महंगाई"],
    "पदोन्नति व वरिष्ठता (Promotion & Seniority)": ["promotion", "seniority", "dpc", "पदोन्नति", "वरिष्ठता"],
    "आरक्षण रोस्टर व विशेष वर्ग (Reservation & Roster)": ["reservation", "roster", "आरक्षण", "रोस्टर"],
    "सूचना का अधिकार व जन शिकायत (RTI & Grievances)": ["rti", "right to information", "grievance", "complaint", "सूचना का अधिकार", "शिकायत", "लोक सूचना अधिकारी", "pio"]
}

EXCLUDED_ENTITIES = [
    "visakhapatnam port", "mormugao", "kolkata port", "smpk", "port trust", "syama prasad mookerjee port"
]


def extract_search_terms(query: str) -> Tuple[List[str], List[str], List[str]]:
    """Extract cleaned core tokens, expanded synonym tokens, and government order codes."""
    clean = re.sub(r'[^\w\s\u0900-\u097F]', ' ', query)
    tokens = [t.strip() for t in clean.split() if len(t.strip()) >= 2]
    order_matches = re.findall(r'[A-Za-z0-9\u0900-\u097F]+(?:[-/][A-Za-z0-9\u0900-\u097F]+)+|\b\d{3,6}\b', query)

    core_tokens = [t for t in tokens if t.lower() not in STOP_WORDS]
    if not core_tokens:
        core_tokens = tokens

    expanded = set(core_tokens)
    for t in list(core_tokens):
        low = t.lower()
        if low in SYNONYM_MAP:
            expanded.update(SYNONYM_MAP[low])

    return core_tokens, list(expanded), order_matches


def search_gad_database(
    query: str,
    limit: int = 10,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """Execute high-precision multi-tier retrieval against the production temporal_graph database."""
    if not db_path.exists():
        alt_path = Path("data/fg2/temporal_graph.db")
        if alt_path.exists():
            db_path = alt_path

    core_tokens, expanded_tokens, order_matches = extract_search_terms(query)
    doc_url_index = get_doc_url_index()

    # Detect matching category from governance taxonomy
    matched_categories = []
    q_low = query.lower()
    for cat_name, kw_list in CATEGORY_MAP.items():
        if any(kw in q_low for kw in kw_list):
            matched_categories.append(cat_name)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    candidate_map: Dict[str, Dict[str, Any]] = {}

    # Tier 1: Order Code Match
    if order_matches:
        for code in order_matches:
            if len(code) < 3:
                continue
            cur.execute("""
                SELECT chunk_id, doc_id, order_no, title, doc_date, status, content
                FROM chunks
                WHERE order_no LIKE ? OR content LIKE ?
                ORDER BY chunk_index ASC LIMIT 3
            """, (f"%{code}%", f"%{code}%"))
            for r in cur.fetchall():
                cid = r[0]
                content = r[6] or ""
                if any(ex in content.lower() for ex in EXCLUDED_ENTITIES):
                    continue
                if cid not in candidate_map:
                    candidate_map[cid] = {
                        "chunk_id": cid, "doc_id": r[1], "order_no": r[2] or code,
                        "title": r[3] or "", "doc_date": r[4] or "",
                        "status": r[5] or "current", "content": content,
                        "category": "", "source_url": "https://gad.cg.gov.in/",
                        "score": 30.0, "source_tier": "order_exact"
                    }

    # Tier 2: Category Match (Direct GAD Governance Nodes)
    if matched_categories:
        for cat in matched_categories:
            cur.execute("""
                SELECT id, doc_id, order_no, doc_date, category, title_hi, title_en, source_url
                FROM nodes
                WHERE category = ?
                ORDER BY doc_date DESC LIMIT 5
            """, (cat,))
            for n in cur.fetchall():
                d_id = n[1]
                meta = doc_url_index.get(d_id) or doc_url_index.get(f"doc_{d_id}") or {}
                doc_dir = meta.get("doc_dir") or d_id[:16]
                cur.execute("""
                    SELECT chunk_id, doc_id, order_no, title, doc_date, status, content
                    FROM chunks WHERE doc_id = ? OR doc_id = ? LIMIT 2
                """, (d_id, doc_dir))
                for cr in cur.fetchall():
                    cid = cr[0]
                    content = cr[6] or ""
                    if any(ex in content.lower() for ex in EXCLUDED_ENTITIES):
                        continue
                    if cid not in candidate_map:
                        candidate_map[cid] = {
                            "chunk_id": cid, "doc_id": d_id, "order_no": cr[2] or n[2] or "",
                            "title": cr[3] or n[5] or n[6] or "", "doc_date": cr[4] or n[3] or "",
                            "status": cr[5] or "current", "content": content,
                            "category": cat, "source_url": n[7] or meta.get("source_url") or "https://gad.cg.gov.in/",
                            "score": 25.0, "source_tier": "category_exact"
                        }

    # Tier 3: Search fts_nodes with expanded tokens
    if expanded_tokens and len(candidate_map) < limit:
        node_expr = " OR ".join([f'"{t}"*' for t in list(expanded_tokens)[:8]])
        try:
            cur.execute("""
                SELECT n.id, n.doc_id, n.order_no, n.doc_date, n.category, n.title_hi, n.title_en, n.source_url,
                       bm25(fts_nodes) as score
                FROM fts_nodes f JOIN nodes n ON f.id = n.id
                WHERE fts_nodes MATCH ? ORDER BY score LIMIT 5
            """, (node_expr,))
            for n in cur.fetchall():
                d_id = n[1]
                meta = doc_url_index.get(d_id) or doc_url_index.get(f"doc_{d_id}") or {}
                doc_dir = meta.get("doc_dir") or d_id[:16]
                cur.execute("SELECT chunk_id, doc_id, order_no, title, doc_date, status, content FROM chunks WHERE doc_id = ? OR doc_id = ? LIMIT 2", (d_id, doc_dir))
                for cr in cur.fetchall():
                    cid = cr[0]
                    content = cr[6] or ""
                    if any(ex in content.lower() for ex in EXCLUDED_ENTITIES):
                        continue
                    if cid not in candidate_map:
                        candidate_map[cid] = {
                            "chunk_id": cid, "doc_id": d_id, "order_no": cr[2] or n[2] or "",
                            "title": cr[3] or n[5] or n[6] or "", "doc_date": cr[4] or n[3] or "",
                            "status": cr[5] or "current", "content": content,
                            "category": n[4] or "", "source_url": n[7] or meta.get("source_url") or "https://gad.cg.gov.in/",
                            "score": 20.0 - (float(n[8]) if n[8] else 0), "source_tier": "node_match"
                        }
        except Exception:
            pass

    # Tier 4: FTS Chunks search
    if len(candidate_map) < limit and expanded_tokens:
        hi_tokens = [t for t in expanded_tokens if any('\u0900' <= ch <= '\u097F' for ch in t)]
        query_terms = hi_tokens if hi_tokens else core_tokens
        fts_expr = " AND ".join([f'"{t}"*' for t in query_terms[:2]]) if len(query_terms) >= 2 else f'"{query_terms[0]}"*'
        try:
            cur.execute("""
                SELECT c.chunk_id, c.doc_id, c.order_no, c.title, c.doc_date, c.status, c.content,
                       bm25(fts_chunks) as score, n.category, n.source_url
                FROM fts_chunks f
                JOIN chunks c ON f.chunk_id = c.chunk_id
                LEFT JOIN nodes n ON c.doc_id = n.doc_id
                WHERE fts_chunks MATCH ? ORDER BY score LIMIT 6
            """, (fts_expr,))
            for r in cur.fetchall():
                cid = r[0]
                content = r[6] or ""
                if any(ex in content.lower() for ex in EXCLUDED_ENTITIES):
                    continue
                if cid not in candidate_map:
                    candidate_map[cid] = {
                        "chunk_id": cid, "doc_id": r[1], "order_no": r[2] or "",
                        "title": r[3] or "", "doc_date": r[4] or "",
                        "status": r[5] or "current", "content": content,
                        "category": r[8] or "", "source_url": r[9] or "https://gad.cg.gov.in/",
                        "score": 15.0 - (float(r[7]) if r[7] else 0), "source_tier": "chunk_match"
                    }
        except Exception:
            pass

    conn.close()

    # Enrich and Normalize
    enriched = []
    for c in candidate_map.values():
        d_id = c["doc_id"]
        meta = doc_url_index.get(d_id) or doc_url_index.get(f"doc_{d_id}") or {}
        c["body"] = c["content"]
        c["text"] = c["content"]
        c["source_url"] = meta.get("source_url") or c.get("source_url") or "https://gad.cg.gov.in/"
        c["actual_pdf"] = meta.get("original_name") or meta.get("filename") or f"{d_id[:8]}.pdf"
        c["doc_title"] = meta.get("title") or c.get("title") or c["actual_pdf"]
        clean_content = c["content"].replace("\n", " ").strip()
        c["excerpt"] = clean_content[:250] + "..." if len(clean_content) > 250 else clean_content
        c["label"] = f"आदेश {c['order_no']}" if c.get('order_no') else c['doc_title']
        if c.get('doc_date'):
            c["label"] += f" ({c['doc_date']})"
        enriched.append(c)

    enriched.sort(key=lambda x: x["score"], reverse=True)
    return enriched[:limit]
