"""
Multi-Arm Hybrid Retriever for Chhattisgarh GAD Production Corpus
=================================================================
Queries `data/fg2/temporal_graph.db` across:
  1. Direct Order / Reference / Document-Dir Matching
  2. Bilingual FTS5 Search over Enriched `nodes` (`fts_nodes` + `doc_dir` bridge)
  3. Multi-Query Bilingual FTS5 Search over `chunks` (`fts_chunks`)
  4. Intra-Document Substantive Clause Selection (skipping Krutidev TOCs / cover filler)
  5. Policy Sub-Topic & Rulebook Affinity + Anti-Noise Demotion + Twin Deduplication + RRF
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "fg2"
DB_PATH = DATA_DIR / "temporal_graph.db"
DOC_URL_INDEX_PATH = PROJECT_ROOT / "gad_kg" / "data" / "work" / "doc_url_index.json"

_DOC_URL_INDEX: Optional[Dict[str, Any]] = None
_DUPLICATE_MAP_CACHE: Optional[Dict[str, str]] = None


def _load_doc_url_index() -> Dict[str, Any]:
    global _DOC_URL_INDEX
    if _DOC_URL_INDEX is None:
        if DOC_URL_INDEX_PATH.exists():
            try:
                _DOC_URL_INDEX = json.loads(DOC_URL_INDEX_PATH.read_text(encoding="utf-8"))
            except Exception:
                _DOC_URL_INDEX = {}
        else:
            _DOC_URL_INDEX = {}
    return _DOC_URL_INDEX


def _load_duplicate_map(cur: sqlite3.Cursor) -> Dict[str, str]:
    """Map duplicate node/doc_dir -> canonical node/doc_dir using DUPLICATE_OF edges."""
    global _DUPLICATE_MAP_CACHE
    if _DUPLICATE_MAP_CACHE is not None:
        return _DUPLICATE_MAP_CACHE
    mapping: Dict[str, str] = {}
    try:
        rows = cur.execute("""
            SELECT n1.doc_dir, n2.doc_dir
            FROM edges e
            JOIN nodes n1 ON e.source = n1.id
            JOIN nodes n2 ON e.target = n2.id
            WHERE e.relation = 'DUPLICATE_OF' AND n1.doc_dir != '' AND n2.doc_dir != ''
        """).fetchall()
        for dup_dir, canon_dir in rows:
            mapping[dup_dir] = canon_dir
    except Exception:
        pass
    _DUPLICATE_MAP_CACHE = mapping
    return mapping


STOP_WORDS = {
    "what", "is", "are", "was", "were", "the", "this", "that", "which", "who", "whom",
    "whose", "where", "when", "why", "how", "can", "could", "would", "should", "shall",
    "may", "might", "will", "do", "does", "did", "done", "tell", "me", "about", "for",
    "with", "from", "into", "unto", "by", "on", "at", "to", "in", "of", "and", "or",
    "not", "no", "yes", "all", "any", "some", "many", "much", "give", "show", "find",
    "get", "list", "provide", "explain", "describe", "please", "latest", "current",
    "new", "old", "order", "letter", "circular", "notification", "document", "gad",
    "government", "chhattisgarh", "state", "rules", "rule", "policy", "under", "act",
    "details", "information", "regarding", "related", "many", "days", "entitled", "allowed",
    "read", "see", "apply", "submit", "deadline", "process", "between", "after", "before",
    "क्या", "है", "हैं", "था", "थे", "थी", "यह", "वह", "इस", "उस", "कब", "कैसे", "कहाँ",
    "कौन", "किसे", "किसको", "कितने", "कितनी", "के", "की", "का", "को", "में", "से", "पर",
    "और", "या", "तथा", "एवं", "बताएं", "बताइए", "दीजिए", "जानकारी", "संबंधित", "नवीनतम",
    "आदेश", "परिपत्र", "अधिसूचना", "नियम", "शासन", "छत्तीसगढ़", "विभाग", "सामान्य", "प्रशासन",
}

SYNONYM_MAP: Dict[str, List[str]] = {
    "leave": ["अवकाश", "छुट्टी", "अवकाश नियम"],
    "casual": ["आकस्मिक", "आकस्मिक अवकाश"],
    "earned": ["अर्जित", "अर्जित अवकाश"],
    "medical": ["चिकित्सा", "चिकित्सा प्रमाणपत्र", "लघुकृत", "अर्धवेतन"],
    "commuted": ["लघुकृत", "अर्धवेतन"],
    "half": ["अर्धवेतन", "अर्धवेतन अवकाश"],
    "maternity": ["प्रसूति", "मातृत्व", "प्रसूति अवकाश"],
    "paternity": ["पितृत्व", "पितृत्व अवकाश"],
    "adoption": ["दत्तक", "दत्तक ग्रहण"],
    "study": ["अध्ययन", "अध्ययन अवकाश"],
    "extraordinary": ["असाधारण", "असाधारण अवकाश"],
    "child": ["संतान", "बाल्य", "प्रसूति", "मातृत्व"],
    "care": ["पालन", "देखभाल", "प्रसूति"],
    "holiday": ["अवकाश", "सार्वजनिक", "सामान्य", "ऐच्छिक"],
    "holidays": ["अवकाश", "सार्वजनिक", "ऐच्छिक"],
    "contract": ["संविदा", "संविदा नियुक्ति"],
    "samvida": ["संविदा", "संविदा नियुक्ति"],
    "transfer": ["स्थानांतरण", "तबादला", "स्थानांतरण नीति"],
    "transfers": ["स्थानांतरण", "तबादला"],
    "posting": ["पदस्थापना", "स्थानांतरण"],
    "window": ["अवधि", "तिथि", "आवेदन", "जून", "स्थानांतरण नीति"],
    "husband": ["पति", "पत्नी", "एक ही स्थान"],
    "wife": ["पत्नी", "पति", "एक ही स्थान"],
    "spouse": ["पति", "पत्नी", "एक ही स्थान"],
    "joining": ["कार्यभार ग्रहण", "पदग्रहण", "पदग्रहण काल", "कार्यमुक्त"],
    "compassionate": ["अनुकंपा", "अनुकम्पा", "अनुकंपा नियुक्ति"],
    "anukampa": ["अनुकंपा", "अनुकम्पा"],
    "dependents": ["आश्रित", "परिवार", "अनुकंपा"],
    "retirement": ["सेवानिवृत्ति", "अधिवार्षिकी", "अनिवार्य सेवानिवृत्ति", "पुनर्नियुक्ति"],
    "superannuation": ["अधिवार्षिकी", "सेवानिवृत्ति"],
    "extension": ["पुनर्नियुक्ति", "सेवावृद्धि", "कार्यविस्तार", "अधिवार्षिकी"],
    "dearness": ["महंगाई", "महंगाई भत्ता"],
    "allowance": ["भत्ता", "महंगाई भत्ता", "निर्वाह भत्ता"],
    "da": ["महंगाई भत्ता", "महंगाई"],
    "suspension": ["निलंबन", "निलम्बन", "अनुशासनिक", "विभागीय जांच"],
    "subsistence": ["निर्वाह भत्ता", "जीवन निर्वाह", "निलंबन"],
    "disciplinary": ["अनुशासनिक", "विभागीय जांच", "लघुशास्ति"],
    "conduct": ["आचरण", "सिविल सेवा आचरण नियम", "आचरण नियम"],
    "property": ["अचल संपत्ति", "संपत्ति", "वार्षिक विवरण"],
    "immovable": ["अचल संपत्ति", "संपत्ति विवरण"],
    "return": ["विवरण", "अचल संपत्ति", "वार्षिक विवरण"],
    "rti": ["सूचना का अधिकार", "जन सूचना", "अधिनियम 2005"],
    "information": ["सूचना का अधिकार", "सूचना"],
    "grievance": ["शिकायत", "जन शिकायत", "जनदर्शन"],
    "promotion": ["पदोन्नति", "डीपीसी", "वरिष्ठता"],
    "seniority": ["वरिष्ठता", "पदक्रम", "पदोन्नति"],
    "gradation": ["पदक्रम", "वरिष्ठता", "सूची"],
    "reservation": ["आरक्षण", "रोस्टर", "अनुसूचित"],
    "roster": ["रोस्टर", "आरक्षण"],
    "caste": ["जाति", "प्रमाणीकरण", "अनुसूचित"],
    "salary": ["वेतन", "भत्ते", "पे मैट्रिक्स"],
    "pay": ["वेतन", "मैट्रिक्स", "भत्ते"],
    "pension": ["पेंशन", "सेवानिवृत्ति"],
    "recruitment": ["भर्ती", "नियुक्ति", "चयन"],
    "election": ["निर्वाचन", "आचार संहिता", "मतदान"],
    "secretary": ["सचिव", "मुख्य सचिव"],
    "collector": ["कलेक्टर", "जिलाध्यक्ष"],
    "disabled": ["निःशक्त", "दिव्यांग"],
    "women": ["महिला", "प्रसूति"],
    "woman": ["महिला", "प्रसूति"],
    "female": ["महिला", "प्रसूति"],
}

CATEGORY_MAP: Dict[str, List[str]] = {
    "अवकाश व कार्यदिवस (Leave & Holidays)": ["leave", "holiday", "holidays", "casual", "earned", "maternity", "paternity", "child care", "study leave", "अवकाश", "छुट्टी", "आकस्मिक", "अर्जित", "प्रसूति", "मातृत्व", "पितृत्व"],
    "अनुकंपा नियुक्ति (Compassionate Appointment)": ["compassionate", "anukampa", "अनुकंपा", "अनुकम्पा"],
    "स्थानांतरण व पदस्थापना (Transfer & Postings)": ["transfer", "posting", "relieving", "spouse", "joining time", "स्थानांतरण", "तबादला", "पदस्थापना", "पदग्रहण"],
    "वेतन, पेंशन व भत्ते (Pay, Pension & Allowances)": ["dearness", "da ", "salary", "pay", "pension", "allowance", "retirement", "superannuation", "महंगाई", "वेतन", "पेंशन", "भत्ता", "सेवानिवृत्ति", "अधिवार्षिकी"],
    "विभागीय जांच व अनुशासन (Discipline & Vigilance)": ["suspension", "subsistence", "conduct", "property return", "disciplinary", "vigilance", "निलंबन", "निलम्बन", "निर्वाह भत्ता", "आचरण", "अचल संपत्ति", "विभागीय जांच"],
    "सूचना का अधिकार व जन शिकायत (RTI & Grievances)": ["rti", "right to information", "grievance", "सूचना का अधिकार", "शिकायत"],
    "पदोन्नति व वरिष्ठता (Promotion & Seniority)": ["promotion", "seniority", "dpc", "gradation", "पदोन्नति", "वरिष्ठता", "पदक्रम"],
    "आरक्षण रोस्टर व विशेष वर्ग (Reservation & Roster)": ["reservation", "roster", "caste", "obc", "sc/st", "आरक्षण", "रोस्टर", "जाति"],
    "निर्वाचन व आचार संहिता (Elections & Model Code)": ["election", "voting", "model code", "निर्वाचन", "चुनाव", "मतदान"],
    "भर्ती व सेवा नियम (Recruitment & Rules)": ["recruitment", "contract", "samvida", "भर्ती", "नियुक्ति", "संविदा"],
}

CATEGORY_ID_HINTS: Dict[str, str] = {
    "अवकाश व कार्यदिवस (Leave & Holidays)": "leave",
    "अनुकंपा नियुक्ति (Compassionate Appointment)": "compassionate",
    "स्थानांतरण व पदस्थापना (Transfer & Postings)": "transfer",
    "वेतन, पेंशन व भत्ते (Pay, Pension & Allowances)": "pension",
    "विभागीय जांच व अनुशासन (Discipline & Vigilance)": "vigilance",
    "सूचना का अधिकार व जन शिकायत (RTI & Grievances)": "rti",
    "पदोन्नति व वरिष्ठता (Promotion & Seniority)": "promotion",
    "आरक्षण रोस्टर व विशेष वर्ग (Reservation & Roster)": "reservation",
    "निर्वाचन व आचार संहिता (Elections & Model Code)": "elections",
    "भर्ती व सेवा नियम (Recruitment & Rules)": "recruitment",
}

EXCLUDED_ENTITIES = {
    "chhattisgarh", "government", "india", "general", "administration", "department",
    "order", "letter", "circular", "notification", "rules", "rule", "policy",
    "casual", "earned", "medical", "maternity", "paternity", "child", "care", "leave",
    "transfer", "compassionate", "appointment", "retirement", "dearness", "allowance",
    "suspension", "subsistence", "conduct", "property", "return", "joining", "time",
}

# Domain policy profiles: target policy_topics, high-precision Hindi query phrases, and boost anchors
DOMAIN_PROFILES: List[Dict[str, Any]] = [
    {
        "name": "leave_policy",
        "triggers": ["leave", "अवकाश", "छुट्टी", "maternity", "paternity", "प्रसूति", "मातृत्व", "पितृत्व", "casual leave", "आकस्मिक", "earned leave", "अर्जित", "child care", "संतान पालन", "commuted", "लघुकृत", "अध्ययन अवकाश", "study leave"],
        "policy_topics": {"leave_rules", "casual_leave", "maternity_paternity_leave", "study_leave"},
        "hi_queries": [
            '"सिविल सेवाएं (अवकाश) नियम" OR "अर्जित अवकाश" OR "आकस्मिक अवकाश" OR "प्रसूति अवकाश" OR "पितृत्व अवकाश"',
            '"संविदा नियुक्ति" AND "अवकाश"',
        ],
        "anchors": ["4f672641e687a86e", "2dd89eba555cd50a", "0d216bddd7d0dc85", "99e5df9c5882d856", "bc13bd928e8cc2be"],
    },
    {
        "name": "holiday_calendar",
        "triggers": ["holiday", "holidays", "सार्वजनिक अवकाश", "सामान्य अवकाश", "ऐच्छिक अवकाश", "त्योहार"],
        "policy_topics": {"holiday_calendar"},
        "hi_queries": ['"सार्वजनिक" AND "अवकाश"', '"ऐच्छिक अवकाश" OR "अवकाश घोषित"'],
        "anchors": ["6fc5ba9d98d9f083"],
    },
    {
        "name": "transfer_policy",
        "triggers": ["transfer", "स्थानांतरण", "तबादला", "spouse", "husband", "wife", "पति", "पत्नी", "application window", "joining time", "पदग्रहण", "कार्यभार ग्रहण"],
        "policy_topics": {"transfer_policy", "spouse_posting", "joining_time"},
        "hi_queries": [
            '"स्थानांतरण नीति"',
            '"पति" AND "पत्नी"',
            '"पदग्रहण काल" OR "कार्यभार ग्रहण"',
        ],
        "anchors": ["c9c88208e9d47d6f", "7f3d799683f4d82a", "6ec6ecb5d8a8a5b1", "4f672641e687a86e"],
    },
    {
        "name": "compassionate",
        "triggers": ["compassionate", "anukampa", "अनुकंपा", "अनुकम्पा"],
        "policy_topics": {"compassionate_appointment"},
        "hi_queries": ['"अनुकंपा नियुक्ति" OR "अनुकम्पा नियुक्ति"', '"एफ 7-1/2012" OR "RULE/112/2025"'],
        "anchors": ["ac820304db0fff00", "6a0a801892ce41e3", "ab69aa8a5d0d748a", "622e3839587ede54"],
    },
    {
        "name": "retirement_age",
        "triggers": ["retirement", "superannuation", "सेवानिवृत्ति", "अधिवार्षिकी", "extension", "पुनर्नियुक्ति"],
        "policy_topics": {"retirement_compulsory_review", "retirement_pension"},
        "hi_queries": ['"अधिवार्षिकी" OR "सेवानिवृत्ति आयु" OR "पेंशन नियम"', '"50 वर्ष की आयु" AND "20 वर्ष की सेवा"'],
        "anchors": ["349d95bcbd0e9a10", "6401404ff6377b37", "7f03488ab39f3632", "1fe7adec1b7fe639"],
    },
    {
        "name": "dearness_allowance",
        "triggers": ["dearness allowance", "dearness", "महंगाई भत्ता", "da rate", "da circular", "da rates"],
        "policy_topics": {"dearness_allowance"},
        "hi_queries": ['"महंगाई भत्ता" OR "महंगाई भत्ते"'],
        "anchors": ["6d8daa2180525e19", "ef9ba74af3fe838c", "592d783dd24da4e9", "e9f3b0a8d93e12d3"],
    },
    {
        "name": "suspension_subsistence",
        "triggers": ["suspension", "subsistence", "निलंबन", "निलम्बन", "निर्वाह भत्ता", "जीवन निर्वाह"],
        "policy_topics": {"suspension_inquiry"},
        "hi_queries": ['"निलंबन" OR "निलम्बन" OR "निर्वाह भत्ता" OR "जीवन निर्वाह"'],
        "anchors": ["8b6f4ef6a89cbaf6", "f160619708a54002", "d55755e409324b51"],
    },
    {
        "name": "conduct_and_property",
        "triggers": ["conduct rules", "conduct", "आचरण नियम", "आचरण", "property return", "immovable property", "अचल संपत्ति", "संपत्ति विवरण"],
        "policy_topics": {"conduct_rules", "property_return"},
        "hi_queries": ['"आचरण नियम" OR "अचल संपत्ति"', '"ESTB-1019" OR "एफ-2-1/2017"'],
        "anchors": ["3b2fd23e74466e9b", "fec9f9c9e9038e1e", "02189770b5fa400f", "538a67399e68e93a"],
    },
    {
        "name": "rti",
        "triggers": ["rti", "right to information", "सूचना का अधिकार", "जन सूचना"],
        "policy_topics": {"rti_act"},
        "hi_queries": ['"सूचना का अधिकार अधिनियम" OR "rtionline" OR "लोक सूचना"'],
        "anchors": ["b0337ab454c00f4e", "caebef766afb4fa0", "adc8a9fe74174df7"],
    },
]

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _extract_order_numbers(query: str) -> List[str]:
    q_norm = (query or "").translate(DEV_DIGITS)
    patterns = [
        r"(?:RULE|ESTB|GENS|GENCOR)[-/A-Za-z0-9\(\)]{3,}",
        r"[A-Za-z\u0900-\u097F]{1,8}[-\s]*\d+(?:[-/][A-Za-z0-9\u0900-\u097F\(\)]+){1,5}",
        r"\d{4,}\.\w+",
    ]
    hits: List[str] = []
    for p in patterns:
        for m in re.findall(p, q_norm, re.IGNORECASE):
            s = m.strip()
            if len(s) >= 5 and s not in hits:
                hits.append(s)
    return hits


def _detect_profiles(query: str) -> List[Dict[str, Any]]:
    q_low = (query or "").lower()
    matched = []
    for prof in DOMAIN_PROFILES:
        if any(trig in q_low for trig in prof["triggers"]):
            matched.append(prof)
    return matched


def _is_policy_question(query: str) -> bool:
    q_low = (query or "").lower()
    return any(k in q_low for k in [
        "policy", "rule", "rules", "process", "procedure", "eligible", "eligibility", "who can apply",
        "how do i", "how to", "how are", "how many days", "limit", "entitled", "allowed", "window",
        "deadline", "submit", "where can i read", "act", "updated", "extension", "extensions",
        "नीति", "नियम", "प्रक्रिया", "पात्रता", "अवधि", "सीमा", "कितने दिन", "कैसे", "कहां",
    ])


def _is_krutidev_toc_chunk(content: str) -> bool:
    txt = content or ""
    # Skip only true TOC/index tables ("Columns: uke ... 1: 2; 1: 3;") or pure Krutidev gibberish without Hindi prose
    if "columns: uke" in txt[:350].lower() or txt[:500].count("; ") >= 4 and "1:" in txt[:500]:
        return True
    if "iwB / 1 dks" in txt[:400] or ("table: data table" in txt[:350].lower() and ("iwB /" in txt[:350] or "लिया गया" in txt[:400])):
        return True
    dev_chars = sum(1 for ch in txt if "\u0900" <= ch <= "\u097F")
    if ("| uke |" in txt[:280] or "v/;k;" in txt[:280]) and dev_chars < 240:
        return True
    return False


def _substantive_clause_score(content: str, focus_terms: List[str]) -> float:
    """Score a chunk within a document, rewarding concrete numeric rule clauses over cover/forwarding memos or blank forms."""
    broad_terms = {"अवकाश", "छुट्टी", "leave", "स्थानांतरण", "transfer", "अनुकंपा", "नियम", "नीति", "आदेश"}
    specific_terms = [t.lower() for t in focus_terms[:16] if len(t) >= 2 and t.lower() not in broad_terms]
    broad_list = [t.lower() for t in focus_terms[:10] if t.lower() in broad_terms]
    txt_low = (content or "").lower()

    # Check if user explicitly asked for forms / bonds
    wants_form = any(t in ("form", "proforma", "bond", "प्रपत्र", "बन्ध") for t in specific_terms)

    spec_hits = sum(1 for t in specific_terms if t in txt_low)
    broad_hits = sum(1 for t in broad_list if t in txt_low)
    num_rule_hits = len(re.findall(r"\b\d+\s*(?:दिन|दिनों|दिवस|माह|वर्ष|प्रतिशत|%|days|months|years)\b", txt_low))

    score = spec_hits * 3.0 + broad_hits * 0.5 + min(num_rule_hits, 4) * 1.4
    if any(k in txt_low for k in ("संशोधन", "प्रतिस्थापित", "अधिसूचना", "substituted", "amendment")):
        score += 3.0
    if any(
        k in txt_low
        for k in (
            "15–15 दिनों", "15-15 दिनों", "300 दिनों", "135 दिन", "13 दिवस", "10 दिनों का विशेष",
            "30 दिनों", "नियम 13", "नियम 19", "25-", "38–", "38&", "अर्जित अवकाश", "प्रसूति अवकाश",
            "पितृत्व अवकाश", "आकस्मिक अवकाश", "अर्धवेतन अवकाश", "लघुकृत अवकाश", "स्थानांतरण नीति",
            "15 दिनों की अवधि", "180 दिनों तक का अर्धवेतन", "अध्ययन अवकाश",
            "अग्रिम अर्जित अवकाश", "पहली जनवरी और पहली जुलाई", "135 दिन तक की अवधि के लिये प्रसूति अवकाश",
            "31 जनवरी के पूर्व", "sparrow portal",
        )
    ):
        score += 5.5
    if any(k in txt_low for k in ("15–15 दिनों", "15-15 दिनों", "135 दिन तक की अवधि के लिये प्रसूति अवकाश", "दो से कम जीवित संतान", "31 जनवरी")):
        score += 4.5
    if len(content or "") >= 400:
        score += 0.6

    # Strongly penalize blank Appendix Forms / Bonds / Data Tables / Krutidev proforma headers / tiny stamp-cover pages
    body_only = re.sub(r"^Document:.*?Section:[^\n]*\n*", "", content or "", flags=re.DOTALL).lower()
    if len(body_only.strip()) < 130:
        score -= 18.0
    if "पुनर्नियुक्त" in txt_low and "नियोग्यता पेंशन" in txt_low and not any(t in ("reemploy", "pension", "पुनर्नियुक्ति", "पेंशन") for t in specific_terms):
        score -= 12.0
    if not wants_form and (
        any(
            k in txt_low
            for k in (
                "table: data table", "columns:", "unit: not stated", "िि=",
                "section: &3", "17½", "बन्ध पत्र", "बाध्यताधारी", "मुद्रांक शुल्क", "अवकाश लेखा के विवरण",
                "टीप 1–", "टीप 2–", "प्रपत्र–1", "प्रपत्र-1", "इस प्रपत्र का", "प्रकरण की सावधानी", "सिविल / सर्जन",
                "हस्ताक्षर ……", "लिया गया / चिकित्सा प्रमाणपत्र",
            )
        )
        or ((len(body_only) < 1100 or txt_low.count("....") >= 3) and ("......." in txt_low or "……" in txt_low))
        or ("v/;k;" in body_only)
        or "¼" in body_only[:250] and "38&" not in body_only and "प्रसूति" not in body_only
    ):
        score -= 30.0

    # Penalize pure NIC upload / forwarding / addressee cover memos without substantial rule body
    if len(body_only) < 950 and any(
        k in txt_low
        for k in ("वेबसाइट पर अपलोड", "राज्य सूचना विज्ञान अधिकारी", "अतिरिक्त प्रतियां", "बिजनेस पोस्ट", "आवश्यक कार्यवाही हेतु अग्रेषित")
    ) and not any(k in txt_low for k in ("प्रतिस्थापित", "संशोधन", "नियम-19", "31 जनवरी", "sparrow portal")):
        score -= 22.0
    if "प्रति," in txt_low and "भारत के संविधान" not in txt_low and "संशोधन" not in txt_low and "प्रतिस्थापित" not in txt_low and "13 दिवस" not in txt_low and "निर्णय लिया गया" not in txt_low and len(txt_low) < 950:
        score -= 18.0
    if re.match(r"^\s*Document:.*?Section:[^\n]*\s*(?:219|\d{1,3})?\s*$", content or "", re.DOTALL):
        score -= 12.0
    return score


def extract_search_terms(query: str) -> Tuple[List[str], List[str], List[str]]:
    """Extract clean tokens, bilingual Hindi/English expansions, and matched category labels."""
    clean_q = re.sub(r"[^\w\s\u0900-\u097F/-]", " ", query or "")
    raw_tokens = [t.strip("/-") for t in clean_q.split() if len(t.strip("/-")) > 1]

    core_tokens: List[str] = []
    for t in raw_tokens:
        t_low = t.lower()
        if t_low not in STOP_WORDS and t_low not in core_tokens:
            core_tokens.append(t)

    expanded: List[str] = list(core_tokens)
    for t in core_tokens:
        t_low = t.lower()
        if t_low in SYNONYM_MAP:
            for syn in SYNONYM_MAP[t_low]:
                for part in syn.split():
                    if part not in expanded:
                        expanded.append(part)

    matched_categories: List[str] = []
    q_low = (query or "").lower()
    for cat_name, kw_list in CATEGORY_MAP.items():
        if any(kw in q_low for kw in kw_list):
            matched_categories.append(cat_name)

    return core_tokens, expanded, matched_categories


def _resolve_doc_url(doc_url_index: Dict[str, Any], doc_dir: str, canonical_id: str, db_url: str) -> str:
    if db_url:
        return db_url
    for k in (doc_dir, f"doc_{doc_dir}", canonical_id, f"doc_{canonical_id}"):
        if k and k in doc_url_index:
            u = doc_url_index[k].get("url") or doc_url_index[k].get("source_url")
            if u:
                return u
    return ""


def search_gad_database(
    query: str,
    limit: int = 12,
    extra_terms: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    4-arm hybrid retrieval over `temporal_graph.db` with:
      - Full `doc_dir <-> nodes.doc_id` joining (5,295 / 5,295)
      - Bilingual `fts_nodes` + `fts_chunks` BM25 + phrase queries
      - Domain profile topic & anchor boosting (no blind Tier-2 category dump!)
      - Intra-document substantive clause selection for rulebooks
      - Anti-noise filtering (Central/Port vacancies, ISTM workshops, grant slips) & duplicate collapse
    """
    if not DB_PATH.exists():
        return []

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    doc_url_index = _load_doc_url_index()
    dup_map = _load_duplicate_map(cur)

    core_tokens, expanded_tokens, matched_categories = extract_search_terms(query)
    if extra_terms:
        for et in extra_terms:
            for tok in re.findall(r"[A-Za-z0-9\u0900-\u097F]{2,}", str(et)):
                if tok.lower() not in STOP_WORDS and tok not in expanded_tokens:
                    expanded_tokens.append(tok)

    profiles = _detect_profiles(query)
    target_cat_ids = {CATEGORY_ID_HINTS[c] for c in matched_categories if c in CATEGORY_ID_HINTS}
    target_policy_topics: Set[str] = set()
    profile_anchors: List[str] = []
    profile_hi_queries: List[str] = []
    for prof in profiles:
        target_policy_topics.update(prof["policy_topics"])
        for anc in prof["anchors"]:
            if anc not in profile_anchors:
                profile_anchors.append(anc)
        for hq in prof["hi_queries"]:
            if hq not in profile_hi_queries:
                profile_hi_queries.append(hq)

    q_low = (query or "").lower()
    wants_historical = bool(re.search(r"\b(old|previous|earlier|history|before|201[0-9]|200\d|पहले|पूर्व|इतिहास)\b", q_low))
    wants_noise_docs = any(k in q_low for k in ["istm", "workshop", "port", "visakhapatnam", "deputation", "vacancy", "voluntary grant", "स्वेच्छानुदान", "प्रतिनियुक्ति"])
    wants_holidays = any(k in q_low for k in ["holiday", "holidays", "सार्वजनिक", "ऐच्छिक", "त्योहार", "calendar", "2026 holiday"])
    is_policy_q = _is_policy_question(query)
    requested_years = re.findall(r"\b(20\d{2}|19\d{2})\b", query or "")

    # Track per-chunk RRF score + signals
    rrf_scores: Dict[str, float] = {}
    chunk_rows: Dict[str, Dict[str, Any]] = {}
    node_cache: Dict[str, Dict[str, Any]] = {}

    def _load_node_by_dir(doc_dir: str) -> Dict[str, Any]:
        if doc_dir in node_cache:
            return node_cache[doc_dir]
        r = cur.execute(
            """
            SELECT id, doc_id, doc_dir, order_no, doc_date, year, status,
                   title_hi, title_en, doc_type, summary, source_url,
                   category, category_id, policy_topics, is_cg_policy
            FROM nodes
            WHERE doc_dir = ? OR doc_id = ?
            LIMIT 1
            """,
            (doc_dir, doc_dir),
        ).fetchone()
        d = dict(r) if r else {}
        node_cache[doc_dir] = d
        return d

    def _record_chunk(
        row_obj: Any,
        rank: int,
        arm_weight: float,
        tier_name: str,
        extra_boost: float = 0.0,
        vec_sim: Optional[float] = None,
    ) -> None:
        r = dict(row_obj)
        cid = r["chunk_id"]
        doc_dir = r["doc_id"]
        node_info = _load_node_by_dir(doc_dir) if doc_dir else {}
        if cid not in chunk_rows:
            chunk_rows[cid] = {
                "chunk_id": cid,
                "doc_id": doc_dir,
                "canonical_doc_id": node_info.get("doc_id") or r.get("canonical_doc_id") or doc_dir,
                "order_no": r.get("order_no") or node_info.get("order_no") or "",
                "title": node_info.get("title_hi") or r.get("title") or node_info.get("title_en") or "",
                "title_en": node_info.get("title_en") or "",
                "doc_date": r.get("doc_date") or node_info.get("doc_date") or "",
                "status": node_info.get("status") or r.get("status") or "current",
                "content": r.get("content") or "",
                "category": node_info.get("category") or r.get("category") or "General",
                "category_id": node_info.get("category_id") or r.get("category_id") or "general",
                "policy_topics": node_info.get("policy_topics") or "[]",
                "is_cg_policy": node_info.get("is_cg_policy") if node_info.get("is_cg_policy") is not None else 1,
                "doc_type": node_info.get("doc_type") or "",
                "source_url": _resolve_doc_url(
                    doc_url_index, doc_dir, node_info.get("doc_id") or "", r.get("source_url") or node_info.get("source_url") or ""
                ),
                "tier": tier_name,
            }
        elif vec_sim is not None and "semantic_vector" not in chunk_rows[cid].get("tier", ""):
            chunk_rows[cid]["tier"] = f"{chunk_rows[cid]['tier']}+semantic_vector"
        if vec_sim is not None:
            chunk_rows[cid]["vec_sim"] = max(float(chunk_rows[cid].get("vec_sim") or 0.0), float(vec_sim))
        rrf_val = arm_weight / (35.0 + rank) + extra_boost
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_val

    def _pick_best_chunks_for_doc(doc_dir: str, focus_terms: List[str], max_chunks: int = 2) -> List[Any]:
        """Select substantive query-relevant chunks inside a document via idx_chunks_doc_id (O(1) lookup)."""
        rows = cur.execute(
            """
            SELECT chunk_id, doc_id, canonical_doc_id, chunk_index, order_no, title, doc_date, status, content
            FROM chunks
            WHERE doc_id = ?
            ORDER BY chunk_index ASC
            LIMIT 80
            """,
            (doc_dir,),
        ).fetchall()
        if not rows:
            return []
        scored: List[Tuple[float, int, Any]] = []
        wants_form = any(t.lower() in ("form", "proforma", "bond", "प्रपत्र", "बन्ध") for t in focus_terms)
        for idx, row in enumerate(rows):
            # In Leave Rules 2010 (4f672641e687a86e), positions <5 are TOC and >=47 are Appendix forms/tables
            if doc_dir == "4f672641e687a86e" and not wants_form and (idx < 5 or idx >= 47):
                continue
            txt = row["content"] or ""
            if _is_krutidev_toc_chunk(txt):
                continue
            sc = _substantive_clause_score(txt, focus_terms)
            scored.append((sc, -idx, row))
        if not scored:
            return list(rows[:max_chunks])
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if doc_dir == "4f672641e687a86e" and max_chunks >= 3:
            # Spread selections across core chapters: EL(18-25), HPL/Med(26-33), Mat135d(34-35), Pat15d(36-39), Study(40+), Gen(5-17)
            chosen: List[Any] = []
            seen_bands: Set[int] = set()
            for sc_v, neg_i, rw in scored:
                pos_i = -neg_i
                band = 1 if 18 <= pos_i <= 25 else (2 if 26 <= pos_i <= 33 else (3 if 34 <= pos_i <= 35 else (4 if 36 <= pos_i <= 39 else (5 if pos_i >= 40 else 0))))
                if band not in seen_bands:
                    seen_bands.add(band)
                    chosen.append(rw)
                    if len(chosen) >= max_chunks:
                        return chosen
            for _, _, rw in scored:
                if rw not in chosen:
                    chosen.append(rw)
                    if len(chosen) >= max_chunks:
                        break
            return chosen
        return [item[2] for item in scored[:max_chunks]]

    def _fts_term(t: str) -> str:
        # Avoid expensive prefix wildcards on short/Hindi tokens inside 94k FTS5 scans
        if all(ord(c) < 128 for c in t) and len(t) >= 5:
            return f'"{t}"*'
        return f'"{t}"'

    # ── Arm 1: Direct Order / Doc-Dir Number Match ────────────────────────────────
    order_refs = _extract_order_numbers(query)
    # Also check raw hex doc_dir / PDF filenames in query
    hex_refs = re.findall(r"\b([a-f0-9]{16})\b", q_low)
    pdf_refs = re.findall(r"\b(\d{6,10}_?\d*\.pdf)\b", q_low)
    for idx, oref in enumerate(order_refs[:4]):
        like_pat = f"%{oref}%"
        matched_n = cur.execute(
            "SELECT doc_dir FROM nodes WHERE order_no LIKE ? LIMIT 5",
            (like_pat,),
        ).fetchall()
        for rank_i, mn in enumerate(matched_n):
            dd = mn["doc_dir"]
            if dd:
                for ch in _pick_best_chunks_for_doc(dd, core_tokens, max_chunks=2):
                    _record_chunk(ch, rank_i + 1, 2.5, "exact_order", extra_boost=0.05)

    for dd in hex_refs[:4]:
        for ch in _pick_best_chunks_for_doc(dd, core_tokens, max_chunks=2):
            _record_chunk(ch, 1, 2.5, "exact_doc_id", extra_boost=0.05)

    for pdf_name in pdf_refs[:3]:
        matched_n = cur.execute(
            "SELECT doc_dir FROM nodes WHERE source_url LIKE ? LIMIT 3",
            (f"%{pdf_name}",),
        ).fetchall()
        for mn in matched_n:
            for ch in _pick_best_chunks_for_doc(mn["doc_dir"], core_tokens, max_chunks=2):
                _record_chunk(ch, 1, 2.5, "exact_pdf", extra_boost=0.05)

    # ── Prepare clean FTS expressions ─────────────────────────────────────────────
    hi_tokens = [t for t in expanded_tokens if any("\u0900" <= ch <= "\u097F" for ch in t)]
    en_tokens = [t for t in core_tokens if all(ord(c) < 128 for c in t) and t.lower() not in STOP_WORDS]

    all_safe_terms = []
    for t in hi_tokens + en_tokens:
        clean_t = re.sub(r'["*\'()]', "", t).strip()
        if len(clean_t) >= 2 and clean_t not in all_safe_terms:
            all_safe_terms.append(clean_t)

    or_expr = " OR ".join(_fts_term(t) for t in all_safe_terms[:8])
    hi_or_expr = " OR ".join(_fts_term(t) for t in hi_tokens[:8]) if hi_tokens else or_expr
    hi_and_expr = " AND ".join(_fts_term(t) for t in hi_tokens[:2]) if len(hi_tokens) >= 2 else ""

    # ── Arm 2: Bilingual FTS5 on Enriched `fts_nodes` ─────────────────────────────
    matched_node_dirs: List[Tuple[str, float]] = []
    seen_node_dirs: Set[str] = set()
    node_queries: List[Tuple[str, float]] = []
    for hq in profile_hi_queries[:2]:
        node_queries.append((hq, 1.8))
    if hi_and_expr:
        node_queries.append((hi_and_expr, 1.5))
    if len(en_tokens) >= 2:
        node_queries.append((" AND ".join(_fts_term(t) for t in en_tokens[:3]), 1.3))
    if or_expr:
        node_queries.append((or_expr, 1.0))

    for n_expr, weight in node_queries:
        try:
            n_rows = cur.execute(
                """
                SELECT n.doc_dir, n.is_cg_policy, bm25(fts_nodes) AS bscore
                FROM fts_nodes f
                JOIN nodes n ON f.id = n.id
                WHERE fts_nodes MATCH ?
                ORDER BY bscore ASC
                LIMIT 18
                """,
                (n_expr,),
            ).fetchall()
            for rank_i, nr in enumerate(n_rows, 1):
                dd = nr["doc_dir"]
                if not dd:
                    continue
                if nr["is_cg_policy"] == 0 and not wants_noise_docs:
                    continue
                if dd not in seen_node_dirs:
                    seen_node_dirs.add(dd)
                    matched_node_dirs.append((dd, weight / (25.0 + rank_i)))
        except Exception:
            continue

    # Pull substantive chunks from top FTS-matched nodes (up to 10 unique docs)
    for idx, (dd, n_score) in enumerate(matched_node_dirs[:10], 1):
        per_doc = 4 if dd == "4f672641e687a86e" else 2
        best_chs = _pick_best_chunks_for_doc(dd, all_safe_terms, max_chunks=per_doc)
        for ch_rank, ch in enumerate(best_chs):
            _record_chunk(ch, idx + ch_rank, 1.6, "fts_node", extra_boost=n_score)

    # ── Arm 3: Multi-Query FTS5 over `fts_chunks` ─────────────────────────────────
    chunk_queries: List[Tuple[str, float]] = []
    for hq in profile_hi_queries[:2]:
        chunk_queries.append((hq, 1.9))
    if hi_and_expr:
        chunk_queries.append((hi_and_expr, 1.4))
    elif len(en_tokens) >= 2:
        chunk_queries.append((" AND ".join(_fts_term(t) for t in en_tokens[:3]), 1.1))
    elif or_expr:
        chunk_queries.append((or_expr, 1.0))

    for c_expr, weight in chunk_queries:
        try:
            c_rows = cur.execute(
                """
                SELECT c.chunk_id, c.doc_id, c.canonical_doc_id, c.order_no, c.title,
                       c.doc_date, c.status, c.content, bm25(fts_chunks) AS bscore
                FROM fts_chunks f
                JOIN chunks c ON f.chunk_id = c.chunk_id
                WHERE fts_chunks MATCH ?
                ORDER BY bscore ASC
                LIMIT 20
                """,
                (c_expr,),
            ).fetchall()
            for rank_i, cr in enumerate(c_rows, 1):
                if _is_krutidev_toc_chunk(cr["content"]):
                    continue
                _record_chunk(cr, rank_i, weight, "fts_chunk")
        except Exception:
            continue

    # Run broad OR fallback ONLY if < 6 chunks found so far
    if len(chunk_rows) < 6 and or_expr:
        try:
            c_rows = cur.execute(
                """
                SELECT c.chunk_id, c.doc_id, c.canonical_doc_id, c.order_no, c.title,
                       c.doc_date, c.status, c.content
                FROM fts_chunks f
                JOIN chunks c ON f.chunk_id = c.chunk_id
                WHERE fts_chunks MATCH ?
                ORDER BY bm25(fts_chunks) ASC
                LIMIT 16
                """,
                (or_expr,),
            ).fetchall()
            for rank_i, cr in enumerate(c_rows, 1):
                if not _is_krutidev_toc_chunk(cr["content"]):
                    _record_chunk(cr, rank_i, 0.85, "fts_chunk_or")
        except Exception:
            pass

    # ── Arm 4: Rulebook / Anchor Coverage Backstop ─────────────────────────────
    if profiles and is_policy_q:
        for dd in profile_anchors[:5]:
            per_anc = 4 if dd == "4f672641e687a86e" else 2
            anchor_chs = _pick_best_chunks_for_doc(dd, all_safe_terms, max_chunks=per_anc)
            for ch_rank, ch in enumerate(anchor_chs):
                _record_chunk(ch, ch_rank + 1, 1.45, "policy_anchor", extra_boost=0.015)

    # ── Arm 5: Dense Vector Semantic Search (BAAI/bge-m3, 1024-d) ───────────────
    try:
        from src.semantic_search import is_semantic_index_ready, semantic_vector_search
        if is_semantic_index_ready():
            hi_semantic_q = " ".join(hi_tokens[:6]) if hi_tokens else None
            vec_hits = semantic_vector_search(
                query,
                limit=28,
                extra_query=hi_semantic_q,
                allow_non_policy=wants_noise_docs,
            )
            seen_vec_dirs: Set[str] = set()
            for rank_i, vh in enumerate(vec_hits, 1):
                v_dd = vh["doc_dir"]
                v_cid = vh["chunk_id"]
                v_sim = vh["similarity"]
                v_node = _load_node_by_dir(v_dd) if v_dd else {}
                if not wants_noise_docs and v_node.get("is_cg_policy", 1) == 0:
                    continue
                v_row = cur.execute(
                    """
                    SELECT chunk_id, doc_id, canonical_doc_id, order_no, title,
                           doc_date, status, content
                    FROM chunks
                    WHERE chunk_id = ?
                    LIMIT 1
                    """,
                    (v_cid,),
                ).fetchone()
                if v_row is not None and not _is_krutidev_toc_chunk(v_row["content"]):
                    _record_chunk(
                        v_row,
                        rank_i,
                        2.15,
                        "semantic_vector",
                        extra_boost=max(0.0, (v_sim - 0.42) * 0.14),
                        vec_sim=v_sim,
                    )
                if v_dd and v_dd not in seen_vec_dirs and len(seen_vec_dirs) < 4:
                    seen_vec_dirs.add(v_dd)
                    per_v = 3 if v_dd == "4f672641e687a86e" else 1
                    for ch_r, best_ch in enumerate(_pick_best_chunks_for_doc(v_dd, all_safe_terms, max_chunks=per_v)):
                        if best_ch["chunk_id"] != v_cid:
                            _record_chunk(best_ch, rank_i + ch_r, 1.45, "semantic_vector", vec_sim=round(v_sim * 0.92, 4))
    except Exception:
        pass

    # ── Rerank Candidates with Policy, Topic, Anti-Noise, Temporal & Dedup ────────
    scored_list: List[Tuple[float, Dict[str, Any]]] = []
    for cid, cdict in chunk_rows.items():
        score = rrf_scores.get(cid, 0.0)
        title_blob = f"{cdict.get('title', '')} {cdict.get('title_en', '')}".lower()
        content_low = (cdict.get("content") or "").lower()
        ono_low = (cdict.get("order_no") or "").lower()
        doc_date = cdict.get("doc_date") or ""
        try:
            p_topics = set(json.loads(cdict.get("policy_topics") or "[]"))
        except Exception:
            p_topics = set()

        # 1. Policy topic & category affinity bonus (relevance-gated)
        if target_policy_topics and (p_topics & target_policy_topics):
            overlap_cnt = len(p_topics & target_policy_topics)
            score *= 1.0 + 0.25 * min(overlap_cnt, 3)
        elif target_cat_ids and cdict.get("category_id") in target_cat_ids:
            score *= 1.18

        # 2. Exact query / Hindi core term presence inside content + title
        term_hits = 0
        for tok in all_safe_terms[:10]:
            tl = tok.lower()
            if tl in title_blob:
                term_hits += 2
            elif tl in content_low:
                term_hits += 1
        score += 0.004 * min(term_hits, 10)

        # 3. Rulebook / Policy prioritization over individual posting notices & seniority rosters
        if is_policy_q:
            is_individual_posting = (
                "estb-102" in ono_low
                or ("श्री " in title_blob and ("स्थानांतरण" in title_blob or "पदस्थापना" in title_blob or "सेवानिवृत्ति आदेश" in title_blob))
                or "अनंतिम पदक्रम सूची" in title_blob
                or "gradation list" in title_blob
                or ("seniority" in title_blob and "seniority" not in q_low and "वरिष्ठता" not in q_low)
            )
            if is_individual_posting:
                score *= 0.25
            if cdict.get("doc_id") in profile_anchors:
                score *= 1.45

        # 4. Anti-noise penalties (discretionary grants, port vacancies, ISTM workshops, central deputation, portal stubs)
        is_central_vacancy = any(
            k in title_blob
            for k in ("filling up", "deputation", "ministry of", "port authority", "visakhapatnam", "c&a -", "workshop on")
        )
        if not wants_noise_docs and (cdict.get("is_cg_policy", 1) == 0 or is_central_vacancy):
            score *= 0.10

        # If user asked about leave rules/policy (not holidays), demote 1-day festival/local holidays
        if profiles and any(p["name"] == "leave_policy" for p in profiles) and not wants_holidays:
            if "holiday_calendar" in p_topics and "leave_rules" not in p_topics and "casual_leave" not in p_topics:
                score *= 0.22

        # Demote superseded 2004/2008/2019 contract copies below 2024 30-day amendment unless historical year asked
        if cdict.get("doc_id") == "def19af28b141485" and not wants_historical and "2004" not in q_low and "2008" not in q_low:
            score *= 0.42
        if cdict.get("doc_id") == "99e5df9c5882d856" and not wants_historical and "2012" not in q_low and "2019" not in q_low:
            score *= 0.62
        if cdict.get("doc_id") == "0d216bddd7d0dc85" and any(k in q_low for k in ("leave", "अवकाश", "casual", "contract", "संविदा")):
            score *= 1.35
        # Demote December-only CL warning order below the 13-day general CL circular unless Dec asked
        if cdict.get("doc_id") == "bc13bd928e8cc2be" and not any(k in q_low for k in ("december", "दिसम्बर", "दिसंबर")):
            score *= 0.58
        # Demote narrow leave side-memos (union SCL 1972, CR period 2007, 1971 CL act snippet) on broad leave queries
        if cdict.get("doc_id") == "360d3e5a8a94d455" and not any(k in q_low for k in ("union", "federation", "संघ", "तहसील")):
            score *= 0.45
        if cdict.get("doc_id") == "b03bbc08d0d82e7c" and not any(k in q_low for k in ("confidential", "acr", "cr ", "गोपनीय")):
            score *= 0.45
        if cdict.get("doc_id") == "6cd99f8d55c70326" and "1971" not in q_low:
            score *= 0.55

        # If user specifically asks for contractual leave rules, keep regular Leave Rulebook 2010 secondary
        wants_contract = any(k in q_low for k in ("contract", "contractual", "संविदा"))
        if wants_contract and not any(k in q_low for k in ("regular", "permanent", "नियमित", "compare")):
            if cdict.get("doc_id") == "4f672641e687a86e":
                score *= 0.25
            elif cdict.get("doc_id") in ("0d216bddd7d0dc85", "53404853ad6ea068", "99e5df9c5882d856"):
                score *= 1.65

        # Boost 13-day General CL circular + Leave Rule 25/38 EL/Maternity when casual/earned/maternity is asked
        if any(k in q_low for k in ("casual", "आकस्मिक")) and cdict.get("doc_id") == "2dd89eba555cd50a":
            score *= 1.55

        # Extra relevance boost for specific sub-queries (e.g., spouse posting -> clause 3.13, joining time -> Leave Rule 23)
        if any(k in q_low for k in ("husband", "wife", "spouse", "पति", "पत्नी")):
            if "पति" in content_low or "पत्नी" in content_low or "एक ही स्थान" in content_low:
                score *= 1.55
            if cdict.get("doc_id") == "c9c88208e9d47d6f":
                score *= 1.45
        if any(k in q_low for k in ("joining time", "पदग्रहण", "कार्यभार ग्रहण")):
            if "पदग्रहण" in content_low or "कार्यभार ग्रहण" in content_low:
                score *= 1.50
            if cdict.get("doc_id") == "4f672641e687a86e":
                score *= 1.35

        # 5. Temporal validity, recency preference & explicit year match
        status = (cdict.get("status") or "current").lower()
        if status == "superseded" and not wants_historical:
            score *= 0.55
        elif status == "current":
            score *= 1.08

        if requested_years:
            if any(doc_date.startswith(y) or y in title_blob or y in ono_low for y in requested_years):
                score *= 1.65
            elif not wants_historical and doc_date:
                score *= 0.65
        elif not wants_historical and doc_date:
            if doc_date >= "2024-01-01":
                score *= 1.32
            elif doc_date >= "2020-01-01":
                score *= 1.15
            elif doc_date < "2010-01-01" and cdict.get("doc_id") not in profile_anchors:
                score *= 0.85

        # Boost substantive rule/numeric chunks over cover/NIC/form memos
        clause_bonus = _substantive_clause_score(cdict.get("content") or "", all_safe_terms)
        if clause_bonus > 0:
            score += 0.015 * min(clause_bonus, 12.0)
        elif clause_bonus < 0:
            score *= 0.25

        cdict["score"] = round(score, 6)
        scored_list.append((score, cdict))

    scored_list.sort(key=lambda x: x[0], reverse=True)

    # Collapse twin duplicates & enforce per-document diversity
    final_results: List[Dict[str, Any]] = []
    doc_chunk_counts: Dict[str, int] = {}
    seen_canon_content: Set[str] = set()

    for _, cdict in scored_list:
        raw_dir = cdict["doc_id"]
        canon_dir = dup_map.get(raw_dir, raw_dir)
        # Skip duplicate PDFs covering the same order/content
        content_sig = f"{canon_dir}:{re.sub(r'\s+', '', (cdict.get('content') or '')[:140])}"
        if content_sig in seen_canon_content:
            continue

        # Collapse identical (canon_dir) twin versions with clean cap (up to 4 for comprehensive 2010 Leave Rulebook)
        cap = 4 if canon_dir == "4f672641e687a86e" else 2
        if doc_chunk_counts.get(canon_dir, 0) >= cap:
            continue
        seen_canon_content.add(content_sig)
        doc_chunk_counts[canon_dir] = doc_chunk_counts.get(canon_dir, 0) + 1
        final_results.append(cdict)
        if len(final_results) >= limit:
            break

    conn.close()
    return final_results
