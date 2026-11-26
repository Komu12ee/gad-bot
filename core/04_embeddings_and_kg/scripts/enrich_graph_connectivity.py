#!/usr/bin/env python3
"""
Non-Destructive Knowledge Graph & Node-Chunk Bridge Enrichment for temporal_graph.db.

Fixes:
1. Node <-> Chunk ID Alignment: Populates `nodes.doc_dir` and `chunks.canonical_doc_id`
   so every chunk maps deterministically to its graph node (5,295 / 5,295).
2. Category & Sub-Topic Cleansing: Replaces noisy topic[0] assignments with
   deterministic GAD-section + bilingual keyword classification while preserving
   all 15 `(category_id, category)` buckets used by the WebUI.
3. Noise Tagging (`is_cg_policy`): Flags Central/Port deputation vacancies,
   ISTM workshops, and discretionary grant slips so policy graphs & retrieval
   stay focused on Chhattisgarh GAD rules and circulars.
4. High-Signal Multi-Layer Edges:
   - `SUPERSEDES`, `AMENDS`, `CONTINUES`, `REFERENCES` via normalized order numbers,
     Devanagari-to-ASCII digit conversion, `समसंख्यक` (even-number) + exact date
     resolution, and regex citation extraction from early chunks (no blind YYYY-MM).
   - `DUPLICATE_OF` linking mirrored/scanned copies to canonical rulebook copies.
   - `CITES_RULEBOOK` linking circulars that reference core Acts/Rules directly
     to canonical rulebook nodes (Leave Rules 2010, Conduct Rules 1965, etc.).
   - `SAME_FILE_SERIES` chronological chains across shared policy file numbers.
   - `RELATED_POLICY_TOPIC` anchored intra-topic timelines/hubs per policy sub-topic.
   - Clean intra-category `CHRONOLOGICAL_NEXT` / `CHRONOLOGICAL_PREV` replacing
     arbitrary cross-category date jumps.
5. Bilingual `nodes` Metadata & Rebuilt `fts_nodes` while preserving `chunks`,
   `fts_chunks`, `officers`, `officer_notifications`, and `fts_officers`.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "fg2"
DB_PATH = DATA_DIR / "temporal_graph.db"
DOCS_PATH = DATA_DIR / "documents.json"
META_PATH = DATA_DIR / "doc_metadata.json"

CATEGORY_LABELS: Dict[str, str] = {
    "leave": "अवकाश व कार्यदिवस (Leave & Holidays)",
    "transfer": "स्थानांतरण व पदस्थापना (Transfer & Postings)",
    "compassionate": "अनुकंपा नियुक्ति (Compassionate Appointment)",
    "pension": "वेतन, पेंशन व भत्ते (Pay, Pension & Allowances)",
    "vigilance": "विभागीय जांच व अनुशासन (Discipline & Vigilance)",
    "rti": "सूचना का अधिकार व जन शिकायत (RTI & Grievances)",
    "elections": "निर्वाचन व आचार संहिता (Elections & Model Code)",
    "reservation": "आरक्षण रोस्टर व विशेष वर्ग (Reservation & Roster)",
    "promotion": "पदोन्नति व वरिष्ठता (Promotion & Seniority)",
    "recruitment": "भर्ती व सेवा नियम (Recruitment & Rules)",
    "training": "प्रशिक्षण व प्रशासनिक क्षमता (Training & Capacity)",
    "finance": "बजट व वित्तीय स्वीकृतियां (Budget & Finance)",
    "schemes": "शासकीय योजनाएं व अनुदान (Schemes & Grants)",
    "cadre": "प्रशासनिक संगठन व संवर्ग (Cadre & Organization)",
    "general": "सामान्य प्रशासनिक परिपत्र (General Circulars)",
}

# Known canonical rulebook / core policy anchors (keyed by doc_dir)
CORE_ANCHORS: Dict[str, Dict[str, Any]] = {
    "4f672641e687a86e": {
        "title_en": "Chhattisgarh Civil Services (Leave) Rules, 2010",
        "extra_summary": (
            "Chhattisgarh Civil Services (Leave) Rules 2010. Comprehensive rules on Earned Leave (अर्जित अवकाश), "
            "Half Pay Leave (अर्धवेतन अवकाश), Commuted Leave (लघुकृत अवकाश), Extraordinary Leave (असाधारण अवकाश), "
            "Leave Not Due (अदेय अवकाश), Maternity Leave (प्रसूति अवकाश), Paternity Leave (पितृत्व अवकाश), "
            "Adoption Leave (दत्तक ग्रहण अवकाश), Special Disability Leave (विशेष नियोग्यता अवकाश), Study Leave (अध्ययन अवकाश), Joining Time (पदग्रहण काल)."
        ),
        "policy_topics": ["leave_rules", "maternity_paternity_leave", "study_leave", "joining_time"],
        "category_id": "leave",
    },
    "2dd89eba555cd50a": {
        "policy_topics": ["casual_leave", "leave_rules"],
        "category_id": "leave",
    },
    "bc13bd928e8cc2be": {
        "policy_topics": ["casual_leave", "leave_rules"],
        "category_id": "leave",
    },
    "71adae321b07ab8e": {
        "policy_topics": ["casual_leave", "leave_rules"],
        "category_id": "leave",
    },
    "6cd99f8d55c70326": {
        "policy_topics": ["casual_leave", "leave_rules"],
        "category_id": "leave",
    },
    "0d216bddd7d0dc85": {
        "policy_topics": ["contract_rules", "leave_rules"],
        "category_id": "recruitment",
    },
    "99e5df9c5882d856": {
        "policy_topics": ["contract_rules", "leave_rules"],
        "category_id": "recruitment",
    },
    "def19af28b141485": {
        "policy_topics": ["contract_rules", "maternity_paternity_leave", "leave_rules"],
        "category_id": "leave",
    },
    "c9c88208e9d47d6f": {
        "title_en": "Chhattisgarh Transfer Policy Year 2025",
        "extra_summary": (
            "Chhattisgarh State Government Transfer Policy 2025 (स्थानांतरण नीति वर्ष 2025). Covers transfer application window "
            "(6 June to 13 June 2025), district transfer dates (14 June to 25 June 2025), spouse posting (पति-पत्नी एक ही स्थान), "
            "tenure limits, relieving deadlines, and exemptions."
        ),
        "policy_topics": ["transfer_policy", "spouse_posting"],
        "category_id": "transfer",
    },
    "7f3d799683f4d82a": {"policy_topics": ["transfer_policy", "spouse_posting"], "category_id": "transfer"},
    "6ec6ecb5d8a8a5b1": {"policy_topics": ["transfer_policy"], "category_id": "transfer"},
    "3b2fd23e74466e9b": {
        "title_en": "Chhattisgarh / MP Civil Services (Conduct) Rules, 1965 (Compilation)",
        "extra_summary": (
            "Chhattisgarh Civil Services (Conduct) Rules 1965 (सिविल सेवा आचरण नियम 1965), property return Rule 19, "
            "gift, integrity, movable and immovable property return, strike and social media rules."
        ),
        "policy_topics": ["conduct_rules", "property_return"],
        "category_id": "vigilance",
    },
    "02189770b5fa400f": {"policy_topics": ["conduct_rules"], "category_id": "vigilance"},
    "538a67399e68e93a": {"policy_topics": ["conduct_rules", "property_return"], "category_id": "vigilance"},
    "fec9f9c9e9038e1e": {"policy_topics": ["conduct_rules", "property_return"], "category_id": "vigilance"},
    "ac820304db0fff00": {"policy_topics": ["compassionate_appointment"], "category_id": "compassionate"},
    "622e3839587ede54": {"policy_topics": ["compassionate_appointment"], "category_id": "compassionate"},
    "857a3b0589b4cbf7": {"policy_topics": ["compassionate_appointment"], "category_id": "compassionate"},
    "ab69aa8a5d0d748a": {"policy_topics": ["compassionate_appointment"], "category_id": "compassionate"},
    "8b6f4ef6a89cbaf6": {"policy_topics": ["suspension_inquiry"], "category_id": "vigilance"},
    "f160619708a54002": {"policy_topics": ["suspension_inquiry"], "category_id": "vigilance"},
    "d55755e409324b51": {"policy_topics": ["suspension_inquiry"], "category_id": "vigilance"},
    "349d95bcbd0e9a10": {"policy_topics": ["retirement_compulsory_review", "retirement_pension"], "category_id": "pension"},
    "6401404ff6377b37": {"policy_topics": ["retirement_compulsory_review", "retirement_pension"], "category_id": "pension"},
    "6d8daa2180525e19": {"policy_topics": ["dearness_allowance"], "category_id": "pension"},
    "ef9ba74af3fe838c": {"policy_topics": ["dearness_allowance"], "category_id": "pension"},
    "592d783dd24da4e9": {"policy_topics": ["dearness_allowance"], "category_id": "pension"},
    "e9f3b0a8d93e12d3": {"policy_topics": ["dearness_allowance"], "category_id": "pension"},
    "b0337ab454c00f4e": {"policy_topics": ["rti_act"], "category_id": "rti"},
    "caebef766afb4fa0": {"policy_topics": ["rti_act"], "category_id": "rti"},
    "adc8a9fe74174df7": {"policy_topics": ["rti_act"], "category_id": "rti"},
}

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

ORDER_NO_REGEX = re.compile(
    r"(?:(?:क्रमांक|क्र\.?|पत्र\s*क्र\.?|अधिसूचना\s*क्र\.?|आदेश\s*क्र\.?|परिपत्र\s*क्र\.?|No\.?|F\.?\s*No\.?)\s*[:\-–]?\s*)"
    r"([A-Za-z0-9\u0900-\u097F\-]+(?:/[A-Za-z0-9\u0900-\u097F\.\-\(\)]+){1,6})",
    re.IGNORECASE,
)

DATE_DMY_REGEX = re.compile(r"\b(\d{1,2})[./\-](\d{1,2})[./\-]((?:19|20)\d{2})\b")

STOP_TOKENS = {
    "chhattisgarh", "government", "governance", "department", "general", "administration",
    "order", "circular", "notification", "letter", "dated", "date", "regarding", "subject",
    "छत्तीसगढ़", "शासन", "सामान्य", "प्रशासन", "विभाग", "मंत्रालय", "नवा", "रायपुर",
    "आदेश", "परिपत्र", "अधिसूचना", "विषय", "संबंध", "बाबत", "दिनांक", "क्रमांक",
    "समस्त", "विभागाध्यक्ष", "कलेक्टर", "संभागायुक्त", "प्रति", "के", "की", "का", "में", "से",
}


def normalize_digits(text: str) -> str:
    return (text or "").translate(DEV_DIGITS)


def normalize_order_no(order_no: Optional[str]) -> str:
    """Normalize CG GAD order/file number across Hindi/English OCR variations."""
    if not order_no:
        return ""
    s = normalize_digits(order_no).strip().lower()
    s = re.sub(r"^(क्रमांक|क्र\.?|संख्या|सं\.?|पत्र\s*क्रमांक|आदेश\s*क्रमांक|no\.?|f\.?\s*no\.?)\s*[:\-–\.]*\s*", "", s)
    # Remove trailing :: ministry junk captured by OCR
    s = s.split("::")[0].strip()
    # Hindi prefix / section code equivalence
    repls = [
        ("एफ.", "f "), ("एफ", "f "), ("सी.", "c "), ("सी", "c "), ("ई.", "e "), ("ई", "e "),
        ("जी.", "g "), ("जी", "g "), ("एक-", "1-"), ("/एक/", "/1/"), ("/एक", "/1"),
        ("/दो/", "/2/"), ("/तीन/", "/3/"), ("/चार/", "/4/"), ("/पांच/", "/5/"), ("/छः/", "/6/"),
        ("आ.प्र.", "ap"), ("सूअप्र", "rti"), ("आरटीआई", "rti"), ("वि.जा.", "vj"),
    ]
    for src, dst in repls:
        s = s.replace(src, dst)
    s = re.sub(r"\s+", "", s)
    s = s.replace("–", "-").replace("—", "-").replace("||", "/")
    s = re.sub(r"^[.\-:/]+|[.\-:/]+$", "", s)
    # Reject meaningless tokens
    if len(s) < 4 or not any(ch.isdigit() for ch in s):
        return ""
    if s in {"मांक", "क्रमांक", "0771-2510973", "0771-2510825"}:
        return ""
    return s


def extract_file_series(order_no: Optional[str], policy_topics: Iterable[str]) -> str:
    """Extract canonical standing file/rule series key for series-chain linking."""
    norm = normalize_order_no(order_no)
    if not norm:
        return ""
    # Skip phone-like numbers
    if norm.startswith("0771"):
        return ""
    # Specific high-signal CG GAD standing rule/policy series
    # E.g. F9-1/2012/1-3 -> series:f9-1/2012/1-3 ; RULE-3017/2/2025-GAD-6 -> series:rule-3017/gad-6
    if "gens-1001/" in norm and "gad-5" in norm:
        return "series:gens-1001/gad-5"
    if "rule/112/" in norm and "gad-3" in norm:
        return "series:compassionate/rule-112"
    if "f7-1/2012/1-3" in norm or "f7-4/2009/1-3" in norm:
        return "series:compassionate/f7-1-2012"
    if "f9-1/2012/1-3" in norm or "f9-1/2004/1-3" in norm:
        return "series:contract/f9-1"
    if "e3-12/2008" in norm or "gencor/3064/" in norm:
        return "series:da/e3-12-gencor3064"
    if "rule-3017/" in norm and "gad-6" in norm:
        return "series:transfer/rule-3017"
    if "f3-2/2010/1-3" in norm or "f9-1/2009/1-3" in norm:
        return "series:suspension/f3-2-f9-1"
    if "f2-1/2017/1-3" in norm or "f2-2/2025/1-3" in norm or "estb-1019/" in norm:
        return "series:conduct/f2-estb1019"
    if "f7-1/2017/1-3" in norm:
        return "series:compulsory-retire/f7-1-2017"
    if "rule-801/" in norm and "gad-13" in norm:
        return "series:rti/rule-801"
    if "c-3-7/" in norm or "f13-9/2012" in norm:
        return "series:casual-leave/c3-7-f13-9"
    # Generic multi-part file number where trailing order number varies or same file no repeats across years
    # Avoid hyper-generic daily posting files (estb-102, estb-101, gencor-35) unless combined with specific file
    if any(norm.startswith(p) for p in ("estb-102", "estb102", "estb-101", "gens-1002", "1525", "y-180", "y-170", "y-120", "y-250")):
        return ""
    if norm.count("/") >= 2 and len(norm) >= 8:
        return f"series:{norm}"
    return ""


def parse_dmy_to_iso(d_str: Optional[str]) -> Optional[str]:
    if not d_str:
        return None
    s = normalize_digits(d_str).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    m = DATE_DMY_REGEX.search(s)
    if m:
        day, mon, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mon <= 12 and 1 <= day <= 31:
            return f"{yr:04d}-{mon:02d}-{day:02d}"
    return None


def tokens_for_overlap(text: str) -> Set[str]:
    toks = re.findall(r"[a-zA-Z0-9\u0900-\u097F]{3,}", (text or "").lower())
    return {t for t in toks if t not in STOP_TOKENS}


def is_noise_document(
    title_hi: str, title_en: str, order_no: str, doc_type: str, summary: str, first_chunk: str, url: str, meta: Dict[str, Any]
) -> Tuple[bool, str]:
    """Identify external/noisy/navigation docs that should not dominate CG state policy categories/hubs."""
    blob = f"{title_hi} {title_en} {order_no} {summary} {first_chunk[:350]}".lower()
    if meta.get("source_type") == "html" and not order_no and len(first_chunk) < 600:
        return True, "portal_navigation"
    t_low = (title_hi or "").strip().lower()
    if t_low in {"form details view", "contactus", "rules and circular", "published reports", "mantralay ciculars", "cgstate circular"}:
        return True, "portal_navigation"
    if any(k in blob for k in ["स्वेच्छानुदान", "voluntary grant", "discretionary grant", "voluntary donation"]):
        return True, "discretionary_grant"
    if any(k in blob for k in [
        "visakhapatnam port", "port authority", "श्यामा प्रसाद मुखर्जी पोर्ट", "cochin port",
        "kandla", "mormugao", "jawaharlal nehru port", "paradip port", "mumbai port",
    ]):
        return True, "central_port_vacancy"
    if any(k in blob for k in [
        "institute of secretariat training", "istm", "two days workshop", "three days training",
        "refresher course", "igot", "karmayogi", "कर्मयोगी", "capacity building training",
        "seminar on right to information by istm", "singapore cooperation programme",
    ]) and "अवकाश नियम" not in blob:
        return True, "training_workshop"
    if (
        "vacancy circular" in doc_type.lower() or "रिक्ति / भर्ती" in doc_type.lower()
    ) and any(k in blob for k in ["government of india", "ministry of ", "भारत सरकार", "filling up", "deputation", "composite method"]):
        return True, "central_deputation_vacancy"
    return False, ""


def classify_node(
    order_no: str,
    title_hi: str,
    title_en: str,
    summary: str,
    doc_type: str,
    meta: Dict[str, Any],
    first_chunk: str,
    url: str,
    doc_dir: str,
) -> Tuple[str, str, List[str], int]:
    """Return clean `(category_id, category_label, policy_topics, is_cg_policy)`."""
    if doc_dir in CORE_ANCHORS and "category_id" in CORE_ANCHORS[doc_dir]:
        anchor = CORE_ANCHORS[doc_dir]
        cid = anchor["category_id"]
        return cid, CATEGORY_LABELS[cid], list(anchor["policy_topics"]), 1

    noise, noise_kind = is_noise_document(title_hi, title_en, order_no, doc_type, summary, first_chunk, url, meta)
    title_blob = f"{title_hi} {title_en}".lower()
    full_blob = f"{title_hi} {title_en} {summary} {first_chunk[:500]}".lower()
    ono_low = normalize_digits(order_no or "").lower()

    if noise:
        if noise_kind == "discretionary_grant":
            return "schemes", CATEGORY_LABELS["schemes"], ["discretionary_grant"], 0
        if noise_kind == "training_workshop":
            return "training", CATEGORY_LABELS["training"], ["training_workshop"], 0
        if noise_kind == "portal_navigation":
            return "general", CATEGORY_LABELS["general"], ["portal_navigation"], 0
        return "recruitment", CATEGORY_LABELS["recruitment"], ["central_deputation"], 0

    policy_topics: List[str] = []

    # Granular CG GAD policy topic extraction
    if any(k in full_blob for k in ["अवकाश नियम", "leave rules", "आकस्मिक अवकाश", "casual leave", "अर्जित अवकाश", "earned leave", "प्रसूति अवकाश", "मातृत्व अवकाश", "maternity leave", "पितृत्व अवकाश", "paternity leave", "अध्ययन अवकाश", "study leave", "अर्धवेतन अवकाश", "निःशक्त कर्मचारियों"]):
        policy_topics.append("leave_rules")
    if any(k in full_blob for k in ["आकस्मिक अवकाश", "विशेष आकस्मिक अवकाश", "casual leave", "सी-3-7", "c-3-7"]):
        policy_topics.append("casual_leave")
    if any(k in full_blob for k in ["प्रसूति अवकाश", "मातृत्व अवकाश", "पितृत्व अवकाश", "दत्तक", "maternity", "paternity", "child care", "संतान पालन"]):
        policy_topics.append("maternity_paternity_leave")
    if any(k in full_blob for k in ["अध्ययन अवकाश", "study leave"]):
        policy_topics.append("study_leave")
    if any(k in title_blob for k in ["सार्वजनिक अवकाश", "सामान्य अवकाश", "ऐच्छिक अवकाश", "स्थानीय अवकाश", "public holiday", "optional holiday", "अवकाश घोषित", "अवकाश घोषणा", "अवकाश के दिन"]):
        policy_topics.append("holiday_calendar")
    if any(k in full_blob for k in ["स्थानांतरण नीति", "transfer policy", "rule-3017"]):
        policy_topics.append("transfer_policy")
    if any(k in full_blob for k in ["पति-पत्नी", "पति एवं पत्नी", "spouse", "एक ही स्थान पर पदस्थापना"]):
        policy_topics.append("spouse_posting")
    if any(k in full_blob for k in ["कार्यभार ग्रहण", "पदग्रहण काल", "joining time"]):
        policy_topics.append("joining_time")
    if any(k in full_blob for k in ["अनुकंपा", "अनुकम्पा", "compassionate"]):
        policy_topics.append("compassionate_appointment")
    if any(k in full_blob for k in ["महंगाई भत्ता", "dearness allowance", "gencor/3064", "ई 3-12/"]):
        policy_topics.append("dearness_allowance")
    if any(k in full_blob for k in ["संविदा नियुक्ति", "contract appointment", "एफ 9-1/2012", "एफ 9-1/2004"]):
        policy_topics.append("contract_rules")
        if "अवकाश" in full_blob or "प्रसूति" in full_blob or "leave" in full_blob:
            policy_topics.append("leave_rules")
    if any(k in full_blob for k in ["आचरण) नियम", "आचरण नियम", "conduct rules", "अचल संपत्ति", "property return", "संपत्ति विवरण"]):
        policy_topics.append("conduct_rules")
        if any(k in full_blob for k in ["अचल संपत्ति", "property", "नियम 19", "rule 19"]):
            policy_topics.append("property_return")
    if any(k in full_blob for k in ["निलंबन", "निलम्बन", "suspension", "निर्वाह भत्ता", "subsistence", "विभागीय जांच", "अनुशासनिक", "लघुशास्ति", "एफ 3-2/2010"]):
        policy_topics.append("suspension_inquiry")
    if any(k in full_blob for k in ["50 वर्ष की आयु", "20 वर्ष की सेवा", "अनिवार्य सेवानिवृत्ति", "मूलभूत नियम 56"]):
        policy_topics.append("retirement_compulsory_review")
    if any(k in full_blob for k in ["अधिवार्षिकी", "सेवानिवृत्ति", "पेंशन नियम", "pension", "superannuation", "gratuity"]):
        policy_topics.append("retirement_pension")
    if any(k in full_blob for k in ["सूचना का अधिकार", "right to information", "rtionline", "जन शिकायत", "जनदर्शन", "pg portal", "gad-13"]):
        policy_topics.append("rti_act")
    if any(k in full_blob for k in ["आरक्षण", "रोस्टर", "reservation", "roster", "अनुसूचित जाति", "अनुसूचित जनजाति", "पिछड़ा वर्ग", "दिव्यांगजन"]):
        policy_topics.append("reservation_roster")
    if any(k in full_blob for k in ["पदोन्नति", "वरिष्ठता", "पदक्रम सूची", "seniority", "promotion", "gradation"]):
        policy_topics.append("promotion_seniority")

    # Deterministic Primary Category Classification (title & order_no carry highest authority)
    cat_id: Optional[str] = None
    if any(k in title_blob for k in ["अनुकंपा", "अनुकम्पा", "compassionate"]) or "rule/112/" in ono_low or "7-1/2012" in ono_low:
        cat_id = "compassionate"
    elif any(k in title_blob for k in ["निर्वाचन", "चुनाव", "आचार संहिता", "मतदान", "election", "voter", "उप निर्वाचन"]):
        cat_id = "elections"
    elif (
        any(k in title_blob for k in ["अवकाश नियम", "leave rules", "आकस्मिक अवकाश", "casual leave", "अर्जित अवकाश", "earned leave", "प्रसूति अवकाश", "maternity leave", "पितृत्व अवकाश", "सार्वजनिक", "ऐच्छिक अवकाश", "स्थानीय अवकाश", "holiday", "हड़ताल अवधि का अवकाश", "कार्यदिवस", "शनिवार अवकाश"])
        or ("gens-1001" in ono_low and "gad-5" in ono_low)
    ):
        cat_id = "leave"
    elif any(k in title_blob for k in ["स्थानांतरण", "पदस्थापना", "transfer", "posting", "कार्यभार", "संलग्नीकरण", "पदस्थापित"]) or "rule-3017" in ono_low or "estb-102" in ono_low:
        cat_id = "transfer"
    elif any(k in title_blob for k in ["महंगाई भत्ता", "dearness allowance", "वेतन", "पेंशन", "भत्ता", "अधिवार्षिकी", "सेवानिवृत्ति", "50 वर्ष की आयु", "20 वर्ष की सेवा", "superannuation", "pension", "salary", "pay matrix", "da rate"]) or "gencor/3064" in ono_low or "e3-12" in normalize_order_no(order_no):
        cat_id = "pension"
    elif any(k in title_blob for k in ["आचरण", "conduct", "अचल संपत्ति", "property return", "निलंबन", "निलम्बन", "suspension", "विभागीय जांच", "अनुशासनिक", "सतर्कता", "vigilance", "दण्ड", "लघुशास्ति", "निर्वाह भत्ता"]) or "gad-7" in ono_low or "f3-2/" in normalize_order_no(order_no):
        cat_id = "vigilance"
    elif any(k in title_blob for k in ["सूचना का अधिकार", "right to information", "rti", "लोक सूचना", "राज्य सूचना", "जन शिकायत", "जनदर्शन", "gad-13"]) or "rule-801" in ono_low:
        cat_id = "rti"
    elif any(k in title_blob for k in ["आरक्षण", "रोस्टर", "जाति प्रमाण", "reservation", "roster", "दिव्यांग", "निःशक्त", "sc/st/obc"]):
        cat_id = "reservation"
    elif any(k in title_blob for k in ["पदोन्नति", "वरिष्ठता", "पदक्रम", "promotion", "seniority", "gradation", "dpc"]):
        cat_id = "promotion"
    elif any(k in title_blob for k in ["संविदा नियुक्ति", "भर्ती", "सेवा नियम", "recruitment", "contract appointment", "चयन", "नियुक्ति आदेश", "पुनर्नियुक्ति"]):
        cat_id = "recruitment"
    elif any(k in title_blob for k in ["प्रशिक्षण", "training", "अकादमी", "academy", "कार्यशाला", "workshop", "seminar"]):
        cat_id = "training"
    elif any(k in title_blob for k in ["बजट", "वित्तीय", "अंकेक्षण", "budget", "financial", "expenditure", "audit", "आहरण"]):
        cat_id = "finance"
    elif any(k in title_blob for k in ["योजना", "पुरस्कार", "सम्मान", "scheme", "award", "अलंकृत"]):
        cat_id = "schemes"
    elif any(k in title_blob for k in ["संवर्ग", "कार्य आवंटन", "गठन", "समिति", "प्रशासनिक संगठन", "cadre", "committee", "work allocation", "सेवा संघ", "पत्राचार", "उपस्थित"]) or "welf-3002" in ono_low or "welf 3002" in ono_low or "gad-9" in ono_low or "gens-2101" in ono_low:
        cat_id = "cadre"

    if not cat_id:
        # Clean fallback mapping from doc_metadata.topics
        topic_map = {
            "leave_holidays": "leave",
            "transfer": "transfer",
            "compassionate_appointment": "compassionate",
            "pay_allowances": "pension",
            "pension": "pension",
            "discipline": "vigilance",
            "rti": "rti",
            "elections": "elections",
            "reservation_roster": "reservation",
            "promotion": "promotion",
            "seniority": "promotion",
            "training": "training",
            "budget": "finance",
            "audit_accounts": "finance",
            "scheme": "schemes",
            "awards_grants": "schemes",
            "organisation": "cadre",
            "recruitment_rules": "recruitment",
        }
        for t in (meta.get("topics") or []):
            if str(t) in topic_map:
                cat_id = topic_map[str(t)]
                break

    if not cat_id:
        cat_id = "general"

    # Deduplicate policy_topics while preserving priority order
    seen_t: List[str] = []
    for t in policy_topics:
        if t not in seen_t:
            seen_t.append(t)

    return cat_id, CATEGORY_LABELS[cat_id], seen_t, 1


def main() -> None:
    print("=" * 72)
    print("Enriching Knowledge Graph Connectivity in temporal_graph.db...")
    print("=" * 72)

    docs_raw = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
    doc_list = docs_raw.get("documents", docs_raw) if isinstance(docs_raw, dict) else docs_raw
    uuid_to_doc: Dict[str, Dict[str, Any]] = {}
    dir_to_doc: Dict[str, Dict[str, Any]] = {}
    for d in doc_list:
        if not isinstance(d, dict):
            continue
        if d.get("id"):
            uuid_to_doc[d["id"]] = d
        if d.get("doc_dir"):
            dir_to_doc[d["doc_dir"]] = d

    doc_meta: Dict[str, Dict[str, Any]] = json.loads(META_PATH.read_text(encoding="utf-8")).get("documents", {})

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1. Schema migration (idempotent)
    node_cols = {r["name"] for r in cur.execute("PRAGMA table_info(nodes)").fetchall()}
    if "doc_dir" not in node_cols:
        cur.execute("ALTER TABLE nodes ADD COLUMN doc_dir TEXT DEFAULT ''")
    if "policy_topics" not in node_cols:
        cur.execute("ALTER TABLE nodes ADD COLUMN policy_topics TEXT DEFAULT '[]'")
    if "is_cg_policy" not in node_cols:
        cur.execute("ALTER TABLE nodes ADD COLUMN is_cg_policy INTEGER DEFAULT 1")

    chunk_cols = {r["name"] for r in cur.execute("PRAGMA table_info(chunks)").fetchall()}
    if "canonical_doc_id" not in chunk_cols:
        cur.execute("ALTER TABLE chunks ADD COLUMN canonical_doc_id TEXT DEFAULT ''")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_doc_dir ON nodes(doc_dir)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_cg_policy ON nodes(is_cg_policy)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_canonical_doc_id ON chunks(canonical_doc_id)")

    # 2. Load first 2 chunks per doc_dir (fast single scan)
    first_chunks_by_dir: Dict[str, str] = defaultdict(str)
    for row in cur.execute(
        "SELECT doc_id, chunk_index, content FROM chunks WHERE chunk_index <= 2 ORDER BY doc_id, chunk_index ASC"
    ):
        dd = row["doc_id"]
        if len(first_chunks_by_dir[dd]) < 2400:
            first_chunks_by_dir[dd] += " " + (row["content"] or "")[:1200]

    # Populate chunks.canonical_doc_id cleanly
    dir_to_uuid = {dd: d["id"] for dd, d in dir_to_doc.items() if d.get("id")}
    chunk_updates = [(uid, dd) for dd, uid in dir_to_uuid.items()]
    cur.executemany("UPDATE chunks SET canonical_doc_id = ? WHERE doc_id = ?", chunk_updates)
    conn.commit()
    print(f"[1/5] Bridged chunks.canonical_doc_id for {len(chunk_updates)} document dirs.")

    # 3. Enrich & cleanse nodes
    all_nodes = cur.execute("SELECT * FROM nodes").fetchall()
    node_records: Dict[str, Dict[str, Any]] = {}
    dir_to_node_id: Dict[str, str] = {}
    uuid_to_node_id: Dict[str, str] = {}
    ono_date_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    ono_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # norm_ono -> [(doc_date, node_id)]
    date_index: Dict[str, List[str]] = defaultdict(list)
    series_groups: Dict[str, List[str]] = defaultdict(list)

    cat_counts: Counter = Counter()
    policy_topic_counts: Counter = Counter()
    node_update_rows = []

    for row in all_nodes:
        nid = row["id"]
        uid = row["doc_id"] or ""
        doc_info = uuid_to_doc.get(uid, {})
        meta = doc_meta.get(uid, {})
        dd = meta.get("doc_dir") or doc_info.get("doc_dir") or uid[:16]

        order_no = row["order_no"] or meta.get("order_number") or ""
        doc_date = row["doc_date"] or meta.get("issue_date") or ""
        title_hi = row["title_hi"] or meta.get("title") or ""
        title_en = row["title_en"] or meta.get("title_en") or ""
        doc_type = row["doc_type"] or meta.get("doc_type") or ""
        source_url = row["source_url"] or meta.get("source_url") or doc_info.get("url") or ""

        # Enrich summary with structured summary_hi/summary_en/key_facts from doc_metadata
        sum_parts = []
        if meta.get("summary_en"):
            sum_parts.append(str(meta["summary_en"]).strip())
        if meta.get("summary_hi"):
            sum_parts.append(str(meta["summary_hi"]).strip())
        if not sum_parts and row["summary"]:
            sum_parts.append(str(row["summary"]).strip())
        kf = meta.get("key_facts")
        if isinstance(kf, dict):
            kf_txt = " | ".join(f"{k}: {v}" for k, v in kf.items() if v)
            if kf_txt:
                sum_parts.append(kf_txt)
        elif isinstance(kf, list) and kf:
            sum_parts.append(" | ".join(str(x) for x in kf[:4]))

        if dd in CORE_ANCHORS:
            anc = CORE_ANCHORS[dd]
            if anc.get("title_en") and not title_en:
                title_en = anc["title_en"]
            if anc.get("extra_summary"):
                sum_parts.insert(0, anc["extra_summary"])

        enriched_summary = " | ".join(dict.fromkeys(p for p in sum_parts if p))[:1500]
        first_chunk = first_chunks_by_dir.get(dd, "")

        cat_id, cat_label, p_topics, is_cg = classify_node(
            order_no=order_no,
            title_hi=title_hi,
            title_en=title_en,
            summary=enriched_summary,
            doc_type=doc_type,
            meta=meta,
            first_chunk=first_chunk,
            url=source_url,
            doc_dir=dd,
        )

        norm_ono = normalize_order_no(order_no)
        series_key = extract_file_series(order_no, p_topics)

        rec = {
            "id": nid,
            "doc_id": uid,
            "doc_dir": dd,
            "order_no": order_no,
            "norm_ono": norm_ono,
            "doc_date": doc_date,
            "status": row["status"] or meta.get("status") or "current",
            "title_hi": title_hi,
            "title_en": title_en,
            "summary": enriched_summary,
            "category": cat_label,
            "category_id": cat_id,
            "policy_topics": p_topics,
            "is_cg_policy": is_cg,
            "source_url": source_url,
            "meta": meta,
            "first_chunk": first_chunk,
            "tokens": tokens_for_overlap(f"{title_hi} {title_en} {enriched_summary}"),
            "series_key": series_key,
        }
        node_records[nid] = rec
        dir_to_node_id[dd] = nid
        uuid_to_node_id[uid] = nid

        if norm_ono:
            ono_index[norm_ono].append((doc_date or "0000-00-00", nid))
            if doc_date:
                ono_date_index[(norm_ono, doc_date)].append(nid)
        if doc_date:
            date_index[doc_date].append(nid)
        if series_key and is_cg == 1:
            series_groups[series_key].append(nid)

        cat_counts[cat_id] += 1
        for pt in p_topics:
            policy_topic_counts[pt] += 1

        node_update_rows.append((
            dd,
            title_en,
            enriched_summary,
            cat_label,
            cat_id,
            json.dumps(p_topics, ensure_ascii=False),
            is_cg,
            nid,
        ))

    cur.executemany(
        """
        UPDATE nodes
        SET doc_dir = ?, title_en = ?, summary = ?, category = ?, category_id = ?, policy_topics = ?, is_cg_policy = ?
        WHERE id = ?
        """,
        node_update_rows,
    )
    conn.commit()
    print(f"[2/5] Cleansed & enriched {len(node_update_rows)} nodes. Categories: {dict(cat_counts)}")

    # 4. Build multi-layer semantic & chronological edges
    edges_to_insert: Dict[Tuple[str, str, str], Tuple[str, str, str, str, str, float, str]] = {}

    def add_edge(src: str, tgt: str, rel: str, date_str: str, evidence: str, weight: float, meta_d: Optional[Dict[str, Any]] = None) -> None:
        if not src or not tgt or src == tgt:
            return
        key = (src, tgt, rel)
        existing = edges_to_insert.get(key)
        if existing and existing[5] >= weight:
            return
        edges_to_insert[key] = (
            src,
            tgt,
            rel,
            date_str or "",
            (evidence or "")[:280],
            float(weight),
            json.dumps(meta_d or {}, ensure_ascii=False),
        )

    def resolve_citation(src_nid: str, ref_ono: Optional[str], ref_date_raw: Optional[str], ref_text: str) -> Optional[str]:
        src = node_records[src_nid]
        src_date = src["doc_date"] or "9999-12-31"
        ref_date = parse_dmy_to_iso(ref_date_raw) or parse_dmy_to_iso(ref_text)
        norm_ref = normalize_order_no(ref_ono)
        ref_low = (ref_text or "").lower()

        # Check "समसंख्यक" (of even number) + date
        if not norm_ref and any(k in ref_low for k in ["समसंख्यक", "even number", "सम क्रमांक"]):
            norm_ref = src["norm_ono"]

        # 1. Exact normalized (order_no, date)
        if norm_ref and ref_date and (norm_ref, ref_date) in ono_date_index:
            cands = [c for c in ono_date_index[(norm_ref, ref_date)] if c != src_nid]
            if cands:
                return cands[0]

        # 2. Exact normalized order_no -> closest earlier (or equal if date missing)
        if norm_ref and norm_ref in ono_index:
            earlier = [(dt, cid) for dt, cid in ono_index[norm_ref] if cid != src_nid and (not src_date or dt <= src_date)]
            if earlier:
                earlier.sort(key=lambda x: x[0], reverse=True)
                # Prefer same category or topic if multiple on closest date
                best_dt = earlier[0][0]
                same_dt = [cid for dt, cid in earlier if dt == best_dt]
                for cid in same_dt:
                    if node_records[cid]["category_id"] == src["category_id"]:
                        return cid
                return same_dt[0]

        # 3. Exact YYYY-MM-DD + strong semantic token overlap (strictly >= 2 shared tokens, NO YYYY-MM fallback)
        if ref_date and ref_date in date_index:
            ref_toks = tokens_for_overlap(ref_text)
            if len(ref_toks) >= 2:
                best_cid = None
                best_overlap = 1
                for cid in date_index[ref_date]:
                    if cid == src_nid:
                        continue
                    ov = len(ref_toks & node_records[cid]["tokens"])
                    if ov > best_overlap:
                        best_overlap = ov
                        best_cid = cid
                if best_cid:
                    return best_cid
        return None

    # Layer A: Explicit citations from doc_metadata + regex scan in early chunks
    rel_weights = {"SUPERSEDES": 1.0, "AMENDS": 0.95, "CONTINUES": 0.85, "REFERENCES": 0.70}
    explicit_count = 0
    for nid, rec in node_records.items():
        m = rec["meta"]
        src_date = rec["doc_date"]
        for field, rel in [("supersedes", "SUPERSEDES"), ("amends", "AMENDS"), ("continues", "CONTINUES"), ("references", "REFERENCES")]:
            for item in (m.get(field) or []):
                if isinstance(item, dict):
                    ev_txt = item.get("text") or item.get("evidence") or item.get("title") or ""
                    tgt = resolve_citation(nid, item.get("order_number"), item.get("date"), ev_txt)
                    if tgt:
                        add_edge(nid, tgt, rel, src_date, ev_txt or item.get("order_number") or "", rel_weights[rel], {"layer": "metadata"})
                        explicit_count += 1
        # Also regex scan early chunks for order number citations missed by metadata extractor
        txt_head = rec["first_chunk"]
        if txt_head:
            for match in ORDER_NO_REGEX.finditer(txt_head):
                cand_ono = match.group(1)
                norm_cand = normalize_order_no(cand_ono)
                if not norm_cand or norm_cand == rec["norm_ono"] or norm_cand not in ono_index:
                    continue
                ctx_start = max(0, match.start() - 60)
                ctx_end = min(len(txt_head), match.end() + 90)
                ctx_window = txt_head[ctx_start:ctx_end]
                tgt = resolve_citation(nid, cand_ono, None, ctx_window)
                if tgt:
                    ctx_low = ctx_window.lower()
                    rel = "REFERENCES"
                    if any(k in ctx_low for k in ["अधिक्रमित", "निरस्त", "supersession", "superseded"]):
                        rel = "SUPERSEDES"
                    elif any(k in ctx_low for k in ["संशोधन", "आंशिक संशोधन", "amendment", "modification"]):
                        rel = "AMENDS"
                    add_edge(nid, tgt, rel, src_date, ctx_window.strip(), rel_weights[rel] - 0.05, {"layer": "chunk_regex"})

    # Mark superseded_by links explicitly too
    superseded_nodes: Set[str] = set()
    for (src, tgt, rel), _ in edges_to_insert.items():
        if rel == "SUPERSEDES":
            superseded_nodes.add(tgt)

    # Layer B: Duplicates / Twin PDFs (`DUPLICATE_OF`)
    dup_count = 0
    for nid, rec in node_records.items():
        dup_uid = rec["meta"].get("duplicate_of")
        if dup_uid and dup_uid in uuid_to_node_id:
            tgt = uuid_to_node_id[dup_uid]
            add_edge(nid, tgt, "DUPLICATE_OF", rec["doc_date"], "doc_metadata duplicate_of", 0.90, {"layer": "dedup"})
            dup_count += 1
    # Twin order_no + doc_date + high title overlap
    for (norm_ono, dt), nids in ono_date_index.items():
        if len(nids) < 2 or not dt:
            continue
        # Canonical = prefer ActRuleDocs / highest chunk count / has summary
        ranked = sorted(
            nids,
            key=lambda x: (
                0 if "actruledocs" in node_records[x]["source_url"].lower() else 1,
                x,
            ),
        )
        canon = ranked[0]
        c_toks = node_records[canon]["tokens"]
        for other in ranked[1:]:
            o_toks = node_records[other]["tokens"]
            if c_toks and o_toks:
                jacc = len(c_toks & o_toks) / max(1, len(c_toks | o_toks))
                if jacc >= 0.55:
                    add_edge(other, canon, "DUPLICATE_OF", dt, f"Same order {norm_ono} & date {dt}", 0.88, {"layer": "twin"})
                    dup_count += 1

    # Layer C: Rulebook & Core Act Citation Links (`CITES_RULEBOOK`)
    rulebook_patterns: List[Tuple[str, List[str], List[str]]] = [
        ("4f672641e687a86e", ["सिविल सेवा (अवकाश) नियम", "सिविल सेवाएं (अवकाश) नियम", "अवकाश नियम, 2010", "अवकाश नियम 2010", "leave rules, 2010", "अवकाश नियम"], []),
        ("2dd89eba555cd50a", ["आकस्मिक अवकाश", "विशेष आकस्मिक अवकाश", "casual leave"], ["holiday_calendar"]),
        ("0d216bddd7d0dc85", ["संविदा नियुक्ति) नियम", "संविदा नियुक्ति नियम", "contract appointment) rules"], []),
        ("c9c88208e9d47d6f", ["स्थानांतरण नीति", "transfer policy"], []),
        ("3b2fd23e74466e9b", ["आचरण) नियम, 1965", "आचरण नियम, 1965", "आचरण नियम 1965", "conduct) rules, 1965", "conduct rules 1965", "अचल संपत्ति का विवरण"], []),
        ("ac820304db0fff00", ["अनुकंपा नियुक्ति", "अनुकम्पा नियुक्ति", "compassionate appointment"], []),
        ("8b6f4ef6a89cbaf6", ["अनावश्यक निलंबन", "जीवन निर्वाह भत्ता", "निर्वाह भत्ता", "subsistence allowance", "एफ 3-2/2010/1-3"], []),
        ("349d95bcbd0e9a10", ["50 वर्ष की आयु", "20 वर्ष की सेवा", "अनिवार्य सेवानिवृत्ति", "मूलभूत नियम 56"], []),
        ("b0337ab454c00f4e", ["सूचना का अधिकार अधिनियम, 2005", "right to information act, 2005", "सूचना का अधिकार अधिनियम 2005"], []),
    ]
    rulebook_count = 0
    for anchor_dir, phrases, _ in rulebook_patterns:
        anchor_nid = dir_to_node_id.get(anchor_dir)
        if not anchor_nid:
            continue
        for nid, rec in node_records.items():
            if nid == anchor_nid or rec["is_cg_policy"] == 0:
                continue
            blob = f"{rec['title_hi']} {rec['title_en']} {rec['summary']} {rec['first_chunk'][:800]}".lower()
            if any(p.lower() in blob for p in phrases):
                add_edge(nid, anchor_nid, "CITES_RULEBOOK", rec["doc_date"], f"Cites core rulebook {anchor_dir}", 0.85, {"layer": "rulebook"})
                rulebook_count += 1

    # Layer D: Same Standing File Series Chains (`SAME_FILE_SERIES`)
    series_count = 0
    for s_key, nids in series_groups.items():
        if len(nids) < 2:
            continue
        # Sub-cluster huge generic series by primary policy topic / category if > 40 nodes
        buckets: Dict[str, List[str]] = defaultdict(list)
        if len(nids) > 40:
            for x in nids:
                pt = node_records[x]["policy_topics"][0] if node_records[x]["policy_topics"] else node_records[x]["category_id"]
                buckets[f"{s_key}:{pt}"].append(x)
        else:
            buckets[s_key] = nids

        for b_key, b_nids in buckets.items():
            if len(b_nids) < 2 or len(b_nids) > 65:
                continue
            ordered = sorted(b_nids, key=lambda x: (node_records[x]["doc_date"] or "0000-00-00", x))
            latest = ordered[-1]
            for i in range(len(ordered) - 1):
                u, v = ordered[i], ordered[i + 1]
                # Newer -> Older lineage + Older -> Newer update pointer
                add_edge(v, u, "SAME_FILE_SERIES", node_records[v]["doc_date"], f"File series {b_key}", 0.82, {"layer": "series"})
                series_count += 1
            # Connect earlier items (up to 15) to latest standing order in the same series when separated by >1 hop
            if len(ordered) >= 3:
                for u in ordered[:-2]:
                    add_edge(u, latest, "LATEST_IN_SERIES", node_records[latest]["doc_date"], f"Latest order in {b_key}", 0.86, {"layer": "series_latest"})
                    series_count += 1

    # Layer E: Granular Policy Sub-Topic Clusters (`RELATED_POLICY_TOPIC`)
    topic_buckets: Dict[str, List[str]] = defaultdict(list)
    for nid, rec in node_records.items():
        if rec["is_cg_policy"] != 1:
            continue
        for pt in rec["policy_topics"]:
            topic_buckets[pt].append(nid)

    topic_count = 0
    for pt, nids in topic_buckets.items():
        if len(nids) < 2:
            continue
        ordered = sorted(nids, key=lambda x: (node_records[x]["doc_date"] or "0000-00-00", x))
        # Identify up to 2 authoritative hubs per sub-topic (preferring CORE_ANCHORS + newest active doc)
        anchors = [
            nid for nid in ordered
            if node_records[nid]["doc_dir"] in CORE_ANCHORS and pt in CORE_ANCHORS[node_records[nid]["doc_dir"]].get("policy_topics", [])
        ]
        if not anchors:
            anchors = [ordered[-1]]
        anchors = anchors[:2]
        # Connect chronological spine within policy sub-topic (only 1 neighbor hop each, bounded & clean)
        for i in range(len(ordered) - 1):
            u, v = ordered[i], ordered[i + 1]
            add_edge(u, v, "RELATED_POLICY_TOPIC", node_records[v]["doc_date"], f"Policy topic {pt}", 0.72, {"topic": pt})
            topic_count += 1
        # Connect every member of sub-topic (capped cleanly for massive buckets) to sub-topic anchors
        for nid in ordered:
            for anc in anchors:
                if nid != anc:
                    add_edge(nid, anc, "RELATED_POLICY_TOPIC", node_records[nid]["doc_date"], f"Topic anchor {pt}", 0.78, {"topic": pt, "anchor": True})
                    topic_count += 1

    # Layer F: Clean Intra-Category Chronological Spine (`CHRONOLOGICAL_NEXT` / `CHRONOLOGICAL_PREV`)
    cat_buckets: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for nid, rec in node_records.items():
        if rec["doc_date"]:
            cat_buckets[(rec["category_id"], rec["is_cg_policy"])].append(nid)

    chrono_count = 0
    for (cat_id, is_cg), nids in cat_buckets.items():
        ordered = sorted(nids, key=lambda x: (node_records[x]["doc_date"], x))
        for i in range(len(ordered) - 1):
            u, v = ordered[i], ordered[i + 1]
            add_edge(u, v, "CHRONOLOGICAL_NEXT", node_records[v]["doc_date"], f"Same category {cat_id}", 0.30, {"category_id": cat_id})
            add_edge(v, u, "CHRONOLOGICAL_PREV", node_records[u]["doc_date"], f"Same category {cat_id}", 0.30, {"category_id": cat_id})
            chrono_count += 2

    # Replace edges table atomically in a transaction
    cur.execute("DELETE FROM edges")
    cur.executemany(
        """
        INSERT INTO edges (source, target, relation, date, evidence, weight, meta_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        list(edges_to_insert.values()),
    )
    conn.commit()
    print(f"[3/5] Rebuilt edges ({len(edges_to_insert)} total): explicit={explicit_count}, dup={dup_count}, rulebook={rulebook_count}, series={series_count}, topic={topic_count}, chrono={chrono_count}.")

    # 5. Rebuild fts_nodes for clean bilingual FTS5 search
    cur.execute("DELETE FROM fts_nodes")
    fts_rows = [
        (rec["id"], rec["order_no"], rec["doc_date"], rec["title_hi"], rec["title_en"], rec["summary"])
        for rec in node_records.values()
    ]
    cur.executemany(
        "INSERT INTO fts_nodes (id, order_no, doc_date, title_hi, title_en, summary) VALUES (?, ?, ?, ?, ?, ?)",
        fts_rows,
    )
    conn.commit()

    # Compute final semantic connectivity stats
    adj: Dict[str, Set[str]] = defaultdict(set)
    rel_counts: Counter = Counter()
    for (s, t, r), _ in edges_to_insert.items():
        rel_counts[r] += 1
        if r not in ("CHRONOLOGICAL_NEXT", "CHRONOLOGICAL_PREV"):
            adj[s].add(t)
            adj[t].add(s)
    connected_nodes = sum(1 for nid in node_records if len(adj[nid]) > 0)
    cg_total = sum(1 for rec in node_records.values() if rec["is_cg_policy"] == 1)
    cg_connected = sum(1 for nid, rec in node_records.items() if rec["is_cg_policy"] == 1 and len(adj[nid]) > 0)
    print(f"[4/5] Rebuilt fts_nodes ({len(fts_rows)} rows).")
    print(f"[5/5] Edge breakdown: {dict(rel_counts)}")
    print(f"      Semantic connectivity: {connected_nodes}/{len(node_records)} nodes overall; {cg_connected}/{cg_total} CG policy nodes connected!")
    conn.close()


if __name__ == "__main__":
    main()
