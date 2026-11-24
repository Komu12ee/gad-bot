"""Read a question before searching — METADATA_LAYER_PLAN.md step 4 (§7).

plan(question) -> Plan: the fiscal years, plain years, document types, topics and
districts the question STATES, and what kind of answer it wants:

  catalogue   "GAD ke published reports kaun se hain?", "latest notifications"
              -> the matching documents from doc_metadata.json join the evidence
  count       "Durg domicile ke kitne Section Officers hain?" -> a count over the table
  number      "2026-27 में पूंजीगत व्यय कितना है?" -> table rows rank higher
  general     everything else

Rules only (milliseconds, no model). Only values the question names are used, and
search uses them to RANK (score_factor), never to exclude — a filter that finds
nothing must not turn into "not in the documents".
"""

from __future__ import annotations

from datetime import date
from importlib import import_module
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
aliases = import_module("02_optimization.aliases")
dm = import_module("01_preprocessing.doc_metadata")

_DOC_WORDS = [
    ("circular", r"परिपत्र|सर्कुलर|circulars?"),
    ("order", r"आदेश|\borders?\b"),
    ("notification", r"अधिसूचना|notifications?"),
    ("memo", r"ज्ञाप|memorandum|\bO\.?M\.?\b"),
    ("list", r"सूची|\blists?\b|civil\s*list"),
    ("annual_report", r"प्रतिवेदन|रिपोर्ट|reports?"),
    ("act_rule", r"अधिनियम|नियमावली|\bacts?\b|\brules\b"),
    ("policy", r"नीति|\bpolic(?:y|ies)\b"),
    ("budget_speech", r"बजट\s*भाषण|budget\s*speech"),
    ("tender", r"निविदा|tenders?"),
    ("vacancy_notice", r"रिक्त|vacanc|भर्ती\s*विज्ञापन"),
    ("application_form", r"प्रपत्र|प्रारूप|\bforms?\b"),
    ("press_note", r"प्रेस\s*(?:नोट|विज्ञप्ति)|press\s*(?:note|release)"),
]
_TOPIC_WORDS = [
    ("transfer", r"स्थानांतरण|स्थानान्तरण|tabadla|transfer"),
    ("promotion", r"पदोन्नति|promotion"),
    ("seniority", r"वरिष्ठता|पदक्रम|seniority|gradation"),
    ("deputation", r"प्रतिनियुक्ति|deputation"),
    ("compassionate_appointment", r"अनुकंपा|अनुकम्पा|compassionate|anukampa"),
    ("reservation_roster", r"आरक्षण|रोस्टर|reservation|roster"),
    ("recruitment_rules", r"भर्ती\s*नियम|recruitment\s*rules?"),
    ("leave_holidays", r"अवकाश|छुट्टी|\bleave\b|holiday"),
    ("pay_allowances", r"वेतन|भत्त|\bpay\b|allowance|\bDA\b"),
    ("pension", r"पेंशन|pension"),
    ("discipline", r"अनुशासन|विभागीय\s*जांच|discipline"),
    ("training", r"प्रशिक्षण|training"),
    ("awards_grants", r"पुरस्कार|सम्मान|अनुदान|award|grant"),
    ("rti", r"सूचना\s*का\s*अधिकार|\bRTI\b"),
    ("organisation", r"संरचना|structure|organi[sz]ation|कक्ष"),
    ("budget", r"बजट|budget|व्यय|expenditure|राजस्व|revenue|deficit|घाटा|GSDP"),
    ("scheme", r"योजना|yojana|scheme|mission|मिशन"),
]
_LATEST = re.compile(r"latest|नवीनतम|ताज़ा|ताजा|हाल\s*(?:ही|में)|recent|नए|naye|nayi|current|वर्तमान", re.I)
# "What changed / what was it before" — NOT a bare "पहले": "आवेदन से पहले क्या करना
# होगा" (what to do BEFORE applying) is not a question about old orders.
_HISTORY = re.compile(r"पुरान|पिछल|history|इतिहास|बदलाव|परिवर्तन\s+(?:हुआ|हुए|किया|क्या)|"
                      r"changed|earlier|previous|पहले\s+(?:\S+\s+){0,2}?(?:था|थी|थे)(?![ऀ-ॿ])|"
                      r"(?:was|were)\s+(?:\S+\s+){0,3}?before", re.I)
_COUNT = re.compile(r"कितने|कितनी\s+(?:संख्या|बार)|kitne|how\s+many|संख्या\s+(?:कितनी|बताएं)|number\s+of", re.I)
_MONEY_COUNT = re.compile(r"कितने\s+(?:रुपये|रुपए|करोड़|लाख|प्रतिशत)|kitne\s+(?:rupaye|rupees|crore|lakh)", re.I)
_NUMBER = re.compile(r"कितना|कितनी|kitna|kitni|how\s+much|राशि|बजट|प्रतिशत|%|percent|allocation|आवंटन|"
                     r"प्रावधान|deficit|घाटा|GSDP|receipts?|expenditure|व्यय|surplus", re.I)
# An AMOUNT is asked for: such a question is never a list of documents. (Not "बजट":
# "2025-26 के बजट दस्तावेज़ कौन से हैं?" asks for documents.)
_AMOUNT = re.compile(r"कितना|कितनी|kitna|kitni|how\s+much|राशि|प्रतिशत|%|percent|allocation|आवंटन|"
                     r"प्रावधान|कितने\s+(?:रुपये|रुपए|करोड़|लाख)", re.I)
# Documents NEEDED for something: "अनुकंपा नियुक्ति के लिए कौन से दस्तावेज़ चाहिए"
# asks what to attach to an application, not which orders the department issued.
_NEEDS = re.compile(r"चाहिए|chahiye|chahie|आवश्यक|ज़रूरी|जरूरी|zaruri|required|needed|\bneed\b|"
                    r"लगते|लगेंगे|लगता|संलग्न|attach", re.I)


@dataclass
class Plan:
    intent: str = "general"
    latest: bool = False
    history: bool = False
    years: list = field(default_factory=list)          # fiscal years "2026-2027"
    plain_years: list = field(default_factory=list)    # "2025"
    doc_types: list = field(default_factory=list)
    topics: list = field(default_factory=list)
    districts: list = field(default_factory=list)      # Hindi canonical names


def _word(rx: str) -> str:
    """No match inside a longer Devanagari word ("अधोसंरचना" is not "संरचना");
    suffixes stay allowed ("भत्त" -> भत्ता / भत्ते)."""
    return rf"(?<![ऀ-ॿ])(?:{rx})"


_DOCW = "|".join(rx for _, rx in _DOC_WORDS) + r"|दस्तावेज़?|documents?"
_WHICH = r"kaun\s*-?\s*(?:kaun\s*)?s[ei]|कौन\s*-?\s*(?:कौन\s*)?(?:से|सी)|which"
# A question that asks WHICH DOCUMENTS there are. "civil list kya hoti hai" and
# "seniority list me kaun se details" name a document type but ask about its content.
_ASKS_WHICH = re.compile(
    rf"(?:{_WHICH})\s+(?:\S+\s+){{0,2}}?(?:{_DOCW})"                    # कौन से परिपत्र
    rf"|(?:{_DOCW})\S*\s+(?:{_WHICH}|kaun|कौन)"                         # circulars kaun se
    rf"|(?:latest|नवीनतम|recent|हाल\s+(?:ही\s+)?(?:के|में)|published|प्रकाशित)\s+(?:\S+\s+){{0,2}}?(?:{_DOCW})"
    rf"|list\s+of\s+(?:\S+\s+){{0,2}}?(?:{_DOCW})|(?:{_DOCW})\S*\s+की\s+सूची"
    rf"|(?:show|list|give)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:\S+\s+){{0,2}}?(?:{_DOCW})"
    rf"|(?:सभी|all)\s+(?:\S+\s+){{0,2}}?(?:{_DOCW})\S*\s+(?:दिखा|दें|दीजिए|बताएं|बताइए|show|list)", re.I)


def plan(question: str) -> Plan:
    q = dm.prep(question or "")
    p = Plan()
    p.years = sorted(dm.fiscal_years(q))
    p.plain_years = sorted({y for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d|-)", q)
                            if not any(y in fy for fy in p.years)})
    p.doc_types = [k for k, rx in _DOC_WORDS if re.search(_word(rx), q, re.I)]
    p.topics = [k for k, rx in _TOPIC_WORDS if re.search(_word(rx), q, re.I)]
    p.districts = [aliases.hindi("district", k) for k in aliases.find(q, "district")]
    p.latest = bool(_LATEST.search(q))
    p.history = bool(_HISTORY.search(q))
    if _ASKS_WHICH.search(q) and not _AMOUNT.search(q) and not _NEEDS.search(q):
        p.intent = "catalogue"
    elif _COUNT.search(q) and not _MONEY_COUNT.search(q):
        p.intent = "count"
    elif _NUMBER.search(q):
        p.intent = "number"
    return p


# ── using the plan to rank chunks ────────────────────────────────────────────
# Newer documents rank a little higher, for every question that names no year and
# does not ask about the past. Government orders replace older ones, and a question
# such as "who is the CM?" means today: with no date preference at all, the 2013
# swearing-in notification ranked first for it and the answer depended on which of
# three Chief Ministers' documents the model trusted (live GAD traces, 2026-10-21).
# Mild — a ranking nudge, never a filter: an old rule still in force is still found.
# RECENCY_BOOST=0 turns it off; a question saying "latest/current" gets twice it.
RECENCY_BOOST = float(os.getenv("RECENCY_BOOST", "0.15"))
_RECENCY_FROM = 2000


def recency(payload: dict) -> float | None:
    """0 (2000 or older) .. 1 (this year) from the document's date; a web page with
    no date of its own is the live site as read — current (1)."""
    issued = str(payload.get("issue_date") or "")[:4]
    if issued.isdigit():
        span = max(1, date.today().year - _RECENCY_FROM)
        return max(0.0, min(1.0, (int(issued) - _RECENCY_FROM) / span))
    if payload.get("doc_type") == "web_page" or payload.get("source_type") == "web":
        return 1.0
    return None


def score_factor(pl: Plan, payload: dict) -> float:
    """Multiplier for a candidate chunk's retrieval score. Matches on what the
    question STATES lift a chunk a little; a superseded order sinks (unless the
    question asks about history); a chunk from a year the question did not ask
    about sinks for figure questions. Never zero: nothing is excluded."""
    f = 1.0
    if payload.get("status") == "superseded" and not pl.history:
        f *= 0.7       # mild: if a supersession link were ever wrong, a current rule must still surface
    fys = set(payload.get("fiscal_years") or [])
    if pl.years and fys:
        if fys & set(pl.years):
            f *= 1.15 if payload.get("primary_year") in pl.years else 1.08
        elif pl.intent in ("number", "count"):
            f *= 0.8
    issued = str(payload.get("issue_date") or "")[:4]
    if pl.plain_years and issued in pl.plain_years:
        f *= 1.1       # "2023 के स्थानांतरण आदेश": orders issued in 2023
    if pl.doc_types and payload.get("doc_type") in pl.doc_types:
        f *= 1.1
    if pl.topics and set(payload.get("topics") or []) & set(pl.topics):
        f *= 1.1
    if pl.districts and set(payload.get("districts") or []) & set(pl.districts):
        f *= 1.1
    if pl.intent in ("number", "count") and payload.get("chunk_type") == "table_rows":
        f *= 1.1
    if RECENCY_BOOST and not pl.history and not pl.years and not pl.plain_years:
        r = recency(payload)
        if r is not None:
            f *= 1.0 + RECENCY_BOOST * (2 if pl.latest else 1) * r
    return f
