"""Per-document metadata from rules — METADATA_LAYER_PLAN.md step 1.

One record per document in data/<pipeline>/doc_metadata.json, built from what is
already on disk, cheapest and most reliable source first:

  listing row   what the department's own listing page says about the file —
                subject, order number, date, issuer, last date, category
                (Scraper: listing_rows.jsonl; else the saved listing page)
  manifest      URL, listing page, fetch time, Last-Modified, file hash
  text rules    order number, dates, "विषय:" subject, "संदर्भ:" references,
                supersede / amend / continue phrases (with the sentence as
                evidence), issuing office, central vs state, signatory, fiscal
                years, unit statements, last date, districts, posts, doc type
  OCR outputs   pages, pages the gate dropped, legacy-font pages, tables

Nothing here is guessed by a model (step 3 fills title/type/topic gaps and writes
summaries). A field the rules cannot read stays null, and every derived field
says where it came from (`*_source`). A corpus pass then marks exact duplicates
and orders that a later order says it supersedes.

    PIPELINE=fg2 uv run python core/01_preprocessing/doc_metadata.py              # build / refresh
    PIPELINE=fg2 uv run python core/01_preprocessing/doc_metadata.py --report     # + coverage report
    PIPELINE=fg2 uv run python core/01_preprocessing/doc_metadata.py --show <doc_dir|url>
"""

from __future__ import annotations

from importlib import import_module
import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
aliases = import_module("02_optimization.aliases")

SCHEMA_VERSION = 2
RULES_VERSION = "doc_metadata rules 2"   # 2: Gazette registration line, listing order numbers

_DEV = "ऀ-ॿ"
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
DOC_TYPES = ("circular", "order", "notification", "memo", "letter", "list", "vacancy_notice",
             "application_form", "budget_summary", "budget_table", "budget_speech", "act_rule",
             "policy", "annual_report", "press_note", "tender", "web_page", "other")


# ── text helpers ─────────────────────────────────────────────────────────────
def prep(text: str) -> str:
    """NFC, ASCII digits, one kind of dash and space — what every rule below reads."""
    t = unicodedata.normalize("NFC", text or "").translate(_DEV_DIGITS)
    t = t.replace(" ", " ").replace("–", "-").replace("—", "-").replace("​", "")
    # the VLM OCR writes Markdown emphasis: "**विषय:-** …" must read as "विषय:- …"
    t = t.replace("**", "").replace("__", "")
    return re.sub(r"[ \t]+", " ", t)


def pages_of(text: str) -> list[str]:
    parts = re.split(r"<!--\s*Page\s+\d+\s*-->", text)
    parts = [p for p in parts if p.strip()]
    return parts or [text]


def squash(s: str, limit: int = 300) -> str:
    s = re.sub(r"[#*_`>|]+", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip(" .,:;-|।")
    return s[:limit].strip()


def _clip(m_start: int, text: str, before: int = 0, after: int = 300) -> str:
    return squash(text[max(0, m_start - before):m_start + after], before + after)


# ── dates ────────────────────────────────────────────────────────────────────
_HI_MONTHS = {"जनवरी": 1, "फरवरी": 2, "फ़रवरी": 2, "मार्च": 3, "अप्रैल": 4, "अप्रेल": 4, "मई": 5,
              "जून": 6, "जुलाई": 7, "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10,
              "अक्तूबर": 10, "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12}
_EN_MONTHS = {m: i for i, ms in enumerate(
    [("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",),
     ("jun", "june"), ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"),
     ("oct", "october"), ("nov", "november"), ("dec", "december")], 1) for m in ms}
_HI_M = "|".join(sorted(_HI_MONTHS, key=len, reverse=True))
_EN_M = "|".join(sorted(_EN_MONTHS, key=len, reverse=True))
# same separator twice: "1-2/2026" inside an order number is not a date
_D_NUM = re.compile(r"(?<![\d/-])(\d{1,2})\s?([./-])\s?(\d{1,2})\s?\2\s?((?:19|20)\d{2})(?!\d)")
_D_ISO = re.compile(r"(?<!\d)((?:19|20)\d{2})-(\d{2})-(\d{2})(?!\d)")
_D_HI = re.compile(rf"(?<!\d)(\d{{1,2}})\s*(?:st|nd|rd|th)?\s*({_HI_M})\s*,?\s*((?:19|20)\d{{2}})(?!\d)")
_D_EN_DMY = re.compile(rf"(?<!\d)(\d{{1,2}})\s*(?:st|nd|rd|th)?\s*(?:of\s+)?({_EN_M})\.?\s*,?\s*((?:19|20)\d{{2}})(?!\d)", re.I)
_D_EN_MDY = re.compile(rf"(?<![A-Za-z])({_EN_M})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*((?:19|20)\d{{2}})(?!\d)", re.I)


def _mk(y, m, d):
    try:
        return date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None


def find_dates(text: str) -> list[tuple[int, int, date]]:
    """[(start, end, date)] in text order; impossible dates (31/02, day 61) skipped."""
    out = []
    for m in _D_NUM.finditer(text):
        out.append((m.start(), m.end(), _mk(m.group(4), m.group(3), m.group(1))))
    for m in _D_ISO.finditer(text):
        out.append((m.start(), m.end(), _mk(m.group(1), m.group(2), m.group(3))))
    for m in _D_HI.finditer(text):
        out.append((m.start(), m.end(), _mk(m.group(3), _HI_MONTHS[m.group(2)], m.group(1))))
    for m in _D_EN_DMY.finditer(text):
        out.append((m.start(), m.end(), _mk(m.group(3), _EN_MONTHS[m.group(2).lower()], m.group(1))))
    for m in _D_EN_MDY.finditer(text):
        out.append((m.start(), m.end(), _mk(m.group(3), _EN_MONTHS[m.group(1).lower()], m.group(2))))
    out = [o for o in out if o[2] is not None]
    out.sort()
    keep, last_end = [], -1
    for o in out:                      # overlapping forms of one date: keep the first
        if o[0] >= last_end:
            keep.append(o)
            last_end = o[1]
    return keep


def parse_date(s: str):
    d = find_dates(prep(s or ""))
    return d[0][2] if d else None


def sane(d, max_year: int):
    """Plan §5: 1990 <= year <= fetch year. OCR-garbled years (2071) are dropped."""
    return d if (d is not None and 1990 <= d.year <= max_year) else None


# ── order numbers ────────────────────────────────────────────────────────────
_ORDER_KW = (r"(?:(?P<copy>पृ\.?\s*क्र(?:मांक)?\.?|पृष्ठांकन\s*क्र(?:मांक)?\.?|प\.\s*क्र\.?|पु\.\s*क्र\.?)"
             r"|क्रमांक|क्रमाक|कमांक|कमाक|फ्रमांक|क्रंमाक|क्रं\.?|क्र\."
             r"|F\.?\s*No\.?|File\s+No\.?|Memo\s+No\.?|Order\s+No\.?|O\.\s?M\.\s*No\.?|Letter\s+No\.?"
             r"|(?<![A-Za-z])N[oO0]\.?)")
_ORDER_RE = re.compile(_ORDER_KW + r"\s*[:\-ः]?\s*(?P<num>[^\n]{2,90})")
_ORDER_STOP = re.compile(r"\s+(?:नवा|नया|रायपुर|दिनांक|दिनाँक|दि\.|Dated|dated|DATED|Date|Raipur|"
                         r"New\s+Delhi|नई\s+दिल्ली|प्रति|To|भोपाल|Government|GOVERNMENT|भारत)"
                         rf"(?![A-Za-z{_DEV}])|,\s|\s{{2,}}")   # (no \b: matras are not \w)


# The Gazette's postal registration line — "रजिस्ट्रेशन नंबर … क्रमांक जी. 2-22-
# छत्तीसगढ़ गजट/38 सि. से. भिलाई, दिनांक 30-05-2001" — is printed on EVERY Gazette
# page. It is neither the document's number nor its date (seen on the live GAD bot:
# rules of 2008, 2014 and 2017 all "issued" 30.05.2001).
_GAZETTE_REG = re.compile(r"गजट|राजट|gazette|रजिस्ट्रेशन|रजि\.|पंजीयन|regd\.?\s*no|registration", re.I)


def _gazette_date(page: str, s: int) -> bool:
    """Is the date at `s` the Gazette registration's? Looked for in the same block of
    lines (back to the last blank line, at most 120 characters) — the registration
    line can wrap, but the real notification date sits below a blank line."""
    start = max(0, s - 120)
    blank = list(re.finditer(r"\n[ \t]*\n", page[start:s]))
    if blank:
        start += blank[-1].end()
    return bool(_GAZETTE_REG.search(page[start:s]))


def display_order(s: str | None) -> str | None:
    """An order number as shown after "क्र.": without a keyword of its own (stored
    numbers from before 2026-10-12 can begin with "क्र", giving "क्र. क्र एफ …")."""
    if not s:
        return s
    return re.sub(r"^(?:क्रमांक|क्रं?\.?|No\.?)\s*[:\-ः]?\s*", "", s.strip()) or s


def clean_order(raw: str) -> str | None:
    s = _ORDER_STOP.split(raw or "", maxsplit=1)[0]
    if _GAZETTE_REG.search(s):
        return None
    # a Latin file number followed by Devanagari is the number + the next words
    # ("12040/26/2018-FTCIIR कैब्रिषट"): keep the number
    m = re.match(rf"^([A-Za-z0-9][^{_DEV}]*?/[^{_DEV}]*?)\s+[{_DEV}]", s)
    if m:
        s = m.group(1)
    s = re.sub(r"^(?:N[oO0]\.?|F\.?\s*No\.?|क्रमांक|क्रं?\.?(?=\s|[:\-ः]))\s*[:\-ः]?\s*", "", s.strip())
    s = s.strip(" .,:;|-ः)(][")
    if not (3 <= len(s) <= 60) or not re.search(r"\d", s):
        return None
    if not re.search(r"[/\-]", s):
        return None
    if _D_NUM.fullmatch(s) or re.fullmatch(r"[\d\s\-]{6,}", s):      # a date or a phone number
        return None
    if re.match(r"0\d{2,4}-\d{5,}", s):                               # STD-code phone
        return None
    return s


def norm_order(s: str | None) -> str:
    """Comparable form: no spaces, one dash, upper-case Latin, no leading keyword."""
    if not s:
        return ""
    s = prep(s).upper()
    s = re.sub(r"^(?:NO\.?|N0\.?|F\.?\s*NO\.?|क्रमांक|क्र\.)\s*", "", s)
    return re.sub(r"\s+", "", s).strip(".,:;-")


def find_order_number(page1: str):
    """(order number, match start) — the first order number on page 1; a copy's
    number (पृ.क्र., प.क्र.) only when the original's is not readable."""
    first_copy = None
    for m in _ORDER_RE.finditer(page1):
        kw = page1[m.start():m.start("num")].strip()
        num = clean_order(m.group("num"))
        if not num:
            continue
        if re.match(r"N[oO0]", kw) and "/" not in num:     # "No. of posts 07-12" is not an order no.
            continue
        if m.group("copy"):
            first_copy = first_copy or (num, m.start())
            continue
        return num, m.start()
    # e-office file numbers often stand alone at the start of the order's first
    # sentence: "ESTB-102(1)/4/2024-GAD-8 एतद्द्वारा …"
    m = _EOFFICE.search(page1[:3000])
    if m:
        return m.group(1), m.start()
    return first_copy or (None, -1)


_EOFFICE = re.compile(r"(?<![A-Za-z0-9/])([A-Z][A-Z0-9]{2,}-\s?\d+(?:\(\d+\))?/\d+/(?:19|20)\d{2}-[A-Z][A-Z0-9]*"
                      r"(?:[- ](?:\d+|[A-Z]+(?:\s[IVX]+)?))?)")


def lead_sentence(page1: str, opos: int) -> str | None:
    """The first sentence of an order that has no subject line — "सचिवालय सेवा के
    निम्नलिखित अधिकारियों को … पदस्थ किया जाता है" — after its number and date."""
    starts = [opos] if opos >= 0 else []
    for _, rx in _FORM_WORDS:
        fm = rx.search(page1[:2500])
        if fm:
            starts.append(fm.end())
    if not starts:
        return None
    seg = page1[max(starts):max(starts) + 900]
    seg = _EOFFICE.sub(" ", seg)
    seg = re.sub(r"^[^\n]*?(?:दिनांक|दिनाक|दिनाँक|Dated?)[^\n]*\n", " ", seg, count=1)
    seg = re.split(r"[।|]|:-|हैः|\n\s*\n|\n\|", seg)[0]
    seg = re.sub(r"^[\s\W\d]*(?:क्रमांक|File\s+No\.?)?[^\s]*\d[^\s]*\s+", " ", seg)
    seg = squash(seg, 170)
    # strict: an address line, a date, an order number or a letterhead is not a
    # title — better none (the model step writes one) than a noisy one
    if (len(seg.split()) < 8 or find_dates(seg[:60]) or _ORDER_RE.match(seg) or _EOFFICE.search(seg[:40])
            or re.match(r"(?:नवा|नया)\s*रायपुर|प्रतिलिपि|No\.|Government|Govt|भारत\s+सरकार", seg, re.I)
            or re.search(r"आदेशानुसार|नाम\s+से\s+तथा?|प्रतिलिपि|(?:^|\s)0\d\.\s", seg)
            or not re.match(rf"[{_DEV}A-Za-z]", seg)):
        return None
    return seg


# ── subject, references, links ───────────────────────────────────────────────
_SUBJ_RE = re.compile(r"(?:^|\n|\s)(?:विषय|विपय|सब्जेक्ट|S[uv]b[ij]e[ce]t|SUBJECT|Sub|SUB)\.?\s*[:ःः\-]+\s*")
_REF_RE = re.compile(r"(?:^|\n|\s)(?:संदर्भ|सन्दर्भ|Reference|REFERENCE|Ref|REF)\.?\s*[:ः\-]+\s*")
_BLOCK_END = re.compile(r"\n\s*\n|\n\s*[-*]?\s*(?:संदर्भ|सन्दर्भ|Ref|REF|Reference|महोदय|Sir|Madam|"
                        r"विषयांतर्गत|---|\.-0)")


def _block_after(text: str, rx: re.Pattern, limit: int = 400) -> tuple[str | None, int]:
    m = rx.search(text)
    if not m:
        return None, -1
    rest = text[m.end():m.end() + limit + 200]
    e = _BLOCK_END.search(rest)
    block = squash(rest[:e.start()] if e else rest, limit)
    return (block if len(block) >= 5 else None), m.start()


def find_subject(page1: str) -> str | None:
    return _block_after(page1, _SUBJ_RE, 300)[0]


def find_references(page1: str) -> list[dict]:
    block, _ = _block_after(page1, _REF_RE, 500)
    if not block:
        return []
    out, prev = [], 0
    for s, e, d in find_dates(block):
        piece = block[prev:e]
        om = _ORDER_RE.search(piece)
        out.append({"order_number": clean_order(om.group("num")) if om else None,
                    "date": d.isoformat(), "text": squash(piece, 200)})
        prev = e
    if not out:
        om = _ORDER_RE.search(block)
        out.append({"order_number": clean_order(om.group("num")) if om else None,
                    "date": None, "text": squash(block, 200)})
    return out


_DOC_WORD = re.compile(r"आदेश|परिपत्र|ज्ञाप|अधिसूचना|पत्र|नियम|निर्देश|order|circular|O\.\s?M\.|"
                       r"notification|memorandum|letter|instructions?|guidelines?", re.I)
_LINK_PATTERNS = {
    "supersedes": re.compile(r"अधिक्रमित|अधिक्रमण|अधिक्रांत|अतिष्ठित|निरस्त|in\s+supersession|"
                             r"supersed|rescind|withdrawn", re.I),
    "amends": re.compile(r"आंशिक\s*संशोधन|संशोधन\s+(?:किया|किए|करते)|संशोधित\s+(?:किया|किए|करते)|"
                         r"में\s+(?:निम्नानुसार\s+)?संशोधन|amendment|(?:is|are|stands|hereby)\s+amended|"
                         r"partial\s+modification", re.I),
    "continues": re.compile(r"के\s+(?:अनु)?क्रम\s+में|in\s+continuation\s+of|निरंतरता\s+में", re.I),
}


def find_links(text: str, own: str | None = None) -> dict:
    """Sentences that supersede / amend / continue an earlier order. Kept only when
    the sentence names an order (a document word plus a date or an order number), with
    the sentence itself as evidence."""
    out = {k: [] for k in _LINK_PATTERNS}
    body = text[:12000]
    for sent in re.split(r"(?<=[।|])\s+|\n\s*\n|(?<=\.)\s+(?=[A-Z0-9(])", body):
        if len(sent) < 15 or not _DOC_WORD.search(sent):
            continue
        dates = find_dates(sent)
        # the order the sentence names: not the document's own number (a "File No."
        # often opens the same sentence)
        onum = None
        for om in _ORDER_RE.finditer(sent):
            cand = clean_order(om.group("num"))
            if cand and norm_order(cand) != norm_order(own):
                onum = cand
                break
        if onum is None:
            em = next((m for m in _EOFFICE.finditer(sent) if norm_order(m.group(1)) != norm_order(own)), None)
            onum = em.group(1) if em else None
        if not dates and not onum:
            continue
        for kind, rx in _LINK_PATTERNS.items():
            if rx.search(sent):
                item = {"order_number": onum, "date": dates[0][2].isoformat() if dates else None,
                        "evidence": squash(sent, 300)}
                if item not in out[kind]:
                    out[kind].append(item)
    return {k: v[:5] for k, v in out.items()}


# ── office, scope, signatory ─────────────────────────────────────────────────
_CG_MARK = re.compile(r"छत्तीसगढ़?\s*शासन|छ\.?\s?ग\.?\s*शासन|Government\s+of\s+Chhatt?isgarh|"
                      r"GOVERNMENT\s+OF\s+CHHATT?ISGARH|Chhattisgarh\s+Government", re.I)
_GOI_MARK = re.compile(r"भारत\s+सरकार|Government\s+of\s+India|GOVERNMENT\s+OF\s+INDIA|Govt\.?\s+of\s+India|"
                       r"Ministry\s+of|MINISTRY\s+OF|Department\s+of\s+Personnel|New\s+Delhi|नई\s+दिल्ली")
_DEPT_HI = re.compile(rf"([{_DEV}][{_DEV} ,.()]{{1,60}}?\s*विभाग)(?![{_DEV}])")
_MINISTRY_EN = re.compile(r"(Ministry\s+of\s+[A-Z][A-Za-z,&\s]{2,60}?)(?=\s{2,}|\n|,|\.|\s+(?:Department|"
                          r"Govt|Government|New|North|South|Shastri|Nirman|Kartavya|Room|Dated))", re.I)
_SECTION = re.compile(rf"\(\s*([{_DEV} ]{{2,30}}(?:शाखा|अनुभाग|प्रकोष्ठ))\s*\)")


def find_office(head: str) -> tuple[str | None, str | None, str]:
    """(issuing office, section, scope). Central when a Government-of-India letterhead
    comes before any Chhattisgarh one in the first ~800 characters."""
    h = head[:1500]
    cg, goi = _CG_MARK.search(h), _GOI_MARK.search(h[:800])
    scope = "central" if goi and (not cg or goi.start() < cg.start()) else "state"
    office = None
    if scope == "central":
        m = _MINISTRY_EN.search(h)
        office = squash(m.group(1), 80) if m else None
    else:
        start = cg.end() if cg else 0
        depts = aliases.find(h[start:start + 600], "department")
        if depts:
            office = aliases.hindi("department", max(depts, key=depts.get))
        else:
            m = _DEPT_HI.search(h[start:start + 400])
            if m and len(m.group(1).split()) <= 8:
                office = squash(m.group(1), 80)
    sec = _SECTION.search(h)
    return office, (squash(sec.group(1), 40) if sec else None), scope


_SIG_DIGITAL = re.compile(r"Digitally\s+signed\s+by\s*:?\s*([A-Z][A-Za-z .]{2,60}?)\s*(?:Date|\n|$)")
# "(नाम)" then the post — on the next line or the same one
_SIG_HI = re.compile(rf"\(\s*([{_DEV}][{_DEV} .]{{2,40}})\s*\)\s*([^\n]{{2,90}})")
_SIG_EN = re.compile(r"\(\s*([A-Z][A-Za-z .]{3,40})\s*\)\s*([^\n]{2,90})")
_COPY = re.compile(r"प्रतिलिपि|Copy\s+(?:to|forwarded)|पृ\.?\s*क्र|पृष्ठांकन|प\.\s*क्र", re.I)


def find_signatory(text: str) -> dict | None:
    body = text[:15000]
    cut = _COPY.search(body, 200)
    region = body[:cut.start()] if cut else body
    for rx in (_SIG_HI, _SIG_EN):
        cands = []
        for m in rx.finditer(region):
            posts = aliases.find(m.group(2), "post")
            if posts:
                cands.append({"name": squash(m.group(1), 60),
                              "post": aliases.hindi("post", next(iter(posts)))})
        if cands:
            return cands[-1]
    m = _SIG_DIGITAL.search(body)
    if m:
        return {"name": squash(m.group(1), 60).title(), "post": None}
    return None


# ── years, units, deadlines ──────────────────────────────────────────────────
_FY = re.compile(r"(?<![\d/])((?:19|20)\d{2})\s?[-/]\s?(\d{2}|(?:19|20)\d{2})(?![\d/])")


def fiscal_years(text: str) -> collections.Counter:
    c = collections.Counter()
    for m in _FY.finditer(text):
        a, b = int(m.group(1)), m.group(2)
        end = int(b) if len(b) == 4 else (a // 100) * 100 + int(b)
        if len(b) == 2 and int(b) == 0:
            end = (a // 100 + 1) * 100
        if end == a + 1:
            c[f"{a}-{end}"] += 1
    return c


_UNIT_WORDS = {"करोड़": "crore", "करोड": "crore", "crore": "crore", "crores": "crore",
               "लाख": "lakh", "lakh": "lakh", "lakhs": "lakh", "lac": "lakh", "lacs": "lakh",
               "हजार": "thousand", "हज़ार": "thousand", "thousand": "thousand", "thousands": "thousand",
               "रुपयों": "rupees", "rupees": "rupees"}   # same names as vlm_ocr.find_unit
_UNIT_HI = re.compile(r"(?:राशि|रकम|रुपये|रुपए|रू\.?|रु\.?|₹)\s*[(\[]?\s*(करोड़|करोड|लाख|हजार|हज़ार|रुपयों)"
                      r"\s*(?:रुपये|रुपए|रुपयों)?\s*में|\(\s*(करोड़|करोड|लाख|हजार|हज़ार)\s*(?:रुपये|रुपए)?\s*में\s*\)")
_UNIT_EN = re.compile(r"(?:Rs\.?|₹|INR|Rupees|[Aa]mount|[Ff]igures)\s*\(?\s*in\s+(crores?|lakhs?|lacs?|thousands?|rupees)\b|"
                      r"\(\s*in\s+(crores?|lakhs?|lacs?|thousands?)\s*\)", re.I)


def find_units(text: str) -> tuple[str | None, str | None, str | None, list]:
    """(unit, evidence, confidence, all units) from explicit unit statements only —
    "(राशि करोड़ में)", "(₹ in crore)". Several different units -> no single unit."""
    seen: dict = {}
    for rx in (_UNIT_HI, _UNIT_EN):
        for m in rx.finditer(text):
            word = next(g for g in m.groups() if g)
            unit = _UNIT_WORDS.get(word.lower(), _UNIT_WORDS.get(word))
            if unit and unit not in seen:
                seen[unit] = squash(m.group(0), 60)
    if not seen:
        return None, None, None, []
    if len(seen) == 1:
        u = next(iter(seen))
        return u, seen[u], "high", [u]
    return None, None, "mixed", sorted(seen)


_DEADLINE_KW = re.compile(r"अंतिम\s*(?:तिथि|तारीख|दिनांक)|last\s+date|latest\s+by|on\s+or\s+before|"
                          r"not\s+later\s+than|reach\s+(?:this|the)\s+\w+\s+(?:latest\s+)?by", re.I)
_DEADLINE_HI = re.compile(r"\s*तक\s+[^।\n]{0,50}?(?:प्रस्तुत|भेज|जमा|आवेदन|प्रेषित|उपलब्ध|पहुँच|पहुंच)")


def find_last_date(text: str, issued, max_year: int):
    body = text[:20000]
    cands = []
    for m in _DEADLINE_KW.finditer(body):
        for s, e, d in find_dates(body[m.end():m.end() + 80])[:1]:
            cands.append(d)
    for s, e, d in find_dates(body):
        if _DEADLINE_HI.match(body, e):
            cands.append(d)
    cands = [d for d in cands if sane(d, max_year + 1) and (issued is None or d >= issued)]
    return min(cands) if cands else None


# ── listing rows ─────────────────────────────────────────────────────────────
_H_SUBJECT = re.compile(r"विषय|subject|description|title|विवरण|नाम|name|details|particulars", re.I)
_H_ORDER = re.compile(r"सर्कुलर\s*नंबर|आदेश\s*क्र|क्रमांक\s*एवं|order\s*no|circular\s*no|letter\s*no|"
                      r"memo\s*no|notification\s*no|^no\.?$|reference", re.I)
_H_DATE = re.compile(r"^(?:दिनांक|date|dated|issue\s*date|जारी\s*दिनांक|दिनांक\s*/\s*date)$", re.I)
_H_LAST = re.compile(r"अंतिम|last\s*date|closing", re.I)
_H_BY = re.compile(r"द्वारा|issued\s*by|department|विभाग|office", re.I)
_H_KIND = re.compile(r"प्रकार|type|category|श्रेणी", re.I)
_JUNK_CELL = re.compile(r"^(?:download|view|open|click here|pdf|hindi|english|हिंदी|हिन्दी|डाउनलोड|देखें|-|\d{1,5}\.?|"
                        r"[\d./\-\s]{6,12}|new|नया)$", re.I)


def read_row(row: dict) -> dict:
    """What a listing row says, by column: title, order number, date, issuer, last
    date, category. Without a header row: the longest real text cell is the title and
    a date-like cell the date."""
    cells, heads = row.get("cells") or [], row.get("headers") or []
    out = {"title": None, "order_number": None, "date": None, "issued_by": None,
           "last_date": None, "category": None, "kind": None}
    joined = " ".join(heads)
    for kind, rx in (("circular", r"सर्कुलर|circular"), ("notification", r"अधिसूचना|notification"),
                     ("order", r"आदेश|\border\b")):
        if re.search(rx, joined, re.I):
            out["kind"] = kind            # the listing says what its documents are
            break
    if cells and heads and len(heads) == len(cells):
        for h, c in zip(heads, cells):
            c = squash(c, 400)
            if not c or c in ("-", "--"):
                continue
            if _H_LAST.search(h):
                out["last_date"] = out["last_date"] or parse_date(c)
            elif _H_DATE.search(h.strip()):
                out["date"] = out["date"] or parse_date(c)
            elif _H_ORDER.search(h.strip()):
                # the same test as a number found in the text: a cell that merely holds
                # a digit somewhere ("वर्तमान व्यवसाय : … 2024") became the order number
                out["order_number"] = out["order_number"] or clean_order(c)
            elif _H_SUBJECT.search(h):
                out["title"] = out["title"] or c
            elif _H_BY.search(h):
                out["issued_by"] = out["issued_by"] or c
            elif _H_KIND.search(h):
                out["category"] = out["category"] or c
    if not out["title"]:
        texts = [squash(c, 400) for c in cells]
        texts = [t for t in texts if t and not _JUNK_CELL.match(t)
                 and not (parse_date(t) and len(t) <= 25) and not clean_order(t) == t]
        if texts:
            out["title"] = max(texts, key=len)
        elif not cells:
            out["title"] = squash(row.get("text") or "", 400) or None
    if row.get("near") and not heads:
        # no column headers: the scraper's "near" text (the link's own item, or the
        # label before a "Hindi"/date link) names it better than the row's cells
        out["title"] = squash(row["near"], 300)
    elif out["title"] and len(out["title"]) > 200:
        out["title"] = squash(row["near"], 300) if row.get("near") else None  # a page section, not a row
    if not out["date"]:
        for c in cells:
            d = parse_date(c)
            if d and len(squash(c)) <= 25:
                out["date"] = d
                break
    anchor = squash(row.get("anchor") or "", 200)
    # one row can link several files ("All Books" / "Schedule" / "Scheme"): the link
    # text is what tells them apart
    if (anchor and not _JUNK_CELL.match(anchor) and not clean_order(anchor) and len(anchor) >= 4
            and (not out["title"] or not out["title"].lower().endswith(anchor.lower()))):
        out["title"] = f"{out['title']} — {anchor}" if out["title"] else anchor
    return out


def pick_row(rows: list[dict], parent_url: str) -> dict | None:
    """The most informative row for one document: from its own listing page if
    possible, with headers, the longest."""
    if not rows:
        return None
    def score(r):
        return ((r.get("listing_url") or "") == parent_url, bool(r.get("headers")),
                len(r.get("text") or ""))
    return max(rows, key=score)


# ── doc type, site section, title ────────────────────────────────────────────
_FORM_WORDS = [
    ("memo", re.compile(r"^\W*CIRCULAR\s+MEMORANDUM\W*$", re.M)),
    ("circular", re.compile(r"^\W*(?:परिपत्र|CIRCULAR|Circular|C[TI]RCULAR|TRAINING\s+CIRCUL\w*)\W*$", re.M)),
    # OCR drops the matra ("आदश"); a revised order prints "संशोधित आदेश"; the line
    # often runs on after "//" ("आदेश// नवा रायपुर, दिनांक …")
    ("order", re.compile(r"^\W*(?:/+\s*)?(?:संशोधित\s+)?(?:आदेश|आदश|ORDER|Order)\s*(?:/+|$)", re.M)),
    ("notification", re.compile(r"^\W*(?:अधिसूचना|NOTIFICATION|Notification)\W*$", re.M)),
    ("memo", re.compile(r"^\W*(?:ज्ञाप|ज्ञापन|OFFICE\s+MEMORANDUM|Office\s+Memorandum|MEMORANDUM)\W*$", re.M)),
    ("press_note", re.compile(r"^\W*(?:प्रेस\s+विज्ञप्ति|PRESS\s+(?:NOTE|RELEASE))\W*$", re.M)),
]
_CONTENT_TYPES = [
    ("application_form", re.compile(r"आवेदन\s*(?:पत्र\s*का\s*)?(?:प्रारूप|प्रपत्र)|application\s+form|"
                                    r"^\W*(?:प्रपत्र|प्रारूप|proforma|PROFORMA)\b", re.I | re.M)),
    ("tender", re.compile(r"निविदा|\btender\b|e-procurement|\bRFP\b|\bEOI\b|expression\s+of\s+interest", re.I)),
    ("vacancy_notice", re.compile(r"भर्ती\s*(?:विज्ञापन|हेतु)|रिक्त\s*पद|vacanc|filling\s+up|"
                                  r"on\s+deputation\s+(?:\(.*?\)\s+)?basis|प्रतिनियुक्ति\s+पर\s+(?:भरे|भरने|नियुक्ति)", re.I)),
    ("annual_report", re.compile(r"वार्षिक\s*(?:प्रतिवेदन|रिपोर्ट)|प्रशासकीय\s*प्रतिवेदन|annual\s+report", re.I)),
    ("budget_speech", re.compile(r"बजट\s*भाषण|budget\s+speech", re.I)),
    ("budget_summary", re.compile(r"बजट\s*(?:पत्रों\s*का\s*)?(?:संक्षिप्त\s*विवरण|सारांश|एक\s*दृष्टि)|"
                                  r"budget\s+(?:at\s+a\s+glance|summary|in\s+brief)|key\s+to\s+budget", re.I)),
    ("list", re.compile(r"(?:पदक्रम|वरिष्ठता|ग्रेडेशन|सिविल|चयन|पात्रता)\s*सूची|सूची|civil\s+list|"
                        r"gradation\s+list|seniority\s+list|list\s+of|transfer\s+list", re.I)),
    ("policy", re.compile(r"नीति\b|नीति\s|policy", re.I)),
    ("act_rule", re.compile(r"अधिनियम|नियम,?\s*(?:19|20)\d{2}|नियमावली|\bRules?,?\s*(?:19|20)\d{2}|\bAct,?\s*(?:19|20)\d{2}", re.I)),
]


def doc_type(*, source_type: str, url: str, title: str, subject: str, head: str,
             listing_kind: str | None = None) -> tuple[str, str]:
    """(type, why). The FORM word printed on the document (परिपत्र / आदेश / अधिसूचना…)
    decides, unless what it is about is more specific (a vacancy, a form, a list, a
    budget book)."""
    if source_type == "web":
        return "web_page", "web page"
    path = unquote(urlsplit(url).path).lower()
    about = " ".join(x for x in (title, subject) if x)
    books = ("budget_summary", "budget_speech", "annual_report")
    for kind, rx in _CONTENT_TYPES:
        if kind in ("policy", "act_rule"):
            continue
        # a LETTER (it has a subject line) about an annual report or a budget is not
        # the report or the budget book ("…वार्षिक प्रतिवेदन की अनुशंसाओं पर कार्यवाही")
        if kind in books and subject:
            continue
        if rx.search(about) or (kind in ("application_form",) + books and rx.search(head[:600])):
            return kind, f"about: {kind}"
    if "budget_doc" in path or "/budget" in path:
        if re.search(r"speech|bhashan|भाषण", path):
            return "budget_speech", "url"
        if re.search(r"key\s*to\s*budget|summary|glance|budget-viv", path):
            return "budget_summary", "url"
        return "budget_table", "url"
    if "transfer list" in path or "transfer_list" in path:
        return "list", "url"
    for kind, rx in _FORM_WORDS:
        if rx.search(head[:2500]):
            return kind, f"printed: {kind}"
    for kind, rx in _CONTENT_TYPES:
        if kind in ("policy", "act_rule") and rx.search(about):
            return kind, f"about: {kind}"
    if re.search(r"परिपत्र|circular", about, re.I):
        return "circular", "title"
    if re.search(r"आदेश|\border\b", about, re.I):
        return "order", "title"
    if re.search(r"अधिसूचना|notification", about, re.I):
        return "notification", "title"
    if listing_kind:
        return listing_kind, "listing column"
    if re.search(r"circular|criculars", path):
        return "circular", "url"
    if re.search(r"(?:^|\n)\s*(?:प्रति|To),?\s*$|Dear\s+Sir|महोदय", head[:3000], re.M):
        return "letter", "addressed"
    if re.search(r"rules?_act|/acts?/|/rules?/", path):
        return "act_rule", "url"
    if re.search(r"annual", path):
        return "annual_report", "url"
    if re.search(r"form", path):
        return "application_form", "url"
    return "other", "no rule matched"


_SECTIONS = [("about", r"about|introduction|parichay|परिचय|profile|history|vision|mission|function|work"),
             ("structure", r"structure|organi[sz]ation|organogram|setup|hierarchy|who'?s\s*who|officers"),
             ("rti", r"rti|right\s*to\s*information"), ("contact", r"contact|address|directory|phone"),
             ("tenders", r"tender|nivida|procurement"), ("recruitment", r"recruit|vacanc|bharti|career|job"),
             ("acts_rules", r"act|rule|niyam"), ("schemes", r"scheme|yojana|yojna"),
             ("notices", r"notice|notification|circular|order|news|announcement"),
             ("budget", r"budget"), ("gallery", r"gallery|photo|video|media"), ("faq", r"faq"),
             ("home", r"^/?(?:index|default|home)?(?:\.\w+)?$")]


def site_section(url: str) -> str:
    path = unquote(urlsplit(url).path + "?" + urlsplit(url).query).lower()
    for name, rx in _SECTIONS:
        if re.search(rx, path if name != "home" else urlsplit(url).path.lower()):
            return name
    return "other"


_LETTERHEAD = re.compile(r"छत्तीसगढ़?\s*शासन|government\s+of|मंत्रालय|महानदी\s+भवन|कार्यालय|"
                         r"^(?:परिपत्र|आदेश|अधिसूचना|ज्ञाप|circular|order|notification|अनुक्रमणिका|"
                         r"अनुकमणिका|index|contents|विषय\s*सूची)$|ministry\s+of|department\s+of|"
                         r"^(?:to|प्रति)$|@|www\.|\bph\.?|phone|fax|"
                         r"^\(?[^()]{0,40}(?:शाखा|प्रकोष्ठ|अनुभाग|कक्ष|सेल|cell|section)\)?$", re.I)


def is_letterhead(h: str) -> bool:
    """A letterhead line (the VLM OCR marks them as headings): "छत्तीसगढ़ शासन",
    "सामान्य प्रशासन विभाग" on its own, "(सूचना का अधिकार प्रकोष्ठ)", "मंत्रालय",
    an address, an e-mail/phone line."""
    h = squash(h, 200)
    if _LETTERHEAD.search(h):
        return True
    depts = aliases.find(h, "department")
    if depts:      # a department's name and little else ("सामान्य प्रशासन विभाग (लेखा शाखा)")
        rest = h
        for k in depts:
            for v in [aliases.hindi("department", k), aliases.english("department", k)]:
                rest = rest.replace(v, " ")
        return len(re.findall(rf"[A-Za-z{_DEV}]{{2,}}", re.sub(r"\(.*?\)", " ", rest))) <= 1
    return False


def first_heading(text: str) -> str | None:
    """The first real heading ("## …", or a "*…*" line), not a letterhead line."""
    for m in re.finditer(r"^(?:#{1,3}\s+(.+)|\*([^*\n]{8,200})\*)\s*$", text[:6000], re.M):
        h = squash(m.group(1) or m.group(2), 200)
        if re.fullmatch(r"[A-Z]{3,14}", h) or re.search(r"attachment\(s\)|the\s+image\s+is", h, re.I):
            continue                                  # "CTRCULAR", e-mail / image-caption artefacts
        if 8 <= len(h) <= 200 and not is_letterhead(h) and not re.fullmatch(r"[\d\W]+", h):
            return h
    return None


def url_context(url: str, short: bool = False) -> str:
    """Readable path context for titles that lack it: "2023-2024 · 2nd Supplementry".
    For a very short title ("2. अध्याय- 2 Hindi") also the last folders of the path
    ("FSFC · Panchayats")."""
    segs = [unquote(s) for s in urlsplit(url).path.split("/")[1:-1]]
    keep = [s for s in segs if fiscal_years(s) or re.search(r"supplement|budget|demand", s, re.I)]
    if short:
        keep += [s for s in segs[-2:] if s not in keep and not re.fullmatch(r"(?i)hindi|english|pdf|docs?", s)]
    keep = [s for s in keep if s.lower() not in ("budget_doc",)]
    return " · ".join(keep[-3:])


def url_date(url: str):
    name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    m = _D_NUM.search(name)
    return _mk(m.group(4), m.group(3), m.group(1)) if m else None


# ── OCR facts ────────────────────────────────────────────────────────────────
def ocr_facts(doc_dir: Path, text: str, doc_name: str) -> dict:
    pages = dropped = None
    legacy: list = []
    side = doc_dir / f"{doc_name}.pdf_confidence.json"
    if not side.exists():
        side = next(iter(doc_dir.glob("*_confidence.json")), None) if doc_dir.exists() else None
    if side and side.exists():
        try:
            data = json.loads(side.read_text(encoding="utf-8"))
            plist = data.get("pages") or []
            pages = len(plist) or None
            if data.get("engine") == "vlm":
                dropped = sum(1 for p in plist if p.get("status") == "reject")
                legacy = [p["page_number"] for p in plist
                          if any("legacy" in str(r).lower() or "krutidev" in str(r).lower()
                                 for r in (p.get("reasons") or []))]
        except (OSError, ValueError):
            pass
    if pages is None:
        n = len(re.findall(r"<!--\s*Page\s+\d+\s*-->", text))
        pages = n or None
    n_tables = 0
    tj = doc_dir / "tables.json"
    if tj.exists():
        try:
            n_tables = len(json.loads(tj.read_text(encoding="utf-8")).get("tables") or [])
        except (OSError, ValueError):
            n_tables = 0
    else:
        n_tables = len(re.findall(r"^\|.*\|\s*\n\|[\s:|-]+\|\s*$", text, re.M))
    return {"pages": pages, "pages_dropped": dropped, "legacy_font_pages": legacy,
            "has_tables": n_tables > 0, "n_tables": n_tables}


# ── one document ─────────────────────────────────────────────────────────────
def _strip_copies(text: str) -> str:
    """The text without its distribution lists ("प्रतिलिपि: … 1. कलेक्टर, दुर्ग …"),
    which name half the state's districts and posts on every order."""
    return re.sub(r"(?:प्रतिलिपि|Copy\s+(?:to|forwarded))[\s\S]*?(?=<!--\s*Page|\Z)", " ", text, flags=re.I)


_ADDRESS = re.compile(r"(?:नवा|नया)\s*रायपुर|अटल\s*नगर|रायपुर\s*[-(]?\s*(?:\(?छ\.?\s?ग\.?\)?|492\d*)|"
                      r"Nava\s*Raipur|Naya\s*Raipur|Atal\s*Nagar|Raipur\s*[-(]?\s*(?:\(?C\.?\s?G\.?\)?|492\d*)|"
                      r"महानदी\s*भवन|इंद्रावती\s*भवन|Mahanadi\s*Bhawan", re.I)


# Documents with a dated letterhead. A budget book, report or act carries many
# dates (the state's formation on 1 Nov 2000, the base year of a table) and none of
# them is its issue date — only the listing, the URL or a signature is trusted there.
_LETTER_TYPES = {"circular", "order", "notification", "memo", "letter", "vacancy_notice",
                 "application_form", "press_note", "tender", "policy", "other"}


def title_agreement(title: str | None, text: str) -> float:
    """Share of a listing title's words that the document itself contains — a
    website label that the PDF does not bear out is not used as its title."""
    words = {w for w in re.findall(rf"[A-Za-z{_DEV}]{{3,}}", prep(title or "").lower())}
    if not words:
        return 0.0
    body = prep(text[:12000]).lower()
    return sum(1 for w in words if w in body) / len(words)


def issue_date(page1: str, t: str, opos: int, listing: dict, url: str, max_year: int, dtype: str):
    """(date, source): a "दिनांक/Dated" date near the order number, else any keyed
    date on page 1; the digital signature; the listing's date column; the file
    name; for a letter type, the first date at the top."""
    letter = dtype in _LETTER_TYPES
    keyed = [(s, x) for s, e, x in find_dates(page1)
             if re.search(r"(?:दिनांक|दिनाक|दिनाँक|दि\.|dated|date|dt\.?)\s*[:\-ः,]?\s*(?:the\s+)?$",
                          page1[max(0, s - 22):s], re.I) and sane(x, max_year)
             and not _gazette_date(page1, s)]      # the Gazette's registration date
    near = [x for s, x in keyed if opos >= 0 and -60 <= s - opos <= 400]
    # "क्रमांक … दिनांक …" on one line is the issue date of ANY document; other keyed
    # dates on page 1 only for letter types (a book's page 1 has many dates)
    d = (near or ([x for _, x in keyed] if letter else []) or [None])[0]
    if d:
        return d, "text"
    m = re.search(r"Digitally\s+signed\s+by[\s\S]{0,120}?Date\s*:?\s*([\d./-]{8,10})", t)
    d = sane(parse_date(m.group(1)), max_year) if m else None
    if d:
        return d, "signature"
    d = sane(listing.get("date"), max_year)
    if d:
        return d, "listing"
    d = sane(url_date(url), max_year)
    if d:
        return d, "url"
    if letter:
        top = [x for s, e, x in find_dates(page1[:700])
               if sane(x, max_year) and not _gazette_date(page1, s)]
        if top:
            return top[0], "text-top"
    return None, None


def build_record(*, doc: dict, manifest: dict | None, row: dict | None, text: str,
                 doc_dir: Path) -> dict:
    t = prep(text)
    pgs = pages_of(t)
    page1 = pgs[0][:6000]
    head = page1[:3000]
    fetched = (manifest or {}).get("fetched_at") or doc.get("uploaded_at") or ""
    max_year = int(fetched[:4]) if fetched[:4].isdigit() else datetime.now().year
    url = doc.get("source_url") or ""
    listing = read_row(row) if row else {}

    # The DOCUMENT is the source of truth: government letters follow one format
    # (क्रमांक / दिनांक / विषय), department websites do not. The listing row is a
    # hint — kept as evidence, used only where it agrees with the text or the
    # text has nothing (and marked so); clean titles are the model step's job.
    subject = find_subject(page1)
    onum, opos = find_order_number(page1)
    if onum:
        order_number, order_src = onum, "text"
    elif listing.get("order_number"):
        order_number, order_src = listing["order_number"], "listing"
    else:
        order_number, order_src = None, None

    office, section, scope = find_office(head)
    if not office and listing.get("issued_by"):
        office = listing["issued_by"]

    # title: the subject line; else the listing's title IF the document's own words
    # confirm it; else the first real heading; else the order's first sentence
    lt = listing.get("title")
    agree = title_agreement(lt, t) if lt else None
    title, title_src = None, None
    if subject:
        title, title_src = subject, "subject_line"
    elif lt and agree >= 0.6:
        title, title_src = lt, "listing"
    if not title:
        h = first_heading(t)
        if h:
            title, title_src = h, "heading"
    if not title and doc.get("source_type") != "web":
        lead = lead_sentence(page1, opos)
        if lead:
            title, title_src = lead, "lead_sentence"
    if not title:
        stem = Path(unquote(urlsplit(url).path)).stem or doc.get("original_name") or ""
        stem = re.sub(r"[_\-]+", " ", stem).strip()
        if stem and not re.fullmatch(r"[\d\s]+", stem):
            title, title_src = stem, "filename"
        else:
            title, title_src = None, None
    if title and doc.get("source_type") != "web":
        ctx = url_context(url, short=len(title) < 30)
        if ctx and (len(title) < 30 or not re.search(r"(?:19|20)\d{2}", title)):
            title = f"{title} ({ctx})"

    dtype, dtype_why = doc_type(source_type=doc.get("source_type") or "", url=url,
                                title=title or "", subject=subject or "", head=head,
                                listing_kind=listing.get("kind"))
    issue, issue_src = issue_date(page1, t, opos, listing, url, max_year, dtype)

    fys = fiscal_years(t)
    for src_text in (title or "", subject or "", unquote(url)):
        for fy in fiscal_years(src_text):          # a year only in the title / URL still counts
            fys.setdefault(fy, 0)
    primary = None
    for src_text in (title or "", subject or "", unquote(urlsplit(url).path)):
        f = fiscal_years(src_text)
        if f:
            primary = f.most_common(1)[0][0]
            break
    if not primary and fys:
        fy, n = fys.most_common(1)[0]
        if n >= 2 and n >= 0.4 * sum(fys.values()):
            primary = fy

    unit, unit_ev, unit_conf, units = find_units(t)
    body = _strip_copies(t)
    districts = aliases.find(_ADDRESS.sub(" ", body), "district")
    posts = aliases.find(body, "post")
    links = find_links(t, order_number)
    last = find_last_date(t, issue, max_year) or sane(listing.get("last_date"), max_year + 1)

    rec = {
        "doc_id": doc["id"], "doc_dir": doc.get("doc_dir"), "schema_version": SCHEMA_VERSION,
        "source_url": url, "listing_url": (manifest or {}).get("parent_url") or None,
        "fetched_at": fetched or None, "last_modified": (manifest or {}).get("last_modified") or None,
        "sha256": (manifest or {}).get("sha256") or None,
        "source_type": doc.get("source_type"), "text_source": doc.get("text_source"),

        "title": title, "title_source": title_src, "title_en": None,
        "doc_type": dtype, "doc_type_source": dtype_why,
        "subject": subject,
        "order_number": order_number, "order_number_source": order_src,
        "issue_date": issue.isoformat() if issue else None, "issue_date_source": issue_src,
        "issuing_office": office, "section": section,
        "signatory": find_signatory(t),

        "references": find_references(page1),
        "supersedes": links["supersedes"], "amends": links["amends"], "continues": links["continues"],
        "status": "current", "superseded_by": None, "duplicate_of": None,

        "scope": scope, "applies_from": None,
        "last_date": last.isoformat() if last else None,

        # what the website says, kept as evidence for the model step (never trusted alone)
        "listing": ({"title": lt, "agrees_with_text": round(agree, 2) if agree is not None else None,
                     "order_number": listing.get("order_number"),
                     "date": listing["date"].isoformat() if listing.get("date") else None,
                     "last_date": listing["last_date"].isoformat() if listing.get("last_date") else None,
                     "issued_by": listing.get("issued_by"), "category": listing.get("category"),
                     "text": squash((row or {}).get("text") or "", 400)} if listing else None),
        "topics": [],
        "fiscal_years": sorted(fys)[:12], "primary_year": primary,
        "unit": unit, "unit_evidence": unit_ev, "unit_confidence": unit_conf, "units_seen": units,
        "entities": {
            "districts": [aliases.hindi("district", k) for k, _ in
                          sorted(districts.items(), key=lambda kv: -kv[1])][:10],
            "posts": [aliases.hindi("post", k) for k, _ in sorted(posts.items(), key=lambda kv: -kv[1])][:10],
            "schemes": [], "departments": [], "people": [],
        },
        "site_section": site_section(url) if doc.get("source_type") == "web" else None,

        "summary_hi": None, "summary_en": None, "key_facts": None, "section_summaries": [],
        **ocr_facts(doc_dir, text, doc.get("doc_dir") or ""),
        "generated_by": {"rules": RULES_VERSION, "model": None},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if signer := rec["signatory"]:
        rec["entities"]["people"] = [{"name": signer["name"], "post": signer["post"],
                                      "as_of": rec["issue_date"], "role": "signatory"}]
    return rec


# ── the corpus pass ──────────────────────────────────────────────────────────
def corpus_pass(records: dict) -> dict:
    """Duplicates (same file bytes or same text) and orders a later order supersedes."""
    stats = {"duplicates": 0, "superseded": 0}
    for r in records.values():
        r["status"], r["superseded_by"], r["duplicate_of"] = "current", None, None
    groups = collections.defaultdict(list)
    for r in records.values():
        if r.get("sha256") and r.get("source_type") != "web":
            groups[("sha", r["sha256"])].append(r)
        if r.get("_text_hash"):
            groups[("text", r["_text_hash"])].append(r)
    for _, grp in groups.items():
        if len(grp) < 2:
            continue
        grp.sort(key=lambda r: (r.get("title_source") != "listing", r.get("fetched_at") or "", r["doc_id"]))
        keep = grp[0]
        for r in grp[1:]:
            if r["status"] != "duplicate" and r is not keep:
                r["status"], r["duplicate_of"] = "duplicate", keep["doc_id"]
                stats["duplicates"] += 1
    by_order = collections.defaultdict(list)
    for r in records.values():
        if r.get("order_number") and r["status"] == "current":
            by_order[norm_order(r["order_number"])].append(r)
    for a in records.values():
        own = norm_order(a.get("order_number"))
        for link in a.get("supersedes") or []:
            key = norm_order(link.get("order_number"))
            # one e-office file number carries many orders: a link to the document's
            # own number is not a supersession of its file-mates
            if len(key) < 5 or key == own:
                continue
            for b in by_order.get(key, []):
                if b is a or b["status"] != "current":
                    continue
                if a.get("issue_date") and b.get("issue_date") and b["issue_date"] > a["issue_date"]:
                    continue
                if link.get("date") and b.get("issue_date") and link["date"] != b["issue_date"]:
                    continue                     # the sentence names a different order's date
                b["status"], b["superseded_by"] = "superseded", a["doc_id"]
                stats["superseded"] += 1
    return stats


# ── IO ───────────────────────────────────────────────────────────────────────
def _load_jsonl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _norm_url(u: str) -> str:
    return (u or "").strip().split("#")[0]


def load_inputs(data_root: Path, corpus: Path):
    reg = json.loads((data_root / "documents.json").read_text(encoding="utf-8"))
    docs = reg if isinstance(reg, list) else reg.get("documents", [])
    manifest = {}
    for r in _load_jsonl(corpus / "manifest.jsonl"):
        if r.get("outcome") in ("fetched_new", "fetched_changed", "not_modified"):
            for k in (r.get("normalized_url"), r.get("url"), r.get("final_url")):
                if k:
                    manifest[_norm_url(k)] = r
    rows = collections.defaultdict(list)
    for r in _load_jsonl(corpus / "listing_rows.jsonl"):
        rows[_norm_url(r.get("url"))].append(r)
    if not rows:
        rows = rows_from_saved_pages(corpus, manifest)
    return docs, manifest, rows


def rows_from_saved_pages(corpus: Path, manifest: dict) -> dict:
    """Before listing_rows.jsonl exists (a crawl made before rows were recorded):
    read the saved copy of each listing page — page 1 of a paged list only."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parents[1] / "Scraper"))
        import extract as ex
    except ImportError:
        return {}
    rows = collections.defaultdict(list)
    parents = {r.get("parent_url") for r in manifest.values() if r.get("doc_kind") not in ("html", "json")}
    for p in parents:
        rec = manifest.get(_norm_url(p or ""))
        if not rec or not rec.get("raw_path"):
            continue
        raw = corpus / rec["raw_path"]
        if not raw.exists():
            continue
        try:
            for r in ex.extract_link_rows(raw.read_bytes(), rec.get("final_url") or p):
                rows[_norm_url(r["url"])].append({**r, "listing_url": p})
        except Exception:  # noqa: BLE001
            continue
    return rows


def _md_path(doc: dict, data_root: Path) -> Path:
    p = Path(doc.get("md_path") or "")
    if p.exists():
        return p
    return data_root / "stage2" / (doc.get("doc_dir") or "") / "structured.md"


def build_all(data_root: Path, corpus: Path, *, force: bool = False, limit: int | None = None,
              log=print) -> dict:
    docs, manifest, rows = load_inputs(data_root, corpus)
    out_path = data_root / "doc_metadata.json"
    old = {}
    if out_path.exists() and not force:
        try:
            old = json.loads(out_path.read_text(encoding="utf-8")).get("documents", {})
        except (OSError, ValueError):
            old = {}
    records, reused, built, skipped = {}, 0, 0, 0
    for i, doc in enumerate(docs):
        if limit and i >= limit:
            break
        if doc.get("status") != "processed":
            skipped += 1
            continue
        md = _md_path(doc, data_root)
        if not md.exists():
            skipped += 1
            continue
        text = md.read_text(encoding="utf-8", errors="replace")
        url = _norm_url(doc.get("source_url") or "")
        man = manifest.get(url)
        row = pick_row(rows.get(url, []), (man or {}).get("parent_url") or "")
        key = hashlib.sha1(json.dumps([RULES_VERSION, hashlib.sha1(text.encode()).hexdigest(), row,
                                       (man or {}).get("sha256"), doc.get("source_url")],
                                      ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
        prev = old.get(doc["id"])
        if prev and prev.get("_key") == key:
            records[doc["id"]] = prev
            reused += 1
            continue
        rec = build_record(doc=doc, manifest=man, row=row, text=text, doc_dir=md.parent)
        rec["_key"] = key
        rec["_text_hash"] = hashlib.sha1(re.sub(r"\s+", "", text).encode()).hexdigest() \
            if len(text) > 200 else None
        records[doc["id"]] = rec
        built += 1
    # the model pass's checked fields (step 3, doc_model_meta.py), when it has run
    doc_model_meta = import_module("02_optimization.doc_model_meta")
    model_cache = doc_model_meta.load_cache(data_root)
    for doc_id, rec in records.items():
        doc_model_meta.apply(rec, model_cache.get(doc_id))
    stats = corpus_pass(records)
    payload = {"schema_version": SCHEMA_VERSION, "rules_version": RULES_VERSION,
               "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "documents": records}
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, out_path)
    log(f"doc_metadata: {len(records)} documents ({built} built, {reused} unchanged, {skipped} skipped: "
        f"not processed) · {stats['duplicates']} duplicate(s), {stats['superseded']} superseded "
        f"-> {out_path}")
    return payload


# ── report ───────────────────────────────────────────────────────────────────
_FIELDS = ["title", "doc_type", "order_number", "issue_date", "subject", "issuing_office", "signatory",
           "primary_year", "unit", "last_date", "references", "supersedes", "amends", "continues"]


def report(payload: dict, data_root: Path, sample: int = 25, seed: int = 7) -> Path:
    recs = list(payload["documents"].values())
    lines = [f"# doc_metadata report — {data_root.name}", "",
             f"{len(recs)} documents · rules: {payload.get('rules_version')} · {payload.get('generated_at')}", ""]
    groups = collections.defaultdict(list)
    for r in recs:
        groups["all"].append(r)
        groups[f"{r.get('source_type')}/{r.get('text_source')}"].append(r)
    lines += ["## Coverage (share of documents with the field)", "",
              "| field | " + " | ".join(f"{g} ({len(v)})" for g, v in groups.items()) + " |",
              "|---|" + "---|" * len(groups)]
    for f in _FIELDS:
        cells = []
        for g, v in groups.items():
            n = sum(1 for r in v if r.get(f) and r.get(f) != "other")
            cells.append(f"{100 * n / max(1, len(v)):.0f}%")
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    for f in ("doc_type", "title_source", "issue_date_source", "order_number_source", "scope", "unit",
              "unit_confidence", "status", "doc_type_source"):
        c = collections.Counter(str(r.get(f)) if f != "doc_type_source" else str(r.get(f)).split(":")[0]
                                for r in recs)
        lines += ["", f"**{f}:** " + ", ".join(f"{k} {v}" for k, v in c.most_common(20))]
    random.Random(seed).shuffle(recs)
    lines += ["", f"## Review sample ({sample} random documents)", ""]
    for r in recs[:sample]:
        lines += [f"### {r.get('title') or '(no title)'}", "",
                  f"- url: {r.get('source_url')}",
                  f"- type: {r.get('doc_type')} ({r.get('doc_type_source')}) · scope {r.get('scope')} · "
                  f"status {r.get('status')}",
                  f"- order: {r.get('order_number')} ({r.get('order_number_source')}) · date: "
                  f"{r.get('issue_date')} ({r.get('issue_date_source')}) · last date: {r.get('last_date')}",
                  f"- office: {r.get('issuing_office')} · section: {r.get('section')} · signatory: {r.get('signatory')}",
                  f"- subject: {r.get('subject')}",
                  f"- years: {r.get('fiscal_years')} primary {r.get('primary_year')} · unit: {r.get('unit')} "
                  f"({r.get('unit_evidence')})",
                  f"- districts: {r['entities']['districts']} · posts: {r['entities']['posts']}",
                  f"- links: supersedes {len(r.get('supersedes') or [])}, amends {len(r.get('amends') or [])}, "
                  f"continues {len(r.get('continues') or [])}, references {len(r.get('references') or [])}", ""]
    path = data_root / "doc_metadata_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", help="chips | fg | fg2 (default: $PIPELINE)")
    ap.add_argument("--corpus", help="scraped corpus dir (default: <data>/scrapped_data/<pipeline>)")
    ap.add_argument("--force", action="store_true", help="rebuild every record (ignore the cache)")
    ap.add_argument("--limit", type=int, help="first N registry documents only (trial)")
    ap.add_argument("--report", action="store_true", help="also write doc_metadata_report.md")
    ap.add_argument("--sample", type=int, default=25, help="documents in the report's review sample")
    ap.add_argument("--show", help="print the record of one document (doc_dir, doc_id or URL)")
    args = ap.parse_args(argv)
    if args.pipeline:
        os.environ["PIPELINE"] = args.pipeline
    import pipeline_config as pc
    corpus = Path(args.corpus) if args.corpus else pc.DATA_BASE / "scrapped_data" / pc.PIPELINE
    if args.show:
        path = pc.DATA_ROOT / "doc_metadata.json"
        recs = json.loads(path.read_text(encoding="utf-8"))["documents"]
        hit = [r for r in recs.values() if args.show in (r["doc_id"], r.get("doc_dir"), r.get("source_url"))]
        print(json.dumps(hit, ensure_ascii=False, indent=1) if hit else f"not found in {path}")
        return 0
    payload = build_all(pc.DATA_ROOT, corpus, force=args.force, limit=args.limit)
    if args.report:
        print(f"report -> {report(payload, pc.DATA_ROOT, args.sample)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
