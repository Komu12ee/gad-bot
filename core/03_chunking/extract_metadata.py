"""Document-level metadata and date extraction engine.

Strictly follows the Date Extraction Rules:
- Header zone (first 25 lines of Page 1) vs Signature block (last page / closing lines)
- Negative context rejection (ref, vide, last date, w.e.f., etc.)
- Day-first parsing with Devanagari and Hindi month support
- Priority: header date > signature date (conflict triggers needs_review=True)
- Pluggable LLM fallback with strict text verification (Rule 3)
- Supersession and order number regex extraction
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Devanagari to Arabic digits mapping
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_HINDI_MONTHS = {
    "जनवरी": 1, "जनवरी,": 1, "january": 1, "jan": 1,
    "फरवरी": 2, "फरवरी,": 2, "february": 2, "feb": 2,
    "मार्च": 3, "मार्च,": 3, "march": 3, "mar": 3,
    "अप्रैल": 4, "अप्रैल,": 4, "april": 4, "apr": 4,
    "मई": 5, "मई,": 5, "may": 5,
    "जून": 6, "जून,": 6, "june": 6, "jun": 6,
    "जुलाई": 7, "जुलाई,": 7, "july": 7, "jul": 7,
    "अगस्त": 8, "अगस्त,": 8, "august": 8, "aug": 8,
    "सितंबर": 9, "सितम्बर": 9, "september": 9, "sep": 9, "sept": 9,
    "अक्टूबर": 10, "अक्तूबर": 10, "october": 10, "oct": 10,
    "नवंबर": 11, "नवम्बर": 11, "november": 11, "nov": 11,
    "दिसंबर": 12, "दिसम्बर": 12, "december": 12, "dec": 12,
}

# Accepted Date Prefix Labels (Rule 2)
_ACCEPTED_DATE_LABELS = re.compile(
    r"(?i)\b(?:Dated|Date|Date\s*of\s*issue|दिनांक|दिनाँक|जारी\s*करने\s*की\s*तिथि)\b[\s:—–-]*"
)

# Reject context phrases (Rule 3: preceding 60 chars)
_REJECT_PRECEDING_REGEX = re.compile(
    r"(?i)\b(?:vide|ref|reference|read\s+with|in\s+continuation\s+of|in\s+partial\s+modification\s+of|"
    r"in\s+supersession\s+of|notified|G\.?S\.?R|OM|letter|circular|order|meeting\s+held\s+on|"
    r"w\.?e\.?f\.?|with\s+effect\s+from|last\s+date|on\s+or\s+before|up\s+to|till|before|"
    r"के\s*संदर्भ\s*में|पढ़े\s*गए|के\s*अनुक्रम\s*में|अंतिम\s*तिथि)\b"
)

# Regex for Order Numbers
_ORDER_NO_PATTERNS = [
    # English Order / Vacancy Circular No.
    re.compile(r"(?i)\b(?:No\.|Order\s*No\.?|F\.?\s*No\.?|Letter\s*No\.?|Notification\s*No\.?|Circular\s*No\.?)[\s:—–-]*([A-Z0-9\-_./]+(?:/[A-Z0-9\-_.]+)+)", re.I),
    re.compile(r"(?i)\b(?:No\.|Order\s*No\.?|F\.?\s*No\.?)[\s:—–-]*([A-Z0-9]+(?:-[A-Z0-9]+)+)", re.I),
    # Hindi क्रमांक / पत्र क्रमांक with optional spaces (e.g. क्रमांक एफ 1-2/2024/1-3)
    re.compile(r"(?:क्रमांक|पत्र\s*क्र\.?|फा\.\s*क्र\.?|आदेश\s*क्र\.?)[\s:—–-]*([A-Za-z\u0900-\u097F0-9\-_./]+(?:\s+[A-Za-z0-9\-_./]+)*)", re.U),
]

# Supersession phrases (Section 7 item 2)
_SUPERSESSION_PATTERNS = [
    re.compile(r"(?i)\bin\s+partial\s+modification\s+of\s+(?:order|circular|notification|memo)?\s*(?:no\.?)?\s*([A-Z0-9\-_./]+)", re.I),
    re.compile(r"(?i)\bin\s+supersession\s+of\s+(?:all\s+previous\s+orders|order|circular|notification)?\s*(?:no\.?)?\s*([A-Z0-9\-_./]+)", re.I),
    re.compile(r"(?i)\bin\s+continuation\s+of\s+(?:this\s+office\s+)?(?:order|circular|memo)?\s*(?:no\.?)?\s*([A-Z0-9\-_./]+)", re.I),
    re.compile(r"(?i)\bsupersedes\s+(?:order|circular)?\s*(?:no\.?)?\s*([A-Z0-9\-_./]+)", re.I),
    re.compile(r"(?i)\bcancelled\s+and\s+withdrawn\b", re.I),
    re.compile(r"(?i)\bvide\s+order\s+no\.?\s*([A-Z0-9\-_./]+)", re.I),
    # Hindi equivalents
    re.compile(r"(?:के\s*अधिक्रमण\s*में|के\s*आंशिक\s*संशोधन\s*में|के\s*तारतम्य\s*में)\s*(?:आदेश\s*क्र\.?)?\s*([^\s,;।\n]+)", re.U),
]


def normalize_digits(text: str) -> str:
    """Convert Devanagari numerals to standard Arabic numerals without modifying other text."""
    if not text:
        return ""
    return text.translate(_DEVA_DIGITS)


def parse_date_string(raw_date: str) -> Optional[str]:
    """Parse day-first date string into ISO YYYY-MM-DD.
    
    Supports: DD.MM.YYYY, DD/MM/YYYY, DD-MM-YYYY, '18 May 2026', '18 मई 2026', Devanagari digits.
    """
    if not raw_date:
        return None
    s = normalize_digits(raw_date).strip().rstrip(".,;।")
    
    # 1. Numeric Day-First: DD.MM.YYYY, DD/MM/YYYY, DD-MM-YYYY (allowing optional spaces around separators)
    m = re.search(r"\b(\d{1,2})\s*[\./-]\s*(\d{1,2})\s*[\./-]\s*(\d{4})\b", s)
    if m:
        d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            dt = datetime(y, mth, d)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 2. Textual Month: 18 May 2026 or 18 मई 2026 or May 18, 2026
    m2 = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z\u0900-\u097F]+)[,\s]+(\d{4})\b", s)
    if m2:
        d, mth_str, y = int(m2.group(1)), m2.group(2).lower(), int(m2.group(3))
        mth = _HINDI_MONTHS.get(mth_str)
        if mth:
            try:
                dt = datetime(y, mth, d)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

    return None


def extract_date_candidates(
    lines: List[str],
    zone_name: str,
    strict_labels: bool = True
) -> List[Dict[str, Any]]:
    """Extract date candidates from a list of text lines within a specific zone."""
    candidates = []
    
    for idx, line in enumerate(lines):
        norm_line = normalize_digits(line)
        
        # Candidate date patterns
        date_matches = list(re.finditer(
            r"\b(?:\d{1,2}[\./-]\d{1,2}[\./-]\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z\u0900-\u097F]+[,\s]+\d{4})\b",
            norm_line
        ))
        
        for match in date_matches:
            start_pos, end_pos = match.span()
            raw_match_text = line[start_pos:end_pos]
            parsed_iso = parse_date_string(raw_match_text)
            if not parsed_iso:
                continue

            # Look back up to 60 characters for negative context (Rule 3)
            preceding_window = line[max(0, start_pos - 60):start_pos]
            is_rejected = bool(_REJECT_PRECEDING_REGEX.search(preceding_window))
            has_accepted_label = bool(_ACCEPTED_DATE_LABELS.search(preceding_window))

            candidates.append({
                "iso": parsed_iso,
                "raw": raw_match_text,
                "zone": zone_name,
                "line_no": idx + 1,
                "rejected": is_rejected,
                "has_label": has_accepted_label,
                "preceding": preceding_window.strip(),
            })

    return candidates


def extract_doc_date(
    pages: List[str],
    is_web: bool = False,
    web_meta_date: Optional[str] = None,
    llm_extractor: Optional[Callable[[str, str], Any]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], str, bool, List[Dict[str, str]]]:
    """Strictly extract doc_date according to Date Extraction Rules.

    Returns:
        (doc_date, doc_date_raw, doc_date_zone, date_confidence, needs_review, referenced_docs)
    """
    referenced_docs: List[Dict[str, str]] = []
    
    # Check Web meta date if applicable
    if is_web and web_meta_date:
        parsed_web = parse_date_string(web_meta_date)
        if parsed_web:
            return parsed_web, web_meta_date, "web", "high", False, []

    # 1. Header zone: first 25 lines of page 1
    page1_lines = (pages[0] if pages else "").splitlines()
    header_zone = page1_lines[:25]
    header_candidates = extract_date_candidates(header_zone, "header")

    # 2. Signature block: last 30 lines of the last page
    last_page_lines = (pages[-1] if pages else "").splitlines()
    sig_zone = last_page_lines[-30:] if len(last_page_lines) > 30 else last_page_lines
    sig_candidates = extract_date_candidates(sig_zone, "signature")

    # Filter out rejected dates and store in referenced_docs
    valid_header = []
    for c in header_candidates:
        if c["rejected"]:
            referenced_docs.append({"ref_no": "body_ref", "ref_date": c["iso"], "raw": c["raw"]})
        else:
            valid_header.append(c)

    valid_sig = []
    for c in sig_candidates:
        if c["rejected"]:
            referenced_docs.append({"ref_no": "body_ref", "ref_date": c["iso"], "raw": c["raw"]})
        else:
            valid_sig.append(c)

    # Date resolution logic (Rule 4)
    # Prefer candidate with explicit label
    best_header = next((c for c in valid_header if c["has_label"]), None) or (valid_header[0] if valid_header else None)
    best_sig = next((c for c in valid_sig if c["has_label"]), None) or (valid_sig[0] if valid_sig else None)

    # Both header and signature found
    if best_header and best_sig:
        if best_header["iso"] == best_sig["iso"]:
            return best_header["iso"], best_header["raw"], "header", "high", False, referenced_docs
        else:
            # Conflict between header and signature -> choose header, flag review
            return best_header["iso"], best_header["raw"], "header", "high", True, referenced_docs

    if best_header:
        confidence = "high" if best_header["has_label"] else "medium"
        return best_header["iso"], best_header["raw"], "header", confidence, False, referenced_docs

    if best_sig:
        return best_sig["iso"], best_sig["raw"], "signature", "medium", False, referenced_docs

    # Rule 6: LLM fallback only if regex finds none or conflicting candidates
    # Send only the header zone and signature block.
    if llm_extractor:
        combined_zones_text = "=== HEADER ZONE ===\n" + "\n".join(header_zone) + "\n\n=== SIGNATURE BLOCK ===\n" + "\n".join(sig_zone)
        llm_prompt = (
            "From the header zone and signature block below, extract the single exact official issuing/signing date.\n"
            "Rules:\n"
            "- Return ONLY an exact quoted substring from the text below.\n"
            "- Do not return deadlines, meeting dates, or dates of referenced letters.\n"
            "- If no issuing date is present, return {\"quoted_date\": null}.\n\n"
            f"{combined_zones_text}\n\n"
            "Respond in JSON format: {\"quoted_date\": \"<exact quoted date text>\"}"
        )
        sys_prompt = "You are an official government document parser. Return JSON with exact quotes only. Never hallucinate or calculate dates."
        try:
            res = llm_extractor(llm_prompt, sys_prompt)
            quoted = None
            if isinstance(res, dict):
                quoted = res.get("quoted_date")
            elif isinstance(res, str):
                m_match = re.search(r'"quoted_date":\s*"([^"]+)"', res)
                quoted = m_match.group(1) if m_match else None
            
            if quoted and verify_llm_value(quoted, combined_zones_text):
                # Verify negative context (Rule 3)
                idx = combined_zones_text.lower().find(quoted.lower())
                prec_window = combined_zones_text[max(0, idx - 60):idx]
                if not _REJECT_PRECEDING_REGEX.search(prec_window):
                    parsed_iso = parse_date_string(quoted)
                    if parsed_iso:
                        zone = "header" if idx < len("\n".join(header_zone)) + 20 else "signature"
                        return parsed_iso, quoted, zone, "medium", False, referenced_docs
        except Exception:
            pass

    # Rule 5: No candidate -> null, confidence="none", needs_review=True (NO file modified fallback!)
    return None, None, None, "none", True, referenced_docs


def extract_order_no(text: str) -> Optional[str]:
    """Extract official circular / order number using regex."""
    lines = text.splitlines()[:35]
    for line in lines:
        for pat in _ORDER_NO_PATTERNS:
            m = pat.search(line)
            if m:
                cand = m.group(1).strip().rstrip(".,;।")
                if len(cand) >= 3 and not cand.lower().startswith("http"):
                    return cand
    return None


def extract_supersedes_references(text: str) -> List[str]:
    """Extract list of superseded or modified order numbers."""
    refs = []
    for pat in _SUPERSESSION_PATTERNS:
        for match in pat.finditer(text):
            if match.groups():
                ref = match.group(1).strip().rstrip(".,;।")
                if ref and ref not in refs:
                    refs.append(ref)
    return refs


def detect_language(text: str) -> str:
    """Classify language into 'en', 'hi', or 'mixed' based on character script counts."""
    hi_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    en_count = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    total = hi_count + en_count
    if total == 0:
        return "mixed"
    hi_ratio = hi_count / total
    if hi_ratio > 0.75:
        return "hi"
    elif hi_ratio < 0.25:
        return "en"
    return "mixed"


def classify_doc_type(text: str, filename: str = "") -> str:
    """Classify document type into standardized taxonomy."""
    t_lower = text[:1500].lower()
    fn_lower = filename.lower()
    if "vacancy" in t_lower or "भर्ती" in t_lower or "रिक्ति" in t_lower or "recruitment" in t_lower:
        return "vacancy_circular"
    if "अधिसूचना" in t_lower or "notification" in t_lower:
        return "notification"
    if "परिपत्र" in t_lower or "circular" in t_lower:
        return "circular"
    if "आदेश" in t_lower or "order" in t_lower:
        return "order"
    if "बजट" in t_lower or "budget" in t_lower:
        return "budget_table"
    if "अधिनियम" in t_lower or "नियम" in t_lower or "act" in t_lower or "rule" in t_lower:
        return "act_rule"
    if fn_lower.endswith(".html") or "web_page" in fn_lower:
        return "web_page"
    return "other"


def extract_org_structure(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract Ministry, Department, and Office from document header."""
    lines = [ln.strip() for ln in text.splitlines()[:25] if ln.strip()]
    ministry, dept, office = None, None, None
    for ln in lines:
        if not ministry and re.search(r"(?i)\b(?:Ministry\s+of|मंत्रालय)\b", ln):
            ministry = ln
        elif not dept and re.search(r"(?i)\b(?:Department\s+of|विभाग|विभागीय)\b", ln):
            dept = ln
        elif not office and re.search(r"(?i)\b(?:Directorate|कार्यालय|Office\s+of|निदेशालय|Commission)\b", ln):
            office = ln
    return ministry, dept, office


def build_topic_key(dept: Optional[str], doc_type: str, title: Optional[str]) -> str:
    """Construct a deterministic topic_key for latest-version deduplication."""
    d = re.sub(r"[^\w]", "", (dept or "general").lower())
    dt = doc_type.lower()
    t = re.sub(r"[^\w]", "", (title or "untitled").lower()[:40])
    return f"{d}:{dt}:{t}"


def verify_llm_value(value: Optional[str], source_text: str) -> bool:
    """Rule 3: Check that an LLM extracted string actually appears in the source text."""
    if not value or not value.strip():
        return False
    v = value.strip().lower()
    return v in source_text.lower()


def call_llm_json_extractor(
    prompt: str,
    system_prompt: str,
    llm_url: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> Optional[Any]:
    """Pluggable OpenAI-compatible LLM extraction call (temperature 0)."""
    base_url = (llm_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    model = model_name or os.getenv("LLM_MODEL", "Qwen/Qwen3-32B-FP8")
    
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }
    
    headers = {"Content-Type": "application/json"}
    token = api_key or os.getenv("LLM_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            return content.strip()
    except Exception:
        return None
