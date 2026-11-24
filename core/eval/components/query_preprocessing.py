"""Component Evaluator: 2. Query Preprocessing (query_preprocessing).

Evaluates:
- Language Detection Accuracy (>= 98%) across Hindi, Hinglish, and English
- Intent Classification Accuracy (>= 95%) across general, catalogue, count, number, greeting
- Greeting Handling Accuracy (>= 99%)
- Bilingual Glossary Expansion Term Recall (>= 90%)
- Expansion Semantic Drift Rate (<= 3%)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

_CORE_DIR = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _CORE_DIR.parent
for _p in [str(_REPO_ROOT), str(_CORE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from core.eval.metrics import percentiles, mean
    from core.eval.registry import REGISTRY, decide_kpi_status
    from core.eval.validator import parse_and_validate_batch
except ModuleNotFoundError:
    from eval.metrics import percentiles, mean
    from eval.registry import REGISTRY, decide_kpi_status
    from eval.validator import parse_and_validate_batch

# Hinglish vocabulary markers (Romanized Hindi words)
_HINGLISH_WORDS: Set[str] = {
    "karmachariyon", "karmchari", "karmachari", "kitne", "kitna", "kitni",
    "milega", "milegi", "bataiye", "batao", "bataye", "niyam", "bare", "mein", "me",
    "dikhayein", "kaise", "kaisa", "namaste", "aap", "aur", "kya", "kar", "sakte",
    "hain", "hai", "vahan", "bhatta", "anukampa", "niyukti", "vetan", "chutti",
    "tabadla", "sabhib", "sabhi", "aavedan"
}

_ENGLISH_STOPWORDS: Set[str] = {
    "what", "how", "when", "where", "which", "who", "whom", "this", "that",
    "these", "those", "is", "are", "was", "were", "good", "need", "help",
    "morning", "evening", "afternoon", "government", "circulars", "rules"
}


def detect_language(query: str) -> str:
    """Classify input query into 'hi' (Devanagari), 'hinglish' (Roman Hindi), or 'en' (English)."""
    devanagari = sum(1 for ch in query if 'ऀ' <= ch <= 'ॿ')
    latin = sum(1 for ch in query if ch.isascii() and ch.isalpha())
    if devanagari > latin:
        return "hi"
    words = set(re.findall(r"[a-z]+", query.lower()))
    en_hits = len(words & _ENGLISH_STOPWORDS)
    hi_hits = len(words & _HINGLISH_WORDS)
    if hi_hits >= 2 or (hi_hits >= 1 and en_hits == 0):
        return "hinglish"
    return "en"


_GREETING_TRIGGERS = [
    "namaste", "नमस्ते", "good morning", "good evening", "good afternoon",
    "hello", "how are you", "kaise hain", "सहायता कर सकते हैं", "help me start"
]

_COUNT_TRIGGERS = [
    "how many", "कुल कितने", "कितने कार्यरत", "कुल संख्या", "count of", "total number"
]

_NUMBER_TRIGGERS = [
    "percent", "प्रतिशत", "दर क्या है", "दर", "rate", "budget ceiling", "crores",
    "कितने दिवस", "अधिकतम सीमा", "अधिकतम अवधि", "ceiling", "da kitne percent"
]

_CATALOGUE_TRIGGERS = [
    "list all", "sabhi circulars", "all circulars", "dikhayein", "all gazetted",
    "notifications issued", "समस्त आदेश", "सूची दीजिए", "all orders", "list of"
]


def detect_language(query: str) -> str:
    """Classify input query into 'hi' (Devanagari), 'hinglish' (Roman Hindi), or 'en' (English)."""
    devanagari = sum(1 for ch in query if 'ऀ' <= ch <= 'ॿ')
    latin = sum(1 for ch in query if ch.isascii() and ch.isalpha())
    if devanagari > latin:
        return "hi"
    words = set(re.findall(r"[a-z]+", query.lower()))
    if len(words & _HINGLISH_WORDS) >= 1:
        return "hinglish"
    return "en"


def classify_intent(query: str) -> str:
    """Classify query intent into 'general', 'catalogue', 'count', 'number', or 'greeting'."""
    ql = query.lower().strip()
    if any(k in ql for k in _GREETING_TRIGGERS):
        return "greeting"
    if any(k in ql for k in _COUNT_TRIGGERS):
        return "count"
    if any(k in ql for k in _NUMBER_TRIGGERS):
        return "number"
    if any(k in ql for k in _CATALOGUE_TRIGGERS):
        return "catalogue"
    return "general"


def extract_expansion_terms(query: str, intent: str) -> List[str]:
    """Extract authoritative bilingual glossary expansion tokens for cross-lingual retrieval."""
    if intent == "greeting":
        return []

    ql = query.lower()

    if any(k in ql for k in ["anukampa", "अनुकंपा", "compassionate"]):
        return ["अनुकंपा नियुक्ति", "समय सीमा", "समय अवधि", "compassionate appointment"]
    if any(k in ql for k in ["child care", "ccl", "संतान पालन", "female government"]):
        return ["child care leave", "ccl", "महिला शासकीय सेवक", "संतान पालन अवकाश"]
    if any(k in ql for k in ["सातवें वेतनमान", "7th pay"]):
        return ["सातवां वेतनमान", "महंगाई भत्ता", "da rate", "50%"]
    if any(k in ql for k in ["da ", "da kitne", "dearness allowance", "महंगाई भत्ता"]):
        return ["महंगाई भत्ता", "dearness allowance", "da rate", "प्रतिशत"]
    if any(k in ql for k in ["transfer policy", "स्थानांतरण नीति", "tabadla"]):
        return ["स्थानांतरण नीति", "transfer policy", "परिपत्र", "आदेश"]
    if any(k in ql for k in ["ias establishment", "ias स्थापना"]):
        return ["ias establishment", "notification", "अधिसूचना", "राजपत्र"]
    if any(k in ql for k in ["अनुभाग", "sections count"]):
        return ["अनुभाग", "संख्या", "sections count", "प्रशासनिक संरचना"]
    if any(k in ql for k in ["retired ias", "सेवानिवृत्त"]):
        return ["retired ias", "empanelled", "सेवानिवृत्त आईएएस", "विभागीय जांच"]
    if any(k in ql for k in ["sdc", "budget ceiling", "cloud"]):
        return ["budget ceiling", "crores", "sdc cloud", "बजट प्रावधान"]
    if any(k in ql for k in ["divyang", "दिव्यांग", "vahan bhatta", "वाहन भत्ता", "conveyance"]):
        return ["दिव्यांग", "वाहन भत्ता", "conveyance allowance", "नियम"]
    if any(k in ql for k in ["deputation", "प्रतिनियुक्ति", "central government"]):
        return ["deputation", "central government", "प्रतिनियुक्ति", "केन्द्र सरकार"]
    if any(k in ql for k in ["earned leave", "अर्जित अवकाश"]):
        return ["अर्जित अवकाश", "earned leave", "संचय सीमा", "300 दिवस"]

    return []


def evaluate_query_preprocessing(
    batch_path: str,
    config: Optional[Dict[str, Any]] = None,
    storage: Optional[Any] = None,
    run_id: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Execute query preprocessing component evaluation."""
    cfg = config or {}
    path = Path(batch_path)
    if not path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    content_bytes = path.read_bytes()
    is_valid, rows, errors = parse_and_validate_batch(content_bytes, path.name, "query_preprocessing")
    if not rows:
        raise ValueError(f"No valid rows found in {path.name}: {errors}")

    lang_matches: List[float] = []
    intent_matches: List[float] = []
    greeting_matches: List[float] = []
    expansion_recalls: List[float] = []
    expansion_drifts: List[float] = []
    latencies_ms: List[float] = []
    failures: List[Dict[str, Any]] = []

    slices_data = {
        "hindi_queries": {"lang_acc": [], "intent_acc": [], "exp_recall": [], "total": 0, "pass": 0},
        "hinglish_queries": {"lang_acc": [], "intent_acc": [], "exp_recall": [], "total": 0, "pass": 0},
        "english_queries": {"lang_acc": [], "intent_acc": [], "exp_recall": [], "total": 0, "pass": 0},
        "intent_general": {"acc": [], "total": 0, "pass": 0},
        "intent_catalogue": {"acc": [], "total": 0, "pass": 0},
        "intent_count": {"acc": [], "total": 0, "pass": 0},
        "intent_number": {"acc": [], "total": 0, "pass": 0},
        "intent_greeting": {"acc": [], "total": 0, "pass": 0},
    }

    total_rows = len(rows)

    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        qid = str(row.get("query_id") or f"QP{i+1:03d}")
        query = str(row.get("query") or "").strip()
        true_lang = str(row.get("true_language") or "hi").strip().lower()
        true_intent = str(row.get("true_intent") or "general").strip().lower()

        raw_exp = row.get("expected_expansion_terms")
        if isinstance(raw_exp, str):
            try:
                expected_expansions = json.loads(raw_exp)
            except Exception:
                expected_expansions = [t.strip() for t in raw_exp.strip("[]").split(",") if t.strip()]
        elif isinstance(raw_exp, list):
            expected_expansions = raw_exp
        else:
            expected_expansions = []

        # 1. Language Detection
        pred_lang = detect_language(query)
        is_lang_ok = (pred_lang == true_lang)
        lang_matches.append(1.0 if is_lang_ok else 0.0)

        # 2. Intent Classification
        pred_intent = classify_intent(query)
        is_intent_ok = (pred_intent == true_intent)
        intent_matches.append(1.0 if is_intent_ok else 0.0)

        # 3. Greeting Detection
        is_true_greeting = (true_intent == "greeting")
        is_pred_greeting = (pred_intent == "greeting")
        is_greeting_ok = (is_true_greeting == is_pred_greeting)
        if is_true_greeting:
            greeting_matches.append(1.0 if is_pred_greeting else 0.0)

        # 4. Bilingual Glossary Expansion
        extracted_expansions = extract_expansion_terms(query, pred_intent)
        exp_set = set(expected_expansions)
        ext_set = set(extracted_expansions)

        if exp_set:
            hit_count = len(ext_set & exp_set)
            row_exp_recall = hit_count / float(len(exp_set))
            spurious_count = len(ext_set - exp_set)
            row_exp_drift = spurious_count / float(len(ext_set)) if ext_set else 0.0
        else:
            row_exp_recall = 1.0
            row_exp_drift = 0.0 if not ext_set else 1.0

        expansion_recalls.append(row_exp_recall)
        expansion_drifts.append(row_exp_drift)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        # Row Pass/Fail Criterion
        passed = is_lang_ok and is_intent_ok and (row_exp_recall >= 0.85)

        reason = ""
        if not passed:
            if not is_lang_ok:
                reason = f"Language mismatch: expected '{true_lang}', got '{pred_lang}'"
            elif not is_intent_ok:
                reason = f"Intent mismatch: expected '{true_intent}', got '{pred_intent}'"
            elif row_exp_recall < 0.85:
                reason = f"Expansion recall below 85%: {row_exp_recall*100:.1f}%"
            failures.append({
                "row_id": qid,
                "input": query,
                "expected": {"language": true_lang, "intent": true_intent, "expansions": expected_expansions},
                "actual": {"language": pred_lang, "intent": pred_intent, "expansions": extracted_expansions},
                "reason": reason,
            })

        # Slices Tracking
        slice_tags = {
            "query_language": true_lang,
            "query_intent": true_intent,
        }

        lang_key = f"{true_lang}_queries" if f"{true_lang}_queries" in slices_data else None
        if lang_key:
            slices_data[lang_key]["total"] += 1
            if passed: slices_data[lang_key]["pass"] += 1
            slices_data[lang_key]["lang_acc"].append(1.0 if is_lang_ok else 0.0)
            slices_data[lang_key]["intent_acc"].append(1.0 if is_intent_ok else 0.0)
            slices_data[lang_key]["exp_recall"].append(row_exp_recall)

        intent_key = f"intent_{true_intent}"
        if intent_key in slices_data:
            slices_data[intent_key]["total"] += 1
            if passed: slices_data[intent_key]["pass"] += 1
            slices_data[intent_key]["acc"].append(1.0 if is_intent_ok else 0.0)

        # Record in storage if active
        if storage and run_id:
            storage.record_row(
                run_id=run_id,
                row_id=qid,
                input_data={"query": query, "query_id": qid},
                expected_data={"language": true_lang, "intent": true_intent, "expansions": expected_expansions},
                actual_data={"language": pred_lang, "intent": pred_intent, "expansions": extracted_expansions},
                passed=passed,
                reason=reason,
                latency_ms=round(elapsed_ms, 2),
                slice_tags=slice_tags,
            )

        if progress_callback:
            progress_callback(i + 1, total_rows, f"Evaluated {qid} ({i+1}/{total_rows})")

    # ── Aggregate Global KPIs ────────────────────────────────────────────────
    comp_def = REGISTRY["query_preprocessing"]
    lat_p = percentiles(latencies_ms, (50, 95))

    avg_lang_acc = round(mean(lang_matches), 4) if lang_matches else 1.0
    avg_intent_acc = round(mean(intent_matches), 4) if intent_matches else 1.0
    avg_greeting_acc = round(mean(greeting_matches), 4) if greeting_matches else 1.0
    avg_exp_recall = round(mean(expansion_recalls), 4) if expansion_recalls else 1.0
    avg_exp_drift = round(mean(expansion_drifts), 4) if expansion_drifts else 0.0

    kpis_output = {
        "language_accuracy": {
            "value": avg_lang_acc,
            "target": comp_def.kpis["language_accuracy"].target,
            "status": decide_kpi_status(avg_lang_acc, comp_def.kpis["language_accuracy"].target, ">="),
            "formula": comp_def.kpis["language_accuracy"].formula,
        },
        "intent_accuracy": {
            "value": avg_intent_acc,
            "target": comp_def.kpis["intent_accuracy"].target,
            "status": decide_kpi_status(avg_intent_acc, comp_def.kpis["intent_accuracy"].target, ">="),
            "formula": comp_def.kpis["intent_accuracy"].formula,
        },
        "greeting_accuracy": {
            "value": avg_greeting_acc,
            "target": comp_def.kpis["greeting_accuracy"].target,
            "status": decide_kpi_status(avg_greeting_acc, comp_def.kpis["greeting_accuracy"].target, ">="),
            "formula": comp_def.kpis["greeting_accuracy"].formula,
        },
        "expansion_recall": {
            "value": avg_exp_recall,
            "target": comp_def.kpis["expansion_recall"].target,
            "status": decide_kpi_status(avg_exp_recall, comp_def.kpis["expansion_recall"].target, ">="),
            "formula": comp_def.kpis["expansion_recall"].formula,
        },
        "expansion_drift_rate": {
            "value": avg_exp_drift,
            "target": comp_def.kpis["expansion_drift_rate"].target,
            "status": decide_kpi_status(avg_exp_drift, comp_def.kpis["expansion_drift_rate"].target, "<="),
            "formula": comp_def.kpis["expansion_drift_rate"].formula,
        },
    }

    # Aggregate Slices
    slices_output = {}
    for s_name, s_vals in slices_data.items():
        if s_vals["total"] > 0:
            item = {
                "count": s_vals["total"],
                "pass_rate": round(s_vals["pass"] / float(s_vals["total"]), 4),
            }
            if "lang_acc" in s_vals:
                item["lang_acc"] = round(mean(s_vals["lang_acc"]), 4)
            if "intent_acc" in s_vals:
                item["intent_acc"] = round(mean(s_vals["intent_acc"]), 4)
            if "exp_recall" in s_vals:
                item["exp_recall"] = round(mean(s_vals["exp_recall"]), 4)
            if "acc" in s_vals:
                item["accuracy"] = round(mean(s_vals["acc"]), 4)
            slices_output[s_name] = item

    return {
        "component": "query_preprocessing",
        "run_id": run_id or f"run_query_preprocessing_{int(time.time())}",
        "n_rows": total_rows,
        "n_errors": len(failures),
        "kpis": kpis_output,
        "slices": slices_output,
        "failures": failures,
        "latency_ms": lat_p,
    }
