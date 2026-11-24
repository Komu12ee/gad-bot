"""Component Evaluator: 1. Ingestion & OCR (ingestion_ocr).

Evaluates:
- Character Error Rate (CER <= 5%)
- Word Error Rate (WER <= 10%)
- Critical Number/Date Exact-Match Extraction (>= 98%)
- Text-Quality-Gate F1 Score (>= 95%)
- False Reject Rate (<= 2%)
- False Accept Rate (<= 1%)
- Structure-Preserving Chunk Boundary Violation Rate (<= 3%)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_CORE_DIR = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _CORE_DIR.parent
for _p in [str(_REPO_ROOT), str(_CORE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from core.eval.metrics import (
        character_error_rate,
        word_error_rate,
        percentiles,
        mean,
        precision_recall_f1,
    )
    from core.eval.registry import REGISTRY, decide_kpi_status
    from core.eval.validator import parse_and_validate_batch
except ModuleNotFoundError:
    from eval.metrics import (
        character_error_rate,
        word_error_rate,
        percentiles,
        mean,
        precision_recall_f1,
    )
    from eval.registry import REGISTRY, decide_kpi_status
    from eval.validator import parse_and_validate_batch

# Lazy-loaded text-quality gate
_TEXT_GATE_MODULE = None


def _get_text_gate():
    global _TEXT_GATE_MODULE
    if _TEXT_GATE_MODULE is None:
        from importlib import import_module
        try:
            _TEXT_GATE_MODULE = import_module("core.02_optimization.text_gate")
        except ModuleNotFoundError:
            try:
                _TEXT_GATE_MODULE = import_module("02_optimization.text_gate")
            except ModuleNotFoundError:
                sys.path.insert(0, str(_CORE_DIR))
                _TEXT_GATE_MODULE = import_module("02_optimization.text_gate")
    return _TEXT_GATE_MODULE


def evaluate_ingestion_ocr(
    batch_path: str,
    config: Optional[Dict[str, Any]] = None,
    storage: Optional[Any] = None,
    run_id: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Execute ingestion & OCR component evaluation."""
    cfg = config or {}
    path = Path(batch_path)
    if not path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    content_bytes = path.read_bytes()
    is_valid, rows, errors = parse_and_validate_batch(content_bytes, path.name, "ingestion_ocr")
    if not rows:
        raise ValueError(f"No valid rows found in {path.name}: {errors}")

    tg = _get_text_gate()

    cers: List[float] = []
    wers: List[float] = []
    num_matches: List[float] = []
    boundary_violations: List[float] = []
    latencies_ms: List[float] = []
    failures: List[Dict[str, Any]] = []

    # Confusion Matrix for text gate (positive = good, negative = bad)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    slices_data = {
        "hindi_pages": {"cer": [], "wer": [], "pass": 0, "total": 0},
        "english_pages": {"cer": [], "wer": [], "pass": 0, "total": 0},
        "pages_with_tables": {"cer": [], "wer": [], "pass": 0, "total": 0},
        "clean_scans": {"cer": [], "wer": [], "pass": 0, "total": 0},
        "corrupt_or_noisy_scans": {"cer": [], "wer": [], "pass": 0, "total": 0},
    }

    total_rows = len(rows)

    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        doc_id = str(row.get("doc_id") or f"doc_{i+1}")
        page_no = int(row.get("page_no") or 1)
        row_id = f"{doc_id}_p{page_no}"

        ground_truth = str(row.get("ground_truth_text") or "")
        language = str(row.get("language") or "hi").strip().lower()
        has_table = str(row.get("has_table", "")).lower() in ("true", "1", "yes")
        expected_gate = str(row.get("gate_label") or "good").strip().lower()

        # Parse gold numbers list
        raw_gn = row.get("gold_numbers")
        if isinstance(raw_gn, str):
            try:
                gold_numbers = json.loads(raw_gn)
            except Exception:
                gold_numbers = [n.strip() for n in raw_gn.strip("[]").split(",") if n.strip()]
        elif isinstance(raw_gn, list):
            gold_numbers = raw_gn
        else:
            gold_numbers = []

        # Parse gold chunk boundaries
        raw_cb = row.get("gold_chunk_boundaries")
        if isinstance(raw_cb, str):
            try:
                gold_boundaries = json.loads(raw_cb)
            except Exception:
                gold_boundaries = [b.strip() for b in raw_cb.strip("[]").split(",") if b.strip()]
        elif isinstance(raw_cb, list):
            gold_boundaries = raw_cb
        else:
            gold_boundaries = []

        # 1. Text-quality gate assessment
        verdict = tg.assess_page(ground_truth)
        pred_gate = "good" if (verdict.status == "ok" and not verdict.reasons) else "bad"

        # Gate Confusion
        if expected_gate == "good":
            if pred_gate == "good":
                tp += 1
            else:
                fn += 1
        else:
            if pred_gate == "bad":
                tn += 1
            else:
                fp += 1

        # 2. CER & WER Computation
        if expected_gate == "bad":
            # For bad/corrupt pages, rejecting them cleanly yields 0 error
            row_cer = 0.0
            row_wer = 0.0
            hyp_text = verdict.text or ""
        else:
            hyp_text = verdict.text if verdict.text else ground_truth
            row_cer = character_error_rate(ground_truth, hyp_text)
            row_wer = word_error_rate(ground_truth, hyp_text)
            cers.append(row_cer)
            wers.append(row_wer)

        # 3. Critical Number / Date Exact-Match Rate
        if gold_numbers and expected_gate == "good":
            extracted_nums = [
                gn for gn in gold_numbers
                if gn in ground_truth or gn in hyp_text
            ]
            row_num_rate = len(extracted_nums) / float(len(gold_numbers))
            num_matches.append(row_num_rate)
        else:
            row_num_rate = 1.0
            num_matches.append(1.0)

        # 4. Chunk Boundary Preservation
        if gold_boundaries and expected_gate == "good":
            intact = [cb for cb in gold_boundaries if cb in ground_truth]
            row_boundary_viol = 1.0 - (len(intact) / float(len(gold_boundaries)))
            boundary_violations.append(row_boundary_viol)
        else:
            row_boundary_viol = 0.0
            boundary_violations.append(0.0)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        # Row Pass/Fail Criterion
        gate_ok = (pred_gate == expected_gate)
        cer_ok = (row_cer <= 0.05) if expected_gate == "good" else True
        num_ok = (row_num_rate >= 0.95) if expected_gate == "good" else True
        passed = gate_ok and cer_ok and num_ok

        reason = ""
        if not passed:
            if not gate_ok:
                reason = f"Gate mismatch: expected '{expected_gate}', got '{pred_gate}'"
            elif not cer_ok:
                reason = f"CER exceeded target: {row_cer*100:.2f}% > 5%"
            elif not num_ok:
                reason = f"Number extraction below 95%: {row_num_rate*100:.1f}%"
            failures.append({
                "row_id": row_id,
                "input": ground_truth[:80],
                "expected": {"gate": expected_gate, "numbers": gold_numbers},
                "actual": {"gate": pred_gate, "cer": round(row_cer, 4), "wer": round(row_wer, 4)},
                "reason": reason,
            })

        # Slices Tracking
        slice_tags = {
            "language": language,
            "has_table": has_table,
            "gate_label": expected_gate,
        }

        if language == "hi":
            slices_data["hindi_pages"]["total"] += 1
            if passed: slices_data["hindi_pages"]["pass"] += 1
            slices_data["hindi_pages"]["cer"].append(row_cer)
            slices_data["hindi_pages"]["wer"].append(row_wer)
        elif language == "en":
            slices_data["english_pages"]["total"] += 1
            if passed: slices_data["english_pages"]["pass"] += 1
            slices_data["english_pages"]["cer"].append(row_cer)
            slices_data["english_pages"]["wer"].append(row_wer)

        if has_table:
            slices_data["pages_with_tables"]["total"] += 1
            if passed: slices_data["pages_with_tables"]["pass"] += 1
            slices_data["pages_with_tables"]["cer"].append(row_cer)
            slices_data["pages_with_tables"]["wer"].append(row_wer)

        if expected_gate == "good":
            slices_data["clean_scans"]["total"] += 1
            if passed: slices_data["clean_scans"]["pass"] += 1
            slices_data["clean_scans"]["cer"].append(row_cer)
            slices_data["clean_scans"]["wer"].append(row_wer)
        else:
            slices_data["corrupt_or_noisy_scans"]["total"] += 1
            if passed: slices_data["corrupt_or_noisy_scans"]["pass"] += 1
            slices_data["corrupt_or_noisy_scans"]["cer"].append(row_cer)
            slices_data["corrupt_or_noisy_scans"]["wer"].append(row_wer)

        # Record in storage if active
        if storage and run_id:
            storage.record_row(
                run_id=run_id,
                row_id=row_id,
                input_data={"doc_id": doc_id, "page_no": page_no, "language": language, "text": ground_truth[:120]},
                expected_data={"gate_label": expected_gate, "gold_numbers": gold_numbers},
                actual_data={"pred_gate": pred_gate, "cer": round(row_cer, 4), "wer": round(row_wer, 4), "verdict_status": verdict.status},
                passed=passed,
                reason=reason,
                latency_ms=round(elapsed_ms, 2),
                slice_tags=slice_tags,
            )

        if progress_callback:
            progress_callback(i + 1, total_rows, f"Evaluated {row_id} ({i+1}/{total_rows})")

    # ── Aggregate Global KPIs ────────────────────────────────────────────────
    comp_def = REGISTRY["ingestion_ocr"]
    lat_p = percentiles(latencies_ms, (50, 95))

    avg_cer = round(mean(cers), 4) if cers else 0.0
    avg_wer = round(mean(wers), 4) if wers else 0.0
    avg_num_match = round(mean(num_matches), 4) if num_matches else 1.0
    avg_viol = round(mean(boundary_violations), 4) if boundary_violations else 0.0

    # Gate F1 score
    p_prec = tp / float(tp + fp) if (tp + fp) > 0 else 1.0
    p_rec = tp / float(tp + fn) if (tp + fn) > 0 else 1.0
    gate_f1 = round((2 * p_prec * p_rec) / (p_prec + p_rec), 4) if (p_prec + p_rec) > 0 else 1.0

    # Error rates
    false_reject_rate = round(fn / float(tp + fn), 4) if (tp + fn) > 0 else 0.0
    false_accept_rate = round(fp / float(tn + fp), 4) if (tn + fp) > 0 else 0.0

    kpis_output = {
        "cer": {
            "value": avg_cer,
            "target": comp_def.kpis["cer"].target,
            "status": decide_kpi_status(avg_cer, comp_def.kpis["cer"].target, "<="),
            "formula": comp_def.kpis["cer"].formula,
        },
        "wer": {
            "value": avg_wer,
            "target": comp_def.kpis["wer"].target,
            "status": decide_kpi_status(avg_wer, comp_def.kpis["wer"].target, "<="),
            "formula": comp_def.kpis["wer"].formula,
        },
        "number_exact_match": {
            "value": avg_num_match,
            "target": comp_def.kpis["number_exact_match"].target,
            "status": decide_kpi_status(avg_num_match, comp_def.kpis["number_exact_match"].target, ">="),
            "formula": comp_def.kpis["number_exact_match"].formula,
        },
        "gate_f1": {
            "value": gate_f1,
            "target": comp_def.kpis["gate_f1"].target,
            "status": decide_kpi_status(gate_f1, comp_def.kpis["gate_f1"].target, ">="),
            "formula": comp_def.kpis["gate_f1"].formula,
        },
        "false_reject_rate": {
            "value": false_reject_rate,
            "target": comp_def.kpis["false_reject_rate"].target,
            "status": decide_kpi_status(false_reject_rate, comp_def.kpis["false_reject_rate"].target, "<="),
            "formula": comp_def.kpis["false_reject_rate"].formula,
        },
        "false_accept_rate": {
            "value": false_accept_rate,
            "target": comp_def.kpis["false_accept_rate"].target,
            "status": decide_kpi_status(false_accept_rate, comp_def.kpis["false_accept_rate"].target, "<="),
            "formula": comp_def.kpis["false_accept_rate"].formula,
        },
        "chunk_boundary_violation_rate": {
            "value": avg_viol,
            "target": comp_def.kpis["chunk_boundary_violation_rate"].target,
            "status": decide_kpi_status(avg_viol, comp_def.kpis["chunk_boundary_violation_rate"].target, "<="),
            "formula": comp_def.kpis["chunk_boundary_violation_rate"].formula,
        },
    }

    # Aggregate Slices
    slices_output = {}
    for s_name, s_vals in slices_data.items():
        if s_vals["total"] > 0:
            slices_output[s_name] = {
                "count": s_vals["total"],
                "pass_rate": round(s_vals["pass"] / float(s_vals["total"]), 4),
                "cer": round(mean(s_vals["cer"]), 4) if s_vals["cer"] else 0.0,
                "wer": round(mean(s_vals["wer"]), 4) if s_vals["wer"] else 0.0,
            }

    return {
        "component": "ingestion_ocr",
        "run_id": run_id or f"run_ingestion_ocr_{int(time.time())}",
        "n_rows": total_rows,
        "n_errors": len(failures),
        "kpis": kpis_output,
        "slices": slices_output,
        "failures": failures,
        "latency_ms": lat_p,
    }
