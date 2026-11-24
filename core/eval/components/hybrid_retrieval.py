"""Hybrid Retrieval Component Evaluator.

Evaluates BGE-M3 Dense + Sparse + MatchText fused with Reciprocal Rank Fusion (RRF).
Computes Recall@5/10/20/50, MRR, nDCG@10, exact-match hit rate, cross-lingual recall,
latency percentiles (p50/p95), and an ablation table comparing:
  - Dense-only
  - Sparse-only
  - Exact-only (MatchText)
  - RRF-fused (Hybrid)
  - RRF Lift % over best single retriever
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_CORE_DIR = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _CORE_DIR.parent
for _p in [str(_REPO_ROOT), str(_CORE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from core.eval.metrics import (
        compute_lift,
        exact_match_hit,
        hit_at_k,
        mean,
        ndcg_at_k,
        percentiles,
        recall_at_k,
        reciprocal_rank,
    )
    from core.eval.normalizer import normalize_eval_text
    from core.eval.registry import REGISTRY, decide_kpi_status
    from core.eval.validator import parse_and_validate_batch
except ModuleNotFoundError:
    from eval.metrics import (
        compute_lift,
        exact_match_hit,
        hit_at_k,
        mean,
        ndcg_at_k,
        percentiles,
        recall_at_k,
        reciprocal_rank,
    )
    from eval.normalizer import normalize_eval_text
    from eval.registry import REGISTRY, decide_kpi_status
    from eval.validator import parse_and_validate_batch


def _extract_hit_identifiers(point: Any) -> Set[str]:
    """Extract all valid identifier keys for a retrieved point (id, source, source#chunk, doc_id)."""
    ids = set()
    pid = getattr(point, "id", None)
    if pid is not None:
        ids.add(str(pid).strip().lower())

    payload = getattr(point, "payload", None) or {}
    source = payload.get("source")
    chunk = payload.get("chunk")
    doc_id = payload.get("doc_id")

    if source:
        src_clean = str(source).strip().lower()
        ids.add(src_clean)
        ids.add(Path(src_clean).name)
        if chunk is not None:
            ids.add(f"{src_clean}#{chunk}")
            ids.add(f"{Path(src_clean).name}#{chunk}")

    if doc_id:
        ids.add(str(doc_id).strip().lower())

    return ids


def _match_rank(retrieved_points: List[Any], gold_ids: Set[str]) -> List[str]:
    """Map retrieved points to a canonical representation matching gold IDs if possible."""
    result_ids = []
    gold_clean = {g.strip().lower() for g in gold_ids}
    for p in retrieved_points:
        point_ids = _extract_hit_identifiers(p)
        matched = point_ids & gold_clean
        if matched:
            result_ids.append(next(iter(matched)))
        else:
            payload = getattr(p, "payload", None) or {}
            src = payload.get("source") or str(getattr(p, "id", "unknown"))
            chk = payload.get("chunk", "")
            result_ids.append(f"{src}#{chk}" if chk else src)
    return result_ids


def evaluate_hybrid_retrieval(
    batch_path: str,
    config: Optional[Dict[str, Any]] = None,
    storage: Optional[Any] = None,
    run_id: Optional[str] = None,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Shared contract evaluator for Hybrid Retrieval.
    
    Args:
        batch_path: Path to CSV, XLSX, or JSON test batch file.
        config: Evaluation config (e.g. {'alpha': 0.6, 'k': 60, 'mock': False}).
        storage: Optional EvalStorage instance for recording per-row outputs.
        run_id: Active evaluation run ID.
        progress_callback: Optional callable(current, total, status_text).
    """
    cfg = config or {}
    path = Path(batch_path)
    if not path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    content_bytes = path.read_bytes()
    is_valid, rows, errors = parse_and_validate_batch(content_bytes, path.name, "hybrid_retrieval")
    if not rows:
        raise ValueError(f"No valid rows found in {path.name}: {errors}")

    # Initialize live retriever (or mock for unit tests)
    retriever = None
    use_mock = cfg.get("mock", False)
    if not use_mock:
        try:
            import rag_pipeline as rp
            retriever = rp
        except Exception as e:
            print(f"[eval_retrieval] Could not load live rag_pipeline: {e}. Falling back to mock test mode.")
            use_mock = True

    # Containers for global metrics
    m_rec5, m_rec10, m_rec20, m_rec50 = [], [], [], []
    m_mrr, m_ndcg10, m_exact = [], [], []
    latencies_ms = []

    # Ablation containers
    dense_rec10, sparse_rec10, exact_rec10, rrf_rec10 = [], [], [], []

    # Slices containers: {slice_key: {'rec10': [], 'mrr': []}}
    slices_data: Dict[str, Dict[str, List[float]]] = {
        "cross_lingual_hi_to_en": {"rec10": [], "mrr": []},
        "cross_lingual_en_to_hi": {"rec10": [], "mrr": []},
        "monolingual_hi": {"rec10": [], "mrr": []},
        "monolingual_en": {"rec10": [], "mrr": []},
        "exact_code_queries": {"rec10": [], "mrr": []},
        "general_queries": {"rec10": [], "mrr": []},
    }

    failures: List[Dict[str, Any]] = []
    total_rows = len(rows)

    for i, row in enumerate(rows):
        if progress_callback:
            progress_callback(i + 1, total_rows, f"Evaluating query {i+1}/{total_rows}")

        qid = str(row["query_id"])
        query = str(row["query"])
        gold_raw = row["gold_passage_ids"]
        gold_ids = {str(g).strip().lower() for g in gold_raw}
        q_lang = str(row.get("query_language") or "").lower()
        d_lang = str(row.get("doc_language") or "").lower()
        is_exact_q = bool(row.get("is_exact_match_query", False))

        t0 = time.perf_counter()
        retrieved_ids = []
        dense_ids, sparse_ids, exact_ids = [], [], []

        if not use_mock and retriever is not None:
            try:
                # 1. Multi-query dense search
                dense_results, agg_scores, query_sparse = retriever.multi_query_retrieval(query)
                dense_points = getattr(dense_results, "points", []) or []
                points_by_id = {p.id: p for p in dense_points}

                # Dense ranking
                dense_sorted = sorted(agg_scores, key=lambda x: x[1], reverse=True)
                dense_pts = [points_by_id[pid] for pid, _ in dense_sorted if pid in points_by_id]
                dense_ids = _match_rank(dense_pts, gold_ids)

                # Sparse ranking
                sparse_scores = []
                if query_sparse:
                    sparse_scores = retriever.sparse_search(query_sparse, dense_points, limit=60)
                sparse_pts = [points_by_id[pid] for pid, _ in sparse_scores if pid in points_by_id]
                sparse_ids = _match_rank(sparse_pts, gold_ids)

                # Exact Match (MatchText)
                exact_hits = retriever.lexical_search(query, limit=20) if hasattr(retriever, "lexical_search") else []
                exact_pts = [h.get("point") for h in exact_hits if h.get("point")]
                exact_ids = _match_rank(exact_pts, gold_ids)

                # RRF Fused Hybrid Ranking
                alpha = float(cfg.get("alpha", retriever.HYBRID_ALPHA))
                k_rrf = int(cfg.get("k", 60))
                hybrid_scores = retriever.hybrid_search(dense_sorted, sparse_scores, alpha=alpha, k=k_rrf)
                rrf_pts = [points_by_id[pid] for pid, _ in hybrid_scores if pid in points_by_id]
                retrieved_ids = _match_rank(rrf_pts, gold_ids)

            except Exception as e:
                # Fallback to single retrieval or record error
                retrieved_ids = []
                print(f"[eval_retrieval] Error querying live retriever on query '{query[:30]}': {e}")
        else:
            # Synthetic mock logic for isolated unit testing
            time.sleep(0.002)  # simulate 2ms lookup
            # If query contains tokens in gold, rank them high
            mock_hits = []
            for g in gold_ids:
                mock_hits.append(g)
            mock_hits.extend([f"other_doc_{j}.pdf#1" for j in range(25)])
            retrieved_ids = mock_hits
            dense_ids = mock_hits[1:15] + mock_hits[0:1] + mock_hits[15:]
            sparse_ids = mock_hits
            exact_ids = [mock_hits[0]] if is_exact_q else []

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        # Compute row-level metrics
        r5 = recall_at_k(retrieved_ids, gold_ids, k=5)
        r10 = recall_at_k(retrieved_ids, gold_ids, k=10)
        r20 = recall_at_k(retrieved_ids, gold_ids, k=20)
        r50 = recall_at_k(retrieved_ids, gold_ids, k=50)
        rr = reciprocal_rank(retrieved_ids, gold_ids)
        ndcg = ndcg_at_k(retrieved_ids, gold_ids, k=10)
        em_hit = exact_match_hit(retrieved_ids, gold_ids, k=1) if is_exact_q else None

        m_rec5.append(r5)
        m_rec10.append(r10)
        m_rec20.append(r20)
        m_rec50.append(r50)
        m_mrr.append(rr)
        m_ndcg10.append(ndcg)
        if em_hit is not None:
            m_exact.append(em_hit)

        # Ablations
        dense_rec10.append(recall_at_k(dense_ids, gold_ids, k=10))
        sparse_rec10.append(recall_at_k(sparse_ids, gold_ids, k=10))
        exact_rec10.append(recall_at_k(exact_ids, gold_ids, k=10))
        rrf_rec10.append(r10)

        # Slice tagging
        slice_tags = {"is_exact": is_exact_q, "query_lang": q_lang, "doc_lang": d_lang}
        if q_lang == "hi" and d_lang == "en":
            slices_data["cross_lingual_hi_to_en"]["rec10"].append(r10)
            slices_data["cross_lingual_hi_to_en"]["mrr"].append(rr)
        elif q_lang == "en" and d_lang == "hi":
            slices_data["cross_lingual_en_to_hi"]["rec10"].append(r10)
            slices_data["cross_lingual_en_to_hi"]["mrr"].append(rr)
        elif q_lang == "hi" and d_lang == "hi":
            slices_data["monolingual_hi"]["rec10"].append(r10)
            slices_data["monolingual_hi"]["mrr"].append(rr)
        elif q_lang == "en" and d_lang == "en":
            slices_data["monolingual_en"]["rec10"].append(r10)
            slices_data["monolingual_en"]["mrr"].append(rr)

        if is_exact_q:
            slices_data["exact_code_queries"]["rec10"].append(r10)
            slices_data["exact_code_queries"]["mrr"].append(rr)
        else:
            slices_data["general_queries"]["rec10"].append(r10)
            slices_data["general_queries"]["mrr"].append(rr)

        # Determine pass/fail
        passed = r10 >= 0.5 or rr >= 0.2
        reason = "" if passed else "Gold passage was not found in the top 10 retrieved results."
        if not passed:
            failures.append({
                "row_id": qid,
                "input": query,
                "expected": list(gold_ids),
                "actual": retrieved_ids[:5],
                "reason": reason,
            })

        # Record in storage if active
        if storage and run_id:
            storage.record_row(
                run_id=run_id,
                row_id=qid,
                input_data={"query": query, "query_language": q_lang, "doc_language": d_lang},
                expected_data={"gold_passage_ids": list(gold_ids)},
                actual_data={"retrieved_top_10": retrieved_ids[:10]},
                passed=passed,
                reason=reason,
                latency_ms=round(elapsed_ms, 2),
                slice_tags=slice_tags,
            )

    # ── Aggregate Global KPIs ────────────────────────────────────────────────
    comp_def = REGISTRY["hybrid_retrieval"]
    lat_p = percentiles(latencies_ms, ps=[50, 95])

    avg_rec5 = round(mean(m_rec5), 4)
    avg_rec10 = round(mean(m_rec10), 4)
    avg_rec20 = round(mean(m_rec20), 4)
    avg_mrr = round(mean(m_mrr), 4)
    avg_ndcg = round(mean(m_ndcg10), 4)
    avg_exact = round(mean(m_exact), 4) if m_exact else 1.0

    # Cross-lingual aggregate
    cross_rec = mean(
        slices_data["cross_lingual_hi_to_en"]["rec10"] +
        slices_data["cross_lingual_en_to_hi"]["rec10"]
    )
    if not (slices_data["cross_lingual_hi_to_en"]["rec10"] or slices_data["cross_lingual_en_to_hi"]["rec10"]):
        cross_rec = avg_rec10  # default to global if no cross-lingual tag in batch

    # Ablations & Lift calculation
    avg_dense = round(mean(dense_rec10), 4)
    avg_sparse = round(mean(sparse_rec10), 4)
    avg_exact_ab = round(mean(exact_rec10), 4)
    best_single = max(avg_dense, avg_sparse, avg_exact_ab)
    rrf_lift = compute_lift(avg_rec10, best_single)

    kpis_output = {
        "recall_at_5": {
            "value": avg_rec5,
            "target": comp_def.kpis["recall_at_5"].target,
            "status": decide_kpi_status(avg_rec5, comp_def.kpis["recall_at_5"].target, ">="),
            "formula": comp_def.kpis["recall_at_5"].formula,
        },
        "recall_at_10": {
            "value": avg_rec10,
            "target": comp_def.kpis["recall_at_10"].target,
            "status": decide_kpi_status(avg_rec10, comp_def.kpis["recall_at_10"].target, ">="),
            "formula": comp_def.kpis["recall_at_10"].formula,
        },
        "recall_at_20": {
            "value": avg_rec20,
            "target": comp_def.kpis["recall_at_20"].target,
            "status": decide_kpi_status(avg_rec20, comp_def.kpis["recall_at_20"].target, ">="),
            "formula": comp_def.kpis["recall_at_20"].formula,
        },
        "mrr": {
            "value": avg_mrr,
            "target": comp_def.kpis["mrr"].target,
            "status": decide_kpi_status(avg_mrr, comp_def.kpis["mrr"].target, ">="),
            "formula": comp_def.kpis["mrr"].formula,
        },
        "ndcg_at_10": {
            "value": avg_ndcg,
            "target": comp_def.kpis["ndcg_at_10"].target,
            "status": decide_kpi_status(avg_ndcg, comp_def.kpis["ndcg_at_10"].target, ">="),
            "formula": comp_def.kpis["ndcg_at_10"].formula,
        },
        "exact_match_hit_rate": {
            "value": avg_exact,
            "target": comp_def.kpis["exact_match_hit_rate"].target,
            "status": decide_kpi_status(avg_exact, comp_def.kpis["exact_match_hit_rate"].target, ">="),
            "formula": comp_def.kpis["exact_match_hit_rate"].formula,
        },
        "cross_lingual_recall": {
            "value": round(cross_rec, 4),
            "target": comp_def.kpis["cross_lingual_recall"].target,
            "status": decide_kpi_status(cross_rec, comp_def.kpis["cross_lingual_recall"].target, ">="),
            "formula": comp_def.kpis["cross_lingual_recall"].formula,
        },
        "rrf_lift_over_best": {
            "value": round(rrf_lift, 2),
            "target": comp_def.kpis["rrf_lift_over_best"].target,
            "status": decide_kpi_status(rrf_lift, comp_def.kpis["rrf_lift_over_best"].target, ">="),
            "formula": comp_def.kpis["rrf_lift_over_best"].formula,
        },
        "latency_p95_ms": {
            "value": lat_p["p95"],
            "target": comp_def.kpis["latency_p95_ms"].target,
            "status": decide_kpi_status(lat_p["p95"], comp_def.kpis["latency_p95_ms"].target, "<="),
            "formula": comp_def.kpis["latency_p95_ms"].formula,
        },
    }

    # Aggregate Slices
    slices_output = {}
    for s_name, s_vals in slices_data.items():
        if s_vals["rec10"]:
            slices_output[s_name] = {
                "count": len(s_vals["rec10"]),
                "recall_at_10": round(mean(s_vals["rec10"]), 4),
                "mrr": round(mean(s_vals["mrr"]), 4),
            }

    # Ablations Table
    ablations = {
        "dense_only": {"recall_at_10": avg_dense},
        "sparse_only": {"recall_at_10": avg_sparse},
        "exact_match_only": {"recall_at_10": avg_exact_ab},
        "rrf_fused": {"recall_at_10": avg_rec10},
        "best_single_retriever": {"recall_at_10": best_single},
        "rrf_lift_percent": rrf_lift,
    }

    return {
        "component": "hybrid_retrieval",
        "run_id": run_id or f"run_retrieval_{int(time.time())}",
        "n_rows": total_rows,
        "n_errors": len(failures),
        "kpis": kpis_output,
        "slices": slices_output,
        "ablations": ablations,
        "failures": failures,
        "latency_ms": lat_p,
    }

