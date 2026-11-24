"""Pure, unit-testable evaluation metrics for RAG components.

Zero external heavy dependencies; mathematically precise implementations
of Information Retrieval, Classification, OCR, and Latency metrics.
"""
from __future__ import annotations

import math
from typing import Sequence, Set, Any, Iterable


# ── Retrieval Metrics ─────────────────────────────────────────────────────────

def recall_at_k(retrieved: Sequence[str], gold: Sequence[str] | Set[str], k: int) -> float:
    """Proportion of gold documents present in top-k retrieved list.
    
    Formula: |retrieved[:k] ∩ gold| / |gold|
    """
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & gold_set) / float(len(gold_set))


def hit_at_k(retrieved: Sequence[str], gold: Sequence[str] | Set[str], k: int) -> float:
    """Binary hit indicator: 1.0 if at least one gold document is in top-k, else 0.0."""
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    for doc_id in retrieved[:k]:
        if doc_id in gold_set:
            return 1.0
    return 0.0


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str] | Set[str]) -> float:
    """Reciprocal rank of the first relevant document.
    
    Formula: 1 / rank_first_relevant (1-indexed), or 0.0 if none found.
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str] | Set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k with binary relevance.
    
    DCG@k = sum_{i=1}^k [rel_i / log2(i + 1)]
    IDCG@k = sum_{i=1}^{min(k, |gold|)} [1 / log2(i + 1)]
    nDCG@k = DCG@k / IDCG@k
    """
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in gold_set:
            dcg += 1.0 / math.log2(i + 1)

    idcg_count = min(k, len(gold_set))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, idcg_count + 1))

    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def exact_match_hit(retrieved: Sequence[str], gold: Sequence[str] | Set[str], k: int = 1) -> float:
    """Exact match hit rate: 1.0 if top-k contains the primary gold target, else 0.0."""
    return hit_at_k(retrieved, gold, k=k)


def compute_lift(target_metric: float, baseline_metric: float) -> float:
    """Percentage lift of target over baseline.
    
    Formula: ((target - baseline) / baseline) * 100
    """
    if baseline_metric <= 0:
        return 100.0 if target_metric > 0 else 0.0
    return round(((target_metric - baseline_metric) / baseline_metric) * 100.0, 4)


# ── Latency Metrics ──────────────────────────────────────────────────────────

def percentiles(values: Sequence[float], ps: Sequence[float] = (50, 95, 99)) -> dict[str, float]:
    """Compute exact percentile ranks (p50, p95, p99) for an array of values."""
    if not values:
        return {f"p{int(p)}": 0.0 for p in ps}
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    results = {}
    
    for p in ps:
        if n == 1:
            results[f"p{int(p)}"] = round(float(sorted_vals[0]), 2)
            continue
        # Nearest rank / linear interpolation
        k = (p / 100.0) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            val = sorted_vals[int(k)]
        else:
            d0 = sorted_vals[int(f)] * (c - k)
            d1 = sorted_vals[int(c)] * (k - f)
            val = d0 + d1
        results[f"p{int(p)}"] = round(float(val), 2)
        
    return results


# ── Classification & Guardrail Metrics ───────────────────────────────────────

def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Calculate precision, recall, and F1 score from raw counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def false_positive_negative_rates(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    """Calculate False Positive Rate (FPR / false block) and False Negative Rate (FNR / miss rate).
    
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    """
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
    }


def confusion_matrix(y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any] | None = None) -> dict[str, Any]:
    """Build a confusion matrix dictionary {label_true: {label_pred: count}}."""
    all_labels = list(labels) if labels is not None else sorted(list(set(y_true) | set(y_pred)))
    matrix = {str(lt): {str(lp): 0 for lp in all_labels} for lt in all_labels}
    for yt, yp in zip(y_true, y_pred):
        st_yt, st_yp = str(yt), str(yp)
        if st_yt in matrix and st_yp in matrix[st_yt]:
            matrix[st_yt][st_yp] += 1
    return {"labels": [str(l) for l in all_labels], "matrix": matrix}


# ── OCR & Edit Distance Metrics ──────────────────────────────────────────────

def levenshtein_distance(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    """Compute standard Levenshtein edit distance between two sequences (chars or words)."""
    if seq1 == seq2:
        return 0
    l1, l2 = len(seq1), len(seq2)
    if l1 == 0:
        return l2
    if l2 == 0:
        return l1

    dp = [list(range(l2 + 1))] + [[i] + [0] * l2 for i in range(1, l1 + 1)]
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )
    return dp[l1][l2]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Character Error Rate (CER): Levenshtein distance on characters / len(reference)."""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return levenshtein_distance(ref_chars, hyp_chars) / float(len(ref_chars))


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word Error Rate (WER): Levenshtein distance on whitespace-tokenized words."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein_distance(ref_words, hyp_words) / float(len(ref_words))


# ── Aggregator Helper ────────────────────────────────────────────────────────

def mean(values: Iterable[float]) -> float:
    """Compute arithmetic mean of an iterable of numbers, returning 0.0 if empty."""
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / float(len(vals))
