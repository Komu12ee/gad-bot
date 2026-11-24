"""Evaluation framework for CHiPS-RAG pipeline components."""
import sys
from pathlib import Path

_CORE_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CORE_DIR.parent
for _p in [str(_REPO_ROOT), str(_CORE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from core.eval.metrics import (
        recall_at_k,
        hit_at_k,
        reciprocal_rank,
        ndcg_at_k,
        exact_match_hit,
        compute_lift,
        percentiles,
        precision_recall_f1,
        character_error_rate,
        word_error_rate,
    )
    from core.eval.registry import REGISTRY, ComponentDef, ColumnDef, KpiDef
    from core.eval.storage import EvalStorage
    from core.eval.validator import parse_and_validate_batch
except ModuleNotFoundError:
    from eval.metrics import (
        recall_at_k,
        hit_at_k,
        reciprocal_rank,
        ndcg_at_k,
        exact_match_hit,
        compute_lift,
        percentiles,
        precision_recall_f1,
        character_error_rate,
        word_error_rate,
    )
    from eval.registry import REGISTRY, ComponentDef, ColumnDef, KpiDef
    from eval.storage import EvalStorage
    from eval.validator import parse_and_validate_batch

__all__ = [
    "recall_at_k",
    "hit_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "exact_match_hit",
    "compute_lift",
    "percentiles",
    "precision_recall_f1",
    "character_error_rate",
    "word_error_rate",
    "REGISTRY",
    "ComponentDef",
    "ColumnDef",
    "KpiDef",
    "EvalStorage",
    "parse_and_validate_batch",
]

