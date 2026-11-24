"""Registry of evaluatable components, schemas, KPI definitions, and default targets.

Defines schemas and targets for all 8 components in the CHiPS-RAG pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List, Optional


@dataclass
class ColumnDef:
    name: str
    required: bool = True
    col_type: str = "string"  # "string", "int", "bool", "json_list", "json_dict"
    description: str = ""


@dataclass
class KpiDef:
    name: str
    target: float
    direction: str = ">="  # ">=" or "<="
    formula: str = ""
    description: str = ""
    unit: str = ""  # "", "%", "ms", "s"


@dataclass
class ComponentDef:
    id: str
    label: str
    description: str
    columns: List[ColumnDef]
    kpis: Dict[str, KpiDef]
    evaluate_fn: Optional[Callable[..., Any]] = None


# Status decider: pass if target met, warn within 5% of target, fail otherwise
def decide_kpi_status(value: float, target: float, direction: str = ">=") -> str:
    if direction == ">=":
        if value >= target:
            return "pass"
        if value >= target * 0.95:  # within 5%
            return "warn"
        return "fail"
    else:  # "<="
        if value <= target:
            return "pass"
        if value <= target * 1.05:  # within 5% margin
            return "warn"
        return "fail"


REGISTRY: Dict[str, ComponentDef] = {
    "ingestion_ocr": ComponentDef(
        id="ingestion_ocr",
        label="1. Ingestion & OCR",
        description="Evaluates VLM OCR, text-quality gate, and structure-preserving chunking.",
        columns=[
            ColumnDef("doc_id", required=True, description="Unique document ID"),
            ColumnDef("page_no", required=True, col_type="int", description="1-indexed page number"),
            ColumnDef("file_path", required=False, description="Relative path to test PDF/image"),
            ColumnDef("ground_truth_text", required=True, description="Verified ground-truth Markdown/text"),
            ColumnDef("language", required=True, description="'hi' (Hindi), 'en' (English), or 'mixed'"),
            ColumnDef("has_table", required=False, col_type="bool", description="Whether page contains tabular data"),
            ColumnDef("gate_label", required=True, description="Expected gate verdict ('good' or 'bad')"),
            ColumnDef("gold_numbers", required=False, col_type="json_list", description="JSON list of critical numbers/dates"),
            ColumnDef("gold_chunk_boundaries", required=False, col_type="json_list", description="Expected chunk boundary markers"),
        ],
        kpis={
            "cer": KpiDef("cer", 0.05, "<=", "Levenshtein(ref_chars, hyp_chars) / len(ref_chars)", "Character Error Rate", "%"),
            "wer": KpiDef("wer", 0.10, "<=", "Levenshtein(ref_words, hyp_words) / len(ref_words)", "Word Error Rate", "%"),
            "number_exact_match": KpiDef("number_exact_match", 0.98, ">=", "|extracted_nums ∩ gold_nums| / |gold_nums|", "Number/Date Exact-Match Rate", "%"),
            "gate_f1": KpiDef("gate_f1", 0.95, ">=", "2 * (P * R) / (P + R)", "Text-Quality-Gate F1 Score"),
            "false_reject_rate": KpiDef("false_reject_rate", 0.02, "<=", "False Rejects / Total Good Pages", "False Reject Rate", "%"),
            "false_accept_rate": KpiDef("false_accept_rate", 0.01, "<=", "False Accepts / Total Bad Pages", "False Accept Rate", "%"),
            "chunk_boundary_violation_rate": KpiDef("chunk_boundary_violation_rate", 0.03, "<=", "Cuts Inside Table or Sentence / Total Chunks", "Chunk Violation Rate", "%"),
        },
    ),
    "query_preprocessing": ComponentDef(
        id="query_preprocessing",
        label="2. Query Preprocessing",
        description="Evaluates language detection, intent classification, and bilingual glossary expansion.",
        columns=[
            ColumnDef("query_id", required=True, description="Unique query identifier"),
            ColumnDef("query", required=True, description="Raw input user query"),
            ColumnDef("true_language", required=True, description="Expected language ('hi', 'hinglish', 'en')"),
            ColumnDef("true_intent", required=True, description="Expected intent ('general', 'catalogue', 'count', 'number', 'greeting')"),
            ColumnDef("expected_expansion_terms", required=False, col_type="json_list", description="Expected bilingual glossary tokens"),
        ],
        kpis={
            "language_accuracy": KpiDef("language_accuracy", 0.98, ">=", "Correct Language Detections / Total Queries", "Language Detection Accuracy", "%"),
            "intent_accuracy": KpiDef("intent_accuracy", 0.95, ">=", "Correct Intent Predictions / Total Queries", "Intent Classification Accuracy", "%"),
            "greeting_accuracy": KpiDef("greeting_accuracy", 0.99, ">=", "Correct Greeting Classifications / Total Greetings", "Greeting Handling Accuracy", "%"),
            "expansion_recall": KpiDef("expansion_recall", 0.90, ">=", "|extracted_expansions ∩ expected| / |expected|", "Expansion Term Recall", "%"),
            "expansion_drift_rate": KpiDef("expansion_drift_rate", 0.03, "<=", "Spurious Expansions / Total Added Terms", "Expansion Semantic Drift Rate", "%"),
        },
    ),
    "hybrid_retrieval": ComponentDef(
        id="hybrid_retrieval",
        label="3. Hybrid Retrieval",
        description="Evaluates BGE-M3 Dense + Sparse + MatchText fused with Reciprocal Rank Fusion (RRF).",
        columns=[
            ColumnDef("query_id", required=True, description="Unique query ID"),
            ColumnDef("query", required=True, description="Search query string"),
            ColumnDef("gold_passage_ids", required=True, col_type="json_list", description="JSON list of relevant chunk/passage IDs"),
            ColumnDef("query_language", required=False, description="'hi', 'hinglish', or 'en'"),
            ColumnDef("doc_language", required=False, description="'hi' or 'en'"),
            ColumnDef("is_exact_match_query", required=False, col_type="bool", description="Whether query tests exact order/act code (e.g. F-4-1/2023)"),
        ],
        kpis={
            "recall_at_5": KpiDef("recall_at_5", 0.85, ">=", "|retrieved[:5] ∩ gold| / |gold|", "Recall @ 5", "%"),
            "recall_at_10": KpiDef("recall_at_10", 0.90, ">=", "|retrieved[:10] ∩ gold| / |gold|", "Recall @ 10", "%"),
            "recall_at_20": KpiDef("recall_at_20", 0.95, ">=", "|retrieved[:20] ∩ gold| / |gold|", "Recall @ 20", "%"),
            "mrr": KpiDef("mrr", 0.80, ">=", "Mean(1 / rank_first_gold)", "Mean Reciprocal Rank (MRR)"),
            "ndcg_at_10": KpiDef("ndcg_at_10", 0.85, ">=", "DCG@10 / IDCG@10", "nDCG @ 10"),
            "exact_match_hit_rate": KpiDef("exact_match_hit_rate", 0.95, ">=", "Exact Hits @ 1 for Code Queries", "Exact Match Hit Rate", "%"),
            "cross_lingual_recall": KpiDef("cross_lingual_recall", 0.85, ">=", "Recall@10 for Cross-Language Queries (Hi<->En)", "Cross-Lingual Recall", "%"),
            "rrf_lift_over_best": KpiDef("rrf_lift_over_best", 5.0, ">=", "((RRF_Recall - Best_Single_Recall) / Best_Single_Recall) * 100", "RRF Fusion Lift Over Best Single", "%"),
            "latency_p95_ms": KpiDef("latency_p95_ms", 350.0, "<=", "95th percentile retrieval latency in ms", "Retrieval Latency p95", "ms"),
        },
    ),
    "reranking": ComponentDef(
        id="reranking",
        label="4. Second-Stage Reranking",
        description="Evaluates BGE-Reranker-v2-M3 cross-encoder precision and candidate promotion.",
        columns=[
            ColumnDef("query_id", required=True, description="Query ID"),
            ColumnDef("query", required=True, description="Query text"),
            ColumnDef("gold_passage_ids", required=True, col_type="json_list", description="JSON list of gold passage IDs"),
            ColumnDef("candidate_passage_ids", required=False, col_type="json_list", description="Candidate IDs (optional; retrieved if omitted)"),
        ],
        kpis={
            "ndcg_at_5": KpiDef("ndcg_at_5", 0.88, ">=", "Post-rerank nDCG@5", "nDCG @ 5"),
            "mrr": KpiDef("mrr", 0.85, ">=", "Post-rerank MRR", "Mean Reciprocal Rank"),
            "rerank_lift_mrr": KpiDef("rerank_lift_mrr", 8.0, ">=", "MRR Lift vs Pre-Rerank First Stage", "MRR Lift", "%"),
            "gold_promotion_rate": KpiDef("gold_promotion_rate", 0.75, ">=", "Gold Chunks Promoted into Top-3 / Total Gold", "Gold Promotion Rate into Top-3", "%"),
            "gold_demotion_rate": KpiDef("gold_demotion_rate", 0.03, "<=", "Gold Chunks Demoted Out of Top-10 / Total Gold", "Gold Demotion Rate", "%"),
            "latency_per_query_ms": KpiDef("latency_per_query_ms", 250.0, "<=", "Average cross-encoder inference latency in ms", "Rerank Latency", "ms"),
        },
    ),
    "context_budgeting": ComponentDef(
        id="context_budgeting",
        label="5. Context Budgeting",
        description="Evaluates dynamic passage trimming, table header preservation, and 10k token cap adherence.",
        columns=[
            ColumnDef("query_id", required=True, description="Query ID"),
            ColumnDef("query", required=True, description="Query text"),
            ColumnDef("ranked_passage_ids", required=False, col_type="json_list", description="Ranked candidate IDs"),
            ColumnDef("gold_passage_ids", required=True, col_type="json_list", description="Gold passage IDs needed"),
            ColumnDef("gold_answer_span", required=True, description="Exact factual sentence or number span required"),
        ],
        kpis={
            "gold_passage_survival_rate": KpiDef("gold_passage_survival_rate", 0.95, ">=", "Gold Passages Packed Within Budget / Total Gold", "Gold Passage Survival Rate", "%"),
            "gold_span_survival_rate": KpiDef("gold_span_survival_rate", 0.98, ">=", "Gold Answer Spans Preserved Intact / Total", "Gold Span Survival Rate", "%"),
            "truncation_damage_rate": KpiDef("truncation_damage_rate", 0.02, "<=", "Cuts Inside Gold Sentence or Table Row / Total", "Truncation Damage Rate", "%"),
            "cap_hit_rate": KpiDef("cap_hit_rate", 0.15, "<=", "Passages Dropped Due to 10k Cap / Total", "Token Cap Hit Rate", "%"),
        },
    ),
    "generation": ComponentDef(
        id="generation",
        label="6. Generation (LLM)",
        description="Evaluates Qwen3-32B answer correctness, citations, faithfulness, and refusal behavior.",
        columns=[
            ColumnDef("query_id", required=True, description="Query ID"),
            ColumnDef("query", required=True, description="Query text"),
            ColumnDef("passages", required=False, col_type="json_list", description="Context passages to provide (optional)"),
            ColumnDef("reference_answer", required=True, description="Ground truth answer"),
            ColumnDef("expected_language", required=True, description="'hi' or 'en'"),
            ColumnDef("answerable", required=True, col_type="bool", description="Whether answer is present in documents"),
            ColumnDef("expected_units", required=False, col_type="json_list", description="Expected units (crore, %, etc.)"),
            ColumnDef("max_words", required=False, col_type="int", description="Max allowed words"),
        ],
        kpis={
            "correctness": KpiDef("correctness", 0.90, ">=", "Factual Accuracy vs Ground Truth", "Answer Correctness", "%"),
            "faithfulness": KpiDef("faithfulness", 0.95, ">=", "Supported Claims / Total Claims", "Answer Faithfulness", "%"),
            "citation_precision": KpiDef("citation_precision", 0.92, ">=", "Valid Cited Docs / Total Citations", "Citation Precision", "%"),
            "citation_recall": KpiDef("citation_recall", 0.90, ">=", "Cited Gold Docs / Total Gold Docs", "Citation Recall", "%"),
            "correct_refusal_rate": KpiDef("correct_refusal_rate", 0.90, ">=", "Refusals on Unanswerable Queries / Total Unanswerable", "Correct Refusal Rate", "%"),
            "false_refusal_rate": KpiDef("false_refusal_rate", 0.02, "<=", "Refusals on Answerable Queries / Total Answerable", "False Refusal Rate", "%"),
            "brevity_compliance": KpiDef("brevity_compliance", 0.95, ">=", "Answers within Word Cap / Total", "Brevity Compliance", "%"),
            "ttft_ms": KpiDef("ttft_ms", 600.0, "<=", "Time to First Token (TTFT) in ms", "Time to First Token", "ms"),
        },
    ),
    "multi_hop": ComponentDef(
        id="multi_hop",
        label="7. Multi-Hop Loop",
        description="Evaluates Thinking Mode planning, iterative retrieval, and deterministic supersession regex.",
        columns=[
            ColumnDef("query_id", required=True, description="Query ID"),
            ColumnDef("query", required=True, description="Query requiring multiple hops"),
            ColumnDef("hops_required", required=True, col_type="int", description="Expected hops (2 or 3)"),
            ColumnDef("gold_chain_passage_ids", required=True, col_type="json_list", description="Ordered chain of passage IDs"),
            ColumnDef("gold_final_answer", required=True, description="Ground truth synthesized answer"),
            ColumnDef("supersession_pairs", required=False, col_type="json_dict", description="JSON map: {replaced_id: current_id}"),
            ColumnDef("expected_current_rule_id", required=False, description="Authoritative active rule ID"),
        ],
        kpis={
            "multi_hop_accuracy": KpiDef("multi_hop_accuracy", 0.88, ">=", "Correct End-to-End Multi-Hop Answers / Total", "Multi-Hop Accuracy", "%"),
            "supersession_f1": KpiDef("supersession_f1", 0.95, ">=", "2 * (P * R) / (P + R) on Superseded Pairs", "Supersession Regex F1"),
            "stale_answer_rate": KpiDef("stale_answer_rate", 0.02, "<=", "Answers Using Superseded Old Order / Total", "Stale Answer Rate", "%"),
            "wasted_loop_rate": KpiDef("wasted_loop_rate", 0.05, "<=", "Extra Hops with 0 Information Gain / Total", "Wasted Loop Rate", "%"),
            "avg_hops": KpiDef("avg_hops", 2.2, "<=", "Average Hops Executed per Multi-Hop Query", "Average Hops"),
        },
    ),
    "grounding_guardrail": ComponentDef(
        id="grounding_guardrail",
        label="8. Grounding Guardrails",
        description="Evaluates sentence-level unsupported-number checker and hallucination detection.",
        columns=[
            ColumnDef("row_id", required=True, description="Row identifier"),
            ColumnDef("answer_text", required=True, description="Generated answer text to test"),
            ColumnDef("passages", required=True, col_type="json_list", description="Evidence passages text"),
            ColumnDef("has_unsupported_number", required=True, col_type="bool", description="Ground truth: whether answer contains a hallucinated figure"),
            ColumnDef("injected_error_type", required=False, description="'number', 'date', 'order_no', 'name', 'none'"),
        ],
        kpis={
            "guardrail_recall": KpiDef("guardrail_recall", 0.95, ">=", "Detected Hallucinations / Total True Hallucinations", "Guardrail Recall (Detection Rate)", "%"),
            "guardrail_precision": KpiDef("guardrail_precision", 0.95, ">=", "True Hallucinations Detected / Total Flags", "Guardrail Precision", "%"),
            "guardrail_f1": KpiDef("guardrail_f1", 0.95, ">=", "2 * (P * R) / (P + R)", "Guardrail F1 Score"),
            "false_block_rate": KpiDef("false_block_rate", 0.03, "<=", "False Blocks on Factual Answers / Total Factual Answers", "False Block Rate", "%"),
            "miss_rate": KpiDef("miss_rate", 0.05, "<=", "Undetected Hallucinations / Total Hallucinations", "Miss Rate", "%"),
        },
    ),
}

