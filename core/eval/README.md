# Component Evaluation Guide (CHiPS-RAG)

## Overview
The **Component Evaluation** module extends CHiPS-RAG with isolated, unit-level evaluation for all 8 core pipeline components. This enables developers and engineers to measure the precise impact of chunker adjustments, embedding model migrations, prompt revisions, and guardrail rules without confounding end-to-end variables.

---

## 1. Pipeline Components & Data Schemas

### 1) Ingestion & OCR (`ingestion_ocr`)
- **Evaluates**: Vision-Language Model (Qwen2.5-VL) OCR accuracy, table structure fidelity, `text_gate.py` statistical filtering, and structure-preserving chunking.
- **Input Columns**:
  - `doc_id` (str): Unique document identifier.
  - `page_no` (int): 1-indexed document page number.
  - `file_path` (str, optional): Relative path to source document.
  - `ground_truth_text` (str): Gold-standard Markdown with tables and headers.
  - `language` (str): `'hi'`, `'en'`, or `'mixed'`.
  - `has_table` (bool): Whether the page contains tabular grids.
  - `gate_label` (str): Expected text quality verdict (`'good'` or `'bad'`).
  - `gold_numbers` (JSON list): Critical numbers/dates that must be extracted exactly.
  - `gold_chunk_boundaries` (JSON list, optional): Expected chunk break points.
- **Key KPIs**:
  - **CER (Character Error Rate)**: $\text{Levenshtein}(chars) / N_{chars}$ (Target: $\le 5\%$).
  - **WER (Word Error Rate)**: $\text{Levenshtein}(words) / N_{words}$ (Target: $\le 10\%$).
  - **Number Exact Match**: $|extracted \cap gold| / |gold|$ (Target: $\ge 98\%$).
  - **Text Gate F1**: Harmonic mean of gate precision & recall (Target: $\ge 0.95$).

---

### 2) Query Preprocessing (`query_preprocessing`)
- **Evaluates**: Language & script detection (Hindi, Hinglish, English), intent classification (`query_plan.py`), and bilingual glossary expansion (`rag_pipeline.py`).
- **Input Columns**:
  - `query_id` (str): Unique query ID.
  - `query` (str): Raw input query text.
  - `true_language` (str): Ground truth language (`'hi'`, `'hinglish'`, `'en'`).
  - `true_intent` (str): Ground truth intent (`'general'`, `'catalogue'`, `'count'`, `'number'`, `'greeting'`).
  - `expected_expansion_terms` (JSON list): Expected bilingual synonyms.
- **Key KPIs**:
  - **Language Accuracy**: Correct language predictions / Total (Target: $\ge 98\%$).
  - **Intent Accuracy**: Correct intent classifications / Total (Target: $\ge 95\%$).
  - **Expansion Recall**: Retrieved expansion terms / Expected terms (Target: $\ge 90\%$).

---

### 3) Hybrid Retrieval (`hybrid_retrieval`)
- **Evaluates**: BGE-M3 Dense (1024-dim) + Sparse lexical weights + exact MatchText full text fused with Reciprocal Rank Fusion (RRF, $k=60$).
- **Input Columns**:
  - `query_id` (str): Unique query ID.
  - `query` (str): User search query.
  - `gold_passage_ids` (JSON list): List of ground-truth passage/chunk IDs.
  - `query_language` (str): `'hi'`, `'hinglish'`, or `'en'`.
  - `doc_language` (str): `'hi'` or `'en'`.
  - `is_exact_match_query` (bool): True if testing exact circular codes (`F-4-1/2023`).
- **Key KPIs**:
  - **Recall@10**: Gold passages retrieved in top 10 / Total gold passages (Target: $\ge 90\%$).
  - **MRR (Mean Reciprocal Rank)**: Mean of $1 / \text{rank}$ of first gold passage (Target: $\ge 0.80$).
  - **nDCG@10**: Normalized Discounted Cumulative Gain at rank 10 (Target: $\ge 0.85$).
  - **Exact Match Hit Rate**: Top-1 hit rate on code queries (Target: $\ge 95\%$).
  - **RRF Lift %**: Percentage improvement of RRF fusion over best single retriever.
  - **Ablation Matrix**: Side-by-side comparison of Dense vs Sparse vs MatchText vs RRF.

---

### 4) Second-Stage Reranking (`reranking`)
- **Evaluates**: BGE-Reranker-v2-M3 cross-encoder precision, score threshold separation, and candidate promotion.
- **Input Columns**:
  - `query_id`, `query`, `candidate_passage_ids` (JSON list), `gold_passage_ids` (JSON list).
- **Key KPIs**:
  - **nDCG@5**: Post-rerank top-5 ranking quality (Target: $\ge 0.88$).
  - **Gold Promotion Rate**: Share of gold candidates promoted into Top 3 (Target: $\ge 75\%$).
  - **Gold Demotion Rate**: Gold candidates pushed below Top 10 (Target: $\le 3\%$).

---

### 5) Context Budgeting (`context_budgeting`)
- **Evaluates**: Dynamic passage trimmer (`answer_prompt.trim_passage`), table header preservation, and 10,000 token budget allocation.
- **Input Columns**:
  - `query_id`, `query`, `ranked_passage_ids`, `gold_passage_ids`, `gold_answer_span`.
- **Key KPIs**:
  - **Gold Passage Survival Rate**: Percentage of gold chunks fitting in budget (Target: $\ge 95\%$).
  - **Gold Span Survival Rate**: Factual sentence preserved intact (Target: $\ge 98\%$).
  - **Truncation Damage Rate**: Cuts inside sentences/tables (Target: $\le 2\%$).

---

### 6) Generation (`generation`)
- **Evaluates**: Qwen3-32B generation on vLLM, citation validity, brevity, and refusal compliance.
- **Input Columns**:
  - `query_id`, `query`, `passages` (JSON), `reference_answer`, `expected_language`, `answerable` (bool), `expected_units`.
- **Key KPIs**:
  - **Answer Correctness**: Factual match against reference (Target: $\ge 90\%$).
  - **Faithfulness**: Supported claims / Total claims (Target: $\ge 95\%$).
  - **Citation Precision & Recall**: Valid citations matching evidence (Target: $\ge 92\%$).
  - **Correct Refusal Rate**: Refusal on unanswerable queries (Target: $\ge 90\%$).

---

### 7) Multi-Hop Loop (`multi_hop`)
- **Evaluates**: Thinking Mode sub-question planning, iterative retrieval, and supersession regex detection.
- **Input Columns**:
  - `query_id`, `query`, `hops_required`, `gold_chain_passage_ids`, `gold_final_answer`, `supersession_pairs`.
- **Key KPIs**:
  - **Multi-Hop Accuracy**: End-to-end task completion (Target: $\ge 88\%$).
  - **Supersession F1**: Detection of replacement clauses (Target: $\ge 0.95$).
  - **Stale Answer Rate**: Erroneously citing superseded orders (Target: $\le 2\%$).

---

### 8) Grounding Guardrails (`grounding_guardrail`)
- **Evaluates**: Sentence-level `unsupported_numbers()` checker and hallucination detection.
- **Input Columns**:
  - `row_id`, `answer_text`, `passages` (JSON), `has_unsupported_number` (bool), `injected_error_type`.
- **Key KPIs**:
  - **Guardrail Recall**: Hallucinations caught / Total true hallucinations (Target: $\ge 95\%$).
  - **False Block Rate**: Factual answers falsely blocked (Target: $\le 3\%$).

---

## 2. How to Prepare a Test Batch
1. **Format**: CSV, TSV, XLSX, or JSON.
2. **Encoding**: Must be valid **UTF-8** (with or without BOM). Devanagari Hindi text (`शासनादेश`, `०-९`) is fully supported.
3. **Required Columns**: Headers must match column names case-insensitively.
4. **Lists & Dictionaries**: In CSV/XLSX, JSON list fields should be encoded as `["doc1", "doc2"]` or comma-separated `doc1, doc2`.

---

## 3. How to Read Results & KPI Status
- **PASS (Green)**: Value meets or exceeds target ($\ge \text{target}$ or $\le \text{target}$).
- **WARN (Amber)**: Value is within $5\%$ of the target threshold.
- **FAIL (Red)**: Value violates target threshold by $> 5\%$.
- **Delta Indicator ($\uparrow/\downarrow$)**: Compares current performance with the most recent prior run on that component.

