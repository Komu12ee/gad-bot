# Upgraded Document Chunking & Metadata Pipeline

This module upgrades the CHiPS-RAG chunking pipeline to produce structured, filterable metadata payloads for Qdrant alongside clean, standardized embedded texts and document cards.

---

## 1. Key Architectural Features

1. **Deterministic Python Chunking**:
   - Zero LLM usage for text splitting or token slicing.
   - Layout-aware parser recognizing headings (`#`), paragraphs, numbered clauses (`(i)`, `1.`), and tables.
   - Devanagari sentence splitting respecting `।` and `॥` while strictly preserving currency (`Rs. 78,800-2,09,200`), order numbers (`A-12025/3/2023-SA`), and decimals (`12.5%`).

2. **Strict Date Extraction Rules**:
   - `doc_date` represents **ONLY** the official date the issuing department released/signed the document.
   - Scans **Header Zone** (first 25 lines of Page 1) and **Signature Block** (last 30 lines).
   - Negative context filter automatically rejects dates preceded by `vide`, `ref`, `last date`, `w.e.f.`, or `in supersession of`, routing them to `referenced_documents`.
   - Priority: `header date > signature date`. Conflicting dates automatically set `needs_review=True`.
   - Never falls back to file modified time. Missing dates stay `null`.

3. **2D Structure-Preserving Table Chunking**:
   - Tables are never merged with prose text.
   - Rows are serialized as `column: value` pairs, dropping blank cells.
   - Groups up to 15 rows per chunk, replicating table caption, fiscal unit (e.g. `Rs. lakh`, `Rs. crore`), and column headers on every chunk.
   - Total/subtotal row protection (never slices inside a subtotal block).
   - Multi-page table linking (`continued_from_previous_page=True`).

4. **Document Cards & Verified Summaries**:
   - One document card record per document with facts and a 2-3 line summary.
   - Strict hallucination check: any number or date in an LLM summary absent from the source text is rejected.

5. **Latest-Version Logic & Supersedes Linking**:
   - Detects `supersedes`, `in partial modification of`, and `vide order no.` references.
   - Post-ingestion job `link_supersedes()` updates target document cards and chunks in Qdrant with `status="superseded"` and `superseded_by=<new_doc_id>`.
   - Unresolved references are logged to `unresolved_supersedes.csv`.
   - Documents with `doc_date=null` never supersede automatically and are logged to `needs_review.csv`.
   - `topic_key` deduplication: at retrieval time, documents sharing a `topic_key` return only the latest active version unless `include_old=True`.

---

## 2. CLI Usage

### Running the Chunking Pipeline
```bash
# Basic run on a directory of documents (PDF text or Markdown)
python run_pipeline.py --input data/chips/stage2 --output data/chips/chunks

# Disable LLM fallback calls completely (regex only)
python run_pipeline.py --input data/chips/stage2 --output data/chips/chunks --no-llm

# Index payloads and create payload indexes directly in Qdrant
python run_pipeline.py --input data/chips/stage2 --output data/chips/chunks --index-qdrant
```

### Outputs Generated in `--output <dir>`:
- `<doc_id>/structured_chunk_001.txt`: Human-readable chunk with YAML-like metadata above `---` and embedded text below.
- `<doc_id>/structured_chunk_card.txt`: Human-readable document card summary.
- `chunks.jsonl`: One JSON object per chunk (complete metadata payload + text).
- `doc_cards.jsonl`: One JSON object per document card.
- `unresolved_supersedes.csv`: Log of cited orders not yet ingested in corpus.
- `needs_review.csv`: Log of documents with null dates or date conflicts.

---

## 3. How to Re-Index Existing Documents

To migrate existing corpus documents to the upgraded schema:

```bash
# Step 1: Run the upgrade chunker against your existing stage2 OCR markdown files
python run_pipeline.py \
  --input data/chips/stage2 \
  --output data/chips/chunks \
  --index-qdrant

# Step 2: Verify unresolved references and flagged documents
cat data/chips/chunks/needs_review.csv
cat data/chips/chunks/unresolved_supersedes.csv
```

---

## 4. Running the Pytest Suite

```bash
uv run pytest tests/test_chunking_upgrade.py -v
```
Verifies:
- Date extraction (header vs body references, last date rejection, conflict review).
- Order number extraction with slashes and hyphens.
- Supersedes linking and status mutation.
- Hindi sentence splitting without breaking currency or order numbers.
- Hard token limits on full embedded text.
- Table chunking, header repetition, and subtotal preservation.
- Determinism and LLM hallucination rejection.
- Topic key latest-version deduplication.
