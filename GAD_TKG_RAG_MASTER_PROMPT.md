# GAD Temporal Knowledge-Graph RAG — Master Build Prompt

**Corpus:** ~5,300 OCR'd `.md` files (Hindi / English / mixed, with tables) from a General Administration Department.
**Goal:** A question-answering system that is correct about *which rule is in force on which date*, cites its sources, handles Hindi + English, and reads tables correctly.
**Hardware:** 1 × NVIDIA H100, a ~22B LLM served with vLLM, unlimited tokens.

---

## 0. How to use this file

This file has two layers.

| Layer | Sections | Who reads it | Purpose |
|---|---|---|---|
| **Builder layer** | 1 – 5, 8, 9, 10, 11, 12 | A coding agent (Claude Code, Cursor, Aider, etc.) or you | Builds the pipeline, scripts, databases, tests |
| **Runtime layer** | 6, 7 | The LLM *inside* the pipeline | The exact prompts, schemas, and lexicons used to process each document |

**Start:** paste Section 1 into your coding agent. Tell it to read this whole file first and to store the Section 6 prompts verbatim under `prompts/`.

**Golden rule of this design:** the LLM never does a job that code can do, and the LLM never does a big job when it can do ten tiny jobs. This is what makes it work even with a small model.

---

## 1. THE BUILDER PROMPT (paste this to the coding agent)

```text
ROLE
You are a senior data/ML engineer. You will build, test, and run an end-to-end
pipeline that converts ~5,300 OCR markdown files of Indian government documents
(General Administration Department; Hindi, English, mixed; with tables) into:
  (a) a temporal knowledge graph of documents, clauses, and their legal
      relationships (supersedes / amends / rescinds / ...), with computed
      "in force on date T" status;
  (b) a hybrid search index over clause-level chunks;
  (c) a structured table store;
  (d) a query service that answers questions with citations and correct
      temporal status;
  (e) an evaluation harness with a gold test set.

SPEC
Read the file GAD_TKG_RAG_MASTER_PROMPT.md completely before writing any code.
It is the specification. Sections 4 (data contracts), 5 (phases), 6 (runtime
prompts), 7 (lexicons), 8 (graph + status algorithm), 9 (query pipeline),
10 (evaluation) are binding. If you believe something in the spec is wrong,
say so and propose a fix; do not silently deviate.

RULES OF ENGAGEMENT
1. Work phase by phase (P0, P1, ...). Do not start phase N+1 until phase N
   passes its ACCEPTANCE TESTS. Print the test results.
2. Every phase must be idempotent and resumable: write JSONL outputs, key every
   record by file_id, skip records already done, cache every LLM call by
   sha256(prompt + model + params).
3. Run every LLM phase first on the PILOT SET (200 files, stratified: ~40%
   Hindi, ~40% English, ~20% mixed; include 20 table-heavy files and 20 files
   with many references). Show me 10 random outputs per phase and the
   accuracy numbers against my manual labels before running on all 5,300.
4. Never let the LLM invent identifiers. Every extracted quote must be verified
   as a substring of the source text (after whitespace normalisation). If it
   fails, the record is rejected or retried, never kept.
5. Never let the LLM retype table numbers. Parse tables with code.
6. Never let the LLM decide "is this rule in force". Compute it with code from
   graph edges (Section 8).
7. Log everything: every rejected record goes to work/rejects/<phase>.jsonl
   with a reason. Ambiguous cases go to work/review/<phase>.csv for me.
8. Use the exact JSON schemas in Section 4 and prompts in Section 6. Use
   structured / constrained decoding on the vLLM server (check the installed
   vLLM version's docs: `response_format` json_schema, or `guided_json`).
9. Write unit tests for every parser, normaliser, and the status algorithm.
10. Ask me only for what you cannot get yourself. At the start, ask me for:
    (i) 5 sample files (2 Hindi, 2 English, 1 with a large table),
    (ii) the vLLM endpoint + model name, (iii) whether Marathi text also
    appears, (iv) 20 real questions my current chatbot answers badly, with the
    correct answers if I know them.

DELIVERABLES
  gad_kg/ repo (layout in Section 5), README with run commands, config.yaml,
  all prompts in prompts/, all schemas in schemas/, tests/, a final
  report/eval_report.md with the numbers from Section 10.

START
Step 1: read the spec. Step 2: ask me the questions in rule 10. Step 3: begin P0.
```

---

## 2. Environment and configuration

```yaml
# config.yaml
paths:
  raw_md: data/raw_md
  clean: data/clean
  work: data/work
llm:
  base_url: http://localhost:8000/v1
  model: "<YOUR-22B-MODEL-NAME>"
  temperature_greedy: 0.0
  temperature_vote: 0.4
  votes: 3                  # 1 greedy + 2 sampled; tokens are free, so we vote
  max_concurrency: 64
  max_tokens_default: 600
  structured_output: true
embedding:
  model: BAAI/bge-m3        # dense (1024-d) + sparse
reranker:
  model: BAAI/bge-reranker-v2-m3
stores:
  graph: neo4j://localhost:7687   # Neo4j 5.x Community is enough
  vectors: qdrant (local or docker)   # dense + sparse hybrid
  tables: duckdb file data/work/tables.duckdb
thresholds:
  ocr_quality_min: 0.80
  dedup_jaccard: 0.90
  fuzzy_number_match: 0.92
  subject_embed_sim: 0.80
```

**vLLM server (suggested starting point; tune for your model):**

```bash
vllm serve <MODEL> --max-model-len 16384 --gpu-memory-utilization 0.85 \
     --enable-prefix-caching --max-num-seqs 128
```

**GPU scheduling:** a 22B model in bf16 needs ~45 GB. Run the LLM phases (P4–P6, P11, P12-prefix) with the LLM server up. Run embedding and reranker phases after stopping the LLM server, or start the LLM with `--gpu-memory-utilization 0.60` so BGE-M3 (~3 GB) and the reranker (~3 GB) fit beside it.

---

## 3. Design principles (why this works with small models)

Apply all of these. They are the "intelligence" of the system.

1. **Candidate-first.** Code (regex, lexicons) finds *candidate* numbers, dates, sentences. The LLM only *chooses or classifies* among them. Choosing from a list is far easier than free extraction.
2. **One task per call.** Never ask for 10 fields at once. Classify type → pick number → pick date → copy subject → each is its own call. Many small calls are cheap on an H100.
3. **Quote first, label second.** In every JSON schema the `quote` field comes before the `label` field. The model must copy the evidence before deciding. Verify the quote with code.
4. **Closed vocabularies.** Wherever possible the answer is an enum or an index into a list.
5. **Allow "null / NONE / UNCLEAR".** A model forced to answer will hallucinate. Make abstaining easy and cheap.
6. **Short windows.** ≤ 1,500 tokens of input per call. Put the schema and rules at the end of the prompt, near the generation point.
7. **Few-shot in the corpus's own style.** Each key prompt has Hindi and English examples. Replace the placeholder examples with 2–3 real ones from your pilot set. This gives the single biggest accuracy gain.
8. **Vote and verify.** 1 greedy + 2 sampled runs; majority for enums; require quote-substring match; a separate verifier call; cross-checks with the lexicon. Disagreement → lower confidence tier → human review queue.
9. **Escalation ladder.** small model → vote → larger model → human. Only low-confidence cases climb.
10. **Do the hard thinking offline.** Spend the unlimited tokens at *indexing time* (extraction, verification, summaries, contextual prefixes, synthetic questions). At *query time*, the model only has to read a small, well-labelled context. That is how a small model can answer well: use the 22B offline, and (optionally) a 7–9B or the same 22B online with the extract-then-compose pattern (Section 9).
11. **Code computes the truth.** Dates are parsed by code. Status ("in force") is computed by code. Numbers in tables are parsed by code. Final answers are checked by code (numbers/dates must appear in cited chunks).
12. **Keep both copies of text.** `clean` (NFC-normalised, human-readable) and `search` (same length, Devanagari digits → ASCII digits) so character offsets are valid in both.

---

## 4. Data contracts

### 4.1 Identifiers

```
file_id      = sha1(raw file bytes)[:16]
doc_id       = "D_" + file_id of the CANONICAL file of a duplicate cluster
number_norm  = normalised document number (see 7.4)
clause_id    = f"{doc_id}#C{index:03d}"
table_id     = f"{doc_id}#T{index:02d}"
stub_id      = "S_" + sha1(number_norm + "|" + date)[:12]   # document referenced but not in corpus
```

### 4.2 Work files (all JSONL, one record per file_id unless stated)

| File | Producer | Contents |
|---|---|---|
| `work/inventory.jsonl` | P0 | file_id, path, n_chars, devanagari_ratio, lang_guess, ocr_quality, flags |
| `clean/<file_id>.md`, `clean/<file_id>.search.txt` | P1 | cleaned text; same-length search copy |
| `work/dedup.jsonl` | P2 | file_id, cluster_id, canonical_file_id, is_duplicate |
| `work/features.jsonl` | P3 | regex candidates: numbers, dates, cue sentences, reference block, tables |
| `work/header.jsonl` | P4 | normalised header (4.3) |
| `work/relations_raw.jsonl` | P5 | one record per candidate relation (4.4), with votes |
| `work/relations_verified.jsonl` | P6 | adds verifier verdict, tier |
| `work/edges.jsonl` | P7 | resolved, canonical-direction edges (4.5) |
| `work/status.jsonl` | P8 | computed status timeline per doc and clause |
| `work/clauses.jsonl` | P9 | clauses with offsets, parent info |
| `work/tables.jsonl` + `tables.duckdb` | P10 | table metadata + rows |
| `work/entities.jsonl` | P11 | entity mentions linked to canonical IDs |
| `work/chunks.jsonl` | P12 | retrieval chunks with contextual prefix |
| `work/review/*.csv` | any | items needing human decision |
| `work/rejects/*.jsonl` | any | rejected LLM outputs with reason |

### 4.3 `header` record (after code normalisation)

```json
{
  "file_id": "a1b2c3d4e5f60718",
  "doc_type": "GR",
  "doc_number_raw": "GAD-2021/CR-45/Ser-3",
  "doc_number_norm": "GAD-2021/CR-45/SER-3",
  "issue_date_raw": "12 March 2021",
  "issue_date": "2021-03-12",
  "subject": "…exact text…",
  "subject_en": "…English rendering…",
  "issuing_dept": "General Administration Department",
  "language": "hi|en|mixed",
  "references": [
    {"idx": 1, "raw": "…", "number_raw": "…", "number_norm": "…",
     "date_raw": "…", "date": "YYYY-MM-DD|null", "subject_raw": "…", "doc_type_guess": "GR"}
  ],
  "effective_from": "YYYY-MM-DD|null",
  "effective_to": "YYYY-MM-DD|null",
  "retrospective": false,
  "topics": ["leave", "pension"],
  "summary_en": "…", "summary_hi": "…",
  "evidence": {"doc_type": "…", "doc_number": "…", "issue_date": "…"},
  "confidence": {"doc_type": 1.0, "doc_number": 0.67, "issue_date": 1.0}
}
```

### 4.4 `relation` record (from the document's own point of view)

```json
{
  "rel_id": "R_<hash>",
  "src_file_id": "…",
  "relation_local": "SUPERSEDES|SUPERSEDED_BY|AMENDS|AMENDED_BY|RESCINDS|RESCINDED_BY|CLARIFIES|CLARIFIED_BY|EXTENDS|EXTENDED_BY|CORRECTS|CORRECTED_BY|IMPLEMENTS|REFERS_TO",
  "target_kind": "DOCUMENT|ACT_RULE|COURT_ORDER|UNKNOWN",
  "target_ref_idx": 3,
  "target_number_raw": null,
  "target_date_raw": null,
  "target_subject_raw": null,
  "scope": "whole|partial|unknown",
  "scope_text": "para 4, sub-para (b)",
  "effective_date_raw": null,
  "quote": "exact sentence",
  "quote_start": 1234, "quote_end": 1390,
  "votes": {"SUPERSEDES": 3},
  "verifier": "SUPPORTED|NOT_SUPPORTED|UNCLEAR",
  "tier": "A|B|C"
}
```

### 4.5 `edge` record (canonical direction: always `source -[TYPE]-> target`)

`_BY` relations are flipped: "this doc is SUPERSEDED_BY X" becomes edge `X -[SUPERSEDES]-> this`.

```json
{
  "edge_id": "E_<hash>",
  "source": "D_… or S_…",
  "target": "D_… or S_…",
  "type": "SUPERSEDES|AMENDS|RESCINDS|CLARIFIES|EXTENDS|CORRECTS|IMPLEMENTS|REFERS_TO",
  "scope": "whole|partial|unknown",
  "scope_text": "…",
  "effective_date": "YYYY-MM-DD|null",
  "quote": "…", "evidence_file_id": "…", "source_clause_id": "…|null",
  "target_clause_id": "…|null",
  "tier": "A|B|C",
  "resolution": "exact|number_only|fuzzy|stub|review"
}
```

**Edge effective date** = explicit date in the sentence → else the superseding/amending document's `effective_from` → else its `issue_date`.

---

## 5. Phases (build order, with acceptance tests)

### Repo layout

```
gad_kg/
  config.yaml
  data/{raw_md,clean,work}/
  prompts/            # Section 6 prompts, one .txt per prompt
  schemas/            # JSON Schemas for each LLM output
  lexicons/           # Section 7 as YAML
  src/
    common/{io.py, llm.py, cache.py, normalize.py, dates.py, numbers.py}
    p00_inventory.py   p01_clean.py        p02_dedup.py
    p03_features.py    p04_header.py       p05_relations.py
    p06_verify.py      p07_resolve.py      p08_status.py
    p09_chunk.py       p10_tables.py       p11_entities.py
    p12_index.py       p13_graph_load.py   p14_query_service.py
    p15_eval.py
  tests/
  report/
```

`llm.py` must: call the vLLM OpenAI-compatible endpoint, enforce the JSON schema, retry up to 2 times with a *repair prompt* (“Your last output was invalid because …; output only the corrected JSON”), run votes concurrently, and cache by hash.

### P0 — Inventory and OCR quality

* For each file: size, Devanagari ratio, language guess, and an **OCR quality score** in [0,1] = weighted mix of: share of tokens that look valid, absence of `�` and stray symbols, ratio of isolated single characters, repeated-character runs, and (for Hindi) share of Devanagari words whose matras are in valid positions.
* Flag `needs_reocr` if score < `ocr_quality_min`. Write these to `review/p0_bad_ocr.csv`.
* **Acceptance:** counts by language; list of 30 worst files shown to me; I confirm the score ranks them correctly.

### P1 — Cleaning and normalisation

1. Read as UTF-8. Unicode NFC. Remove zero-width characters (U+200B, U+200C, U+200D, U+FEFF), soft hyphens; collapse spaces; normalise line endings.
2. Fix common OCR confusions **only inside number/date contexts** (e.g. `O`→`0`, `l`/`I`→`1`, `S`→`5` between digits). Never alter running text.
3. Remove running headers/footers/page numbers: lines (after digit-stripping) that repeat on ≥ 30% of pages or ≥ 5 times in the file and are < 120 chars. Inspect how pages are delimited in the OCR output first (`---`, `<!-- page -->`, form feeds) and adapt.
4. Re-join words/lines broken across lines only when the line does not end in sentence punctuation and the next line starts lowercase/continuation.
5. Keep markdown tables intact. Validate that each table row has the same number of cells as the header; fix or flag (`broken_table`).
6. Write `clean/<id>.md`. Write `clean/<id>.search.txt` = clean text with Devanagari digits (०-९) mapped 1:1 to ASCII (so offsets match).
* **Acceptance:** unit tests for digit mapping, header removal; 20 random before/after diffs shown to me; `len(clean) == len(search)` for every file.

### P2 — Deduplication

* Exact: same sha1 of clean text.
* Near-duplicate: MinHash LSH on 5-word shingles, Jaccard ≥ `dedup_jaccard`.
* Within a cluster pick the canonical file = highest OCR quality, then longest. **Do not delete** others; mark `is_duplicate` and `canonical_file_id`. Later phases run only on canonical files; edges attach to the canonical `doc_id`.
* **Acceptance:** cluster size histogram; 20 sampled clusters shown with a diff summary.

### P3 — Deterministic feature extraction (no LLM)

For each canonical file produce `features.jsonl`:
* `number_candidates`: every match of the document-number patterns (7.4) with line number and context, flagged `in_top`, `in_bottom`, `in_reference_block`.
* `date_candidates`: every parsed date (7.3) with context and the same flags.
* `reference_block`: the text between a reference-start cue (`संदर्भ`, `वाचा`, `पढ़ें`, `Read:`, `Ref:`, `Reference`, `Reference(s):`) and a block-end cue (`प्रस्तावना`, `Preamble`, `विषय`, `Subject`, `शासन निर्णय`, `Government Resolution`, first blank-line-paragraph after items) split into numbered items.
* `cue_sentences`: every sentence containing a relation cue from the lexicon (7.1), with sentence offsets, the previous and next sentence, and the matched cue keys.
* `effective_sentences`: sentences with effective-date cues (7.2).
* `tables`: list of table spans (start/end line), nearest caption above (≤ 3 lines), and nearest paragraph below.
* `clause_markers`: numbering/heading patterns (7.5).
* **Acceptance:** on the pilot set, number-candidate **recall** (true own-number appears among candidates) ≥ 99% and date-candidate recall ≥ 99% against my manual labels. If lower, fix regex until met.

### P4 — LLM header extraction (prompts R1–R5, R8, R9)

Per file, run these *separate* small calls (Section 6): R1 doc type → R2 own number (choose from candidates) → R3 issue date (choose from candidates) → R4 subject → R5 reference items (one call per item) → R6 effective dates (per candidate sentence) → R7 topics → R8 summaries.
Then code: parse dates, normalise numbers, compute `language`, apply sanity checks:
* issue_date in [1950-01-01, today]; not in the future.
* number and date are *not* from the reference block (`in_reference_block == false`).
* referenced documents' dates must be ≤ own issue date (flag otherwise).
* **Acceptance (pilot, vs manual labels):** doc_type ≥ 95%, doc_number ≥ 97%, issue_date ≥ 98%, subject usable ≥ 95%. Below that: improve few-shots, candidate lists, or switch the failing fields to a larger model.

### P5 — LLM relation extraction (prompt R10)

* For every `cue_sentence` and for every reference item with a strong cue in the body, run R10 with 3 votes. A reference-block item with **no** cue anywhere defaults to `REFERS_TO` (no LLM needed).
* Provide the model the `ref_map` (idx → short label) so it can answer with `target_ref_idx` instead of copying messy numbers.
* Reject outputs whose `quote` is not a substring of the clean text.
* **Acceptance:** see P6.

### P6 — Verification and tiers (prompt R11 + code checks)

For each relation:
1. Code check A: `quote` ⊂ clean text. Code check B: the target number (if given) appears in the quote or in the reference item. Code check C: the cue lexicon contains at least one cue for the voted relation type in the quote.
2. LLM verifier R11 (a fresh call; not the same prompt as R10).
3. Tier: **A** = quote OK ∧ ≥ 2/3 votes agree ∧ verifier SUPPORTED ∧ lexicon agrees. **B** = quote OK ∧ ≥ 2/3 votes ∧ verifier SUPPORTED. **C** = anything else.
4. Temporal sanity: a document cannot SUPERSEDE/AMEND/RESCINDS a document dated *after* it; flag → tier C.
* Tier C → `review/p6_relations.csv`. Tier C is **never** used for status computation (it may be used for retrieval expansion with a `weak` marker).
* **Acceptance (pilot, 100 hand-labelled documents):** relation precision on tier A+B ≥ 95%; recall of true SUPERSEDES/AMENDS/RESCINDS ≥ 85%. If recall is low, lower the candidate-sentence threshold (add cues); if precision is low, strengthen verifier few-shots.


### P7 — Reference resolution (code first, LLM last)

Goal: turn "GR No. X dated Y" strings into links to real nodes.

Build an index over all canonical headers: `(number_norm, issue_date)`, `number_norm`, subject embeddings (BGE-M3).

Resolve each relation target in this order and stop at the first success:
1. **exact:** `(number_norm, date)` matches exactly one doc.
2. **number_only:** `number_norm` matches exactly one doc and (date missing or within ±1 day, to tolerate OCR).
3. **fuzzy:** `rapidfuzz` ratio ≥ `fuzzy_number_match` on `number_norm` **and** (dates equal or one missing) **and** subject embedding similarity ≥ `subject_embed_sim`. Exactly one candidate required.
4. **ambiguous** (≥ 2 candidates): write to `review/p7_ambiguous.csv` with the top 5 candidates and the quote. Create **no** edge until resolved. After I fill in the CSV, re-run to apply my decisions.
5. **stub:** no candidates → create a `Stub` node (`stub_id`) holding number/date/subject_raw. Edges to stubs are valid: they let the answer say "superseded by an order that is not in the corpus".

Also:
* If the target kind is `ACT_RULE` or `COURT_ORDER`, create/merge an `Instrument` node keyed by normalised name+year.
* If `target_ref_idx` is given, take number/date/subject from `header.references[idx]`.
* Flip `_BY` relations into canonical direction (4.5).
* Merge duplicate edges (same source, target, type, scope_text); keep the highest tier and all quotes.
* Link `source_clause_id` (the clause containing the quote offsets) and, when `scope_text` names a clause/para number, try to link `target_clause_id` by matching the clause number inside the target document.
* **Acceptance:** auto-accepted links correct ≥ 97% on a 200-edge manual sample; ambiguous rate reported; stub rate reported (a very high stub rate means numbers are mis-normalised: fix `normalize_number`).

### P8 — Temporal status computation (code only; algorithm in Section 8)

* Produce for every doc, and for every clause where partial amendments are known, a **status timeline**: list of `(from_date, to_date, status, by_doc_id, edge_id)`.
* Provide `status_as_of(doc_id, T)` and `in_force_as_of(T)` functions; unit-test them with hand-made graphs (whole supersession, partial amendment, rescission, chain A←B←C, backdated effective date, expiry, corrigendum).
* **Acceptance:** all unit tests pass; on 50 hand-checked documents the status "as of today" is correct ≥ 95%.

### P9 — Clause segmentation and chunking

* Segment with rules (7.5): numbered paragraphs, lettered sub-paragraphs, headings. Keep a parent pointer (clause 4(b) → clause 4).
* Typical sections to label: `preamble`, `reference`, `operative` (the actual order/decision), `annexure`, `signature`.
* Chunk = one clause (or a group of short consecutive clauses up to ~350 tokens). If a clause > 600 tokens, split at sentence boundaries with 1-sentence overlap and keep `parent_clause_id`.
* Tables: one chunk per table (caption + markdown), or if > 600 tokens, row-groups with the header row repeated and caption in every group.
* Record `char_start`, `char_end` into the clean text for exact citation.
* **Acceptance:** no chunk exceeds 700 tokens; every character of the operative section is covered by exactly one chunk (plus overlap); 20 random documents shown to me.

### P10 — Tables (code + one small LLM call)

1. Parse markdown tables to rows with code. Handle merged-looking cells (empty cells = carry down from the cell above only if the column is a "group" column; flag).
2. R12 (LLM) receives only the header row, first 3 rows, caption, and neighbouring text, and returns: bilingual title, column meanings (`name_en`, `type: text|int|float|date|amount|percent`, `unit`), and `table_kind`. **It never receives or returns the full rows.**
3. Convert Devanagari digits and Indian number formatting (`1,23,456`) in numeric columns using code; keep `value_raw` and `value_num`.
4. Load into DuckDB: one table `t_<table_id>` per table **plus** a long-form table `table_cells(table_id, row_idx, col_name_en, col_name_raw, value_raw, value_num, unit)` for cross-table search.
5. Add a catalog `table_catalog(table_id, doc_id, title_en, title_hi, columns_json, n_rows)`.
* **Acceptance:** on 30 hand-checked tables, ≥ 98% of numeric cells match the source; row/column counts match.

### P11 — Entities and topics (controlled ontology)

Ontology types: `Department`, `Post` (designation/cadre), `Scheme`, `Instrument` (Act/Rule/Regulation), `Benefit` (allowance, pay, pension item), `Topic` (document-level subject taxonomy), `Place` (optional).

**Build the ontology bottom-up, then lock it:**
1. Collect candidate mentions with R13 on a 600-file sample.
2. Cluster by embedding (BGE-M3) + string similarity; show me the top 300 clusters per type as `review/p11_ontology.csv` with proposed canonical names (English + Hindi aliases).
3. I approve/merge/rename. Save `ontology.yaml`: `id`, `type`, `name_en`, `name_hi`, `aliases[]`.
4. Topic taxonomy: cluster document subjects, propose 25–40 topics; I approve; lock as `topics.yaml`.
5. Linking pass (all docs): R13 extracts mentions per clause; code maps each mention to canonical id via alias exact match → embedding nearest neighbour (≥ threshold) → else `UNLINKED` (kept for review, **not** added as a node).
6. R7 assigns up to 3 topics per document from the locked list.
* **Acceptance:** linking precision ≥ 95% on a 200-mention sample; topic accuracy ≥ 90% on 100 docs.

### P12 — Contextual prefix, embeddings, indexes

For each chunk:
* **Prefix** = code-built header `"[{doc_type} {doc_number_raw} dt. {issue_date} | {subject}] "` + the LLM line from R9 (what this excerpt covers + EN/HI keywords).
* Store `text_for_embedding = prefix + "\n" + chunk_text`; store `text_for_display = chunk_text`.
* Embed with BGE-M3 (dense 1024-d + sparse weights). Store in Qdrant with payload: `chunk_id, doc_id, clause_id, doc_type, issue_date, doc_number_norm, topics, entities, language, section_type, status_now`.
* Also create a **document-level index** (one vector per doc from `subject + summary_en + summary_hi`) for coarse retrieval and for "list/find documents" queries.
* **Acceptance:** build a 200-question retrieval gold set (Section 10) and report recall@10; must be ≥ 90% at document level before moving on.

### P13 — Load the graph

Neo4j constraints and indexes (Cypher, Neo4j 5.x):

```cypher
CREATE CONSTRAINT doc_id    IF NOT EXISTS FOR (d:Document)   REQUIRE d.doc_id IS UNIQUE;
CREATE CONSTRAINT stub_id   IF NOT EXISTS FOR (s:Stub)       REQUIRE s.stub_id IS UNIQUE;
CREATE CONSTRAINT clause_id IF NOT EXISTS FOR (c:Clause)     REQUIRE c.clause_id IS UNIQUE;
CREATE CONSTRAINT table_id  IF NOT EXISTS FOR (t:Table)      REQUIRE t.table_id IS UNIQUE;
CREATE CONSTRAINT ent_id    IF NOT EXISTS FOR (e:Entity)     REQUIRE e.entity_id IS UNIQUE;
CREATE CONSTRAINT inst_id   IF NOT EXISTS FOR (i:Instrument) REQUIRE i.inst_id IS UNIQUE;
CREATE INDEX doc_number IF NOT EXISTS FOR (d:Document) ON (d.number_norm);
CREATE INDEX doc_date   IF NOT EXISTS FOR (d:Document) ON (d.issue_date);
CREATE INDEX doc_type   IF NOT EXISTS FOR (d:Document) ON (d.doc_type);
```

Load with `UNWIND $rows AS row MERGE …` in batches of 1,000. Node/edge model in Section 8.

* **Acceptance:** counts in Neo4j match the JSONL counts; random 20 documents queried by number return the right node, clauses, and edges.

### P14 — Query service

Implement Section 9 as a FastAPI service with endpoints `/ask` (returns answer + citations + trace), `/doc/{id}`, `/timeline/{id}`, `/status?doc=…&asof=…`, `/in_force?topic=…&asof=…`. Every `/ask` call writes its full trace (query analysis, retrieved chunk ids with scores, graph expansion, status labels, final prompt, answer, verifier result) to `logs/trace.jsonl`.

### P15 — Evaluation

Implement Section 10. Produce `report/eval_report.md`.

---

## 6. RUNTIME PROMPT LIBRARY (small-model-optimised)

Store each prompt as `prompts/<ID>.txt`. All prompts share the `SYSTEM_BASE`. Replace `{{…}}` fields in code. **Replace the example blocks with 2–3 real examples from your pilot set (one Hindi, one English).**

### SYSTEM_BASE

```text
You are a precise extraction engine for Indian government documents from the
General Administration Department. Text can be Hindi (Devanagari), English, or
both. It comes from OCR, so it may contain spelling mistakes, broken words and
wrong digits.

RULES
1. Use ONLY the text given in this message. Never use outside knowledge.
2. If a value is not clearly stated, output null. Never guess.
3. Copy numbers, dates and names EXACTLY as written. Do not translate, correct
   or reformat them in fields that end with "_raw" or are named "quote".
4. A "quote" must be copied character-for-character from the text.
5. Output ONE JSON object that follows the schema. No explanation. No markdown.
```

### R1 — Document type

```text
TASK: Decide the type of this document.
Read the title lines. Choose exactly ONE value:
"GR"           = Government Resolution (शासन निर्णय)
"CIRCULAR"     = परिपत्रक / Circular
"NOTIFICATION" = अधिसूचना / Notification
"ORDER"        = आदेश / Order
"LETTER"       = पत्र / कार्यालयीन ज्ञापन / Office Memorandum / Letter
"CORRIGENDUM"  = शुद्धिपत्रक / शुद्धिपत्र / corrigendum / सुधारणा
"RULES"        = नियम / Rules
"OTHER"

TEXT (first 25 lines):
{{top_lines}}

EXAMPLE
TEXT: "महाराष्ट्र शासन / सामान्य प्रशासन विभाग / शासन निर्णय क्रमांक: …"
OUTPUT: {"quote":"शासन निर्णय क्रमांक","doc_type":"GR"}

OUTPUT SCHEMA: {"quote": string, "doc_type": one of the values above}
```

### R2 — Own document number (choose from candidates)

```text
TASK: Pick the number that identifies THIS document (its own reference number).
It is usually near the top ("क्रमांक", "संख्या", "No.", "F.No.", "GR No.") or
next to the signature at the end.
Do NOT choose numbers from the Reference / संदर्भ / वाचा list, they belong to
OTHER documents. Do NOT choose phone numbers, years, amounts or page numbers.

CANDIDATES (found by a program; each has an id):
{{numbered_candidates_with_context}}

If none is the document's own number, answer null.

EXAMPLE
CANDIDATES:
[1] (line 3, top) "क्रमांक: सामान्य-2021/प्र.क्र.45/18"
[2] (line 12, reference block) "शासन निर्णय क्र. सामान्य-2019/प्र.क्र.9/18"
OUTPUT: {"quote":"क्रमांक: सामान्य-2021/प्र.क्र.45/18","choice":1}

OUTPUT SCHEMA: {"quote": string|null, "choice": integer|null}
```

### R3 — Issue date (choose from candidates)

```text
TASK: Pick the ISSUE DATE of THIS document.
It is the date printed with the document's own number, or the date beside the
signature ("दिनांक", "Date", "Dated", "Place, date").
Do NOT choose dates from the Reference list, dates of events, effective dates
("with effect from"), or deadlines.

CANDIDATES:
{{numbered_date_candidates_with_context}}

OUTPUT SCHEMA: {"quote": string|null, "choice": integer|null}
```

### R4 — Subject

```text
TASK: Copy the subject line of the document.
Look for "विषय", "Subject", "Sub:" or the bold title under the department name.
Copy it exactly in "subject". Then write a short English version in "subject_en"
(if the subject is already English, repeat it).

TEXT (first 30 lines):
{{top_lines}}

OUTPUT SCHEMA: {"subject": string|null, "subject_en": string|null}
```

### R5 — One reference item (called once per item of the reference block)

```text
TASK: This single line is one item from the "References" list of a document.
Extract what it says about the OTHER document it points to.

ITEM:
{{reference_item_text}}

EXAMPLE
ITEM: "3) शासन निर्णय क्रमांक सामान्य-2019/प्र.क्र.9/18, दिनांक 05 जून 2019"
OUTPUT: {"number_raw":"सामान्य-2019/प्र.क्र.9/18","date_raw":"05 जून 2019","subject_raw":null,"doc_type_guess":"GR"}

OUTPUT SCHEMA: {"number_raw": string|null, "date_raw": string|null,
 "subject_raw": string|null,
 "doc_type_guess": "GR"|"CIRCULAR"|"NOTIFICATION"|"ORDER"|"LETTER"|"CORRIGENDUM"|"RULES"|"ACT"|"COURT_ORDER"|"OTHER"}
```

### R6 — Effective date (called per candidate sentence)

```text
TASK: Does this sentence say from or until when a rule applies?
Choose "kind":
"EFFECTIVE_FROM"  = the rule starts on a date ("with effect from", "दिनांक … पासून लागू", "from the date of issue")
"EFFECTIVE_TO"    = the rule ends / is valid till a date ("valid up to", "पर्यंत")
"NONE"            = anything else
"retrospective" = true only if the sentence says the rule applies from a date BEFORE this document was issued
("retrospectively", "पूर्वलक्षी प्रभावाने", "पूर्व प्रभाव से").
If the sentence says "from the date of this order/issue", set date_raw to "ISSUE_DATE".

SENTENCE: {{sentence}}

OUTPUT SCHEMA: {"quote": string, "kind": "EFFECTIVE_FROM"|"EFFECTIVE_TO"|"NONE",
 "date_raw": string|null, "retrospective": boolean}
```

### R7 — Topics (multi-label from the locked taxonomy)

```text
TASK: Choose up to 3 topics that best describe this document. Choose ONLY from the list.
If none fits, return an empty list.

TOPICS (id: English / Hindi):
{{topics_list}}

DOCUMENT SUBJECT: {{subject}}
DOCUMENT SUMMARY (first 600 words of the operative part): {{operative_text}}

OUTPUT SCHEMA: {"topic_ids": [string, ...]}   // max 3, each from the list
```

### R8 — Summary (bilingual)

```text
TASK: Summarise this document in 3-4 sentences for a search index.
Include: what decision/rule it states, who it applies to, any amounts, dates or
deadlines that appear in the text, and what earlier document it changes (if the text says so).
Write "summary_en" in English and "summary_hi" in Hindi. Do not add facts that are not in the text.

TEXT (operative part, max 1,500 words): {{operative_text}}

OUTPUT SCHEMA: {"summary_en": string, "summary_hi": string}
```

### R9 — Contextual line for a chunk

```text
TASK: In ONE sentence, say what this excerpt covers, so a search engine can find it
without the rest of the document. Examples of "covers": eligibility conditions,
amount of allowance, procedure for application, effective date, list of posts (table),
exceptions, definitions.
Then give 4 English keywords and 4 Hindi keywords that appear in or are clearly meant
by the excerpt. Do not add facts that are not in the excerpt.

DOCUMENT: {{doc_type}} {{doc_number_raw}} dated {{issue_date}} — {{subject}}
EXCERPT:
{{chunk_text}}

OUTPUT SCHEMA: {"covers": string, "keywords_en": [string,...], "keywords_hi": [string,...]}
```

### R10 — Relation classification (called once per cue sentence; run 3 votes)

```text
TASK: Decide whether THIS document does something to ANOTHER document in this sentence.
"THIS document" = the document you are reading: {{this_doc_label}}.

RELATION TYPES (choose from these exact words)
SUPERSEDES    THIS document replaces an earlier document.  Cues: supersedes, in supersession of, in place of,
              अधिक्रमित, के स्थान पर, निरस्त करते हुए, सर्व पूर्वीचे आदेश
SUPERSEDED_BY a LATER document replaces THIS one.  Cue: "is superseded by".
AMENDS        THIS changes a PART of an earlier document.  Cues: in partial modification of, amended as follows,
              substitute, omit, insert, read as, आंशिक संशोधन, निम्नानुसार संशोधन, ऐवजी वाचावे
AMENDED_BY    a later document changes part of THIS one.
RESCINDS      THIS cancels / withdraws an earlier document.  Cues: rescinded, cancelled, withdrawn, रद्द, निरसित, मागे घेण्यात
RESCINDED_BY  a later document cancels THIS one.
CLARIFIES     THIS explains how to read an earlier document.  Cues: clarification, स्पष्टीकरण
CLARIFIED_BY  a later document explains THIS one.
EXTENDS       THIS extends the period or coverage of an earlier document.  Cues: extended up to, मुदतवाढ, अवधि बढ़ाई
EXTENDED_BY   a later document extends THIS one.
CORRECTS      THIS fixes an error in an earlier document.  Cues: corrigendum, शुद्धिपत्र, typographical error
CORRECTED_BY  a later document fixes an error in THIS one.
IMPLEMENTS    THIS gives effect to a higher instrument (Act, Rule, court order, higher GR).  Cues: in pursuance of,
              in compliance with, के अनुपालन में, उच्च न्यायालयाच्या आदेशानुसार
REFERS_TO     only mentions another document as background.  This is the DEFAULT if no stronger cue exists.
NONE          the sentence does not point to another document (for example it only mentions a clause of THIS document).

DECISION RULES
1. Ask: who does the action? "This GR supersedes X" -> SUPERSEDES. "X stands superseded by this GR" -> SUPERSEDES.
   "This GR is superseded by Y" -> SUPERSEDED_BY.
2. If several types fit, prefer in this order: RESCINDS, SUPERSEDES, AMENDS, CORRECTS, EXTENDS, CLARIFIES,
   IMPLEMENTS, REFERS_TO.
3. "scope": "whole" only if the sentence replaces/cancels the entire document ("in supersession of the GR dated …",
   "all earlier orders"). "partial" if only a paragraph, clause, table, amount, word or rule number is changed.
   Otherwise "unknown". When "partial", copy the part into "scope_text" (e.g. "para 4(b)").
4. "target_kind": DOCUMENT (GR, circular, notification, order, letter), ACT_RULE (an Act, Rule, Regulation),
   COURT_ORDER, or UNKNOWN.
5. If the sentence points to a numbered reference ("संदर्भ क्र. 3", "reference (3)"), set target_ref_idx to that number
   and use the REFERENCE MAP below. Otherwise put the number/date exactly as written in target_number_raw / target_date_raw.
6. One sentence may contain several relations. Return all (maximum 5).

REFERENCE MAP OF THIS DOCUMENT:
{{ref_map}}

PREVIOUS SENTENCE: {{prev_sentence}}
SENTENCE: {{sentence}}
NEXT SENTENCE: {{next_sentence}}

EXAMPLE 1
SENTENCE: "शासन निर्णय दिनांक 05 जून 2019 मधील परिच्छेद 4 (ब) ऐवजी खालील मजकूर वाचावा."
OUTPUT: {"relations":[{"quote":"शासन निर्णय दिनांक 05 जून 2019 मधील परिच्छेद 4 (ब) ऐवजी खालील मजकूर वाचावा.",
 "relation":"AMENDS","target_kind":"DOCUMENT","target_ref_idx":null,"target_number_raw":null,
 "target_date_raw":"05 जून 2019","target_subject_raw":null,"scope":"partial","scope_text":"परिच्छेद 4 (ब)"}]}

EXAMPLE 2
SENTENCE: "This Government Resolution supersedes the Resolution at reference (2) in its entirety."
OUTPUT: {"relations":[{"quote":"This Government Resolution supersedes the Resolution at reference (2) in its entirety.",
 "relation":"SUPERSEDES","target_kind":"DOCUMENT","target_ref_idx":2,"target_number_raw":null,
 "target_date_raw":null,"target_subject_raw":null,"scope":"whole","scope_text":null}]}

EXAMPLE 3
SENTENCE: "The pay shall be fixed as per clause 3 above."
OUTPUT: {"relations":[]}

OUTPUT SCHEMA: {"relations":[{"quote":string,"relation":<type>,"target_kind":string,
 "target_ref_idx":integer|null,"target_number_raw":string|null,"target_date_raw":string|null,
 "target_subject_raw":string|null,"scope":"whole"|"partial"|"unknown","scope_text":string|null}]}
```

### R11 — Relation verifier (independent second opinion)

```text
TASK: Check a claim against a sentence. Be strict.
CLAIM: In the document "{{this_doc_label}}", the sentence below shows that THIS document {{relation_in_words}}
the other document ({{target_description}}), with scope: {{scope}}.

SENTENCE: {{quote}}
CONTEXT: {{prev_sentence}} {{next_sentence}}

Answer "SUPPORTED" only if the sentence clearly states exactly this. Answer "NOT_SUPPORTED" if the sentence says
something different (wrong direction, wrong relation, wrong scope). Answer "UNCLEAR" if you cannot tell.

OUTPUT SCHEMA: {"reason": string (max 20 words), "verdict": "SUPPORTED"|"NOT_SUPPORTED"|"UNCLEAR"}
```

`relation_in_words` map: SUPERSEDES→"replaces", AMENDS→"partly amends", RESCINDS→"cancels", CLARIFIES→"clarifies", EXTENDS→"extends", CORRECTS→"corrects an error in", IMPLEMENTS→"implements / gives effect to", REFERS_TO→"only refers to", and the `_BY` forms → "is replaced by / amended by / cancelled by / clarified by / extended by / corrected by".

### R12 — Table description (header + 3 rows only)

```text
TASK: Describe this table so it can be stored in a database. You see only the header and the first rows.
Do NOT output any rows.
For each column give: "name_raw" (as written), "name_en" (short English name, snake_case),
"type" (text|int|float|date|amount|percent), "unit" (e.g. "INR", "%", "posts", or null).
"table_kind": "list_of_posts"|"pay_scale"|"allowance_rates"|"eligibility"|"schedule"|"annexure_list"|"other".

CAPTION / HEADING ABOVE: {{caption}}
HEADER + FIRST 3 ROWS (markdown):
{{table_head}}

OUTPUT SCHEMA: {"title_en":string,"title_hi":string|null,
 "columns":[{"name_raw":string,"name_en":string,"type":string,"unit":string|null}],"table_kind":string}
```

### R13 — Entity mentions (per clause)

```text
TASK: List mentions in this text of ONLY these kinds of things. Copy each mention exactly.
POST       = a designation, cadre or post (e.g. "Deputy Secretary", "अवर सचिव")
SCHEME     = a named scheme
INSTRUMENT = a named Act, Rule or Regulation
BENEFIT    = an allowance, pay item, leave type, pension item
DEPARTMENT = a department or office
Ignore everything else (people's names, places, dates, numbers).

TEXT: {{clause_text}}

OUTPUT SCHEMA: {"mentions":[{"text":string,"kind":"POST"|"SCHEME"|"INSTRUMENT"|"BENEFIT"|"DEPARTMENT"}]}
```

### R14 — Query understanding

```text
TASK: Analyse a user's question about government documents. Today's date is {{today}}.
Fill every field.

FIELDS
"language"      : "hi" | "en" | "mixed"   (language of the QUESTION)
"intent"        : one of
   "LOOKUP"        a fact from a document (amount, eligibility, procedure, date)
   "CURRENT_RULE"  what is the rule now / what applies today
   "AS_OF_RULE"    what was the rule on a past date or period
   "HISTORY"       how a rule changed over time / which documents amended it
   "STATUS"        is this document still valid / was it superseded
   "DOC_LOOKUP"    questions naming a document number or asking for its content
   "LIST"          list or count documents matching a filter
   "TABLE_QUERY"   needs numbers from a table (counts, rates, totals)
   "COMPARE"       compare two rules or two time points
   "OTHER"
"as_of_date"    : "YYYY-MM-DD" if the question states or implies a date/year, else null
"doc_refs"      : list of document numbers or dates the user mentions, exactly as written
"filters"       : {"doc_type":string|null,"year_from":int|null,"year_to":int|null,"topic_hint":string|null}
"queries_en"    : 2 short English search queries
"queries_hi"    : 2 short Hindi search queries
"needs_table"   : true/false

EXAMPLE
QUESTION: "2019 में अवर सचिव का वेतनमान क्या था?"
OUTPUT: {"language":"hi","intent":"AS_OF_RULE","as_of_date":"2019-12-31","doc_refs":[],
 "filters":{"doc_type":null,"year_from":null,"year_to":null,"topic_hint":"pay scale"},
 "queries_en":["pay scale Under Secretary 2019","Under Secretary pay scale"],
 "queries_hi":["अवर सचिव वेतनमान","अवर सचिव वेतन श्रेणी 2019"],"needs_table":true}

QUESTION: {{question}}
OUTPUT SCHEMA: (as above)
```

### R15 — Extract facts from ONE chunk (the "extract" half of extract-then-compose)

```text
TASK: Read ONE excerpt and extract only the facts that help answer the question.
If the excerpt does not help, return an empty list.
Each fact must include an exact quote from the excerpt. Do not use knowledge from outside the excerpt.

QUESTION: {{question}}
EXCERPT LABEL: {{chunk_label}}      // e.g. [D3 | GR 45/2021 | 2021-03-12 | clause 4]
EXCERPT:
{{chunk_text}}

OUTPUT SCHEMA: {"facts":[{"fact": string (one short sentence in the language of the question),
 "quote": string, "contains_number_or_date": boolean}]}
```

### R16 — Compose the final answer from facts (the "compose" half)

```text
TASK: Answer the question using ONLY the FACTS below. Each fact has a label.
RULES
1. Answer in the language of the question ({{language}}).
2. After every statement add the label(s) of the fact(s) that support it, like [D3-C4].
3. Use the STATUS of each source:
   - If a source is SUPERSEDED / RESCINDED, do NOT present it as the current rule. Say which document replaced it and when.
   - If a source is PARTIALLY_AMENDED, mention the amendment and its source.
   - For questions about a past date, use the sources that were in force on that date.
4. Never state a number, date or document number that is not written in the FACTS.
5. If the FACTS do not answer the question, say clearly that the answer is not found in the available documents,
   and say what was found instead (if anything). Do not guess.
6. Be brief: 1-3 sentences, then a bullet list only if there are several items.

QUESTION: {{question}}
AS-OF DATE: {{as_of}}
FACTS:
{{facts_with_labels_and_status}}

OUTPUT SCHEMA: {"answer": string, "used_labels":[string,...], "answerable": boolean}
```

### R17 — Answer verifier

```text
TASK: Check an answer against its sources. For each claim in the answer, decide whether the sources support it.
ANSWER: {{answer}}
SOURCES:
{{cited_chunks}}
List the claims that are NOT supported (wrong number, wrong date, wrong status, missing in sources).
OUTPUT SCHEMA: {"unsupported_claims":[string,...],"verdict":"OK"|"HAS_UNSUPPORTED"}
```

If `HAS_UNSUPPORTED`: regenerate once with the unsupported claims listed as "remove or fix these"; if still failing, return a safe answer: the supported part + "some details could not be confirmed".

### R18 — Table text-to-SQL

```text
TASK: Write ONE DuckDB SQL query that answers the question from the tables below.
Use only these tables and columns. Use "value_num" for numbers. Return only the SQL in "sql".
If the tables cannot answer the question, return "sql": null.

TABLE CATALOG:
{{relevant_table_schemas_with_3_sample_rows_each}}

QUESTION: {{question}}
OUTPUT SCHEMA: {"sql": string|null}
```

Run the SQL **read-only**, with a 5-second timeout, `LIMIT 200`. Feed the result rows (not the LLM's memory) into R16 as facts labelled `[T<table_id>]`.

### R19 — Synthetic evaluation questions (offline)

```text
TASK: From the excerpt, write {{n}} different questions that a clerk might ask, whose answers are fully in the excerpt.
Make them: 1 factual (number/date/amount), 1 procedural, 1 using different words than the excerpt (paraphrase).
Write each question in {{language}}. For each give the exact answer and the exact quote that proves it.
EXCERPT [{{chunk_label}}]: {{chunk_text}}
OUTPUT SCHEMA: {"items":[{"question":string,"answer":string,"quote":string,"type":"factual"|"procedural"|"paraphrase"}]}
```

---

## 7. Lexicons and regular expressions (starting points; tune on the pilot set)

Store as YAML in `lexicons/`. All matching is done on the `search` copy (ASCII digits), case-insensitive, after NFC.

### 7.1 Relation cues

```yaml
SUPERSEDES:
  en: ["supersede", "supersedes", "superseded", "in supersession of", "in place of", "stands replaced", "replaces"]
  hi: ["अधिक्रमित", "के स्थान पर", "की जगह", "निरस्त करते हुए", "अतिक्रमित", "प्रतिस्थापित"]
  mr: ["अधिक्रमित", "ऐवजी", "च्या जागी", "अधिक्रमित करण्यात"]
AMENDS:
  en: ["in partial modification", "partly modified", "amended", "is amended", "substitute", "omit", "insert", "shall be read as", "read as", "modified as follows"]
  hi: ["आंशिक संशोधन", "आंशिक रूप से संशोधित", "संशोधन", "निम्नानुसार पढ़ें", "के स्थान पर पढ़ा जाए", "जोड़ा जाए", "विलोपित"]
  mr: ["अंशतः बदल", "सुधारणा", "ऐवजी वाचावे", "खालीलप्रमाणे वाचावे", "समाविष्ट"]
RESCINDS:
  en: ["rescind", "rescinded", "cancelled", "canceled", "withdrawn", "revoked", "hereby cancelled"]
  hi: ["रद्द", "निरस्त", "निरसित", "वापस लिया", "प्रत्याहरित"]
  mr: ["रद्द", "निरसित", "मागे घेण्यात"]
CLARIFIES:
  en: ["clarification", "clarified", "it is clarified", "for the removal of doubts"]
  hi: ["स्पष्टीकरण", "स्पष्ट किया जाता है"]
  mr: ["स्पष्टीकरण", "स्पष्ट करण्यात"]
EXTENDS:
  en: ["extended up to", "extension of", "extended till", "validity extended"]
  hi: ["अवधि बढ़ाई", "विस्तार", "समय सीमा बढ़ा"]
  mr: ["मुदतवाढ", "मुदत वाढविण्यात"]
CORRECTS:
  en: ["corrigendum", "typographical error", "clerical error", "correction"]
  hi: ["शुद्धिपत्र", "शुद्धि पत्र", "त्रुटि"]
  mr: ["शुद्धिपत्रक", "शुद्धिपत्र", "चूक"]
IMPLEMENTS:
  en: ["in pursuance of", "in compliance with", "in accordance with the order of", "in terms of"]
  hi: ["के अनुपालन में", "के अनुसरण में", "के अनुसार"]
  mr: ["च्या अनुषंगाने", "च्या अनुपालनार्थ", "च्या आदेशानुसार"]
REFERS_TO_BLOCK_START:
  en: ["Read:", "Reference:", "References:", "Ref:", "Ref.:", "Refer:"]
  hi: ["संदर्भ", "पढ़ें", "सन्दर्भ"]
  mr: ["वाचा", "संदर्भ"]
```

### 7.2 Effective-date cues

```yaml
EFFECTIVE_FROM:
  en: ["with effect from", "w.e.f", "effective from", "shall come into force", "from the date of issue", "from the date of this"]
  hi: ["से लागू", "दिनांक .* से प्रभावी", "प्रभावी होगा", "जारी होने की दिनांक से"]
  mr: ["पासून लागू", "प्रभावी", "निर्गमित झाल्याच्या दिनांकापासून"]
EFFECTIVE_TO:
  en: ["valid up to", "valid till", "until", "upto"]
  hi: ["तक मान्य", "तक वैध", "पर्यंत"]
  mr: ["पर्यंत वैध", "पर्यंत"]
RETROSPECTIVE:
  en: ["retrospective", "retrospectively", "with retrospective effect"]
  hi: ["पूर्व प्रभाव से", "पूर्वव्यापी"]
  mr: ["पूर्वलक्षी प्रभावाने"]
```

### 7.3 Dates

```python
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
MONTHS = {
 # English
 "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,"may":5,"jun":6,"june":6,
 "jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,"oct":10,"october":10,
 "nov":11,"november":11,"dec":12,"december":12,
 # Hindi
 "जनवरी":1,"फरवरी":2,"फ़रवरी":2,"मार्च":3,"अप्रैल":4,"मई":5,"जून":6,"जुलाई":7,"अगस्त":8,
 "सितंबर":9,"सितम्बर":9,"अक्टूबर":10,"अक्तूबर":10,"नवंबर":11,"नवम्बर":11,"दिसंबर":12,"दिसम्बर":12,
 # Marathi
 "जानेवारी":1,"फेब्रुवारी":2,"एप्रिल":4,"मे":5,"जुलै":7,"ऑगस्ट":8,"सप्टेंबर":9,"ऑक्टोबर":10,"नोव्हेंबर":11,"डिसेंबर":12,
}
DATE_NUMERIC = r"\b(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{2,4})\b"      # day-month-year (Indian order)
DATE_TEXT    = r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s+([A-Za-z\u0900-\u097F]+)[.,]?\s+(\d{4})\b"
FIN_YEAR     = r"\b(\d{4})\s*[-–]\s*(\d{2,4})\b"    # financial year, NOT a date
```

Rules: Indian **day-first** order. Two-digit year: ≥ 50 → 19xx, else 20xx, and flag `two_digit_year`. Reject impossible dates (month 13, day 32). If day ≤ 12 and month ≤ 12 and the file shows both orders elsewhere, flag `ambiguous_order`.

### 7.4 Document numbers

```python
# Normalisation
def normalize_number(s: str) -> str:
    s = unicodedata.normalize("NFC", s).translate(DEV_DIGITS)
    s = re.sub(r"(?i)\b(no|number|क्रमांक|क्र|संख्या|f\.?\s*no)\b\.?\s*[:\-–]?", "", s)
    s = re.sub(r"[\s]+", "", s)                  # drop spaces
    s = re.sub(r"[\\|]", "/", s)                 # unify separators
    s = s.replace("–", "-").replace("—", "-").rstrip(".,;:")
    return s.upper()
```

Candidate patterns (run on the `search` copy): lines containing `क्रमांक|क्र\.|संख्या|No\.?|F\.?\s*No\.?|GR No|Order No|Memo No` followed by `[A-Za-z0-9\u0900-\u097F/\-\.\(\)]{4,}`; also free-standing patterns such as `[A-Za-z\u0900-\u097F]{2,}[-/ ]?\d{2,4}[/\-][^\s,;]{1,15}[/\-]\d{1,4}`. Keep `line_no`, `in_top` (first 25 lines), `in_bottom` (last 25 lines), `in_reference_block`.

### 7.5 Clause markers

```python
CLAUSE_PATTERNS = [
  r"^\s*(\d{1,3})\s*[.)]\s+",                  # 1.  2)
  r"^\s*\((\d{1,3})\)\s+",                     # (1)
  r"^\s*\(([a-z])\)\s+",                       # (a)
  r"^\s*\(((?:i{1,3}|iv|v|vi{0,3}|ix|x))\)\s+",# (i) (ii) (iv)
  r"^\s*([क-ह])\s*[.)]\s+",                    # क)  ख.
  r"^\s*\(([क-ह])\)\s+",                       # (क)
  r"^\s*(परिच्छेद|कलम|अनुच्छेद|धारा|Para(?:graph)?|Clause|Section|Rule)\s+(\d+)",
  r"^\s*#{1,4}\s+",                            # markdown headings
]
```

---

## 8. Graph model and temporal status algorithm

### 8.1 Nodes and edges

```
(:Document {doc_id, doc_type, number_raw, number_norm, issue_date, effective_from, effective_to,
            subject, subject_en, summary_en, summary_hi, language, topics[], file_id, status_now})
(:Stub     {stub_id, number_raw, number_norm, date, subject_raw})
(:Instrument {inst_id, name, year, kind})          // Acts, Rules, court orders
(:Clause   {clause_id, doc_id, number, section_type, text, char_start, char_end})
(:Table    {table_id, doc_id, title_en, title_hi, table_kind})
(:Entity   {entity_id, type, name_en, name_hi})     // Post, Scheme, Benefit, Department, Topic

(:Document)-[:HAS_CLAUSE]->(:Clause)         (:Clause)-[:CHILD_OF]->(:Clause)
(:Document)-[:HAS_TABLE]->(:Table)
(:Clause)-[:MENTIONS]->(:Entity)             (:Document)-[:ABOUT]->(:Entity {type:'Topic'})
(:Document|Stub)-[r:SUPERSEDES|AMENDS|RESCINDS|CLARIFIES|EXTENDS|CORRECTS|IMPLEMENTS|REFERS_TO]->(:Document|Stub|Instrument)
   r {scope, scope_text, effective_date, quote, tier, source_clause_id, target_clause_id}
(:Document)-[:DUPLICATE_OF]->(:Document)
```

### 8.2 Status rules (deterministic)

For a document `d` and date `T`:

```python
WHOLE = {"SUPERSEDES", "RESCINDS"}

def edge_date(e):           # when does this edge take effect?
    return e.effective_date or e.source_doc.effective_from or e.source_doc.issue_date

def status_as_of(d, T, edges):
    if d.effective_from and T < d.effective_from:           return ("NOT_YET_EFFECTIVE", None)
    if d.effective_to   and T > d.effective_to:             return ("EXPIRED", None)

    inc = [e for e in edges.incoming(d) if e.tier in ("A","B") and edge_date(e) and edge_date(e) <= T]

    whole = [e for e in inc if e.type in WHOLE and e.scope == "whole"]
    if whole:
        e = min(whole, key=edge_date)                       # first replacement
        return ("RESCINDED" if e.type == "RESCINDS" else "SUPERSEDED", e)

    partial = [e for e in inc if e.type in {"AMENDS"} or (e.type in WHOLE and e.scope in ("partial","unknown"))]
    corr    = [e for e in inc if e.type == "CORRECTS"]
    ext     = [e for e in inc if e.type == "EXTENDS"]
    if partial or corr or ext:
        return ("IN_FORCE_AMENDED", partial + corr + ext)   # list the amending documents

    # edges whose date is unknown and tier A/B -> be honest
    undated = [e for e in edges.incoming(d) if e.tier in ("A","B") and not edge_date(e) and e.type in WHOLE]
    if undated:                                             return ("LIKELY_SUPERSEDED_DATE_UNKNOWN", undated)

    return ("IN_FORCE", None)
```

Notes:
* A document that **supersedes** another is not automatically "in force": evaluate it with the same function (chains A ← B ← C work naturally).
* **Corrigendum:** the corrected document stays in force, but answers must apply the correction. Retrieval always pulls CORRECTS documents together with their target (Section 9, step 6).
* **Clause-level status:** when an `AMENDS` edge has `target_clause_id`, mark that clause `AMENDED` from the edge date and attach `by_doc`.
* Pre-compute `status_now` (T = today) into Neo4j and Qdrant payloads for speed; compute `status_as_of` on demand for historical questions.

### 8.3 Ad-hoc Cypher examples

```cypher
// Documents in force on a date
MATCH (d:Document)
WHERE d.issue_date <= date($asof)
  AND NOT EXISTS {
    MATCH (n)-[r:SUPERSEDES|RESCINDS]->(d)
    WHERE r.scope = 'whole' AND r.tier IN ['A','B']
      AND coalesce(r.effective_date, n.effective_from, n.issue_date) <= date($asof)
  }
RETURN d.doc_id, d.number_raw, d.subject_en ORDER BY d.issue_date DESC;

// Full lineage of a rule (everything connected by change-edges, any direction, up to 4 hops)
MATCH p = (d:Document {number_norm: $num})-[:SUPERSEDES|AMENDS|RESCINDS|CORRECTS|CLARIFIES|EXTENDS*1..4]-(x)
RETURN p;

// What changed clause 4 of a given GR?
MATCH (n)-[r:AMENDS]->(c:Clause {clause_id: $cid}) RETURN n.number_raw, r.effective_date, r.quote;
```

---

## 9. Query-time pipeline

```
question
 ├─ R14 query understanding            → language, intent, as_of, doc_refs, filters, queries_en/hi
 ├─ Router (code) by intent:
 │    DOC_LOOKUP  → exact graph lookup by number/date (fallback: fuzzy) → load doc + its edges
 │    LIST        → structured filter over Document metadata (Neo4j/DuckDB), no LLM retrieval
 │    TABLE_QUERY → table_catalog retrieval → R18 text-to-SQL → rows as facts
 │    STATUS      → resolve doc → status_as_of(doc, as_of or today) → answer from graph + quotes
 │    HISTORY     → resolve seed docs → lineage subgraph → order by edge dates → timeline answer
 │    others      → full hybrid pipeline below
 └─ Hybrid pipeline
     1. Retrieve:   BGE-M3 dense (top 40) + sparse (top 40) over chunks, for each of queries_en + queries_hi
                    (+ original question). Fuse with Reciprocal Rank Fusion → top 60. Apply payload filters
                    (doc_type / years / topics) as a *boost*, not a hard filter, unless the user stated them explicitly.
     2. Doc-level:  search the document-level index (top 10) to add chunks from the best documents.
     3. Rerank:     bge-reranker-v2-m3 on (question, chunk_text_for_embedding) → keep top 20.
     4. Seed docs:  distinct doc_ids of the top 8 chunks.
     5. Expand via graph (the step that fixes "interrelated documents"):
          for each seed doc: follow SUPERSEDES / AMENDS / RESCINDS / CORRECTS / CLARIFIES / EXTENDS
          in BOTH directions, up to 2 hops (tier A/B; tier C only as 'weak'); REFERS_TO 1 hop outgoing only.
          Cap at 8 extra docs. For each extra doc, take its 2 chunks most similar to the question (re-rank).
     6. Always add: every CORRECTS doc of a selected doc; the doc that SUPERSEDES a selected doc (so the answer can say so).
     7. Temporal labelling: compute status_as_of(doc, as_of or today) for every selected doc.
          CURRENT_RULE / LOOKUP : prefer IN_FORCE docs; keep SUPERSEDED docs only as labelled context.
          AS_OF_RULE            : keep docs that were in force on as_of; label others.
          HISTORY / COMPARE     : keep all, sorted by date.
     8. Context building: ≤ 12,000 tokens. Order: in-force first, newest first. Each chunk starts with a label:
          [D3-C4 | GR 45/2021 | 2021-03-12 | clause 4 | STATUS: SUPERSEDED on 2023-05-02 by GR 112/2023 (D7)]
     9. Generate (two modes; pick by model size in config):
          Large model: one call, R16-style rules with the labelled context.
          Small model (recommended for ≤ 9B): extract-then-compose
              for each chunk run R15 (parallel) → keep non-empty facts (max 25, newest/in-force first)
              → run R16 on the facts only.
    10. Verify: code check — every number, date and document number in the answer must occur in a cited chunk
        (after digit/whitespace normalisation). Then R17. On failure, regenerate once, then fall back to a safe answer.
    11. Return: answer + citations (doc number, date, clause, char offsets, status) + trace id.
```

Special handling:
* **Document-number questions:** exact match on `number_norm` beats any semantic search.
* **Language:** answer in the question's language; cite documents with their original numbers.
* **"Not found":** if reranker top score is below a calibrated threshold AND graph lookup found nothing, answer "not found in the available documents" without calling the generator.
* **Conflicting sources:** if two IN_FORCE documents conflict, say so and cite both.

---

## 10. Evaluation (build the harness before tuning anything)

### 10.1 Gold sets

| Set | Size | How it is built |
|---|---|---|
| **G1 Header gold** | 200 docs | I label doc_type, number, date, subject |
| **G2 Relation gold** | 100 docs (≈ 300 relations) | I label every relation, scope, target |
| **G3 Resolution gold** | 200 edges | I mark the correct target doc / stub |
| **G4 Table gold** | 30 tables | I check numeric cells |
| **G5 QA gold** | **≥ 400 questions** | (a) 20+ real failing questions from my chatbot, (b) R19 synthetic questions, **every one human-verified**, (c) hand-written temporal / multi-document / unanswerable questions |

G5 composition (target): 35% Hindi, 35% English, 10% mixed; 20% table questions; 20% temporal ("as of", "current", "history"); 15% multi-document; 10% unanswerable (must be refused); 5% document-number lookups. Each item stores: question, language, intent type, gold answer, gold source chunk/doc ids, as-of date, and for temporal items the correct status.

### 10.2 Metrics and targets (targets, not guarantees)

| Layer | Metric | Target |
|---|---|---|
| Parsing | doc number / issue date accuracy | ≥ 97% / ≥ 98% |
| Relations | precision (tier A+B) / recall (whole-doc supersede/amend/rescind) | ≥ 95% / ≥ 85% |
| Resolution | correct auto-links | ≥ 97% |
| Status | correct "in force today" on 50 docs; on temporal gold questions | ≥ 95% |
| Tables | numeric cell exactness | ≥ 98% |
| Retrieval | doc-level recall@10 / clause-level recall@10 / MRR | ≥ 92% / ≥ 80% / report |
| Answer | faithfulness (all claims supported) | ≥ 95% |
| Answer | exact match on numbers / dates / doc numbers | ≥ 95% |
| Answer | correct refusal on unanswerable / wrong refusal on answerable | ≥ 90% / ≤ 5% |
| Citations | cited chunk actually contains the claim | ≥ 95% |

### 10.3 Error taxonomy (tag every failure with exactly one)

`PARSE` (OCR/cleaning) · `HEADER` · `RELATION_MISS` · `RELATION_WRONG` · `RESOLVE` · `STATUS_LOGIC` · `CHUNKING` · `RETRIEVAL_MISS` · `EXPANSION_MISS` · `RERANK` · `TABLE` · `GENERATION` · `VERIFIER`. The report must show failure counts per category so the next fix targets the biggest bucket.

### 10.4 Ablations (run once, report in table form)

(1) hybrid vs dense only; (2) with vs without contextual prefix; (3) with vs without graph expansion; (4) with vs without status labels; (5) one-shot vs extract-then-compose on the small model; (6) with vs without verifier. Keep a component only if it improves G5.

---

## 11. Failure playbook

| Symptom | Most likely cause | Fix |
|---|---|---|
| Wrong GR number extracted | picked a number from the reference list | enforce `in_reference_block == false`; show only top/bottom candidates |
| Very high stub rate | number normalisation mismatch (Devanagari digits, `क्र.`, spaces, `/` vs `-`) | print 30 unmatched pairs; extend `normalize_number`; add year-segment aware matching |
| Hindi answers wrong, English fine | few-shots only English; OCR damaged matras | add Hindi few-shots; check OCR quality; try bigger model for Hindi phases |
| Relations inverted (X supersedes Y vs Y supersedes X) | passive voice / "is superseded" | R10 has `_BY` types; check flipping code; add passive examples |
| Rule shown as current but was replaced | edge missing (recall) or tier C | review tier C; add cue words; look at `REFERS_TO` edges whose quote has a cue |
| Answer mixes old and new amounts | superseded docs entered context without labels | enforce status labels; prefer IN_FORCE; check step 7/8 |
| Table numbers wrong | LLM retyped rows or OCR split cells | use code parsing (P10); compare against the source row |
| Retrieval misses obvious documents | chunk lacks context | check contextual prefix; add document-level index; use both `queries_en` and `queries_hi` |
| Model invents a citation | free-form citation | citations are generated from fact labels only; verify in code |
| Pipeline too slow | sequential calls | raise `max_concurrency`; enable prefix caching; batch by prompt type |
| Bad JSON from the model | long outputs or complex schema | shorten schema; lower `max_tokens`; use the repair prompt; split the task |

---

## 12. Definition of done

- [ ] P0–P15 each passed their acceptance tests and results are in `report/`.
- [ ] All 5,300 files accounted for: processed, duplicate-linked, or listed in `review/needs_reocr.csv`.
- [ ] Every edge has `quote`, `tier`, and `resolution`; no tier-C edge influences status.
- [ ] `status_as_of` unit tests pass (whole / partial / chain / backdated / expiry / corrigendum).
- [ ] The 20 real failing chatbot questions now pass, or each has an explained root cause.
- [ ] `/ask` returns answer, citations with status, and a trace; every number in an answer is verified in code.
- [ ] `report/eval_report.md` shows every metric in 10.2 and the error-taxonomy counts.
- [ ] README explains how to re-run a single phase, add new documents incrementally (P0→P13 for new files only, then recompute status for affected docs), and re-run evaluation.

### Incremental updates (new documents later)

For each new file: P1 → P3 → P4 → P5 → P6 → P7 (resolve against the existing index; also re-resolve existing **stubs** whose number/date now match the new doc) → P8 (recompute status only for docs reachable from the new edges) → P9–P13 for the new doc. Re-run the G5 regression set after every batch of additions.
