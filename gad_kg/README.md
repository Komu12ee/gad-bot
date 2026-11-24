# GAD Temporal Knowledge Graph & RAG System (`gad_kg`)

An end-to-end legal temporal knowledge graph and hybrid RAG system for General Administration Department (GAD, Chhattisgarh) government circulars, notifications, and orders.

Built strictly according to `GAD_TKG_RAG_MASTER_PROMPT.md`.

---

## 1. Architecture Overview

- **Phases 0–3**: Inventory, high-fidelity OCR normalization, MinHash LSH deduplication, deterministic cue extraction.
- **Phases 4–6**: Multi-vote consensus LLM header and legal relation extraction (`SUPERSEDES`, `AMENDS`, `RESCINDS`, `CORRECTS`, etc.) with Rule 4 substring verification and 3-tier confidence partitioning.
- **Phases 7–8**: Reference resolution with typed stubs and deterministic legal status evaluation `status_as_of(d, T)` (Spec Section 8.2).
- **Phases 9–10**: Granular clause segmentation ($\le 600$ tokens ceiling) and code-based markdown table parser with SQLite structured store (`tables.db`).
- **Phases 12–13**: Contextual prefix indexing with SQLite FTS5 BM25 lexical engine, Neo4j 5.x Cypher export, and local graph database (`graph.db`).
- **Phases 14–15**: FastAPI Query Service (`/ask`, `/doc/{id}`, `/status`, `/timeline/{id}`, `/in_force`) with 2-hop graph expansion, chronological context assembly, and gold evaluation harness.

---

## 2. Directory Structure

```text
gad_kg/
├── config.yaml                   # Global system configuration (vLLM URL, paths, thresholds)
├── README.md                     # Run commands & pipeline documentation
├── prompts/                      # R04-R19 small-model-optimized runtime prompts
│   ├── SYSTEM_BASE.txt
│   ├── R04.txt (Header)
│   ├── R10.txt (Relations)
│   ├── R12.txt (Table description)
│   ├── R14.txt (Query understanding)
│   └── R16.txt (Answer generation)
├── schemas/                      # JSON schemas for constrained decoding
├── src/                          # Phase implementations
│   ├── common/                   # LLM client, cache, I/O, normalizers
│   ├── p00_inventory.py
│   ├── p01_clean.py
│   ├── p02_dedup.py
│   ├── p03_features.py
│   ├── p04_header.py
│   ├── p05_relations.py
│   ├── p06_verify.py
│   ├── p07_resolution.py
│   ├── p08_status.py
│   ├── p09_chunk.py
│   ├── p10_tables.py
│   ├── p11_entities.py
│   ├── p12_index.py
│   ├── p13_graph_load.py
│   ├── p14_query_service.py
│   └── p15_eval.py
├── tests/                        # 14 unit test suites (55 passing tests)
├── data/
│   ├── clean/                    # Cleaned OCR markdown files
│   └── work/                     # JSONL artifacts, SQLite DBs, Cypher scripts
│       ├── tables.db             # Structured table store (catalog, cells, dynamic tables)
│       ├── chunks_fts.db         # FTS5 lexical index
│       ├── graph.db              # Local graph database
│       └── graph.cypher          # Neo4j 5.x import script
├── logs/
│   └── trace.jsonl               # Query execution traces & citation logs
└── report/
    └── eval_report.md            # Final evaluation metrics & ablations
```

---

## 3. Quick Run Commands

### Run Full Test Suite (55 Unit Tests)
```bash
pytest gad_kg/tests/
```

### Run Query Service (FastAPI)
```bash
python gad_kg/src/p14_query_service.py
# Server runs at http://127.0.0.1:8000
```

### Endpoints
- `POST /ask`: `{"question": "पदोन्नति संबंधी नियम क्या हैं?", "as_of": "2026-11-19"}`
- `GET /doc/{id}`: `http://127.0.0.1:8000/doc/D_33837c29b3e6f4a8`
- `GET /status`: `http://127.0.0.1:8000/status?doc=D_33837c29b3e6f4a8&asof=2026-11-19`
- `GET /timeline/{id}`: `http://127.0.0.1:8000/timeline/D_33837c29b3e6f4a8`
- `GET /in_force`: `http://127.0.0.1:8000/in_force?topic=पदोन्नति`

### Run Evaluation Harness
```bash
python gad_kg/src/p15_eval.py
```
Outputs report to `gad_kg/report/eval_report.md`.
