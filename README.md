# GAD-BOT: Governance Assistant with Temporal Knowledge Graph & Retriever Explorer

An advanced AI assistant and temporal document retrieval platform specifically built for the **General Administration Department (GAD), Government of Chhattisgarh** (`FG2` module).

---

## 🌟 Key Features

1. **Temporal Knowledge Graph & Document History Traversal**:
   - Stores all 5,305 government documents and 94,493 text passages in an integrated SQLite graph and full-text database (`data/fg2/temporal_graph.db`).
   - Maps 10,630 chronological and legal relationships across 36 years (1990–2026):
     - `SUPERSEDES` / `SUPERSEDED_BY`: Explicit cancellation or replacement of older orders.
     - `AMENDS` / `AMENDED_BY`: Partial modifications and rule amendments.
     - `REFERENCES` / `REFERENCED_BY`: Cross-citations between circulars.
     - `CHRONOLOGICAL_NEXT` / `CHRONOLOGICAL_PREV`: Sequential timeline evolution in service rules, pay scales, and recruitment.
   - Allows instant traversal backward to ancestral root policies and forward to the latest standing active order.

2. **Retriever Experimentation Framework**:
   - **Lexical BM25 Retriever**: High-speed keyword search over chunks using SQLite FTS5.
   - **Temporal Graph Retriever**: Knowledge-graph-augmented retriever that performs history traversal, detects obsolete/superseded orders, applies temporal recency/authority reranking, and injects standing active orders into evidence.
   - **Retriever Experiment Playground**: Interactive side-by-side comparison interface testing precision, latency, and temporal accuracy.

3. **Interactive Graphical Visual Representation**:
   - Fully interactive network graph explorer at `http://localhost:3003/graph.html` powered by Vis.js.
   - Color-coded node statuses:
     - 🟢 **Active / Current**: Valid, in-effect circulars.
     - 🟡 **Amended**: Orders modified by subsequent circulars.
     - 🔴 **Superseded**: Cancelled or obsolete historical orders.
   - **Time-Scrubber Slider**: Scrub through history from 1990 to 2026.
   - **Document Traversal Drawer**: Clicking any document displays its entire chronological lineage, predecessor orders, successor amendments, chunk excerpts, and one-click AI inquiry.

4. **OpenAI-Compatible LLM Integration**:
   - Configured via `.env` to connect to `http://103.51.8.88/v1` running `ocr-32b` (Qwen3-32B) through vLLM.

---

## 🚀 Quick Start

### 1. Environment Configuration
The `.env` file at the root contains the required configurations:
```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://103.51.8.88/v1
LLM_MODEL=ocr-32b
LLM_API_KEY=dummy

PIPELINE=fg2
PORT=3003
FLASK_PORT=5003
JWT_SECRET=gad-bot-temporal-graph-secret-key-2026
KMP_DUPLICATE_LIB_OK=TRUE
```

### 2. Run the Application
Run both the Flask backend and the Node.js Web UI with a single command:
```bash
python run.py
```

### 3. Open in Browser
- **Temporal Knowledge Graph & Retriever Explorer**: [http://localhost:3003/graph.html](http://localhost:3003/graph.html)
- **GAD Governance Assistant Chat**: [http://localhost:3003/](http://localhost:3003/)
- **Document Explorer**: [http://localhost:3003/documents.html](http://localhost:3003/documents.html)
- **Flask Internal API Backend**: [http://127.0.0.1:5003](http://127.0.0.1:5003)

---

## 📁 Project Structure

```text
GAD-BOT/
├── .env                              # LLM, Port, and Pipeline configuration
├── pipelines.json                    # GAD pipeline registry
├── run.py                            # Unified launcher for Flask and Node.js
├── package.json                      # Root npm scripts
├── core/
│   ├── pipeline_config.py            # Active pipeline resolver (defaults to fg2)
│   ├── qdrant_store.py               # Vector store owner
│   ├── 04_embeddings_and_kg/scripts/ # Embeddings, Graph, and Retrievers
│   │   ├── graph_store.py            # Temporal Knowledge Graph & SQLite manager
│   │   ├── build_graph.py            # Graph construction script
│   │   ├── retrievers.py             # Modular retriever experimentation framework
│   │   ├── llm_client.py             # Client for 103.51.8.88/v1/models (vLLM)
│   │   └── rag_pipeline.py           # Core RAG pipeline
│   └── 05_webui/                     # Web UI and Flask Backend
│       ├── app.py                    # Flask API with /api/graph and /api/retriever endpoints
│       └── nodejs/                   # Node.js UI server & static assets
│           ├── server.js             # Express proxy to Flask
│           └── public/
│               ├── graph.html        # Interactive Graphical Explorer & Experiment Playground
│               ├── graph.js          # Vis.js graph rendering & traversal logic
│               ├── index.html        # Chat interface
│               └── documents.html    # Document viewer
└── data/
    └── fg2/                          # GAD data module
        ├── doc_metadata.json         # Extracted document facts (dates, orders, supersessions)
        ├── documents.json            # Document registry
        ├── temporal_graph.db         # SQLite Temporal Graph & FTS5 full-text database
        ├── chunks/                   # Semantic text chunks (94k+ chunks)
        └── stage2/                   # OCR structured markdown results
```
