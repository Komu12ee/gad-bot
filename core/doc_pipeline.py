"""Single-document indexing for the verification workspace.

The panel only REVIEWS text: documents arrive through the ingest run (scrape -> OCR
-> import, deploy/ingest-all.sh / deploy/gpu-vm.sh), which happens while the bots
are stopped. Nothing here OCRs — there is no upload and no re-OCR from the panel,
so OCR never competes with the answering model for the GPU.

What it does: chunk + embed a VERIFIED document with the query pipeline's
already-loaded bge-m3 model + Qdrant client, and remove a document's vectors again.
Each point is tagged with `doc_id` so a document's vectors can be replaced or
removed cleanly when its .md is edited (the chunker names every md "structured", so
the default `source` payload can't distinguish documents).
"""

import hashlib
import json
import os
import shutil
import sys
import threading
import uuid
from pathlib import Path

CORE = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE))
import pipeline_config as pc
import doc_registry as registry

SCRIPTS = CORE / "04_embeddings_and_kg" / "scripts"

# Serialize embedding jobs so concurrent verifies don't thrash the models.
_JOB_LOCK = threading.Lock()


def md_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Indexing: chunk the md, embed, upsert into Qdrant (tagged with doc_id) ─────
def _load_rag():
    sys.path.insert(0, str(SCRIPTS))
    import rag_pipeline as rag
    rag.ensure_models_loaded()
    client = rag.ensure_qdrant_client()
    return rag, client


_CHUNKER = None
_CHUNKER_LOCK = threading.Lock()


def _get_chunker():
    """Cached Stage2Chunker so a (bulk) re-index pays the heavy init — tokenizer +
    Docling converter + HybridChunker — ONCE and reuses it, instead of spawning a
    fresh Python process (and reloading all of it) for every single document. On a
    multi-thousand-doc ingest that per-doc process startup was the dominant cost."""
    global _CHUNKER
    if _CHUNKER is None:
        with _CHUNKER_LOCK:
            if _CHUNKER is None:
                import logging
                sys.path.insert(0, str(CORE / "03_chunking"))
                from chunk_stage2_output import Stage2Chunker
                # keep the per-doc chunker/docling chatter out of the ingest log
                for name in ("chunk_stage2_output", "docling"):
                    logging.getLogger(name).setLevel(logging.WARNING)
                _CHUNKER = Stage2Chunker(model="BAAI/bge-m3", max_tokens=1024)
    return _CHUNKER


UPSERT_BATCH = 256
_INDEXES_READY = False


def upsert_batched(client, collection: str, points: list):
    """Upsert in batches. A Qdrant SERVER refuses a request over 32 MB (its default);
    with ~25 KB a point (vector + text + keyword weights) one document of more than
    ~1,300 chunks — a budget book's table rows — failed to index in a single call."""
    for i in range(0, len(points), UPSERT_BATCH):
        client.upsert(collection_name=collection, points=points[i:i + UPSERT_BATCH])


def _delete_by_doc(client, collection: str, doc_id: str):
    from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
    if not client.collection_exists(collection):
        return
    client.delete(
        collection_name=collection,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        ),
    )


# Which chunker indexes documents: "v2" (core/chunk_builder.py — text split at
# headings, table rows under their column names, a doc card, a context line on every
# chunk; measured better on the demo questions, scripts/eval_retrieval.py) or
# "hybrid" (Docling's HybridChunker, the previous one — kept as a rollback).
CHUNKER = os.getenv("CHUNKER", "v2").strip().lower()


def chunker_version(rec: dict | None = None) -> str:
    """What the index's `chunker` payload field holds for this setting (and, for v2,
    this document: a web page's label says its site template was removed) — the
    re-index check (scripts/ingest_corpus.py) compares against it."""
    if CHUNKER == "hybrid":
        return "hybrid"
    import chunk_builder
    return chunk_builder.version_for((rec or {}).get("source_type"))


_META = {"mtime": None, "docs": {}}
_META_LOCK = threading.Lock()


def doc_meta(doc_id: str) -> dict | None:
    """This document's record from doc_metadata.json (step 1), re-read when the file
    changes. None when the metadata phase has not run — chunks then carry only the
    document's own heading as context."""
    path = pc.DATA_ROOT / "doc_metadata.json"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    with _META_LOCK:
        if _META["mtime"] != mtime:
            try:
                _META["docs"] = json.loads(path.read_text(encoding="utf-8")).get("documents", {})
            except (OSError, ValueError):
                _META["docs"] = {}
            _META["mtime"] = mtime
        return _META["docs"].get(doc_id)


def _chunks_v2(rec: dict, md: Path, chunk_dir: Path, rag) -> tuple[list, list]:
    """(texts to embed, payloads) from chunk_builder; chunk files written for review."""
    import chunk_builder as cb
    tok = rag.model.tokenizer
    count = lambda s: len(tok.encode(s, add_special_tokens=False))     # noqa: E731
    meta = doc_meta(rec["id"])
    text = md.read_text(encoding="utf-8", errors="replace")
    if rec.get("source_type") == "web":            # the website's header/menus/footer out
        import web_boilerplate as wb
        text = wb.strip_page(text, wb.for_registry(pc.DATA_ROOT / "documents.json", pc.paths.stage2))
    chunks = cb.build_chunks(text, meta, count, md.parent)
    version = cb.version_for(rec.get("source_type"))
    suspect = cb.suspect_pages(md.parent, rec["doc_dir"])
    chunk_dir.mkdir(parents=True, exist_ok=True)
    texts, payloads = [], []
    for c in chunks:
        name = f"structured_chunk_{c.n.zfill(3) if c.n.isdigit() else c.n}.txt"
        (chunk_dir / name).write_text(c.file_text(rec["doc_dir"]), encoding="utf-8")
        texts.append(c.body)
        payloads.append(cb.payload(c, meta, doc_id=rec["id"], doc_dir=rec["doc_dir"],
                                   file=f"{rec['doc_dir']}/{name}", suspect_pages=suspect,
                                   version=version))
    return texts, payloads


def _chunks_hybrid(rec: dict, md: Path, chunk_dir: Path) -> tuple[list, list]:
    """The previous chunker (Docling HybridChunker): file text embedded as written."""
    conf = md.parent / f"{rec['doc_dir']}.pdf_confidence.json"
    _get_chunker().chunk_document(md, chunk_dir, conf if conf.exists() else None, pdf_name=rec["doc_dir"])
    files = sorted(chunk_dir.glob("*_chunk_*.txt"))
    texts, payloads = [], []
    for f in files:
        t = f.read_text(encoding="utf-8").strip()
        if t:
            texts.append(t)
            payloads.append({"text": t, "source": rec["doc_dir"], "doc_id": rec["id"],
                             "chunk": str(len(texts)), "file": f"{rec['doc_dir']}/{f.name}",
                             "chunker": "hybrid"})
    return texts, payloads


def index_document(doc_id: str):
    """(Re)build this document's chunks and (re)embed them. Deletes any prior
    vectors for the same doc_id first, so an edit never leaves a stale copy."""
    global _INDEXES_READY
    with _JOB_LOCK:
        rec = registry.get(doc_id)
        if not rec:
            return
        # Text another OCR model read is never indexed for a bot that has a chosen
        # one (.env INDEX_OCR_MODEL_<BOT>) — not by the bulk run, not by a "Verify" in
        # the admin panel either. Old vectors of such a document are removed.
        import ocr_source
        why = ocr_source.refusal(pc.paths.stage2 / (rec.get("doc_dir") or ""),
                                 ocr_source.wanted_model(pc.PIPELINE))
        if why:
            try:
                rag, client = _load_rag()
                _delete_by_doc(client, rag.COLLECTION_NAME, doc_id)
            except Exception as e:      # noqa: BLE001 — the registry still says why
                print(f"[doc_pipeline] could not remove old vectors of {doc_id}: {e}")
            registry.update(doc_id, indexed=False, chunk_count=0, error=why)
            return
        try:
            # 1. Chunk this document's md (rebuilds from the current structured.md).
            #    Clear any previous chunk files FIRST — otherwise a shorter edit
            #    leaves stale higher-numbered chunks that get re-embedded as
            #    duplicates (two versions of the doc in the DB). Vectors are keyed
            #    by doc_id and deleted below, but the chunk FILES must be reset too.
            chunk_dir = pc.paths.chunks / rec["doc_dir"]
            if chunk_dir.exists():
                shutil.rmtree(chunk_dir, ignore_errors=True)
            md = pc.paths.stage2 / rec["doc_dir"] / "structured.md"
            rag, client = _load_rag()
            try:
                texts, payloads = (_chunks_hybrid(rec, md, chunk_dir) if CHUNKER == "hybrid"
                                   else _chunks_v2(rec, md, chunk_dir, rag))
            except Exception as e:
                registry.update(doc_id, indexed=False,
                                error=f"chunking failed: {type(e).__name__}: {e}")
                return
            if not texts:
                registry.update(doc_id, indexed=False, error="no chunks produced")
                return

            from qdrant_client.models import PointStruct, VectorParams, Distance
            coll = rag.COLLECTION_NAME
            if not client.collection_exists(coll):
                client.create_collection(coll, vectors_config=VectorParams(size=1024, distance=Distance.COSINE))
            if not _INDEXES_READY:      # filters on source / chunk / doc_id / text stay fast on a server
                import qdrant_store
                qdrant_store.ensure_indexes(client, coll)
                _INDEXES_READY = True

            # 2. Remove any previous vectors for this document, then embed fresh.
            _delete_by_doc(client, coll, doc_id)

            enc = rag.model.encode(texts, batch_size=8, max_length=1024)
            dense = enc["dense_vecs"]
            sparse = enc.get("lexical_weights") or [None] * len(texts)

            points = []
            for i, payload in enumerate(payloads):
                s = sparse[i] if i < len(sparse) else None
                if s is not None:
                    try:
                        payload["sparse_embedding"] = {str(k): float(v) for k, v in dict(s).items()}
                    except Exception:
                        pass
                points.append(PointStruct(id=str(uuid.uuid4()), vector=dense[i].tolist(), payload=payload))

            upsert_batched(client, coll, points)
            registry.update(doc_id, indexed=True, chunk_count=len(points), error=None)
        except Exception as e:
            registry.update(doc_id, indexed=False, error=f"indexing error: {type(e).__name__}: {e}")


def remove_document_vectors(doc_id: str):
    with _JOB_LOCK:
        rec = registry.get(doc_id)
        if not rec:
            return
        try:
            rag, client = _load_rag()
            _delete_by_doc(client, rag.COLLECTION_NAME, doc_id)
            registry.update(doc_id, indexed=False, chunk_count=0)
        except Exception as e:
            registry.update(doc_id, error=f"remove error: {type(e).__name__}: {e}")
