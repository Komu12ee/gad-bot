"""Per-pipeline document registry (data/<pipeline>/documents.json).

Tracks every uploaded document through its lifecycle: uploaded -> processed (OCR
-> .md) -> verified (indexed into the RAG). Plain JSON (no native deps); all
writes are serialized with a lock (the Flask app touches this from request
handlers and background threads).
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # core/ on path
import pipeline_config as pc

_LOCK = threading.Lock()


def _registry_path() -> Path:
    return pc.DATA_ROOT / "documents.json"


def _load() -> list:
    p = _registry_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("documents", [])
    except Exception:
        return []


def _atomic_write(dst: Path, data: str):
    """Write `data` to `dst`, preferring an atomic temp->rename but tolerating
    filesystems that reject it. Some Windows volumes (a D:/secondary drive, a
    VM-shared folder, a network/exFAT mount, or one behind an antivirus filter)
    raise OSError from os.replace — notably WinError 6714, a transactional-file
    error — even though a plain write works. Retry briefly (covers an AV/indexer
    momentarily holding the file), then fall back to writing in place so callers
    keep making progress instead of failing on every document."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp, dst)
            return
        except OSError:
            if attempt < 4:
                time.sleep(0.1 * (attempt + 1))
                continue
            dst.write_text(data, encoding="utf-8")   # non-atomic fallback
            try:
                tmp.unlink()
            except OSError:
                pass


def _save(docs: list):
    _atomic_write(_registry_path(),
                  json.dumps({"documents": docs}, indent=2, ensure_ascii=False))


def all() -> list:
    with _LOCK:
        return _load()


def get(doc_id: str):
    with _LOCK:
        for d in _load():
            if d.get("id") == doc_id:
                return d
    return None


def add(record: dict) -> dict:
    """Register a document. Enforces the INGESTION.md provenance invariant:

    every doc must be traceable to something a human can open — a source URL, an
    uploaded PDF, or a stored raw snapshot. A doc with none of those produces
    answers that cannot be traced back to a source, which is the one thing this
    pipeline must never do.
    """
    check_provenance(record)
    with _LOCK:
        docs = _load()
        docs.append(record)
        _save(docs)
    return record


def check_provenance(record: dict) -> None:
    if not (record.get("source_url") or record.get("pdf_path") or record.get("raw_path")):
        raise ValueError(
            f"refusing to register {record.get('doc_dir') or record.get('id')!r}: no "
            "source_url, pdf_path or raw_path (INGESTION.md provenance invariant)")


# Bulk forms for an import of thousands of documents: add()/update() load and rewrite
# the whole registry each time — quadratic over a corpus (a 5,000-document import
# re-wrote a growing multi-MB file 5,000 times). These do it once per batch.
def add_many(records: list) -> None:
    for r in records:
        check_provenance(r)
    if not records:
        return
    with _LOCK:
        docs = _load()
        docs.extend(records)
        _save(docs)


def update_many(changes: dict) -> int:
    """changes: {doc_id: {field: value}}. Returns how many records were updated."""
    if not changes:
        return 0
    with _LOCK:
        docs = _load()
        n = 0
        for d in docs:
            f = changes.get(d.get("id"))
            if f:
                d.update(f)
                n += 1
        if n:
            _save(docs)
        return n


def update(doc_id: str, **fields) -> dict | None:
    with _LOCK:
        docs = _load()
        updated = None
        for d in docs:
            if d.get("id") == doc_id:
                d.update(fields)
                updated = d
                break
        if updated is not None:
            _save(docs)
        return updated


def get_by_doc_dir(doc_dir: str):
    with _LOCK:
        for d in _load():
            if d.get("doc_dir") == doc_dir:
                return d
    return None


def update_by_doc_dir(doc_dir: str, **fields) -> dict | None:
    """Update the record whose doc_dir matches (the scraper urlhash == OCR stage2 dir).

    Lets core/01_preprocessing/ocr_scanned.py flip a scraped PDF from 'processing' to 'processed' the
    instant its OCR text lands, so the Documents panel updates live instead of only
    after a full re-import. No-op (returns None) if the doc isn't registered yet."""
    with _LOCK:
        docs = _load()
        updated = None
        for d in docs:
            if d.get("doc_dir") == doc_dir:
                d.update(fields)
                updated = d
                break
        if updated is not None:
            _save(docs)
        return updated


def remove(doc_id: str) -> bool:
    with _LOCK:
        docs = _load()
        new = [d for d in docs if d.get("id") != doc_id]
        if len(new) == len(docs):
            return False
        _save(new)
        return True


def unverified() -> list:
    """Documents that finished OCR but are not yet verified."""
    return [d for d in all() if d.get("status") == "processed" and not d.get("verified")]
