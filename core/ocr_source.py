"""Which OCR model read a document — and may its text be indexed for this bot?

GAD and Finance are indexed from the bigger OCR model's text only (user rule
2026-10-11: the 8B's readings must not answer questions). deploy/gpu-vm.sh writes
INDEX_OCR_MODEL_<BOT> into .env for every bot its bigger model re-read; the bulk
indexer (scripts/ingest_corpus.py) and the admin panel's verify path
(doc_pipeline.index_document) both ask here, so neither can index the other model's
text. No dependencies: safe to import before pipeline_config.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def ocr_model_of(doc_dir: Path) -> str | None:
    """The model in this document's VLM sidecar (models.primary.root). None = not
    OCR'd (a web page's own text) — nothing to check. "" = OCR'd, model not recorded
    (sidecars before 2026-10-10: the 8B) or another engine."""
    doc_dir = Path(doc_dir)
    side = doc_dir / f"{doc_dir.name}.pdf_confidence.json"
    if not side.exists():
        return None
    try:
        s = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if s.get("engine") != "vlm":
        return ""
    return str(((s.get("models") or {}).get("primary") or {}).get("root") or "")


def wanted_model(pipeline: str) -> str:
    """The only OCR model whose text is indexed for this bot ("" = any)."""
    return (os.getenv(f"INDEX_OCR_MODEL_{(pipeline or '').upper()}") or "").strip()


def refusal(doc_dir: Path, want: str) -> str | None:
    """Why this document's text must not be indexed, or None when it may be."""
    if not want:
        return None
    by = ocr_model_of(doc_dir)
    if by is None or by == want:
        return None
    return (f"OCR text read by {by or 'an earlier model (not recorded: the 8B)'}, not {want} — "
            f"not indexed; read it again with {want} first")
