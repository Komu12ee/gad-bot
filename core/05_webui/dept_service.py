"""Department bots — the Python half (phase 3 of the department platform).

The department platform keeps its data in PostgreSQL, and only the Node server
talks to it (core/05_webui/nodejs/platform/). This module does the work that
needs the models already loaded in this Flask process (bge-m3, the reranker, the
answer model client), so no second copy goes on the GPU:

    POST /api/dept/extract   a stored file -> page text -> chunks -> embeddings
    POST /api/dept/encode    a question (+ chat turns) -> search question,
                             variations, dense + sparse vectors
    POST /api/dept/answer    a question + candidate passages -> reranked ->
                             the same answer prompt and Sources as the bots

Reading files: a PDF's own text layer (text_gate decides whether it can be
trusted), Word (.docx), PowerPoint (.pptx), plain text; .doc / .ppt through
LibreOffice when it is installed. Scanned PDFs, images and broken text layers are
NOT read here — they come back as "needs_ocr": OCR runs only when the bots are not
answering (user rule), as its own step.

Internal only: every route needs the X-Dept-Internal key (derived from the session
secret, which only this server and the Node server have), and a file path must
lie inside <RAG_DATA_ROOT>/platform/departments/. The Node server also refuses to
forward /api/dept from browsers.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

from flask import Blueprint, jsonify, request

import pipeline_config as pc

DEPT_ROOT = (pc.DATA_BASE / "platform" / "departments").resolve()
MIN_PAGE_CHARS = 30            # a PDF page with less own text than this is a scan
MAX_BLANK_SHARE = 0.2          # more such pages than this: the document needs OCR
EMBED_BATCH = 8
EMBED_MAX_LENGTH = 1024        # as the bots' indexing (doc_pipeline)
MAX_QUESTION_CHARS = 2000


class NeedsOcr(Exception):
    """The file's text has to be read from images (scans, broken text layer)."""


class Unreadable(Exception):
    """The file cannot be read here (a message for the department admin)."""


def internal_key() -> str:
    secret = os.getenv("GAD_SESSION_SECRET") or os.getenv("SESSION_SECRET") or os.getenv("JWT_SECRET") or ""
    if not secret:
        return ""
    return hmac.new(secret.encode(), b"dept-internal-v1", hashlib.sha256).hexdigest()


def inside_root(path: str) -> Path:
    p = Path(path).resolve()
    if DEPT_ROOT not in p.parents or not p.is_file():
        raise Unreadable("The file is not where department files are kept.")
    return p


# ── reading files into page-marked Markdown ─────────────────────────────────────
# chunk_builder.build_chunks reads "<!-- Page N -->" markers, "#" headings and
# Markdown tables — the same input as the bots' OCR output.

def _gate():
    return import_module("02_optimization.text_gate")


def _md_table(rows: list[list[str]]) -> str:
    rows = [[re.sub(r"\s+", " ", (c or "")).replace("|", "/").strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def read_pdf(path: Path) -> tuple[str, int]:
    import fitz
    try:
        fitz.TOOLS.mupdf_display_errors(False)
    except Exception:
        pass
    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise Unreadable(f"The PDF could not be opened ({e}).")
    try:
        if doc.needs_pass:
            raise Unreadable("The PDF is password-protected.")
        pages = [p.get_text("text") for p in doc]
    finally:
        doc.close()
    if not pages:
        raise Unreadable("The PDF has no pages.")
    gate = _gate()
    keep, reason = gate.text_layer_quality("\n".join(pages), len(pages))
    if not keep:
        raise NeedsOcr(f"scanned or unreadable text layer: {reason}")
    blank = sum(1 for t in pages if len(t.strip()) < MIN_PAGE_CHARS)
    if blank and blank / len(pages) > MAX_BLANK_SHARE:
        raise NeedsOcr(f"{blank} of {len(pages)} pages are scanned images")
    parts = []
    for i, text in enumerate(pages, 1):
        cleaned, _repairs = gate.clean_page(text)
        parts.append(f"<!-- Page {i} -->\n{cleaned.strip()}")
    return "\n\n".join(parts), len(pages)


def read_docx(path: Path) -> tuple[str, int]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    try:
        d = docx.Document(str(path))
    except Exception as e:
        raise Unreadable(f"The Word file could not be opened ({e}).")
    out = []
    for el in d.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(el, d)
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            m = re.match(r"Heading\s*(\d)", style, re.I)
            if m or style.lower() == "title":
                level = int(m.group(1)) if m else 1
                out.append(f"{'#' * min(level + 1, 6)} {text}")
            else:
                out.append(text)
        elif tag == "tbl":
            t = Table(el, d)
            out.append(_md_table([[c.text for c in row.cells] for row in t.rows]))
    return "\n\n".join(x for x in out if x), 0


def read_pptx(path: Path) -> tuple[str, int]:
    from pptx import Presentation
    try:
        deck = Presentation(str(path))
    except Exception as e:
        raise Unreadable(f"The PowerPoint file could not be opened ({e}).")
    parts = []
    for i, slide in enumerate(deck.slides, 1):
        lines = []
        for shape in slide.shapes:
            if getattr(shape, "has_table", False) and shape.has_table:
                lines.append(_md_table([[c.text for c in row.cells] for row in shape.table.rows]))
            elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                text = "\n".join(p.text for p in shape.text_frame.paragraphs if p.text.strip())
                if text.strip():
                    lines.append(text.strip())
        if lines:
            parts.append(f"<!-- Page {i} -->\n" + "\n\n".join(lines))
    return "\n\n".join(parts), len(deck.slides)


def read_txt(path: Path) -> tuple[str, int]:
    try:
        return path.read_text(encoding="utf-8-sig"), 0
    except UnicodeDecodeError:
        raise Unreadable("The text is not UTF-8.")


def read_old_office(path: Path, kind: str) -> tuple[str, int]:
    """.doc / .ppt: converted to .docx / .pptx by LibreOffice, when it is installed."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise Unreadable(f"Old {'Word' if kind == 'doc' else 'PowerPoint'} format (.{kind}): save it as "
                         f".{kind}x and upload that, or install LibreOffice on the server.")
    target = "docx" if kind == "doc" else "pptx"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"in.{kind}"
        shutil.copyfile(path, src)
        try:
            subprocess.run([soffice, "--headless", "--norestore", "--convert-to", target, "--outdir", tmp, str(src)],
                           check=True, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           env={**os.environ, "HOME": tmp})
        except (subprocess.SubprocessError, OSError) as e:
            raise Unreadable(f"LibreOffice could not convert the file ({type(e).__name__}).")
        out = Path(tmp) / f"in.{target}"
        if not out.is_file():
            raise Unreadable("LibreOffice could not convert the file.")
        return (read_docx if target == "docx" else read_pptx)(out)


def read_markdown(path: Path, kind: str) -> tuple[str, int]:
    """(page-marked Markdown, page count or 0). Raises NeedsOcr / Unreadable."""
    if kind == "pdf":
        md, pages = read_pdf(path)
    elif kind == "docx":
        md, pages = read_docx(path)
    elif kind == "pptx":
        md, pages = read_pptx(path)
    elif kind == "txt":
        md, pages = read_txt(path)
    elif kind in ("doc", "ppt"):
        md, pages = read_old_office(path, kind)
    elif kind in ("png", "jpg", "tif"):
        raise NeedsOcr("an image: its text has to be read by OCR")
    else:
        raise Unreadable(f"Files of type .{kind} cannot be read.")
    if kind != "pdf":
        # Word/PowerPoint/text typed in a legacy Hindi font (Krutidev, DevLys) is
        # Latin gibberish as text — the same gate as a PDF's text layer.
        keep, reason = _gate().text_layer_quality(md, 0)
        if not keep:
            raise Unreadable(f"The text cannot be used ({reason}). If it was typed in an old Hindi "
                             "font such as Krutidev, convert it to Unicode first.")
    if not re.sub(r"<!--\s*Page\s+\d+\s*-->|\s", "", md):
        raise Unreadable("The file has no text.")
    return md, pages


# ── chunks and vectors ───────────────────────────────────────────────────────────

def _rag():
    return import_module("rag_pipeline")


def chunk_markdown(md: str, name: str, count) -> list:
    """chunk_builder's chunks (the bots' chunker) — without the metadata doc card."""
    import chunk_builder as cb
    title = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", name).strip() or name
    chunks = cb.build_chunks(md, {"title": title}, count)
    return [c for c in chunks if c.chunk_type != "doc_card" and c.content.strip()]


def embed(texts: list[str]) -> list[tuple[list[float], dict]]:
    rag = _rag()
    rag.ensure_models_loaded()
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        with rag.ENCODE_LOCK:                      # queries from the bots interleave between batches
            enc = rag.model.encode(batch, batch_size=EMBED_BATCH, max_length=EMBED_MAX_LENGTH)
        dense = enc["dense_vecs"]
        sparse = enc.get("lexical_weights") or [None] * len(batch)
        for j in range(len(batch)):
            s = sparse[j] if j < len(sparse) and sparse[j] is not None else {}
            out.append(([round(float(x), 6) for x in dense[j]],
                        {str(k): round(float(v), 4) for k, v in dict(s).items()}))
    return out


def extract(path: str, kind: str, name: str) -> dict:
    """{status: done|needs_ocr|failed, reason, pages, chunks: [...]}"""
    t0 = time.time()
    try:
        p = inside_root(path)
        md, pages = read_markdown(p, kind)
    except NeedsOcr as e:
        return {"status": "needs_ocr", "reason": str(e), "pages": 0, "chunks": []}
    except Unreadable as e:
        return {"status": "failed", "reason": str(e), "pages": 0, "chunks": []}
    rag = _rag()
    rag.ensure_models_loaded()
    tok = rag.model.tokenizer
    count = lambda s: len(tok.encode(s, add_special_tokens=False))     # noqa: E731
    chunks = chunk_markdown(md, name, count)
    if not chunks:
        return {"status": "failed", "reason": "No text could be taken from the file.", "pages": pages, "chunks": []}
    texts = [c.body for c in chunks]
    vectors = embed(texts)
    out = []
    for seq, (c, (dense, sparse)) in enumerate(zip(chunks, vectors), 1):
        pg = sorted(set(x for x in c.pages if x))
        out.append({
            "seq": seq, "page_from": pg[0] if pg else None, "page_to": pg[-1] if pg else None,
            "text": c.file_text(name), "context_line": c.context or None,
            "section": c.section_path[-1] if c.section_path else None,
            "dense": dense, "sparse": sparse,
        })
    print(f"[dept] {name}: {len(out)} chunks from {pages or '?'} page(s) in {time.time() - t0:.1f}s", flush=True)
    return {"status": "done", "reason": None, "pages": pages, "chunks": out}


# ── questions ────────────────────────────────────────────────────────────────────

def encode_question(question: str, history) -> dict:
    rag = _rag()
    turns = rag.clean_history(history)
    search = rag.standalone_question(question, turns) if turns else question
    variations = rag.expand_query(search)
    enc = rag.encode_queries(variations)
    out = [{"text": v, "dense": e[0], "sparse": {str(k): float(w) for k, w in e[1].items()}}
           for v, e in zip(variations, enc) if e is not None]
    return {"search_question": search, "variations": out, "alpha": rag.HYBRID_ALPHA,
            "dense_limit": rag.DENSE_CANDIDATE_LIMIT, "candidates": rag.RERANK_CANDIDATES}


def answer(department: str, question: str, search_question: str, history, candidates: list,
           num_context: int = 8) -> dict:
    """The bots' answer path over the department's passages. `candidates` best first:
    [{id, file_id, file_name, seq, text, context_line, section}]"""
    rag = _rag()
    ap = rag.ap
    llm_client = rag.llm_client
    turns = rag.clean_history(history)
    names = {}
    points = {}
    for c in candidates:
        names[c["file_id"]] = c.get("file_name") or "document"
        points[c["id"]] = SimpleNamespace(id=c["id"], payload={
            "text": c.get("text") or "", "source": c["file_id"], "chunk": c.get("seq"),
            "context_line": c.get("context_line"),
            "section_path": [c["section"]] if c.get("section") else [],
            "verified_correction": c.get("verified_correction"),
        })
    order = [(cid, 1.0) for cid in points]
    if rag.RERANK_ENABLED and order:
        try:
            if rag.load_reranker() is not None:
                document_points = [p for p in points.values() if not p.payload.get('verified_correction')]
                if document_points:
                    order = rag.rerank_candidates(search_question, document_points)
        except Exception as e:          # noqa: BLE001 — never fail an answer over the reranker
            print(f"  ⚠ department rerank failed ({type(e).__name__}: {e}); keeping search order", flush=True)
    # A verified correction is extra evidence. Preserve it if the optional
    # reranker drops it, without dropping the department's document candidates.
    approved_ids = [cid for cid,p in points.items() if p.payload.get('verified_correction')]
    order = [(cid,1.0) for cid in approved_ids] + [(cid,s) for cid,s in order if cid not in approved_ids]
    results, per_file = [], {}
    for cid, score in order:
        p = points[cid]
        src = p.payload["source"]
        if rag.MAX_CHUNKS_PER_SOURCE and per_file.get(src, 0) >= rag.MAX_CHUNKS_PER_SOURCE:
            continue
        per_file[src] = per_file.get(src, 0) + 1
        results.append({"point": p, "score": score, "rank": len(results) + 1})
        if len(results) >= num_context:
            break

    lang, rule, closing, answer_in = rag.language_directive(question)
    rules = ap.RULES.replace("{assistant}", f"{department} Document Assistant")
    earlier = rag.history_block(turns)
    budget = ap.EVIDENCE_TOKENS
    if earlier:
        budget = max(2000, budget - llm_client.estimate_tokens(earlier))
    system, user, sources, docs = ap.build_messages(
        search_question, results, rules=rules, language_rule=rule,
        closing_line=f"{closing} Answer in {answer_in}.", budget=budget,
        count=rag._count_tokens_fn(), clean=rag.clean_chunk_text,
        label=lambda src: (names.get(src, "document"), None),
        terms=rag.query_terms(search_question), extra_before=earlier,
        client=llm_client.client(), max_tokens=ap.ANSWER_TOKENS, thinking=rag.ANSWER_THINKING)
    text, truncated = rag.call_llm(system, user, temperature=rag.ANSWER_TEMPERATURE)
    final, _tail, display = rag.finish_answer(
        text, user=user, question=search_question, sources=rag.Sources(sources, [d.source for d in docs]),
        lang=lang, context_results=results, truncated=truncated)
    cited = []
    for r in display:
        pl = r["point"].payload
        correction = pl.get('verified_correction') or {}
        cited.append({"n": r["rank"], "file_id": pl["source"],
                      "name": correction.get('sourceTitle') or names.get(pl["source"], "document"),
                      "url": correction.get('sourceUrl')})
    return {"answer": llm_client.strip_thinking(final).strip(), "sources": cited, "language": lang,
            "model": getattr(llm_client.client(), "model", None)}


# ── routes ───────────────────────────────────────────────────────────────────────

def _internal_only():
    key = internal_key()
    given = request.headers.get("X-Dept-Internal", "")
    if not key or not hmac.compare_digest(given.encode(), key.encode()):
        return jsonify({"success": False, "error": "Forbidden."}), 403
    return None


def correction_blueprint() -> Blueprint:
    """Every bot publishes corrections; department file routes stay separate."""
    bp = Blueprint("correction_embeddings", __name__)
    bp.before_request(_internal_only)

    @bp.post("/api/dept/correction-embeddings")
    def _correction_embeddings():
        questions = (request.get_json(silent=True) or {}).get('questions')
        if (not isinstance(questions, list) or not 1 <= len(questions) <= 9
                or any(not isinstance(q, str) or not q.strip() or len(q) > 2000 for q in questions)):
            return jsonify({'success': False, 'error': 'Use 1–9 questions, up to 2000 characters each.'}), 400
        try:
            encoded = _rag().encode_queries(questions)
            if any(e is None for e in encoded) or len(encoded) != len(questions):
                raise RuntimeError('Embedding unavailable')
            return jsonify({'success': True, 'model': 'BAAI/bge-m3', 'vectors': [e[0] for e in encoded]})
        except Exception as e:
            print(f'[corrections] indexing failed: {type(e).__name__}', flush=True)
            return jsonify({'success': False, 'error': 'Correction indexing failed; nothing was published.'}), 503

    return bp


def blueprint() -> Blueprint:
    bp = Blueprint("dept", __name__)
    lock = threading.Lock()          # one file at a time; questions are not held up by it
    bp.before_request(_internal_only)

    def body():
        return request.get_json(silent=True) or {}

    @bp.post("/api/dept/extract")
    def _extract():
        b = body()
        if not isinstance(b.get("path"), str) or not isinstance(b.get("kind"), str):
            return jsonify({"success": False, "error": "path and kind are required."}), 400
        with lock:
            try:
                out = extract(b["path"], b["kind"], str(b.get("name") or "document")[:200])
            except Exception as e:      # noqa: BLE001 — the admin sees why
                print(f"[dept] extract failed: {type(e).__name__}: {e}", flush=True)
                return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 500
        return jsonify({"success": True, **out})

    @bp.post("/api/dept/encode")
    def _encode():
        b = body()
        q = str(b.get("question") or "").strip()
        if not q or len(q) > MAX_QUESTION_CHARS:
            return jsonify({"success": False, "error": "Ask a question (2000 characters at most)."}), 400
        try:
            return jsonify({"success": True, **encode_question(q, b.get("history"))})
        except Exception as e:          # noqa: BLE001
            print(f"[dept] encode failed: {type(e).__name__}: {e}", flush=True)
            return jsonify({"success": False, "error": "The search model is not available."}), 503

    @bp.post("/api/dept/answer")
    def _answer():
        b = body()
        q = str(b.get("question") or "").strip()
        cands = b.get("candidates") if isinstance(b.get("candidates"), list) else []
        if not q or not cands:
            return jsonify({"success": False, "error": "question and candidates are required."}), 400
        try:
            out = answer(str(b.get("department") or "Department")[:120], q,
                         str(b.get("search_question") or q), b.get("history"), cands,
                         int(b.get("num_context") or 8))
            return jsonify({"success": True, **out})
        except Exception as e:          # noqa: BLE001 — LLMError and friends: say so plainly
            print(f"[dept] answer failed: {type(e).__name__}: {e}", flush=True)
            msg = None
            try:
                import chat_model
                msg = chat_model.friendly_error(e)     # a model switch in progress, else None
            except Exception:           # noqa: BLE001
                pass
            return jsonify({"success": False, "error": msg or
                            "The answer model is not available right now. Please try again shortly."}), 503

    return bp
