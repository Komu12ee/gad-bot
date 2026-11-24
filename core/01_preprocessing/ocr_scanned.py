"""Produce OCR text for every scraped PDF using the GPU VLM server.

Run the full resumable GPU workflow with:
    bash deploy/gpu-vm.sh ocr

Or run this client against an already running OCR server:
    PIPELINE=fg2 uv run python core/01_preprocessing/ocr_scanned.py \
        --corpus data/scrapped_data/fg2 --engine vlm

Every PDF is OCR'd by default; embedded text is only used with the explicit
--trust-text-layer option. --gate-report previews the text-layer quality gate
without running OCR. --max-pages 0 (the default) processes all pages.

The page-image client is vlm_ocr.py in this folder. The quality gate is
core/02_optimization/text_gate.py. Rejected/suspect pages get a second read;
rejected text is never indexed. Output remains stage2/<urlhash>/structured.md,
with structured tables and per-page confidence sidecars alongside it.

Existing output and OCR_FAILED documents are skipped unless --force is given.
--retry-flagged re-reads bad pages, and --reread-other-model migrates documents
read by another model. Run with the chatbot stack stopped.

The unused Docling/EasyOCR preprocessing implementation has been removed.
"""

from importlib import import_module
import argparse
import json
import os
import shutil
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CORE))
import pipeline_config as pc

ENGINES = ("vlm",)
_LIVE = {"fetched_new", "fetched_changed", "not_modified"}

# ── text-quality gate ────────────────────────────────────────────────────────
# The gates live in text_gate.py: text_layer_quality decides "keep a PDF's embedded
# text vs OCR it", assess_page judges every OCR'd page (VLM engine). Re-exported
# here because doc_pipeline and tests/test_text_gate.py reach them through this
# module.
tg = import_module("02_optimization.text_gate")
text_quality = tg.text_layer_quality
_orphan_matra_ratio = tg._orphan_matra_ratio


def embedded_text(pdf: Path) -> tuple[str, int]:
    """(text, page_count) from the PDF's own text layer. Never trusted blindly."""
    import fitz  # PDF access and page counts; default OCR reads rendered images.

    try:
        fitz.TOOLS.mupdf_display_errors(False)   # damaged govt PDFs spew zlib warnings
    except Exception:
        pass
    doc = fitz.open(str(pdf))
    try:
        return "\n".join(p.get_text("text") for p in doc), doc.page_count
    finally:
        doc.close()


def pdf_page_count(pdf: Path) -> int:
    """Page count only — cheap, so the page-budget gate can decide before OCR."""
    import fitz

    try:
        fitz.TOOLS.mupdf_display_errors(False)
    except Exception:
        pass
    doc = fitz.open(str(pdf))
    try:
        return doc.page_count
    finally:
        doc.close()


# ── deferred-work registry: PDFs skipped for exceeding the page budget ────────
# Written per pipeline (under its data root) so a later, dedicated pass can pick
# exactly these up — see --max-pages 0 in the module docstring. Append-only and
# deduplicated by urlhash so re-runs (including --resume) never double-record.
def skip_registry_path() -> Path:
    return pc.DATA_ROOT / "skipped_large_pdfs.jsonl"


def already_skipped(path: Path) -> set:
    seen = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["urlhash"])
            except Exception:
                continue
    return seen


def record_skip(uhash: str, raw: Path, url: str, pages, reason: str) -> Path:
    path = skip_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "pipeline": pc.PIPELINE,
        "urlhash": uhash,
        "url": url,
        "raw_path": str(raw),
        "pages": pages,
        "size_bytes": raw.stat().st_size if raw.exists() else None,
        "reason": reason,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def _write_extraction_marker(uhash: str, method: str, reason: str) -> None:
    """Note how stage2/<uhash>/structured.md was produced.

    OCR leaves a *_confidence.json sidecar; keeping the embedded text layer left
    nothing behind, so downstream could not tell the two apart and assumed OCR.
    This is the missing half of that signal.
    """
    d = pc.paths.stage2 / uhash
    d.mkdir(parents=True, exist_ok=True)
    (d / "extraction.json").write_text(json.dumps({
        "method": method,        # "text_layer" — the OCR path writes *_confidence.json
        "reason": reason,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")



def scraped_pdfs(corpus: Path):
    """(urlhash, raw_pdf_path) for every PDF in the corpus, newest record per URL.

    Selects on the raw file's extension rather than a doc_kind label: `classify`
    detects PDFs by magic bytes too, so this still works when a server lies about
    its content-type. (The old `doc_kind == "pdf_scanned"` label no longer exists —
    the scraper doesn't read PDFs, so it cannot classify their interiors.)
    """
    manifest = corpus / "manifest.jsonl"
    latest = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("outcome") in _LIVE and str(rec.get("raw_path") or "").lower().endswith(".pdf"):
                latest[rec.get("normalized_url") or rec.get("url")] = rec
    out = []
    for rec in latest.values():
        raw = corpus / str(rec["raw_path"]).replace("\\", "/")
        if raw.exists() and raw.suffix.lower() == ".pdf":
            url = rec.get("normalized_url") or rec.get("url") or ""
            out.append((raw.stem, raw, url))   # stem == urlhash
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True, help="scraper corpus dir (holds manifest.jsonl + raw/)")
    ap.add_argument("--force", action="store_true", help="redo docs that already have stage2 output")
    ap.add_argument("--force-ocr-all", action="store_true",
                    help="(default behaviour; kept for compatibility) OCR every PDF")
    ap.add_argument("--trust-text-layer", action="store_true",
                    help="OPT IN to the old quality gate: keep a PDF's embedded text when it "
                         "reads as clean language instead of OCR'ing it. OFF BY DECISION "
                         "(2026-09-04) — no text layer in this corpus is trusted, because the "
                         "mojibake it produced is unrecoverable once indexed, while OCR'ing a "
                         "clean PDF only costs time.")
    ap.add_argument("--purge-intermediates", action="store_true",
                    help="DELETE the stage1 images and staged PDFs left behind by "
                         "earlier runs for documents that are already OCR'd, then exit. "
                         "Safe: only touches docs whose stage2/<hash>/structured.md exists.")
    ap.add_argument("--gate-report", action="store_true",
                    help="DRY RUN: print the keep-vs-OCR decision for every PDF and exit. "
                         "Use this to see the OCR bill before committing to it.")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after PRODUCING N docs this run (newly OCR'd or kept). Unlike a "
                         "slice, this skips already-done/failed docs first, so --limit 1 called "
                         "repeatedly walks forward one NEW doc at a time — the primitive the "
                         "round-robin driver loops on.")
    ap.add_argument("--only", default=None,
                    help="process ONLY these urlhashes (comma-separated, or @file with one "
                         "per line). For repairing specific documents — e.g. re-OCR'ing the "
                         "ones a bad text layer poisoned — without walking the whole corpus. "
                         "Combine with --force to redo docs that already have output.")
    ap.add_argument("--head", type=int, default=None,
                    help="only consider the first N PDFs in crawl order. The crawler is BFS from "
                         "each site's landing/notice pages (which list newest first), so the first "
                         "N are a proxy for 'the N most recent documents' — use it to OCR just a "
                         "recent slice per site for a demo instead of the whole corpus.")
    ap.add_argument("--max-pages", type=int, default=0,
                    help="skip (and record) any PDF with more than N pages, deferring huge scans "
                         "to a dedicated pass. DEFAULT 0 = NO LIMIT: on a dedicated GPU box you "
                         "want the whole corpus, and a page budget silently withholds documents "
                         "(the old default of 25 deferred ~1,300 of them). Set a value only for "
                         "a deliberate trial run.")
    ap.add_argument("--reread-other-model", action="store_true",
                    help="vlm engine: also read again every document that a DIFFERENT model "
                         "read (per its sidecar; documents from before models were recorded "
                         "count as different) or that failed — after all unread documents. "
                         "Used to move a corpus to a better OCR model; resumable.")
    ap.add_argument("--retry-flagged", action="store_true",
                    help="vlm engine, second pass — run it with the bigger fallback model up: "
                         "re-read the dropped or doubtful pages of documents the quality report "
                         "(<data_root>/ocr_quality.jsonl) lists as partial, or whose sidecar holds "
                         "such a page, and patch the better reads in (other pages untouched); "
                         "failed documents are read whole. Implies --force.")
    ap.add_argument("--engine", choices=ENGINES, default=None,
                    help="GPU OCR engine (vlm). Default: STAGE2_ENGINE, else vlm.")
    args = ap.parse_args()

    engine_name = (args.engine or os.getenv("STAGE2_ENGINE") or "vlm").strip().lower()
    if engine_name not in ENGINES:
        sys.exit(f"unsupported OCR engine {engine_name!r}: Docling OCR was removed; set STAGE2_ENGINE=vlm or --engine vlm")
    use_vlm = engine_name == "vlm"

    corpus = Path(os.path.expanduser(args.corpus)).resolve()
    items = scraped_pdfs(corpus)
    print(f"[{pc.PIPELINE}] scraped PDFs: {len(items)}   (data root {pc.DATA_ROOT})")

    if args.retry_flagged:
        # Second pass of the vlm engine: re-OCR only the documents whose latest
        # record in the quality report is partial (pages dropped) or failed —
        # typically with the bigger fallback model up. Implies --force.
        report = pc.DATA_ROOT / "ocr_quality.jsonl"
        latest = {}
        if report.exists():
            for line in report.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    latest[rec["uhash"]] = rec.get("status")
                except Exception:   # noqa: BLE001 — a torn last line from a killed run
                    continue
        flagged = {h for h, st in latest.items() if st in ("partial", "failed")}
        # Only the bad pages are read again and patched in (vlm_ocr.patch_document):
        # the documents with a bad page are mostly long budget books, and reading
        # them whole would spend pass 2 on pages that were fine. A document with no
        # usable VLM output on disk (failed, or another engine's) is read whole.
        # Documents whose record says "ok" but whose sidecar holds a doubtful page
        # (a page cut short by a loop was "ok" before 2026-10-10) or a page read from
        # a turned image (vlm_ocr.read_turned) are included.
        vlm_ocr = import_module("01_preprocessing.vlm_ocr")
        reread_pages = {}
        whole = pages_n = 0
        for uh, _raw, _url in items:
            pages = vlm_ocr.pages_to_reread(pc.paths.stage2 / uh, uh)
            if uh in flagged and (pages is None or latest.get(uh) == "failed"):
                reread_pages[uh] = None
                whole += 1
            elif pages:
                reread_pages[uh] = pages
                pages_n += len(pages)
        items = [it for it in items if it[0] in reread_pages]
        args.force = True
        print(f"[{pc.PIPELINE}] --retry-flagged: {len(items)} document(s) to read again "
              f"({whole} whole, {pages_n} page(s) in the rest) — partial/failed in {report} "
              f"or a doubtful page in the sidecar")
    else:
        reread_pages = {}

    if args.only:
        raw = args.only
        if raw.startswith("@"):
            raw = Path(raw[1:]).read_text(encoding="utf-8").replace("\n", ",")
        wanted = {h.strip() for h in raw.split(",") if h.strip()}
        items = [it for it in items if it[0] in wanted]
        missing = wanted - {it[0] for it in items}
        print(f"[{pc.PIPELINE}] --only: {len(items)} of {len(wanted)} requested doc(s) found"
              + (f"; not in this corpus: {', '.join(sorted(missing))}" if missing else ""))

    if not items:
        print("nothing to do.")
        return

    # ── dry run: decide for every PDF, spend nothing ─────────────────────────
    if args.purge_intermediates:
        # Reclaim scratch left by earlier runs. Only for documents that are DONE
        # (structured.md present) — anything mid-flight or failed keeps its images.
        freed = 0
        removed = 0
        for d in sorted(pc.paths.stage1.iterdir()) if pc.paths.stage1.is_dir() else []:
            if not d.is_dir():
                continue
            if not (pc.paths.stage2 / d.name / "structured.md").exists():
                continue
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
            staged = pc.paths.input_pdfs / f"{d.name}.pdf"
            if staged.exists():
                size += staged.stat().st_size
                staged.unlink()
            freed += size
            removed += 1
        print(f"purged {removed} finished document(s) from "
              f"{pc.paths.stage1} / {pc.paths.input_pdfs}: {freed / 2**30:.2f} GB freed")
        return

    if args.gate_report:
        keep = ocr = unreadable = toobig = 0
        for uhash, raw, url in items:
            try:
                text, pages = embedded_text(raw)
            except Exception as e:
                unreadable += 1
                print(f"  OCR   {uhash}  (cannot open: {type(e).__name__})")
                continue
            if args.max_pages and args.max_pages > 0 and pages > args.max_pages:
                toobig += 1
                print(f"  SKIP  {uhash}  {pages:>4}p  > {args.max_pages}-page budget (deferred)")
                continue
            good, why = text_quality(text, pages)
            if good:
                keep += 1
                print(f"  keep  {uhash}  {pages:>4}p  {why}")
            else:
                ocr += 1
                print(f"  OCR   {uhash}  {pages:>4}p  {why}")
        print(f"\ngate: keep embedded text {keep}, OCR {ocr + unreadable} "
              f"({unreadable} unopenable), skip-too-large {toobig} (> {args.max_pages}p). "
              f"Re-run without --gate-report to do the work.")
        return

    pc.paths.input_pdfs.mkdir(parents=True, exist_ok=True)
    # --head N: the "N most recent" window (crawl order, newest-first — see the flag
    # help). --limit N: stop once N docs have actually been PRODUCED this run. The two
    # compose: `--head 50 --limit 1` walks the 50 most-recent PDFs and produces the
    # next single undone one — exactly one round-robin step.
    candidates = items[: args.head] if args.head else items
    n = len(candidates)
    kept = ocrd = skip = fail = toobig = 0
    produced = 0                       # docs that gained a structured.md this run
    skipped_hashes = already_skipped(skip_registry_path())

    def _mark_failed(uh: str, why: str):
        """Persist an OCR_FAILED marker so a re-run (round-robin re-attempts every
        undone candidate) doesn't burn OCR time on the same broken PDF each pass."""
        m = pc.paths.stage2 / uh / "OCR_FAILED"
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  {why}\n",
                     encoding="utf-8")

    def _clear_failed(uh: str):
        m = pc.paths.stage2 / uh / "OCR_FAILED"
        if m.exists():
            m.unlink()

    # ── vlm engine: documents run in parallel, so results arrive out of order ──
    runner = None
    engine_errors = 0
    if use_vlm:
        vlm_ocr = import_module("01_preprocessing.vlm_ocr")
        engine = vlm_ocr.VlmEngine.from_env()
        try:
            desc = engine.preflight()
        except vlm_ocr.EngineError as e:
            print(f"[{pc.PIPELINE}] {e}")
            sys.exit(2)
        report = pc.DATA_ROOT / "ocr_quality.jsonl"
        runner = vlm_ocr.DocumentRunner(engine, pc.paths.stage2, report_path=report)
        print(f"[{pc.PIPELINE}] OCR engine: vlm — {desc}; {runner.concurrency} pages in flight; "
              f"quality report -> {report}")

    # Ctrl-C stops NOW. Without this the worker threads first drain every queued page
    # (minutes at 64 in flight) and Ctrl-C looked like it did nothing. Safe: every file
    # is written atomically and structured.md last, so a document in progress simply
    # has no structured.md and the next run reads it again.
    if runner is not None:
        import signal

        def _stop_now(signum, frame):
            print(f"\n[{pc.PIPELINE}] stopped (Ctrl-C): finished documents are saved; the ones in "
                  f"progress are read again on the next run", flush=True)
            if os.name == "posix":    # die BY the signal, so the calling scripts stop too
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGINT)
            os._exit(130)

        signal.signal(signal.SIGINT, _stop_now)

    # --reread-other-model: documents another model read (or that failed) are read
    # again by this one — AFTER every unread document, so new text comes first.
    # Resumable: a document this model has read is skipped like any finished one.
    reread = set()
    if args.reread_other_model and runner is not None:
        mine = engine.primary.root
        fresh, again = [], []
        for it in candidates:
            d = pc.paths.stage2 / it[0]
            if (d / "structured.md").exists() or (d / "OCR_FAILED").exists():
                if vlm_ocr.read_by(d, it[0]) != mine:
                    again.append(it)
                    reread.add(it[0])
                # else: read by this model already — the loop skips it
            else:
                fresh.append(it)
        candidates = fresh + again
        n = len(candidates)
        print(f"[{pc.PIPELINE}] --reread-other-model: {len(again)} document(s) read by another model "
              f"(or failed) will be read again by {mine}, after the {len(fresh)} unread one(s)")

    def _handle(results):
        """Book-keep finished VLM documents (markers, counters, one line each)."""
        nonlocal ocrd, produced, fail, engine_errors
        for r in results:
            line = (f"{r.pages}p, kept {r.kept}, rejected {r.rejected}, blank {r.blank}, "
                    f"2nd-read {r.fallback_used}, tables {r.tables}, {r.seconds:.0f}s")
            if r.produced:
                _clear_failed(r.uhash)
                ocrd += 1
                produced += 1
                print(f"  {r.uhash}: OCR {r.status} — {line}"
                      + (f" — {r.summary}" if r.status == "partial" else ""))
            elif r.status == "engine_error":
                # NOT a property of the document: no OCR_FAILED, a re-run retries it.
                engine_errors += 1
                print(f"  {r.uhash}: ENGINE ERROR (will retry on re-run) — {r.summary[:200]}")
            else:
                fail += 1
                if "(earlier text kept)" not in r.summary:   # a failed re-read keeps its text
                    _mark_failed(r.uhash, f"vlm: {r.summary}")
                print(f"  {r.uhash}: {r.status.upper()} — {r.summary}")

    for i, (uhash, raw, url) in enumerate(candidates, 1):
        if runner is not None:
            if runner.aborted:
                break
            # --limit counts PRODUCED docs, as in earlier runs; documents still in flight
            # might produce, so wait for them before deciding to queue another.
            while (args.limit is not None and runner.inflight_docs
                   and produced + runner.inflight_docs >= args.limit):
                _handle(runner.wait_one())
        if args.limit is not None and produced >= args.limit:
            break
        dest_md = pc.paths.stage2 / uhash / "structured.md"
        if dest_md.exists() and not args.force and uhash not in reread:
            skip += 1
            continue
        # Known-bad from a previous pass — skip fast so the driver moves to the next
        # candidate instead of re-failing this one every cycle. --force retries it.
        if (pc.paths.stage2 / uhash / "OCR_FAILED").exists() and not args.force and uhash not in reread:
            skip += 1
            continue

        # 0. Over the page budget? Defer it — OCR at ~45s/page is what blows up a run.
        #    Recorded once (deduped) so a later `--max-pages 0` pass can drain these.
        if args.max_pages and args.max_pages > 0:
            try:
                npages = pdf_page_count(raw)
            except Exception:
                npages = None
            if npages is not None and npages > args.max_pages:
                if uhash not in skipped_hashes:
                    reg = record_skip(uhash, raw, url, npages, f">{args.max_pages} pages")
                    skipped_hashes.add(uhash)
                    print(f"  [{i}/{n}] {uhash}: SKIP — {npages}p > {args.max_pages} "
                          f"(recorded in {reg})")
                else:
                    print(f"  [{i}/{n}] {uhash}: SKIP — {npages}p > {args.max_pages} "
                          f"(already recorded)")
                toobig += 1
                continue

        # 1. Can we keep the PDF's own text layer?
        # Only when explicitly opted into. By default EVERY PDF is OCR'd: the gate
        # below is a good classifier, but it is still a classifier, and the cost of
        # its false-positives (mojibake silently indexed, unrecoverable without a
        # re-ingest) is not symmetric with the cost of OCR'ing a clean PDF (time).
        if args.trust_text_layer:
            try:
                text, pages = embedded_text(raw)
                good, why = text_quality(text, pages)
            except Exception as e:
                good, why, text = False, f"cannot open ({type(e).__name__})", ""
            if good:
                dest_md.parent.mkdir(parents=True, exist_ok=True)
                dest_md.write_text(unicodedata.normalize("NFC", text).strip() + "\n",
                                   encoding="utf-8")
                # Record that this text came from the PDF, NOT from OCR. Without a
                # marker the two paths are indistinguishable downstream — stage2
                # writes structured.md either way — so scrape_import labelled every
                # scraped PDF "ocr", and the Documents panel showed an OCR badge
                # (and matched the Sarvam filter) on text no OCR engine ever touched.
                _write_extraction_marker(uhash, "text_layer", why)
                _clear_failed(uhash)
                kept += 1
                produced += 1
                print(f"  [{i}/{n}] {uhash}: kept text layer — {why}")
                continue
            print(f"  [{i}/{n}] {uhash}: -> OCR — {why}")

        # 2a. vlm engine: queue the document; whatever finished meanwhile is booked.
        if runner is not None:
            pages = reread_pages.get(uhash)
            _handle(runner.submit(uhash, raw, pages=pages, patch=pages is not None))
            continue

    if runner is not None:
        _handle(runner.finish())
        print(f"[{pc.PIPELINE}] vlm throughput: {runner.throughput()}")

    print(f"\ndone: {ocrd} OCR'd, {kept} kept clean text layer, {skip} already present/failed, "
          f"{toobig} skipped (> {args.max_pages}p), {fail} failed"
          + (f", {engine_errors} engine error(s) (left for re-run)" if use_vlm else "") + ".")
    if toobig:
        print(f"{toobig} large PDF(s) deferred -> {skip_registry_path()}\n"
              f"  drain them later with:  PIPELINE={pc.PIPELINE} uv run python core/01_preprocessing/ocr_scanned.py "
              f"--corpus {corpus} --max-pages 0")
    # Machine-readable control line for the round-robin driver (deploy/ingest-all.sh):
    # produced=0 means this pipeline had nothing left to do within its --head window,
    # so the driver stops cycling it. `window` is the candidate count considered.
    print(f"RR produced={produced} window={n} pipeline={pc.PIPELINE}")
    print("Next: scrape_import to register everything with provenance.")
    if runner is not None and runner.aborted:
        print(f"[{pc.PIPELINE}] {runner.aborted}\n"
              f"  Fix the OCR server, then re-run the same command — finished documents are kept.")
        sys.exit(3)


if __name__ == "__main__":
    main()
