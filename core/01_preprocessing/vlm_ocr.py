"""Local GPU OCR: PDF pages -> a vision-language model -> gated text + structured tables.

Replaces the Docling/EasyOCR + Sarvam path for scanned PDFs. No cloud API: every
page is read by a model served on the local GPU through vLLM's OpenAI-compatible
endpoint (deploy/ocr-server.sh starts it).

    page image ──> PRIMARY model (e.g. Qwen3-VL-8B) ──> text_gate ──ok──> keep
                                                          │
                                                   suspect/reject
                                                          ▼
                   FALLBACK model (e.g. Qwen3-VL-32B, higher resolution) ──> text_gate
                                                          │
                                       better of the two attempts; if still
                                       rejected the page is DROPPED (never indexed)
                                       and the reason is written to the report

The fallback is the local replacement for the old Sarvam "second opinion".

TABLES ARE CAPTURED AS STRUCTURE, not flattened text, because the answer quality
of budget questions depends on every number keeping its row label, column header
and unit. The model is asked for HTML tables (merged cells via rowspan/colspan);
each table is parsed into stage2/<doc>/tables.json:

    caption        the title line(s) printed just above the table
    unit           crore | lakh | thousand | rupees — read from the caption or
                   header ("(राशि करोड़ में)", "(Rs. in lakh)"), with the evidence
    header_rows    the header grid, merged cells expanded
    columns        one flattened label per column ("2025-26 / बजट अनुमान")
    rows           the body rows, rowspans repeated so every row stands alone
    continues      for a table that runs over from the previous page, the id of the
                   part it continues; header/caption/unit are inherited from it

structured.md carries the same tables as Markdown, with the inherited header on
continuation pages, so chunking can attach the header + unit to every row chunk.

Outputs per document (stage2/<urlhash>/), structured.md written LAST so an
interrupted run leaves nothing that looks finished:
    tables.json                    structured tables (when the document has any)
    <urlhash>.pdf_confidence.json  per-page engine, verdict, reasons, timings
    structured.md                  only indexable page text, <!-- Page N --> markers

ENGINE FAILURES ARE NOT DOCUMENT FAILURES. If the model server is unreachable, the
document is left untouched (no OCR_FAILED marker) so a re-run picks it up; after a
few consecutive engine failures the run stops instead of burning through the corpus.
"""

from __future__ import annotations

from importlib import import_module
import base64
import concurrent.futures as cf
import html
import html.parser
import json
import os
import re
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
tg = import_module("02_optimization.text_gate")

# ── prompt ───────────────────────────────────────────────────────────────────
OCR_PROMPT = """Transcribe this scanned government document page (Hindi and/or English) exactly.

Rules:
- Output the text in natural reading order as Markdown: '#' for titles and headings, plain paragraphs, one list item per line.
- Copy every word, number, date, order number and amount exactly as printed. Keep Hindi in Devanagari and English in Latin script. Do not translate, correct, summarize or explain anything.
- Tables: output every table as an HTML <table>. Put the header row(s) inside <thead> using <th> cells and the data rows inside <tbody> using <td> cells. Use rowspan and colspan for merged cells. Keep empty cells as empty <td></td>. One table row per printed row.
- Write the table's title and any unit note (for example "(राशि करोड़ में)" or "(Rs. in lakh)") on their own lines immediately before the table.
- Skip stamps, signatures, logos and unreadable handwriting. Do not describe images.
- If the page has no text, output nothing."""

# A page the model loops on at temperature 0 (one in ~17 of the hard Hindi pages in
# scripts/ocr_hindi_eval.py — a wide blank gap on a clean letter was enough) loses
# everything after the loop. Greedy decoding repeats the same loop on every
# re-read; sampling breaks it, and a mild repetition penalty discourages a new one.
# Used only for the re-read of a page that hit the length cap: sampling everywhere
# would add small random errors to every page. (olmOCR re-reads failed pages at
# rising temperatures the same way.) A penalty this mild leaves repeated digits in
# tables alone — every digit is already "seen", so none is favoured.
LOOP_RETRY = {"temperature": 0.5, "top_p": 0.95, "repetition_penalty": 1.05}

FALLBACK_NOTE = ("\n- Read slowly and carefully. Never repeat a line or phrase unless it is "
                 "printed repeatedly on the page.")

# ── configuration ────────────────────────────────────────────────────────────


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip()


@dataclass
class RenderConfig:
    dpi: int
    max_side: int          # cap on the longer image side, in pixels
    jpeg_quality: int = 92


@dataclass
class OcrSettings:
    primary_url: str
    primary_model: str
    fallback_url: str
    fallback_model: str
    fallback_enabled: bool
    primary_render: RenderConfig
    fallback_render: RenderConfig
    concurrency: int
    max_tokens: int
    timeout: float
    api_key: str
    logprobs: bool
    loop_retry: bool = True      # re-read a page that hit the length cap, sampled

    @classmethod
    def from_env(cls) -> "OcrSettings":
        primary_url = _env("OCR_VLM_URL", "http://127.0.0.1:8001/v1").rstrip("/")
        fb = _env("OCR_VLM_FALLBACK", "on").lower()
        return cls(
            primary_url=primary_url,
            primary_model=_env("OCR_VLM_MODEL", ""),
            # Empty fallback URL = a second look by the PRIMARY model at a higher
            # resolution, with a stricter prompt. Set OCR_VLM_FALLBACK_URL to use a
            # bigger model instead (deploy/ocr-server.sh fallback).
            fallback_url=_env("OCR_VLM_FALLBACK_URL", primary_url).rstrip("/"),
            fallback_model=_env("OCR_VLM_FALLBACK_MODEL", ""),
            fallback_enabled=fb not in ("off", "0", "false", "no"),
            primary_render=RenderConfig(int(_env("OCR_VLM_DPI", "200")),
                                        int(_env("OCR_VLM_MAX_SIDE", "2400"))),
            fallback_render=RenderConfig(int(_env("OCR_VLM_FALLBACK_DPI", "300")),
                                         int(_env("OCR_VLM_FALLBACK_MAX_SIDE", "3200"))),
            concurrency=max(1, int(_env("OCR_VLM_CONCURRENCY", "32"))),
            max_tokens=int(_env("OCR_VLM_MAX_TOKENS", "8192")),
            # Per request, and it includes the time a page WAITS in the server's
            # queue (64 in flight, ~15-25 running). At 600 s, once looping pages
            # slowed the queue (2026-10-11), waiting pages hit the limit, were
            # cancelled with their work thrown away and queued again at the back —
            # each retry making the queue slower: 2 documents in 20 minutes. The
            # limit only has to catch a dead server.
            timeout=float(_env("OCR_VLM_TIMEOUT", "3600")),
            api_key=_env("OCR_VLM_API_KEY", ""),
            logprobs=_env("OCR_VLM_LOGPROBS", "1").lower() not in ("0", "off", "false", "no"),
            loop_retry=_env("OCR_LOOP_RETRY", "1").lower() not in ("0", "off", "false", "no"),
        )


# ── errors ───────────────────────────────────────────────────────────────────
class EngineError(RuntimeError):
    """The OCR server is unreachable or broken — says nothing about the page."""


class PageRefused(RuntimeError):
    """The server rejected this particular page (4xx) — a page-level failure."""


# ── rendering ────────────────────────────────────────────────────────────────
# PyMuPDF is not thread-safe, even across separate Document objects, so every
# fitz call goes through one lock. Rendering a page takes ~50-150 ms; the model
# takes far longer per page, so this lock never becomes the bottleneck.
_FITZ_LOCK = threading.Lock()
# Ink = pixels darker than grey 200 on a 72-DPI greyscale render. Measured on real
# corpus pages (scripts/calibrate_text_gate.py era, 278 pages with known OCR text):
# text pages p1 1.0%, median 7%; truly blank pages <= 0.07%. A coarser render or a
# darker cut-off (<128) made light/grey text pages read as ~0% ink — i.e. "blank" —
# which would have let an unread text page be skipped silently.
_DARK = bytes(range(200))
_INK_ZOOM = 1.0


def _quiet_fitz(fitz):
    try:
        fitz.TOOLS.mupdf_display_errors(False)   # damaged govt PDFs spew zlib warnings
    except Exception:
        pass


def page_count(pdf: Path) -> int:
    import fitz
    with _FITZ_LOCK:
        _quiet_fitz(fitz)
        doc = fitz.open(str(pdf))
        try:
            if doc.needs_pass:
                raise ValueError("PDF is password-protected")
            return doc.page_count
        finally:
            doc.close()


# Pages are NEVER turned before OCR. Turning them (2026-10-10) caused more harm than
# good: on the GAD corpus all 12 pages it turned in a check were upright — English
# forms whose underlines, fill-in blanks and table rules look like upside-down
# Hindi headlines, and a narrow-column Hindi table read as sideways — and a turned
# upright page is read upside down. The orientation check now only NOTES its opinion
# (sidecar page "turn_suggested") so the pages can be looked at by a person
# (scripts/turn_review.py). Judged on a ~108-dpi greyscale render, ~30 ms a page.
_ORIENT_ZOOM = 1.5


@dataclass
class Rendered:
    image: bytes          # JPEG, exactly as printed
    ink: float            # dark-pixel share (blank vs unread page)
    turn_suggested: int = 0   # clockwise degrees the orientation check thinks it needs
    turn_reason: str = ""     # (a note for review only — the image is not turned)


def render_page(pdf: Path, index: int, cfg: RenderConfig) -> tuple[bytes, float]:
    """(JPEG bytes, ink ratio) for one page, exactly as printed.

    The ink ratio — share of dark pixels on a small greyscale render — is what lets
    the gate tell a genuinely blank page from one the model failed to read.
    """
    r = render(pdf, index, cfg)
    return r.image, r.ink


def render(pdf: Path, index: int, cfg: RenderConfig, note_orientation: bool = False) -> Rendered:
    """One page as printed; with note_orientation, also the orientation check's
    opinion of it (page_orientation.orientation), which is recorded, not applied."""
    import fitz
    with _FITZ_LOCK:
        _quiet_fitz(fitz)
        doc = fitz.open(str(pdf))
        try:
            page = doc[index]
            r = page.rect
            longest = max(r.width, r.height) or 1.0
            zoom = min(cfg.dpi / 72.0, cfg.max_side / longest)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            small = page.get_pixmap(matrix=fitz.Matrix(_INK_ZOOM, _INK_ZOOM),
                                    colorspace=fitz.csGRAY, alpha=False)
            s = small.samples
            ink = (len(s) - len(s.translate(None, _DARK))) / max(len(s), 1)
            deg, why = 0, ""
            if note_orientation and ink >= tg.INK_CONTENT_RATIO:
                po = import_module("01_preprocessing.page_orientation")
                gray = page.get_pixmap(matrix=fitz.Matrix(_ORIENT_ZOOM, _ORIENT_ZOOM),
                                       colorspace=fitz.csGRAY, alpha=False)
                deg, why = po.orientation(po.gray_array(gray))
            return Rendered(pix.tobytes("jpeg", jpg_quality=cfg.jpeg_quality), ink, deg, why)
        finally:
            doc.close()


# ── the model endpoint ───────────────────────────────────────────────────────
@dataclass
class Transcript:
    text: str
    finish_reason: str | None
    mean_logprob: float | None
    output_tokens: int | None
    seconds: float
    model: str


class VlmEndpoint:
    """One OpenAI-compatible chat endpoint serving a vision-language model."""

    RETRY_WAITS = (2, 5, 15, 30)     # transport errors / 5xx / 429, then EngineError

    def __init__(self, url: str, model: str, *, prompt: str, max_tokens: int,
                 timeout: float, api_key: str, logprobs: bool, label: str):
        import httpx
        self.url, self.model, self.prompt, self.label = url, model, prompt, label
        self.max_tokens, self.logprobs = max_tokens, logprobs
        self.root = ""     # the real model behind the served alias (vLLM's "root")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout, headers=headers,
                                    limits=httpx.Limits(max_connections=256, max_keepalive_connections=64))

    def preflight(self) -> str:
        """Check the server is up and serving the model; resolve the model id.

        deploy/ocr-server.sh serves every model under an alias ("ocr-primary"), so
        the id alone cannot tell the 8B from the 32B; vLLM's "root" names the real
        model and goes into each document's sidecar (which model read it).
        """
        import httpx
        try:
            r = self._client.get(f"{self.url}/models", timeout=20)
            r.raise_for_status()
            cards = r.json().get("data", [])
            ids = [m.get("id") for m in cards]
        except (httpx.HTTPError, ValueError) as e:
            raise EngineError(f"{self.label} OCR server not reachable at {self.url} "
                              f"({type(e).__name__}: {e}). Start it with: bash deploy/ocr-server.sh") from e
        if not ids:
            raise EngineError(f"{self.label} OCR server at {self.url} serves no model")
        if not self.model:
            self.model = ids[0]
        elif self.model not in ids:
            raise EngineError(f"{self.label} OCR server at {self.url} does not serve "
                              f"{self.model!r} (it serves: {', '.join(ids)})")
        card = next((m for m in cards if m.get("id") == self.model), {})
        self.root = str(card.get("root") or self.model)
        return f"{self.label}: {self.model}{f' ({self.root})' if self.root != self.model else ''} @ {self.url}"

    def transcribe(self, image: bytes, extra: dict | None = None) -> Transcript:
        import httpx
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}},
                {"type": "text", "text": self.prompt},
            ]}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if self.logprobs:
            body["logprobs"] = True
        if extra:
            body.update(extra)
        last = None
        for attempt, wait in enumerate((0,) + self.RETRY_WAITS):
            if wait:
                # Said out loud: retries used to be silent, so a timeout storm on the
                # server looked like plain slowness in the log.
                print(f"  ⚠ {self.label} OCR request failed ({last}) — retry {attempt} of "
                      f"{len(self.RETRY_WAITS)} in {wait}s", flush=True)
                time.sleep(wait)
            t0 = time.monotonic()
            try:
                r = self._client.post(f"{self.url}/chat/completions", json=body)
            except httpx.TransportError as e:
                last = f"{type(e).__name__}: {e}"
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                continue
            if r.status_code >= 400:
                raise PageRefused(f"HTTP {r.status_code}: {r.text[:300]}")
            try:
                data = r.json()
                choice = data["choices"][0]
            except (ValueError, KeyError, IndexError) as e:
                last = f"unparseable response ({type(e).__name__})"
                continue
            content = (choice.get("message") or {}).get("content") or ""
            return Transcript(
                text=content,
                finish_reason=choice.get("finish_reason"),
                mean_logprob=_mean_logprob(choice.get("logprobs")),
                output_tokens=(data.get("usage") or {}).get("completion_tokens"),
                seconds=time.monotonic() - t0,
                model=data.get("model") or self.model,
            )
        raise EngineError(f"{self.label} OCR server failed {len(self.RETRY_WAITS) + 1} times: {last}")


def _mean_logprob(lp) -> float | None:
    try:
        vals = [t["logprob"] for t in (lp or {}).get("content") or []
                if t.get("logprob") is not None and t["logprob"] > -1e9]
    except (AttributeError, TypeError, KeyError):
        return None
    return sum(vals) / len(vals) if vals else None


# ── model output -> page text ────────────────────────────────────────────────
_FENCE = re.compile(r"(?m)^[ \t]*```[a-zA-Z]*[ \t]*$\n?")   # [ \t], not \s: see text_gate._MD_STRIP
_WRAPPER = re.compile(r"</?(html|body|head)\b[^<>]*>|<!DOCTYPE[^<>]*>|<meta\b[^<>]*>", re.I)
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize_output(text: str) -> str:
    """Strip wrappers the model adds around the page; NFC; ASCII digits.

    Devanagari digits become ASCII, matching the rest of the corpus — a question
    typed with "2025" must match a page that printed "२०२५".
    """
    t = text or ""
    t = _FENCE.sub("", t)
    t = _WRAPPER.sub("", t)
    t = unicodedata.normalize("NFC", t).translate(_DEV_DIGITS)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


# ── tables ───────────────────────────────────────────────────────────────────
_HTML_TABLE = re.compile(r"<table\b.*?(?:</table\s*>|\Z)", re.I | re.S)
_STRAY_TABLE_TAG = re.compile(r"</?(table|thead|tbody|tfoot|tr|td|th|caption|colgroup|col)\b[^<>]*>", re.I)
_YEARISH = re.compile(r"^\(?\s*(19|20)\d{2}(\s*[-–/]\s*(\d{2}|(19|20)\d{2}))?\s*\)?$")
_NUMBERISH = re.compile(r"^[()\-–+]*\s*(₹|rs\.?|रु\.?)?\s*[\d,.]+\s*%?\s*\)?$", re.I)

_UNIT_WORDS = [
    ("crore", re.compile(r"करोड़|करोड|\bcrores?\b|\bcr\.", re.I)),
    ("lakh", re.compile(r"लाख|\blakhs?\b|\blacs?\b", re.I)),
    ("thousand", re.compile(r"हजार|हज़ार|\bthousands?\b|'000|000'?s\b", re.I)),
]
_UNIT_CONTEXT = re.compile(r"में|\bin\b|\brs\b|₹|रु|रुपय|रूपय|राशि|आँकड़|आंकड़|amount|figures|\(", re.I)
_RUPEES_ONLY = re.compile(r"(रुपयों|रूपयों|रुपये|रूपये)\s*में|\(\s*(rs\.?|₹|rupees)\s*\)|in\s+rupees", re.I)


class _TableParser(html.parser.HTMLParser):
    """Lenient HTML table reader: rows of (text, is_header, rowspan, colspan)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[dict] = []
        self._row = None
        self._cell = None
        self._section = None
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self._depth += 1
        elif self._depth > 1:
            return                                   # nested table: read as cell text
        elif tag in ("thead", "tbody", "tfoot"):
            self._section = tag
        elif tag == "tr":
            self._close_row()
            self._row = {"cells": [], "thead": self._section == "thead"}
        elif tag in ("td", "th"):
            if self._row is None:
                self._row = {"cells": [], "thead": self._section == "thead"}
            self._close_cell()
            self._cell = {"text": [], "th": tag == "th",
                          "rowspan": _span(a.get("rowspan")), "colspan": _span(a.get("colspan"))}
        elif tag == "br" and self._cell is not None:
            self._cell["text"].append(" ")

    def handle_endtag(self, tag):
        if tag == "table":
            if self._depth == 1:
                self._close_row()
            self._depth -= 1
        elif self._depth > 1:
            return
        elif tag in ("td", "th"):
            self._close_cell()
        elif tag == "tr":
            self._close_row()
        elif tag in ("thead", "tbody", "tfoot"):
            self._section = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell["text"].append(data)

    def _close_cell(self):
        if self._cell is not None and self._row is not None:
            self._row["cells"].append(self._cell)
        self._cell = None

    def _close_row(self):
        self._close_cell()
        if self._row is not None and self._row["cells"]:
            self.rows.append(self._row)
        self._row = None

    def close(self):
        super().close()
        self._close_row()


def _span(v) -> int:
    try:
        return max(1, min(int(str(v).strip()), 50))
    except (TypeError, ValueError):
        return 1


def _cell_text(parts, strip_legacy: bool = False) -> str:
    t = re.sub(r"\s+", " ", "".join(parts)).strip()
    return tg.clean_cell(t, strip_legacy)


def _expand(rows: list[dict], strip_legacy: bool = False) -> tuple[list[list[str]], list[bool]]:
    """Grid with merged cells expanded (their text repeated), + per-row header flag."""
    occupied: dict[tuple[int, int], str] = {}
    header_flags: list[bool] = []
    for r, row in enumerate(rows):
        c = 0
        for cell in row["cells"]:
            while (r, c) in occupied:
                c += 1
            text = _cell_text(cell["text"], strip_legacy)
            for dr in range(cell["rowspan"]):
                for dc in range(cell["colspan"]):
                    occupied.setdefault((r + dr, c + dc), text)
            c += cell["colspan"]
        header_flags.append(row["thead"] or all(cell["th"] for cell in row["cells"]))
    if not occupied:
        return [], []
    nrows = len(rows)
    ncols = max(c for (_, c) in occupied) + 1
    grid = [[occupied.get((r, c), "") for c in range(ncols)] for r in range(nrows)]
    return grid, header_flags


def _is_value(cell: str) -> bool:
    c = cell.strip()     # a value is short; the length cap also keeps the \s* runs cheap
    return 0 < len(c) <= 40 and bool(_NUMBERISH.match(c)) and not _YEARISH.match(c)


def _header_count(grid: list[list[str]], flags: list[bool]) -> int:
    """How many leading rows are header rows."""
    n = 0
    while n < len(flags) and flags[n]:
        n += 1
    if n:
        return n
    # No explicit header: leading rows (max 3) without data values, followed by a
    # row that is mostly values, are the header.
    for k in range(min(3, len(grid) - 1)):
        row = grid[k]
        if any(_is_value(c) for c in row):
            break
        nxt = [c for c in grid[k + 1] if c.strip()]
        if nxt and sum(_is_value(c) for c in nxt) / len(nxt) >= 0.5:
            return k + 1
    return 0


def _flatten_columns(header_rows: list[list[str]], ncols: int) -> list[str]:
    cols = []
    for c in range(ncols):
        parts: list[str] = []
        for row in header_rows:
            v = row[c].strip() if c < len(row) else ""
            if v and v not in parts:
                parts.append(v)
        cols.append(" / ".join(parts))
    return cols


def find_unit(*texts: str) -> tuple[str | None, str | None, str | None]:
    """(unit, evidence, confidence) from caption/header text.

    unit: crore | lakh | thousand | rupees. confidence "high" when the unit sits in
    its own parenthetical — "(राशि करोड़ में)", "(Rs. in lakh)" — the way tables
    print it; "low" when it was only found somewhere in a caption line, where a
    title like "50 हजार आवासों के निर्माण में प्रगति" can fool it.
    """
    for text in texts:
        for line in (text or "").splitlines():
            for chunk in re.findall(r"\([^()]{0,80}\)", line):
                for unit, pat in _UNIT_WORDS:
                    if pat.search(chunk):
                        return unit, chunk.strip(), "high"
                if _RUPEES_ONLY.search(chunk):
                    return "rupees", chunk.strip(), "high"
    for text in texts:
        for line in (text or "").splitlines():
            if len(line) <= 160 and _UNIT_CONTEXT.search(line):
                for unit, pat in _UNIT_WORDS:
                    if pat.search(line):
                        return unit, line.strip(), "low"
            m = _RUPEES_ONLY.search(line)
            if m:
                return "rupees", m.group(0).strip(), "low"
    return None, None, None


def _md_cell(v: str) -> str:
    return re.sub(r"\s+", " ", (v or "").replace("|", "/")).strip()


def render_markdown_table(columns: list[str] | None, rows: list[list[str]]) -> str:
    """Markdown table. `columns` None -> the first row is printed as the header line."""
    if not rows and not columns:
        return ""
    ncols = max([len(columns or [])] + [len(r) for r in rows])
    head = list(columns) if columns else list(rows[0])
    body = rows if columns else rows[1:]
    head += [""] * (ncols - len(head))
    lines = ["| " + " | ".join(_md_cell(h) for h in head) + " |",
             "|" + "|".join(["---"] * ncols) + "|"]
    for r in body:
        r = list(r) + [""] * (ncols - len(r))
        lines.append("| " + " | ".join(_md_cell(v) for v in r) + " |")
    return "\n".join(lines)


def _md_table_rows(block: list[str], strip_legacy: bool = False) -> tuple[list[list[str]], int]:
    """Rows of a Markdown pipe table + number of header rows (rows above |---|)."""
    rows, n_head = [], 0
    for ln in block:
        if tg.is_table_rule(ln):
            n_head = len(rows)
            continue
        cells = ln.strip().strip("|").split("|")
        rows.append([tg.clean_cell(c.strip(), strip_legacy) for c in cells])
    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows], n_head


def _caption_before(prefix: str) -> str:
    """The title line(s) printed just above a table (at most 2, at most 300 chars)."""
    out: list[str] = []
    blanks = 0
    for ln in reversed([x.strip() for x in prefix.rstrip().splitlines()]):
        if not ln:
            blanks += 1
            if blanks > 1:            # two blank lines: whatever is above is not the caption
                break
            continue
        blanks = 0
        if ln.startswith("|") or ln.lower().startswith("<t") or ln.startswith("<!--"):
            break
        out.insert(0, ln.lstrip("#*_ ").rstrip("*_ "))
        if len(out) == 2 or sum(len(x) for x in out) > 300:
            break
    return " ".join(out)[:300]


def extract_tables(text: str, page_no: int, strip_legacy: bool = False) -> tuple[str, list[dict]]:
    """Convert every table on the page to Markdown and describe it structurally.

    Returns (page text with Markdown tables, [table dicts]). Rows whose cells are
    legacy-font junk are dropped here, before rendering, so the Markdown and
    tables.json agree and the page gate has nothing left to cut inside tables.
    """
    tables: list[dict] = []

    def add_table(prefix_text: str, grid: list[list[str]], n_head: int, fmt: str) -> str:
        grid = [r for r in grid if any(c.strip() for c in r)]
        if not grid:
            return ""
        header_rows, body = grid[:n_head], grid[n_head:]
        dropped = [r for r in body if tg.is_mojibake_line(" ".join(r))]
        body = [r for r in body if not tg.is_mojibake_line(" ".join(r))]
        ncols = max(len(r) for r in grid)
        columns = _flatten_columns(header_rows, ncols) if header_rows else None
        caption = _caption_before(prefix_text)
        unit, evidence, unit_conf = find_unit(caption, "\n".join(" ".join(r) for r in header_rows))
        t = {
            "id": f"p{page_no}-t{len(tables) + 1}",
            "page": page_no,
            "index": len(tables) + 1,
            "caption": caption,
            "unit": unit,
            "unit_evidence": evidence,
            "unit_confidence": unit_conf,
            "n_cols": ncols,
            "header_rows": header_rows,
            "columns": columns,
            "rows": body,
            "rows_dropped_as_junk": len(dropped),
            "source_format": fmt,
            "continues": None,
            "inherited": [],
        }
        md = render_markdown_table(columns, body)
        t["markdown"] = md
        tables.append(t)
        return md

    # 1. HTML tables
    out, pos = [], 0
    for m in _HTML_TABLE.finditer(text):
        out.append(text[pos:m.start()])
        p = _TableParser()
        try:
            p.feed(m.group(0))
            p.close()
        except Exception:      # noqa: BLE001 — malformed markup: keep it as text
            out.append(re.sub(r"<[^<>]+>", " ", m.group(0)))   # [^<>]: linear on "< < <" loops
            pos = m.end()
            continue
        grid, flags = _expand(p.rows, strip_legacy)
        md = add_table("".join(out), grid, _header_count(grid, flags), "html") if grid else ""
        out.append("\n\n" + md + "\n\n" if md else "\n")
        pos = m.end()
    out.append(text[pos:])
    # Leftover table tags (a table nested inside a cell ends the outer match early)
    # are layout debris, not text.
    text = _STRAY_TABLE_TAG.sub(" ", "".join(out))

    # 2. Markdown pipe tables the model wrote despite the prompt (skip ours)
    ours = {t["markdown"] for t in tables}
    lines = text.split("\n")
    out_lines, i = [], 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            block = lines[i:j]
            if "\n".join(block) not in ours and len(block) >= 2:
                rows, n_head = _md_table_rows(block, strip_legacy)
                if not n_head:
                    n_head = _header_count(rows, [False] * len(rows))
                md = add_table("\n".join(out_lines), rows, n_head, "markdown")
                out_lines.extend(md.split("\n") if md else [])
            else:
                out_lines.extend(block)
            i = j
        else:
            out_lines.append(lines[i])
            i += 1
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()

    # Positions — for linking tables that run across pages — and ids in PAGE order
    # (HTML tables were converted before Markdown ones, so list order is not page
    # order; link_continuations relies on first/last table of a page).
    for t in tables:
        k = text.find(t["markdown"]) if t["markdown"] else -1
        t["_pos"] = k if k >= 0 else len(text)
        if k < 0:
            t["chars_before"] = t["chars_after"] = None
            continue
        before = tg.normalize_for_scoring(text[:k])
        after = tg.normalize_for_scoring(text[k + len(t["markdown"]):])
        t["chars_before"] = sum(c.isalnum() for c in before)
        t["chars_after"] = sum(c.isalnum() for c in after)
    tables.sort(key=lambda t: t.pop("_pos"))
    for n, t in enumerate(tables, 1):
        t["id"], t["index"] = f"p{page_no}-t{n}", n
    return text, tables


EDGE_CHARS = 200    # a table "starts/ends the page" with at most this much text around it


def link_continuations(pages: list[dict]) -> int:
    """Mark tables that continue from the previous page; inherit header/caption/unit.

    pages: [{"page": n, "text": str, "tables": [table dicts]}] in page order, only
    indexable pages. A table continues the previous page's last table when it is
    the first thing on its page, the previous one was the last thing on its page,
    the column counts match, and it has no header of its own (or repeats the same
    one). Rewrites the continuation's Markdown in the page text so the header is
    there for chunking. Returns the number of links made.
    """
    links = 0
    prev_page = None
    for pg in pages:
        tabs = pg["tables"]
        if prev_page is not None and tabs and prev_page["tables"] and pg["page"] == prev_page["page"] + 1:
            cur, prev = tabs[0], prev_page["tables"][-1]
            starts = cur.get("chars_before") is not None and cur["chars_before"] <= EDGE_CHARS
            ended = prev.get("chars_after") is not None and prev["chars_after"] <= EDGE_CHARS
            same_cols = cur["n_cols"] == prev["n_cols"]
            no_own_header = not cur["header_rows"] or cur["columns"] == prev["columns"]
            if starts and ended and same_cols and no_own_header and prev.get("columns"):
                root = prev.get("table_group") or prev["id"]
                cur["continues"] = prev["id"]
                cur["table_group"] = root
                prev.setdefault("table_group", root)
                inherited = []
                if not cur["header_rows"]:
                    cur["header_rows"] = prev["header_rows"]
                    cur["columns"] = prev["columns"]
                    inherited.append("header")
                    # A header-less table printed its first row as the Markdown
                    # header line; re-render with the inherited header and every
                    # row as body (rows already holds all of them).
                    new_md = render_markdown_table(cur["columns"], cur["rows"])
                    if cur["markdown"] and cur["markdown"] in pg["text"]:
                        pg["text"] = pg["text"].replace(cur["markdown"], new_md, 1)
                        cur["markdown"] = new_md
                if not cur["caption"] and prev.get("caption"):
                    cur["caption"] = prev["caption"]
                    inherited.append("caption")
                if not cur["unit"] and prev.get("unit"):
                    cur["unit"], cur["unit_evidence"] = prev["unit"], prev.get("unit_evidence")
                    cur["unit_confidence"] = prev.get("unit_confidence")
                    inherited.append("unit")
                cur["inherited"] = inherited
                links += 1
        prev_page = pg
    return links


# ── one page: primary, then fallback when the gate is not happy ──────────────
@dataclass
class Attempt:
    engine: str                      # "vlm" | "vlm-fallback"
    model: str
    verdict: tg.PageVerdict
    tables: list[dict]
    seconds: float
    finish_reason: str | None = None
    mean_logprob: float | None = None
    output_tokens: int | None = None
    error: str | None = None
    loop_retry: bool = False         # this is the sampled re-read of a looped attempt

    def summary(self) -> dict:
        d = {"engine": self.engine, "model": self.model, "status": self.verdict.status,
             "score": round(self.verdict.score, 3), "reasons": self.verdict.reasons,
             "seconds": round(self.seconds, 2), "finish_reason": self.finish_reason,
             "mean_logprob": None if self.mean_logprob is None else round(self.mean_logprob, 3),
             "output_tokens": self.output_tokens, "error": self.error}
        if self.loop_retry:
            d["loop_retry"] = True
        return d


_RANK = {"ok": 3, "suspect": 2, "blank": 1, "reject": 0}


def _better(a: Attempt, b: Attempt) -> Attempt:
    ka = (_RANK[a.verdict.status], a.verdict.score, a.verdict.stats.get("content_chars", 0))
    kb = (_RANK[b.verdict.status], b.verdict.score, b.verdict.stats.get("content_chars", 0))
    return b if kb > ka else a


@dataclass
class PageOutcome:
    page: int                        # 1-based
    final: Attempt
    attempts: list[Attempt]
    ink_ratio: float
    turn_suggested: int = 0          # the orientation check's opinion, for review only
    turn_reason: str = ""


class VlmEngine:
    def __init__(self, settings: OcrSettings):
        self.s = settings
        common = dict(max_tokens=settings.max_tokens, timeout=settings.timeout,
                      api_key=settings.api_key, logprobs=settings.logprobs)
        self.primary = VlmEndpoint(settings.primary_url, settings.primary_model,
                                   prompt=OCR_PROMPT, label="primary", **common)
        self.fallback = (VlmEndpoint(settings.fallback_url, settings.fallback_model,
                                     prompt=OCR_PROMPT + FALLBACK_NOTE, label="fallback", **common)
                         if settings.fallback_enabled else None)

    @classmethod
    def from_env(cls) -> "VlmEngine":
        return cls(OcrSettings.from_env())

    def preflight(self) -> str:
        parts = [self.primary.preflight()]
        if self.fallback is not None:
            if not self.fallback.model and self.fallback.url == self.primary.url:
                self.fallback.model = self.primary.model   # same server: second look
            parts.append(self.fallback.preflight())
        return "; ".join(parts)

    def describe(self) -> dict:
        d = {"primary": {"url": self.primary.url, "model": self.primary.model, "root": self.primary.root,
                         "dpi": self.s.primary_render.dpi, "max_side": self.s.primary_render.max_side}}
        if self.fallback is not None:
            d["fallback"] = {"url": self.fallback.url, "model": self.fallback.model, "root": self.fallback.root,
                             "dpi": self.s.fallback_render.dpi, "max_side": self.s.fallback_render.max_side}
        return d

    def _read(self, ep: VlmEndpoint, rd: Rendered, index: int, engine: str,
              extra: dict | None = None) -> Attempt:
        t0 = time.monotonic()
        try:
            tr = ep.transcribe(rd.image, extra)
        except PageRefused as e:
            v = tg.PageVerdict("reject", 0.0, [f"engine refused the page: {e}"], {}, "")
            return Attempt(engine, ep.model, v, [], time.monotonic() - t0, error=str(e),
                           loop_retry=bool(extra))
        raw = normalize_output(tr.text)
        # One legacy-font decision per page, shared by the table cleaner and the page
        # gate, so tables.json and structured.md are cleaned identically.
        legacy = tg.legacy_font_signature(raw)
        text, tables = extract_tables(raw, index + 1, strip_legacy=legacy)
        v = tg.assess_page(text, ink_ratio=rd.ink, finish_reason=tr.finish_reason,
                           mean_logprob=tr.mean_logprob, strip_legacy=legacy)
        return Attempt(engine, tr.model, v, tables, tr.seconds, tr.finish_reason,
                       tr.mean_logprob, tr.output_tokens, loop_retry=bool(extra))

    def _attempt(self, ep: VlmEndpoint, cfg: RenderConfig, pdf: Path, index: int,
                 engine: str, note_orientation: bool = False) -> tuple[Attempt, Rendered]:
        """Read one page; if the model looped into its length cap, read it once more
        with sampling (LOOP_RETRY) and keep the better of the two."""
        rd = render(pdf, index, cfg, note_orientation)
        first = self._read(ep, rd, index, engine)
        if not (self.s.loop_retry and first.finish_reason == "length"):
            return first, rd
        second = self._read(ep, rd, index, engine, LOOP_RETRY)
        return _better(first, second), rd

    def ocr_page(self, pdf: Path, index: int) -> PageOutcome:
        """OCR one page. Raises EngineError only when the server itself fails."""
        first, rd = self._attempt(self.primary, self.s.primary_render, pdf, index, "vlm",
                                  note_orientation=True)
        attempts = [first]
        final = first
        if self.fallback is not None and first.verdict.status in ("suspect", "reject"):
            second, _ = self._attempt(self.fallback, self.s.fallback_render, pdf, index, "vlm-fallback")
            attempts.append(second)
            final = _better(first, second)
        return PageOutcome(index + 1, final, attempts, rd.ink, rd.turn_suggested, rd.turn_reason)


# ── a document: assemble pages, write outputs ────────────────────────────────
@dataclass
class DocResult:
    uhash: str
    status: str            # ok | partial | failed | blank | engine_error
    summary: str
    pages: int = 0
    kept: int = 0
    rejected: int = 0
    blank: int = 0
    fallback_used: int = 0
    tables: int = 0
    seconds: float = 0.0

    @property
    def produced(self) -> bool:
        return self.status in ("ok", "partial")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


_STALE = ("structured.json", "extraction.json")


def write_document(out_dir: Path, uhash: str, outcomes: list[PageOutcome], engine: VlmEngine,
                   seconds: float) -> DocResult:
    """Gate the document, then write tables.json, the sidecar and structured.md."""
    outcomes = sorted(outcomes, key=lambda o: o.page)
    verdicts = [o.final.verdict for o in outcomes]
    status, summary = tg.assess_document(verdicts)
    md = out_dir / "structured.md"
    if md.exists():
        # Reading again a document that already has text (--reread-other-model, --force).
        if status not in ("ok", "partial"):
            # Nothing usable this time: keep what is there rather than lose it.
            return DocResult(uhash=uhash, status=status, summary=summary + " (earlier text kept)",
                             pages=len(outcomes), seconds=seconds)
        # structured.md is written LAST again: an interrupted rewrite then leaves the
        # document without one, so the next run reads it again — never old text
        # next to a sidecar that names the new model.
        md.unlink()

    pages = [{"page": o.page, "text": o.final.verdict.text, "tables": o.final.tables}
             for o in outcomes if o.final.verdict.indexable]
    links = link_continuations(pages)
    text_by_page = {p["page"]: p["text"] for p in pages}
    tables = [dict(t) for p in pages for t in p["tables"]]
    for t in tables:
        t.pop("markdown", None)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in _STALE:                       # outputs of a previous engine, now wrong
        (out_dir / name).unlink(missing_ok=True)
    for f in out_dir.glob("page_*_sarvam.*"):
        f.unlink(missing_ok=True)

    sidecar = {
        "engine": "vlm",
        "models": engine.describe(),
        "document": {"status": status, "summary": summary, "table_links": links,
                     "seconds": round(seconds, 1),
                     "at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "pages": [_page_entry(o) for o in outcomes],
    }
    if tables:
        _atomic_write(out_dir / "tables.json", json.dumps({"tables": tables}, ensure_ascii=False, indent=1))
    else:
        (out_dir / "tables.json").unlink(missing_ok=True)
    _atomic_write(out_dir / f"{uhash}.pdf_confidence.json", json.dumps(sidecar, ensure_ascii=False, indent=1))

    if status in ("ok", "partial"):
        parts = []
        for o in outcomes:
            parts.append(f"<!-- Page {o.page} -->")
            body = text_by_page.get(o.page, "").strip()
            if body:
                parts.append(body)
            parts.append("")
        _atomic_write(out_dir / "structured.md", "\n\n".join(parts).strip() + "\n")

    return DocResult(
        uhash=uhash, status=status, summary=summary, pages=len(outcomes),
        kept=sum(1 for v in verdicts if v.indexable and v.status != "blank"),
        rejected=sum(1 for v in verdicts if v.status == "reject"),
        blank=sum(1 for v in verdicts if v.status == "blank"),
        fallback_used=sum(1 for o in outcomes if o.final.engine == "vlm-fallback"),
        tables=len(tables), seconds=seconds)


# ── pass 2: re-read only the bad pages of a document ─────────────────────────
_PAGE_MARKER = re.compile(r"<!--\s*Page\s+(\d+)\s*-->")
REREAD_STATUSES = ("reject", "suspect")


def read_turned(page: dict) -> bool:
    """A sidecar page whose text came from an image turned before OCR (code of
    2026-10-10, since removed — it turned upright pages too). Read again as printed."""
    return any(a.get("rotation") for a in page.get("attempts") or [])


def pages_to_reread(out_dir: Path, uhash: str) -> list[int] | None:
    """Pages of an OCR'd document worth a second pass: dropped (reject), kept with
    a doubt (suspect — e.g. cut short by a loop), or read from a turned image
    (read_turned). None = no usable VLM output on disk, so the whole document must
    be read again; [] = nothing to redo.
    """
    side_p = out_dir / f"{uhash}.pdf_confidence.json"
    if not (out_dir / "structured.md").exists() or not side_p.exists():
        return None
    try:
        side = json.loads(side_p.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if side.get("engine") != "vlm":
        return None
    return sorted(p["page_number"] for p in side.get("pages", [])
                  if p.get("status") in REREAD_STATUSES or read_turned(p))


def read_by(out_dir: Path, uhash: str) -> str:
    """The real model that read this document (sidecar models.primary.root); "" when
    unknown — no VLM sidecar, or written before the model was recorded (2026-10-10:
    everything up to then was Qwen3-VL-8B)."""
    try:
        side = json.loads((out_dir / f"{uhash}.pdf_confidence.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if side.get("engine") != "vlm":
        return ""
    return str(((side.get("models") or {}).get("primary") or {}).get("root") or "")


def _page_entry(o: PageOutcome) -> dict:
    d = {"page_number": o.page, "ocr_score": round(o.final.verdict.score, 3),
         "source": o.final.engine, "status": o.final.verdict.status,
         "reasons": o.final.verdict.reasons, "ink_ratio": round(o.ink_ratio, 4),
         "stats": o.final.verdict.stats, "attempts": [a.summary() for a in o.attempts]}
    if o.turn_suggested:
        d["turn_suggested"] = {"degrees": o.turn_suggested, "reason": o.turn_reason}
    return d


def patch_document(out_dir: Path, uhash: str, outcomes: list[PageOutcome], engine: VlmEngine,
                   seconds: float) -> DocResult:
    """Put re-read pages into a document's existing outputs.

    A page is replaced only when the new read beats the one on disk (gate status,
    then score); every other page — its text in structured.md, its tables — stays
    exactly as pass 1 wrote it. Re-reading whole documents instead would spend most
    of pass 2 on pages that were fine: the documents with a bad page are mostly the
    long budget books.
    """
    side_p = out_dir / f"{uhash}.pdf_confidence.json"
    side = json.loads(side_p.read_text(encoding="utf-8"))
    pages = {p["page_number"]: p for p in side.get("pages", [])}
    md_p = out_dir / "structured.md"
    parts = _PAGE_MARKER.split(md_p.read_text(encoding="utf-8")) if md_p.exists() else [""]
    body = {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    tj_p = out_dir / "tables.json"
    tables = json.loads(tj_p.read_text(encoding="utf-8")).get("tables", []) if tj_p.exists() else []

    improved = []
    for o in sorted(outcomes, key=lambda o: o.page):
        v, prev = o.final.verdict, pages.get(o.page)
        # A page read from a turned image is replaced whatever its score: an upright
        # page read upside down can come out as fluent text that is not on the page.
        if prev is not None and not read_turned(prev) and (_RANK[v.status], v.score) <= (
                _RANK.get(prev.get("status"), 0), prev.get("ocr_score", 0.0)):
            prev["pass2"] = {"kept": "the earlier read was as good or better",
                             "attempts": [a.summary() for a in o.attempts]}
            continue
        improved.append(o.page)
        pages[o.page] = dict(_page_entry(o), pass2=True)
        body[o.page] = v.text.strip() if v.indexable else ""
        tables = [t for t in tables if t.get("page") != o.page]
        if v.indexable:
            for t in o.final.tables:
                t = dict(t)
                t.pop("markdown", None)
                tables.append(t)
    tables.sort(key=lambda t: (t.get("page", 0), t.get("index", 0)))
    ids = {t.get("id") for t in tables}
    for t in tables:                  # a continuation whose first part was replaced keeps
        if t.get("continues") and t["continues"] not in ids:   # its inherited header,
            t["continues"] = None                               # just not the dangling link

    order = sorted(pages)
    verdicts = [tg.PageVerdict(pages[k].get("status", "reject"), pages[k].get("ocr_score", 0.0),
                               pages[k].get("reasons") or [], {}, "") for k in order]
    status, summary = tg.assess_document(verdicts)
    side["pages"] = [pages[k] for k in order]
    side.setdefault("document", {}).update(status=status, summary=summary)
    side["pass2"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "models": engine.describe(), "pages_reread": sorted(o.page for o in outcomes),
                     "pages_improved": improved, "seconds": round(seconds, 1)}
    if tables:
        _atomic_write(tj_p, json.dumps({"tables": tables}, ensure_ascii=False, indent=1))
    else:
        tj_p.unlink(missing_ok=True)
    _atomic_write(side_p, json.dumps(side, ensure_ascii=False, indent=1))
    if status in ("ok", "partial"):
        out = []
        for k in order:
            out.append(f"<!-- Page {k} -->")
            if body.get(k):
                out.append(body[k])
            out.append("")
        _atomic_write(md_p, "\n\n".join(out).strip() + "\n")
    return DocResult(
        uhash=uhash, status=status, summary=summary + f" (pass 2: {len(improved)} of "
        f"{len(outcomes)} re-read page(s) improved)", pages=len(order),
        kept=sum(1 for v in verdicts if v.indexable and v.status != "blank"),
        rejected=sum(1 for v in verdicts if v.status == "reject"),
        blank=sum(1 for v in verdicts if v.status == "blank"),
        fallback_used=sum(1 for o in outcomes if o.page in improved and o.final.engine == "vlm-fallback"),
        tables=len(tables), seconds=seconds)


def _report_line(res: DocResult, outcomes: list[PageOutcome]) -> dict:
    return {
        "uhash": res.uhash, "status": res.status, "summary": res.summary,
        "pages": res.pages, "kept": res.kept, "rejected": res.rejected, "blank": res.blank,
        "fallback_used": res.fallback_used, "tables": res.tables,
        "seconds": round(res.seconds, 1),
        "rejected_pages": [{"page": o.page, "reasons": o.final.verdict.reasons}
                           for o in sorted(outcomes, key=lambda o: o.page)
                           if o.final.verdict.status == "reject"],
        # Pages whose PDF uses a non-Unicode Hindi font (Krutidev-type): their junk
        # was removed, and they can be recovered later by font conversion.
        "legacy_font_pages": sorted(o.page for o in outcomes
                                    if (o.final.verdict.stats.get("repairs") or {}).get("legacy_font")),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ── many documents, pages in parallel ────────────────────────────────────────
@dataclass
class _Job:
    uhash: str
    pdf: Path
    futures: list
    started: float
    pages: list[int] | None = None   # 1-based pages being read; None = all of them
    patch: bool = False              # pass 2: patch them into the existing outputs


class DocumentRunner:
    """Feeds pages from many documents to the model server in parallel.

    Small documents dominate the corpus (median 2-3 pages), so pages from many
    documents are kept in flight together — one document at a time would leave the
    GPU mostly idle. Documents are finalized (gated + written) in the caller's
    thread as their last page completes.
    """

    ABORT_AFTER = 3     # consecutive documents lost to engine errors

    def __init__(self, engine: VlmEngine, out_root: Path, report_path: Path | None = None,
                 concurrency: int | None = None, log=print):
        t0 = time.monotonic()
        tg.warm_up()        # the page check's word lists: once, before any worker starts
        if time.monotonic() - t0 >= 1:
            log(f"word lists for the page check loaded ({time.monotonic() - t0:.0f}s)")
        self.engine = engine
        self.out_root = Path(out_root)
        self.report_path = report_path
        self.concurrency = concurrency or engine.s.concurrency
        self.max_backlog = self.concurrency * 2
        self.pool = cf.ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="ocr")
        self.jobs: list[_Job] = []
        self.log = log
        self.aborted: str | None = None
        self._engine_failures = 0
        self.pages_done = 0
        self._t0 = time.monotonic()
        self._arm_watchdog()

    # Stall watchdog. On 2026-10-10 pass 1 stopped feeding the GPU for ~5 h overnight
    # without an error. If no page finishes for this long, faulthandler writes the
    # stack of every thread to STALL_LOG (local disk, not the data mount) — the
    # place it is stuck, without needing py-spy. Re-armed by every finished page.
    STALL_SECONDS = int(os.getenv("OCR_STALL_DUMP_SECONDS", "900"))
    STALL_LOG = os.getenv("OCR_STALL_LOG", "/tmp/ocr-stall-stacks.log")
    _stall_fh = None

    def _arm_watchdog(self):
        if self.STALL_SECONDS <= 0:
            return
        import faulthandler
        if DocumentRunner._stall_fh is None:
            try:
                DocumentRunner._stall_fh = open(self.STALL_LOG, "a", encoding="utf-8")
            except OSError:
                DocumentRunner.STALL_SECONDS = 0
                return
        fh = DocumentRunner._stall_fh
        fh.write(f"\n# armed {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
                 f"pid {os.getpid()}: a dump below means no page finished for {self.STALL_SECONDS}s\n")
        fh.flush()
        faulthandler.dump_traceback_later(self.STALL_SECONDS, repeat=False, file=fh)

    def _page(self, pdf: Path, index: int) -> PageOutcome:
        try:
            return self.engine.ocr_page(pdf, index)
        finally:
            if self.STALL_SECONDS > 0 and DocumentRunner._stall_fh is not None:
                import faulthandler
                faulthandler.dump_traceback_later(self.STALL_SECONDS, repeat=False,
                                                  file=DocumentRunner._stall_fh)

    @property
    def inflight_docs(self) -> int:
        return len(self.jobs)

    def _backlog(self) -> int:
        return sum(1 for j in self.jobs for f in j.futures if not f.done())

    def submit(self, uhash: str, pdf: Path, pages: list[int] | None = None,
               patch: bool = False) -> list[DocResult]:
        """Queue a document — or, with `pages` + `patch`, only those pages of an
        already-OCR'd one (pass 2, patch_document). Returns documents that finished
        meanwhile."""
        if self.aborted:
            return []
        done: list[DocResult] = []
        while self._backlog() >= self.max_backlog and not self.aborted:
            done.extend(self.wait_one())
        if self.aborted:
            return done
        try:
            n = page_count(pdf)
        except Exception as e:      # noqa: BLE001 — unreadable PDF is a document property
            done.append(DocResult(uhash, "failed", f"cannot open PDF ({type(e).__name__}: {e})"))
            return done
        if n == 0:
            done.append(DocResult(uhash, "blank", "PDF has no pages"))
            return done
        todo = [p for p in pages if 1 <= p <= n] if pages is not None else list(range(1, n + 1))
        if not todo:
            return done
        futures = [self.pool.submit(self._page, pdf, p - 1) for p in todo]
        self.jobs.append(_Job(uhash, pdf, futures, time.monotonic(), todo, patch and pages is not None))
        done.extend(self._collect())
        return done

    def _collect(self) -> list[DocResult]:
        finished = [j for j in self.jobs if all(f.done() for f in j.futures)]
        results = []
        for j in finished:
            self.jobs.remove(j)
            results.append(self._finalize(j))
        return results

    def wait_one(self) -> list[DocResult]:
        """Block until at least one document finishes."""
        while self.jobs:
            res = self._collect()
            if res:
                return res
            pending = [f for j in self.jobs for f in j.futures if not f.done()]
            if pending:
                cf.wait(pending, timeout=60, return_when=cf.FIRST_COMPLETED)
        return []

    def finish(self) -> list[DocResult]:
        """Drain everything (or, after an abort, cancel what has not started)."""
        out = []
        if self.aborted:
            for j in self.jobs:
                for f in j.futures:
                    f.cancel()
        while self.jobs:
            out.extend(self.wait_one())
        self.pool.shutdown(wait=True)
        if self.STALL_SECONDS > 0:
            import faulthandler
            faulthandler.cancel_dump_traceback_later()
        return out

    def _finalize(self, job: _Job) -> DocResult:
        outcomes, engine_err = [], None
        for f in job.futures:
            if f.cancelled():
                engine_err = engine_err or "cancelled after an engine failure"
                continue
            exc = f.exception()
            if isinstance(exc, EngineError):
                engine_err = engine_err or str(exc)
            elif exc is not None:
                # A bug or a page that cannot be rendered: a page-level failure.
                outcomes.append(PageOutcome(
                    job.pages[job.futures.index(f)],
                    Attempt("vlm", self.engine.primary.model,
                            tg.PageVerdict("reject", 0.0, [f"page failed: {type(exc).__name__}: {exc}"], {}, ""),
                            [], 0.0, error=str(exc)),
                    [], 0.0))
            else:
                outcomes.append(f.result())
        seconds = time.monotonic() - job.started
        if engine_err:
            self._engine_failures += 1
            if self._engine_failures >= self.ABORT_AFTER and not self.aborted:
                self.aborted = (f"stopping: {self._engine_failures} documents in a row hit an OCR "
                                f"server failure — {engine_err}")
            return DocResult(job.uhash, "engine_error", engine_err, pages=len(job.futures), seconds=seconds)
        self._engine_failures = 0
        self.pages_done += len(outcomes)
        write = patch_document if job.patch else write_document
        res = write(self.out_root / job.uhash, job.uhash, outcomes, self.engine, seconds)
        if self.report_path is not None:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.report_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(_report_line(res, outcomes), ensure_ascii=False) + "\n")
        return res

    def throughput(self) -> str:
        dt = max(time.monotonic() - self._t0, 1e-6)
        return f"{self.pages_done} pages in {dt / 60:.1f} min ({self.pages_done / dt:.2f} pages/s)"
