"""Index chunks with their document's metadata — METADATA_LAYER_PLAN.md step 2 (§4).

Replaces Docling's HybridChunker in the indexing path. That chunker serialised a
budget table as ",  = मतदय. , राजस्व = 3,60,00,000. , पूंजी = . ,  = ." (header
lost, half the tokens empty cells) and cut 7-token fragments off headings.

Chunk types:
  text         prose, split at headings and a token budget (~400 bge-m3 tokens —
               about 1,000 Qwen tokens of Hindi, so ~10 passages fit the answer
               budget); pages exact, from the OCR page markers
  table_rows   up to 15 rows of one table, each row written "मद — column: value;
               …" with empty cells dropped, under the table's caption, unit and
               column line; the rows of a table that runs across pages share
               `table_group` and the first page's header
  table_whole  a table of 16-30 rows also once as a whole (a smaller one already
               fits in its single table_rows chunk)
  doc_card     one per document: its metadata as a card, so a broad question can
               find the document when no passage matches. Facts only, from the
               rules record (doc_metadata.py) — never a generated summary.

Every chunk starts with a CONTEXT LINE — "[आदेश · सामान्य प्रशासन विभाग · क्र. … ·
14.08.2026] title › section" — embedded with the chunk and shown to the answer
model, so "15 दिनों के भीतर आपत्ति प्रस्तुत करें" knows its order, date and subject.

Pure: the token counter is passed in, so this runs without models in tests.
"""

from __future__ import annotations

from importlib import import_module
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
aliases = import_module("02_optimization.aliases")
dm = import_module("01_preprocessing.doc_metadata")

# v2.1: the context line's order number / date from doc_metadata rules 2 (no Gazette
# registration line, no listing junk) — a bump re-embeds every document on the next
# index run (scripts/ingest_corpus.py compares this with each point's `chunker`).
CHUNKER_VERSION = "chunks v2.1"
# A scraped web page is chunked without the website's template — header, menus,
# footer (web_boilerplate.py). Its own version label, so a re-index after this change
# re-embeds only the web pages, not every document.
WEB_TEMPLATE_VERSION = "site template removed 1 + tables as rows 1"


def version_for(source_type: str | None) -> str:
    """The `chunker` label a document's chunks carry (scripts/ingest_corpus.py compares it)."""
    return f"{CHUNKER_VERSION} + {WEB_TEMPLATE_VERSION}" if source_type == "web" else CHUNKER_VERSION
TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "400"))
MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "512"))
MIN_TOKENS = 60
HEADING_FLUSH_TOKENS = 150      # a heading starts a new chunk only after this much text
ROWS_PER_CHUNK = 15
ROWS_TOKENS = 450
WHOLE_TABLE_MAX_ROWS = 30

TYPE_LABEL = {
    "circular": "परिपत्र", "order": "आदेश", "notification": "अधिसूचना", "memo": "ज्ञाप",
    "letter": "पत्र", "list": "सूची", "vacancy_notice": "रिक्ति/भर्ती सूचना",
    "application_form": "आवेदन प्रपत्र", "budget_summary": "बजट सारांश", "budget_table": "बजट तालिका",
    "budget_speech": "बजट भाषण", "act_rule": "अधिनियम/नियम", "policy": "नीति",
    "annual_report": "वार्षिक प्रतिवेदन", "press_note": "प्रेस विज्ञप्ति", "tender": "निविदा",
    "web_page": "वेब पेज", "other": None,
}
UNIT_LABEL = {"crore": "करोड़ रुपये", "lakh": "लाख रुपये", "thousand": "हजार रुपये", "rupees": "रुपये"}


@dataclass
class Chunk:
    n: str                       # "1", "2", … in document order; "card" for the doc card
    chunk_type: str
    body: str                    # context line + content: what is embedded and shown
    pages: list = field(default_factory=list)
    section_path: list = field(default_factory=list)
    tokens: int = 0
    table: dict | None = None
    context: str = ""            # the context line at the top of `body` ("" when off)

    @property
    def content(self) -> str:
        return self.body[len(self.context):].lstrip("\n") if self.context else self.body

    def file_text(self, doc_name: str) -> str:
        """Same header as the old chunk files (rag_pipeline.clean_chunk_text and
        answer_prompt.chunk_meta read it), plus the chunk type."""
        return (f"# Chunk {self.n}\nDocument: {doc_name}\nPages: {page_range(self.pages)}\n"
                f"Tokens: {self.tokens}\nHeadings: {[h[:60] for h in self.section_path[-3:]]!r}\n"
                f"Type: {self.chunk_type}\n"
                f"\n---\n\n{self.body}\n")


def _join(ctx: str, content: str) -> str:
    return f"{ctx}\n\n{content}" if ctx else content


def page_range(pages: list) -> str:
    if not pages:
        return "Unknown"
    pages = sorted(set(pages))
    return str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"


# ── the context line ─────────────────────────────────────────────────────────
def _ddmmyyyy(iso: str | None) -> str | None:
    if not iso:
        return None
    y, m, d = iso.split("-")
    return f"{d}.{m}.{y}"


CONTEXT_ON = os.getenv("CHUNK_CONTEXT", "1").strip().lower() not in ("0", "off", "false", "no")


def context_line(rec: dict | None, section: str | None = None, fallback_title: str | None = None) -> str:
    """CHUNK_CONTEXT=0 turns it off (for measuring what it adds — scripts/eval_retrieval.py)."""
    if not CONTEXT_ON:
        return ""
    rec = rec or {}
    parts = [TYPE_LABEL.get(rec.get("doc_type")), rec.get("issuing_office"),
             f"क्र. {dm.display_order(rec['order_number'])}" if rec.get("order_number") else None,
             _ddmmyyyy(rec.get("issue_date"))]
    if rec.get("status") == "superseded":
        parts.append("REPLACED / अधिक्रमित")       # the answer rules' CURRENT / REPLACED
    head = " · ".join(p for p in parts if p)
    # Only the document's OWN title goes into the line that is embedded with every
    # chunk: a model-written title (step 3) can be wrong without containing a single
    # checkable number, and here it would mislead the search for the whole document.
    # It is used for display (sources, catalogue) instead.
    own = rec.get("title") if rec.get("title_source") != "model" else (rec.get("_rules") or {}).get("title")
    title = dm.squash(own or fallback_title or "", 140)
    line = (f"[{head}] " if head else "") + title
    sec = dm.squash(section or "", 70)
    if sec and sec.lower() not in title.lower():
        line += f" › {sec}"
    return line.strip()


# ── pages and tables ─────────────────────────────────────────────────────────
_PAGE_MARK = re.compile(r"<!--\s*Page\s+(\d+)\s*-->")


def split_pages(md: str) -> list[tuple[int | None, str]]:
    marks = list(_PAGE_MARK.finditer(md))
    if not marks:
        return [(None, md)]
    out = []
    if md[:marks[0].start()].strip():
        out.append((int(marks[0].group(1)), md[:marks[0].start()]))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(md)
        out.append((int(m.group(1)), md[m.end():end]))
    return out


def load_tables_json(doc_dir: Path | None) -> dict:
    p = (doc_dir / "tables.json") if doc_dir else None
    if not p or not p.exists():
        return {}
    try:
        return {t["id"]: t for t in json.loads(p.read_text(encoding="utf-8")).get("tables") or []}
    except (OSError, ValueError, KeyError):
        return {}


def page_segments(md: str, doc_dir: Path | None = None) -> list[dict]:
    """[{"page", "segments": [("text", str) | ("table", table dict)]}] in order.

    Tables are found with the OCR engine's own parser (vlm_ocr.extract_tables —
    Markdown and HTML tables, junk rows dropped); when the OCR run left a
    tables.json (the GPU VM's engine), its caption / unit / header rows and
    cross-page links replace the re-parsed ones. Without one (older OCR), tables
    that run across pages are linked here the same way."""
    vlm_ocr = import_module("01_preprocessing.vlm_ocr")
    known = load_tables_json(doc_dir)
    pages = []
    for pno, text in split_pages(md):
        new_text, tables = vlm_ocr.extract_tables(text, pno or 1)
        for t in tables:
            k = known.get(t["id"])
            if k and k.get("n_cols") == t.get("n_cols"):
                for f in ("caption", "unit", "unit_evidence", "unit_confidence", "header_rows",
                          "columns", "continues", "table_group", "inherited"):
                    if f in k:
                        t[f] = k[f]
        pages.append({"page": pno, "text": new_text, "tables": tables})
    if not known:
        vlm_ocr.link_continuations([p for p in pages if p["page"] is not None])
    out = []
    for pg in pages:
        segs, rest = [], pg["text"]
        for t in pg["tables"]:
            md_t = t.get("markdown") or ""
            k = rest.find(md_t) if md_t else -1
            if k < 0:
                segs.append(("table", t))
                continue
            if rest[:k].strip():
                segs.append(("text", rest[:k]))
            segs.append(("table", t))
            rest = rest[k + len(md_t):]
        if rest.strip():
            segs.append(("text", rest))
        out.append({"page": pg["page"], "segments": segs})
    return out


# ── prose ────────────────────────────────────────────────────────────────────
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_LETTERHEAD_EXTRA = re.compile(r"e-?mail|फोन|phone|fax|website|www\.|@|\d{3,}-\d{5,}|भवन|नगर\s*$|^\W*$", re.I)


def is_letterhead(h: str) -> bool:
    """Letterhead and address lines that OCR marks as headings ("## छत्तीसगढ़ शासन",
    "## Email ID:- … Phone No. 0771-…", "## महानदी भवन, नवा रायपुर") — not sections."""
    return bool(dm.is_letterhead(h) or _LETTERHEAD_EXTRA.search(h))
_SENT = re.compile(r"(?<=[।|?!.])\s+")


def _pieces(text: str, count, limit: int) -> list[str]:
    """A paragraph over the limit, split at sentence ends, then at words."""
    out, cur = [], ""
    for s in _SENT.split(text):
        cand = f"{cur} {s}".strip()
        if cur and count(cand) > limit:
            out.append(cur)
            cur = s
        else:
            cur = cand
    if cur:
        out.append(cur)
    final = []
    for p in out:
        if count(p) <= limit:
            final.append(p)
            continue
        words, cur = p.split(), []
        for w in words:
            if count(w) > limit:
                # one "word" over the limit — a looped run of dots or underscores with
                # no space: cut it by length (it used to recurse until RecursionError)
                if cur:
                    final.append(" ".join(cur))
                    cur = []
                step = max(100, len(w) * limit // max(count(w), 1))
                final.extend(w[i:i + step] for i in range(0, len(w), step))
                continue
            if cur and count(" ".join(cur + [w])) > limit:
                final.append(" ".join(cur))
                cur = [w]
            else:
                cur.append(w)
        if cur:
            final.append(" ".join(cur))
    return final


# ── tables ───────────────────────────────────────────────────────────────────
_EST = [("BE", r"बजट\s*अनुमान|budget\s*estimate|\bB\.?\s?E\b"),
        ("RE", r"पुनरीक्षित|revised|\bR\.?\s?E\b"),
        ("Actuals", r"वास्तविक|लेखा|actuals?|accounts?"),
        ("Supplementary", r"अनुपूरक|supplementary")]


def estimate_type(col: str) -> str:
    for name, rx in _EST:
        if re.search(rx, col or "", re.I):
            return name
    return ""


def _is_number(v: str) -> bool:
    return bool(re.fullmatch(r"[\s(]*[-+]?[\d,]+(?:\.\d+)?\s*[)%]?\s*", v or ""))


def header_quality(cols: list[str]) -> float:
    """Share of columns with a distinct real name. Old OCR often returns 16 columns,
    most unnamed, the same text repeated across merged cells — naming those
    "स्तंभ 6: …" only adds noise."""
    if not cols:
        return 0.0
    named = {c.strip() for c in cols if c and c.strip() not in ("—", "-", "")}
    return len(named) / len(cols)


def _dedupe(vals: list[str]) -> list[str]:
    out = []
    for v in vals:
        v = (v or "").strip()
        if v and v not in ("-", "--") and v not in out:
            out.append(v)
    return out


def row_line(cols: list[str], row: list[str], structured: bool = True) -> str:
    """Structured: "मद — 2025-26 / बजट अनुमान: 26,500.00; …" — the first column as the
    row's label when it is text, every other cell with its column name. Otherwise
    (no usable header): the row's distinct cells, "a | b | c". Empties dropped."""
    if not cols or not structured:
        return " | ".join(_dedupe(row))
    vals = [(c or "", (v or "").strip()) for c, v in zip(cols + [""] * (len(row) - len(cols)), row)]
    label = ""
    if vals and vals[0][1] and not _is_number(vals[0][1]):
        label, vals = vals[0][1], vals[1:]
    pairs, seen = [], {label}
    for c, v in vals:
        if not v or v in ("-", "--") or v in seen:
            continue
        seen.add(v)
        pairs.append(f"{c}: {v}" if c and c.strip() not in ("—", "-") else v)
    return (f"{label} — " if label else "") + "; ".join(pairs)


def table_unit(t: dict, rec: dict | None) -> str:
    if t.get("unit"):
        ev = t.get("unit_evidence")
        return f"{UNIT_LABEL.get(t['unit'], t['unit'])}" + (f" {ev}" if ev else "")
    if rec and rec.get("unit") and rec.get("unit_confidence") == "high":
        return f"{UNIT_LABEL.get(rec['unit'], rec['unit'])} (दस्तावेज़ में: {rec.get('unit_evidence')})"
    return "इकाई नहीं दी गई (unit not stated)"


# ── the document card ────────────────────────────────────────────────────────
def doc_card(rec: dict) -> str:
    lines = [f"दस्तावेज़: {rec.get('title') or '—'}"]
    facts = [("प्रकार", TYPE_LABEL.get(rec.get("doc_type")) or rec.get("doc_type")),
             ("क्रमांक", rec.get("order_number")), ("दिनांक", _ddmmyyyy(rec.get("issue_date"))),
             ("कार्यालय", rec.get("issuing_office")),
             ("स्थिति", {"superseded": "अधिक्रमित", "duplicate": "प्रतिलिपि"}.get(rec.get("status")))]
    lines.append("; ".join(f"{k}: {v}" for k, v in facts if v))
    if rec.get("subject") and rec["subject"] != rec.get("title"):
        lines.append(f"विषय: {rec['subject']}")
    more = [("वित्तीय वर्ष", ", ".join(rec.get("fiscal_years") or [])[:80] or None),
            ("इकाई", UNIT_LABEL.get(rec.get("unit")) if rec.get("unit_confidence") == "high" else None),
            ("अंतिम तिथि", _ddmmyyyy(rec.get("last_date"))),
            ("जिले", ", ".join((rec.get("entities") or {}).get("districts") or []) or None)]
    if any(v for _, v in more):
        lines.append("; ".join(f"{k}: {v}" for k, v in more if v))
    lst = rec.get("listing") or {}
    if lst.get("title") and lst["title"] != rec.get("title"):
        lines.append(f"वेबसाइट पर: {dm.squash(lst['title'], 200)}")   # the site's label, a finding aid
    return "\n".join(lines)


# ── the whole document ───────────────────────────────────────────────────────
def build_chunks(md: str, rec: dict | None, count, doc_dir: Path | None = None) -> list[Chunk]:
    """All chunks of one document, in reading order, the doc card last."""
    fallback_title = None if rec else dm.first_heading(dm.prep(md))
    chunks: list[Chunk] = []
    section: list[str] = []
    levels: list[int] = []                          # heading level of each section entry
    buf: list[tuple[int | None, str]] = []          # (page, paragraph)
    buf_section: list[str] = []
    buf_tokens = 0

    def ctx():
        return context_line(rec, (buf_section or section)[-1] if (buf_section or section) else None,
                            fallback_title)

    def flush():
        nonlocal buf, buf_tokens, buf_section
        if not buf:
            return
        text = "\n\n".join(p for _, p in buf if p).strip()
        if text:
            c_ctx = ctx()
            body = _join(c_ctx, text)
            chunks.append(Chunk("", "text", body, [p for p, _ in buf if p], list(buf_section),
                                count(body), context=c_ctx))
        buf, buf_tokens, buf_section = [], 0, []

    def add_par(page, par, split=True):
        nonlocal buf_tokens, buf_section
        t = count(par)
        if t > MAX_TOKENS and split:
            flush()
            for piece in _pieces(par, count, TARGET_TOKENS):
                add_par(page, piece, split=False)        # pieces are never split again
            return
        if buf and buf_tokens + t > TARGET_TOKENS and buf_tokens >= MIN_TOKENS:
            flush()
        if not buf:
            buf_section = list(section)
        buf.append((page, par))
        buf_tokens += t

    for pg in page_segments(md, doc_dir):
        page = pg["page"]
        for kind, seg in pg["segments"]:
            if kind == "table":
                if buf and buf_tokens < MIN_TOKENS and chunks and chunks[-1].chunk_type == "text" \
                        and chunks[-1].tokens + buf_tokens <= MAX_TOKENS:
                    prev = chunks.pop()                     # a fragment before a table:
                    pp = prev.pages or [None]
                    buf = [(pp[0], prev.content)] + [(x, "") for x in pp[1:]] + buf
                    buf_section = prev.section_path
                flush()
                chunks.extend(table_chunks(seg, rec, count, section, page, fallback_title))
                continue
            for raw in re.split(r"\n\s*\n", seg):
                par = raw.strip()
                if not par or _PAGE_MARK.fullmatch(par):
                    continue
                m = _HEADING.match(par.split("\n", 1)[0])
                if m and is_letterhead(m.group(2)):
                    par = par.lstrip("# ").strip()
                    m = None                    # "## Email ID… Phone…" is not a section
                if m:
                    level, head = len(m.group(1)), dm.squash(m.group(2), 120)
                    if buf and buf_tokens >= HEADING_FLUSH_TOKENS:
                        flush()                 # a section starts a chunk — unless what
                        # came before is too small to stand alone (a letterhead block:
                        # "छत्तीसगढ़ शासन / … / प्रति, समस्त विभाग"), then it joins it
                    while levels and levels[-1] >= level:      # a sibling or a parent ends
                        levels.pop()
                        section.pop()
                    levels.append(level)
                    section.append(head)
                    par = par.lstrip("# ").strip()
                add_par(page, par)
    flush()

    # a last fragment too small to stand alone joins the chunk before it
    if len(chunks) >= 2 and chunks[-1].chunk_type == "text" and chunks[-1].tokens < MIN_TOKENS \
            and chunks[-2].chunk_type == "text":
        last = chunks.pop()
        prev = chunks[-1]
        prev.body = _join(prev.context, prev.content + "\n\n" + last.content)
        prev.pages = sorted(set(prev.pages + last.pages))
        prev.tokens = count(prev.body)

    for i, c in enumerate(chunks, 1):
        c.n = str(i)
    if rec:
        c_ctx = context_line(rec)
        body = _join(c_ctx, doc_card(rec))
        chunks.append(Chunk("card", "doc_card", body, [], [], count(body), context=c_ctx))
    return chunks


def _looks_like_header(row: list[str]) -> bool:
    """A first row of labels: no amounts, at least two named cells (years and
    "बजट अनुमान"/"लेखा" style labels count as names)."""
    cells = [c.strip() for c in row if c and c.strip()]
    if len(cells) < 2:
        return False
    numeric = sum(1 for c in cells if _is_number(c) and not re.fullmatch(r"(?:19|20)\d{2}", c.strip()))
    return numeric == 0


def _is_numbering_row(row: list[str]) -> bool:
    """"1 | 2 | 3 | 4" or "(1) | (2) | (3)" — the column numbers printed under a
    header, not data."""
    cells = [re.sub(r"[()\s.]", "", c or "") for c in row]
    cells = [c for c in cells if c]
    if len(cells) < 3 or not all(c.isdigit() for c in cells):
        return False
    nums = [int(c) for c in cells]
    return nums == list(range(nums[0], nums[0] + len(nums))) and nums[0] in (0, 1)


def table_chunks(t: dict, rec: dict | None, count, section: list, page, fallback_title) -> list[Chunk]:
    cols = list(t.get("columns") or [])
    rows = [r for r in (t.get("rows") or []) if any((c or "").strip() for c in r) and not _is_numbering_row(r)]
    if not cols and rows and len(rows) > 1 and _looks_like_header(rows[0]):
        # the OCR did not mark a header row, but the first row is one: without it,
        # every chunk after the first is numbers nobody can read ("143000.00" — which
        # year, BE or RE?)
        cols, rows = [c.strip() for c in rows[0]], rows[1:]
    if not rows:
        if header_quality(cols) < 0.6:
            return []
        # a blank form's table: its column names are what a "which details are
        # needed" question asks about — keep them as one short chunk
        ctx = context_line(rec, section[-1] if section else None, fallback_title)
        cap = dm.squash(t.get("caption") or "", 160)
        body = _join(ctx, f"प्रपत्र/तालिका: {cap or '—'}\nकॉलम: " + " | ".join(c or "—" for c in cols))
        return [Chunk("", "table_rows", body, [page] if page else [], list(section), count(body),
                      {"table_id": t.get("id"), "table_group": t.get("table_group") or t.get("id"),
                       "columns": cols, "caption": cap or None, "row_range": [0, 0]}, context=ctx)]
    ctx = context_line(rec, section[-1] if section else None, fallback_title)
    cap = dm.squash(t.get("caption") or "", 160)
    unit = table_unit(t, rec)
    structured = header_quality(cols) >= 0.6
    head = [f"तालिका: {cap or '—'} · इकाई: {unit}" + (" · (पिछले पृष्ठ से जारी)" if t.get("continues") else "")]
    if cols and structured:
        head.append("कॉलम: " + " | ".join(c or "—" for c in cols))
    head_text = "\n".join(head)
    info = {"table_id": t.get("id"), "table_group": t.get("table_group") or t.get("id"),
            "columns": cols if structured else [], "estimate_types": [estimate_type(c) for c in cols],
            "column_years": [(sorted(dm.fiscal_years(c)) or [None])[0] for c in cols],
            "caption": cap or None, "table_unit": t.get("unit")}
    pages = [page] if page else []
    out, lines, start = [], [], 1
    base = count(_join(ctx, head_text))
    used = base
    for i, r in enumerate(rows, 1):
        line = row_line(cols, r, structured)
        if not line:
            continue
        lt = count(line)
        if lines and (len(lines) >= ROWS_PER_CHUNK or used + lt > ROWS_TOKENS):
            out.append((start, i - 1, lines))
            lines, start, used = [], i, base
        lines.append(line)
        used += lt
    if lines:
        out.append((start, len(rows), lines))
    chunks = []
    for a, b, ls in out:
        body = _join(ctx, head_text + "\n" + "\n".join(ls))
        chunks.append(Chunk("", "table_rows", body, pages, list(section), count(body),
                            {**info, "row_range": [a, b]}, context=ctx))
    # No table_whole chunk (plan §4) for now: it repeats the rows chunks, and when both
    # reach the answer they spend the evidence budget twice. Revisit with the eval.
    return chunks


# ── payload ──────────────────────────────────────────────────────────────────
_INHERIT = ("title", "doc_type", "order_number", "issue_date", "issuing_office", "scope", "status",
            "superseded_by", "topics", "primary_year", "fiscal_years", "unit", "listing_category")


def payload(c: Chunk, rec: dict | None, *, doc_id: str, doc_dir: str, file: str,
            suspect_pages: set | None = None, version: str = CHUNKER_VERSION) -> dict:
    p = {"text": c.file_text(doc_dir), "source": doc_dir, "doc_id": doc_id, "chunk": c.n, "file": file,
         "chunk_type": c.chunk_type, "pages": sorted(set(c.pages)), "section_path": c.section_path,
         "context_line": c.context or None, "chunker": version}
    for k in _INHERIT:
        if rec and rec.get(k) not in (None, [], ""):
            p[k] = rec[k]
    content = c.content
    dist = aliases.find(dm._ADDRESS.sub(" ", content), "district")
    if dist:
        p["districts"] = [aliases.hindi("district", k) for k in dist]
    if c.table:
        p.update({k: v for k, v in c.table.items() if v not in (None, [], "")})
    if suspect_pages and set(c.pages) & suspect_pages:
        p["page_flags"] = ["suspect"]
    return p


def suspect_pages(doc_dir: Path | None, doc_name: str) -> set:
    """Pages the OCR gate let through with doubts (vlm sidecar status "suspect")."""
    if not doc_dir:
        return set()
    side = doc_dir / f"{doc_name}.pdf_confidence.json"
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if data.get("engine") != "vlm":
        return set()
    return {p["page_number"] for p in data.get("pages") or [] if p.get("status") == "suspect"}
