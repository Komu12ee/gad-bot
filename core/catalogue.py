"""The document list as a database — METADATA_LAYER_PLAN.md §6.4, §7 (catalogue route).

"GAD ke published reports kaun se hain?", "CHiPS latest notifications kya hain?"
have no passage that answers them; a list of the department's documents by type,
topic and date does. answer(question) builds that list straight from
doc_metadata.json (title, order number, date, link — nothing written by a model, so
no document can be invented), and rag_pipeline.with_catalogue puts it in front of
the passages as one more piece of evidence: the answer model still writes the
answer, so a question misread as a catalogue question costs one extra passage.
"""

from __future__ import annotations

from importlib import import_module
import json
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import query_plan as qp     # noqa: E402
display_order = import_module("01_preprocessing.doc_metadata").display_order   # noqa: E402,F401  (also used by rag_pipeline)

LIMIT = 12
_CACHE = {"path": None, "mtime": None, "docs": []}
_LOCK = threading.Lock()


def records(data_root: Path) -> list[dict]:
    """The document records — only of documents the search index holds (the
    registry's `indexed` flag, kept true by the indexer). A document left out of the
    index (its OCR read by another model than the bot's, or not indexed yet) is not
    listed either: the list must not name what the answers cannot cite. Without a
    registry (tests, a bare data folder) every record is listed."""
    path = Path(data_root) / "doc_metadata.json"
    reg = Path(data_root) / "documents.json"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    try:
        reg_mtime = reg.stat().st_mtime
    except OSError:
        reg_mtime = None
    with _LOCK:
        if _CACHE["path"] != path or _CACHE["mtime"] != (mtime, reg_mtime):
            try:
                docs = list(json.loads(path.read_text(encoding="utf-8")).get("documents", {}).values())
            except (OSError, ValueError):
                docs = []
            if reg_mtime is not None:
                try:
                    r = json.loads(reg.read_text(encoding="utf-8"))
                    r = r if isinstance(r, list) else r.get("documents", [])
                    indexed = {d.get("doc_dir") for d in r if d.get("indexed")}
                    docs = [d for d in docs if d.get("doc_dir") in indexed]
                except (OSError, ValueError):
                    pass
            _CACHE.update(path=path, mtime=(mtime, reg_mtime), docs=docs)
        return _CACHE["docs"]


def find(pl: qp.Plan, docs: list[dict], limit: int = LIMIT, question: str = "") -> list[dict]:
    """Documents matching what the question states: type, topic, year, district —
    each condition only if stated. Newest first; current before replaced; no
    duplicates or web pages."""
    # "list of circulars" / "परिपत्रों की सूची" asks for circulars, not for documents of
    # the type "list"
    types = [t for t in pl.doc_types
             if not (t == "list" and re.search(r"list\s+of|की\s+सूची|ki\s+suchi", question, re.I))]
    out = []
    for r in docs:
        if r.get("status") == "duplicate" or r.get("source_type") == "web":
            continue
        if types and r.get("doc_type") not in types:
            continue
        if pl.topics and not set(pl.topics) & set(r.get("topics") or []):
            # the model pass sets topics; before it has run, the title is the fallback
            if not any(re.search(qp._word(rx), r.get("title") or "", re.I)
                       for k, rx in qp._TOPIC_WORDS if k in pl.topics):
                continue
        if pl.years and not set(pl.years) & set(r.get("fiscal_years") or []):
            continue
        if pl.plain_years and not any((r.get("issue_date") or "").startswith(y) for y in pl.plain_years):
            continue
        if pl.districts and not set(pl.districts) & set((r.get("entities") or {}).get("districts") or []):
            continue
        out.append(r)
    out.sort(key=lambda r: (r.get("status") == "superseded", -(int(r["issue_date"].replace("-", ""))
                                                             if r.get("issue_date") else 0)))
    return out[:limit]


def display_title(r: dict) -> str:
    """A name a reader recognises: the document's own title; else the model's (step
    3); else the department's website label — fine for DISPLAY even where it was not
    trusted as the title; a file name only as the last resort."""
    if r.get("title") and r.get("title_source") != "filename":
        return r["title"]
    for t in (r.get("title_model"), (r.get("listing") or {}).get("title"), r.get("title_en"), r.get("title")):
        if t:
            return t
    return (r.get("source_url") or "").rsplit("/", 1)[-1] or r.get("doc_dir") or "—"


def _ddmmyyyy(iso):
    return ".".join(reversed(iso.split("-"))) if iso else None


def render(items: list[dict], lang: str, total: int) -> str:
    head = {"en": f"Documents found ({total}, newest first):",
            "hi": f"मिले दस्तावेज़ ({total}, नवीनतम पहले):"}.get(lang, f"मिले दस्तावेज़ ({total}, नवीनतम पहले):")
    lines = [head, ""]
    for i, r in enumerate(items, 1):
        title = display_title(r)
        bits = [b for b in (f"क्र. {display_order(r['order_number'])}" if r.get("order_number") else None,
                            _ddmmyyyy(r.get("issue_date")),
                            "अधिक्रमित / replaced" if r.get("status") == "superseded" else None) if b]
        link = f" — [{'link' if lang == 'en' else 'लिंक'}]({r['source_url']})" if r.get("source_url") else ""
        lines.append(f"{i}. **{title}**" + (f" ({', '.join(bits)})" if bits else "") + link)
    if total > len(items):
        lines += ["", {"en": f"…and {total - len(items)} more. Ask with a year or topic to narrow the list."}
                  .get(lang, f"…और {total - len(items)} दस्तावेज़। सूची छोटी करने के लिए वर्ष या विषय के साथ पूछें।")]
    return "\n".join(lines)


def answer(question: str, data_root: Path, lang: str = "hi") -> str | None:
    """The document list for a catalogue question, or None when the question is not
    one or nothing matches (the passages answer it alone)."""
    pl = qp.plan(question)
    if pl.intent != "catalogue":
        return None
    docs = records(data_root)
    if not docs:
        return None
    hits = find(pl, docs, limit=10**6, question=question)
    if not hits:
        return None
    return render(hits[:LIMIT], lang, len(hits))
