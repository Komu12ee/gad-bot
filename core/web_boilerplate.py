"""The scraped website's own page template — header, menus, footer — on its web pages.

A scraped GAD page starts with ~35 lines of site chrome ("भाषा बदलें", "|| सेवाएं ||" and
the whole menu) and ends with a footer ("|| मुख्य लिंक ||", the disclaimer). In the
index that chrome made every web page look like a match for a general question, and
in the prompt it filled most of the evidence (GAD trace, 2026-10-15: 5 of the 8
documents given for "what is gad" were web pages whose first passage was the menu).

A line is template when it stands on at least SHARE of the bot's web pages (and on
MIN_PAGES or more): measured on 50 GAD pages, 0.5 finds the 34-line menu and
disclaimer (on 49 pages) and the footer (on 27), and keeps a median 61% of each page;
0.3 also took real lines and emptied 6 pages. Only the template lines at the START
and END of a page (or chunk) are removed — a common line inside the content ("नाम",
a serial number) stays.

Used when a web page is chunked (doc_pipeline, chunk_builder.version_for) and, for
chunks indexed before that, when the evidence and the reranker read a chunk
(rag_pipeline).
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from pathlib import Path

SHARE = 0.5
MIN_PAGES = 3
MIN_KEEP_CHARS = 20        # a page left with less than this keeps its text as it was

_HEADING = re.compile(r"^#+\s*")
_SPACE = re.compile(r"\s+")


def norm(line: str) -> str:
    return _SPACE.sub(" ", _HEADING.sub("", line or "")).strip()


def template_lines(pages, share: float = SHARE, min_pages: int = MIN_PAGES) -> frozenset:
    """The lines that stand on at least `share` of `pages` (texts) and on `min_pages`."""
    pages = list(pages)
    count = Counter()
    for text in pages:
        count.update({norm(l) for l in (text or "").splitlines() if norm(l)})
    need = max(min_pages, share * len(pages))
    return frozenset(l for l, c in count.items() if c >= need)


def _trim(lines: list[str], template: frozenset) -> list[str]:
    i, j = 0, len(lines)
    while i < j and (not norm(lines[i]) or norm(lines[i]) in template):
        i += 1
    while j > i and (not norm(lines[j - 1]) or norm(lines[j - 1]) in template):
        j -= 1
    return lines[i:j]


def strip_page(text: str, template: frozenset) -> str:
    """A whole page without the template lines at its start and end."""
    if not template or not text:
        return text
    kept = "\n".join(_trim(text.split("\n"), template))
    return kept if len(kept.strip()) >= MIN_KEEP_CHARS else text


def strip_chunk(text: str, template: frozenset, context_line: str | None = None) -> str:
    """A stored chunk ("# Chunk N … ---" header, the context line, the content) without
    the template lines at the start and end of its content. A chunk that is nothing
    but template keeps only its header and context line: the evidence then skips it."""
    if not template or not text:
        return text
    marker = "\n---\n"
    k = text.find(marker)
    head, body = (text[:k + len(marker)], text[k + len(marker):]) if 0 <= k <= 400 else ("", text)
    lines = body.split("\n")
    pre, idx = [], 0
    while idx < len(lines) and not lines[idx].strip():
        pre.append(lines[idx])
        idx += 1
    if context_line and idx < len(lines) and lines[idx].strip() == context_line.strip():
        pre.append(lines[idx])
        idx += 1
    kept = _trim(lines[idx:], template)
    if not kept:
        return head + "\n".join(pre)
    return head + "\n".join(pre + ([""] if pre else []) + kept)


_cache = {"key": None, "lines": frozenset()}
_lock = threading.Lock()


def for_registry(registry: Path, stage2: Path) -> frozenset:
    """Template lines of a bot's web pages (documents.json records with source_type
    "web", their stage2/<doc_dir>/structured.md), cached until documents.json changes."""
    try:
        key = (str(registry), registry.stat().st_mtime)
    except OSError:
        return frozenset()
    with _lock:
        if _cache["key"] == key:
            return _cache["lines"]
        try:
            docs = json.loads(registry.read_text(encoding="utf-8")).get("documents") or {}
        except (OSError, ValueError):
            docs = {}
        recs = docs.values() if isinstance(docs, dict) else docs
        pages = []
        for r in recs:
            if not isinstance(r, dict) or r.get("source_type") != "web" or not r.get("doc_dir"):
                continue
            md = stage2 / r["doc_dir"] / "structured.md"
            try:
                pages.append(md.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
        lines = template_lines(pages) if len(pages) >= MIN_PAGES else frozenset()
        _cache.update(key=key, lines=lines)
        return lines
