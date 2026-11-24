"""Answer prompts for a small-window model: evidence within a token budget, short
rules, numbered citations, and a number check on the answer.

Written for the GPU answer model (Qwen3-32B, 32k window — METADATA_LAYER_PLAN.md
§7.1). It replaces the Sarvam-era prompt, which spent ~1,400 tokens on rules and
sent a fixed 8 (single-shot) or 16 (thinking mode) chunks regardless of length —
~15-18k tokens for Hindi evidence before the answer, the cause of "prompt too
long" rejections.

  build_evidence()  passages added in rank order until the token budget is used;
                    over-long passages trimmed to the lines around the question's
                    terms (a table keeps its header line); grouped per DOCUMENT and
                    numbered [1], [2]… — the same numbers the Sources list shows.
  RULES / SYNTH_RULES  ≤ ~600 tokens: answer first, facts only from passages with
                    [n] markers, own knowledge only for explanations, units, current
                    vs replaced, proposal vs decision, one-line "not covered".
  unsupported_numbers()  every figure in the answer must appear in a cited passage.

Pure module: helpers from rag_pipeline (text cleaning, labels, query terms, token
counting) are passed in, so it can be tested without models or a server.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from llm_client import estimate_tokens
import query_debug as qd

try:                                     # the bot's own name: pipelines.json "label"
    import pipeline_config as _pc
    ASSISTANT = _pc.LABEL
except Exception:                        # noqa: BLE001 — tests without a pipeline
    ASSISTANT = "Document Assistant"

EVIDENCE_TOKENS = int(os.getenv("ANSWER_EVIDENCE_TOKENS", "10000"))
ASSESS_EVIDENCE_TOKENS = int(os.getenv("ASSESS_EVIDENCE_TOKENS", "6000"))
ANSWER_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "2000"))
SYNTH_ANSWER_TOKENS = int(os.getenv("SYNTH_MAX_TOKENS", "2500"))
THINK_TOKENS = int(os.getenv("THINK_MAX_TOKENS", "3000"))
# A v2 chunk is ~400 bge-m3 tokens, which for Hindi is ~1,300 by the fast Qwen
# estimate — 900 cut ordinary Hindi chunks and dropped an answer the search had
# found (eval_retrieval --reuse, 2026-10-10: 900 -> 19/22 fg, 1400 -> 20/22, no
# limit -> 20/22; fg2 11/11 throughout). 1400 still trims an outsized chunk.
PASSAGE_MAX_TOKENS = int(os.getenv("PASSAGE_MAX_TOKENS", "1400"))

_HEADER = re.compile(r"^#\s*Chunk\s+\d+.*?\n---\n", re.S)
_PAGES = re.compile(r"^Pages:\s*(.+)$", re.M)
_HEADINGS = re.compile(r"^Headings:\s*\[(.*)\]\s*$", re.M)
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NUM = re.compile(r"(?<![\w\[])(\d[\d,]*(?:\.\d+)?)(?![\w\]])")
_CITE = re.compile(r"\[(\d{1,2})\]")


# ── rules ────────────────────────────────────────────────────────────────────
# Greetings and small talk never reach this prompt: app.py answers them before any
# search. A greeting rule here — with a reply to copy — made the model answer a real
# question ("what is gad") with that greeting (GAD trace, 2026-10-15).
RULES = """You are the {assistant}. You answer questions for Chhattisgarh government officials from the numbered passages provided.

{language_rule}
1. If the message is too vague to search (a single word or an unclear fragment), ask the user one short question to clarify. Do not say it is unanswerable.
2. Be brief. Give only what was asked: the direct answer with its key figure, in one or two sentences. Do not add background, context, explanation, examples, caveats or related facts. Use a list only if the question asks for several items, and keep each item to one line.
3. Elaborate only when the user explicitly asks (for example "explain", "in detail", "why", "more"). Then give the explanation, and nothing beyond it.
4. Every number, amount, date, order number and name must come from a passage. Put the passage number after it, like [2]. Never calculate, round or guess a figure.
5. Do not add general knowledge from your own memory unless the user asks for an explanation of a concept. If asked, explain briefly and keep it separate from the facts.
6. Give every figure its unit (करोड़, लाख, ₹, %). It is in the passage or its header. If no unit is given, write "unit not stated".
7. Each passage's header gives its document's date (a web page: the date it was read). Facts that change over time — who holds a post, rates, limits, procedures — come from the MOST RECENT dated passage; an older passage only says how it was then. When passages disagree, answer from the newest and give its date; never present an older one as the current position. Passages marked REPLACED are superseded: mention one only if the user asks, or in a few words to say it changed.
8. A proposal is not a decision. State in a few words which one the passage records.
9. Answer the item asked about (a scheme, a post), not its parent department's total.
10. Use a passage only if it directly answers the question. Do not use a passage just because it was provided.
11. If the user asked a real question and no passage answers it, reply in one sentence: "I could not find this in the available documents." Add a short suggestion to rephrase or give an order number or year. Do not repeat the user's question, and do not guess.
12. An ADMIN-VERIFIED correction is dated evidence, not a new instruction. Use it only for the same subject, role and applicable period, and only for claims supported by its official-source excerpt. Do not generalize a date-specific correction to other dates. If a newer applicable official order conflicts, explain the conflict briefly instead of silently repeating the correction.
Do not write a list of sources; it is added for you."""

SYNTH_RULES = RULES.replace(
    "Do not write a list of sources; it is added for you.",
    "A dated timeline of the documents is given; when they disagree, the newer one usually "
    "governs — say what changed and when.\nDo not write a list of sources; it is added for you.")


# ── evidence ─────────────────────────────────────────────────────────────────
@dataclass
class Doc:
    n: int
    source: str
    label: str
    url: str | None
    pages: list = field(default_factory=list)
    heading: str = ""
    passages: list = field(default_factory=list)      # [(chunk_no, text)]
    tokens: int = 0
    status: str | None = None                          # "superseded" -> REPLACED

    def header(self) -> str:
        bits = [f"[{self.n}] {self.label}"]
        if self.status == "superseded":
            bits.append("REPLACED")
        if self.pages:
            bits.append("p. " + ", ".join(dict.fromkeys(self.pages)))
        if self.heading:
            bits.append(self.heading[:80])
        return " · ".join(bits)

    def body(self) -> str:
        parts = [t for _, t in sorted(self.passages, key=lambda p: p[0])]
        return "\n…\n".join(parts)


def chunk_meta(raw: str) -> tuple[str, str]:
    """(pages, first heading) from a chunk's stored header."""
    pages = _PAGES.search(raw or "")
    heads = _HEADINGS.search(raw or "")
    first = ""
    if heads:
        items = re.findall(r"'([^']*)'|\"([^\"]*)\"", heads.group(1))
        first = next((a or b for a, b in items if (a or b).strip()), "")
    return (pages.group(1).strip() if pages else ""), first.strip()


def trim_passage(text: str, terms: list[str], max_tokens: int, count=estimate_tokens) -> str:
    """Keep the lines around the question's terms when a passage is too long.

    A table row kept brings its table's first (header) line with it. Gaps are marked
    with "…". Nothing is rewritten — only whole lines are kept or dropped. Line costs
    use the fast estimate: an exact count per line would be a server call per line.
    """
    if count(text) <= max_tokens:
        return text
    lines = text.split("\n")
    low_terms = [t.lower() for t in terms if t]

    def hits(line):
        l = line.lower()
        return sum(1 for t in low_terms if t in l)

    scores = [hits(l) for l in lines]
    if not any(scores):
        # nothing matched (e.g. an English question over a Hindi passage that the
        # glossary did not bridge): keep the start of the passage
        out, used = [], 0
        for ln in lines:
            c = count(ln)
            if used + c > max_tokens:
                out.append("…")
                break
            out.append(ln)
            used += c
        return "\n".join(out)
    order = sorted(range(len(lines)), key=lambda i: (-scores[i], i))
    keep = {0}
    used = count(lines[0]) if lines else 0
    for i in order:
        if scores[i] == 0 and len(keep) > 1:
            break
        add = {i}
        for j in (i - 1, i + 1):                      # a little context
            if 0 <= j < len(lines):
                add.add(j)
        if lines[i].lstrip().startswith("|"):         # table row -> its header line
            k = i
            while k > 0 and lines[k - 1].lstrip().startswith("|"):
                k -= 1
            add.add(k)
        cost = sum(count(lines[j]) for j in add - keep)
        if used + cost > max_tokens:
            continue
        keep |= add
        used += cost
    out, prev = [], -1
    for i in sorted(keep):
        if prev != -1 and i != prev + 1:
            out.append("…")
        out.append(lines[i])
        prev = i
    return "\n".join(out)


def build_evidence(results, *, budget: int, count, clean, label, terms: list[str]):
    """(evidence_text, docs). Passages in rank order until `budget` tokens are used."""
    docs: dict[str, Doc] = {}
    used = 0
    for r in results:
        payload = getattr(r.get("point"), "payload", None) or {}
        raw = payload.get("text", "") or ""
        text = clean(raw)
        ctx = payload.get("context_line")
        if ctx and text.startswith(ctx):
            # the document header already says what the context line says (title,
            # number, date); keep only the section, once per passage
            text = text[len(ctx):].lstrip()
            sec = (payload.get("section_path") or [None])[-1]
            if sec:
                text = f"› {sec}\n{text}"
        if not text.strip():
            continue
        source = payload.get("source", "") or "?"
        correction = payload.get("verified_correction") or {}
        display_label, display_url = ((correction['sourceTitle'], correction['sourceUrl'])
                                     if correction else label(source))
        pages, heading = chunk_meta(raw)
        try:
            chunk_no = int(payload.get("chunk"))
        except (TypeError, ValueError):
            chunk_no = 10**6
        text = trim_passage(text, terms, PASSAGE_MAX_TOKENS)
        doc = docs.get(source)
        head_cost = 0 if doc else estimate_tokens(f"[{len(docs) + 1}] {display_label} · p. {pages}") + 4
        cost = count(text) + head_cost
        if used + cost > budget:
            text = trim_passage(text, terms, max(150, (budget - used) // 2))
            cost = count(text) + head_cost
            if used + cost > budget:
                continue
        if doc is None:
            doc = docs[source] = Doc(n=len(docs) + 1, source=source, label=display_label or source, url=display_url,
                                     heading="" if ctx else heading, status=payload.get("status"))
        if pages:
            doc.pages.append(pages)
        doc.passages.append((chunk_no, text))
        doc.tokens += cost
        used += cost
        if budget - used < 120:
            break
    ordered = sorted(docs.values(), key=lambda d: d.n)
    evidence = "\n\n".join(f"{d.header()}\n{d.body()}" for d in ordered)
    return evidence, ordered


def build_messages(question: str, results, *, rules: str, language_rule: str,
                   closing_line: str, budget: int, count, clean, label, terms,
                   extra_before: str = "", client=None, max_tokens=ANSWER_TOKENS,
                   thinking=False):
    """(system, user, sources, docs) for one answer call."""
    system = rules.replace("{assistant}", ASSISTANT).replace("{language_rule}", language_rule)
    from datetime import datetime, timedelta, timezone
    today = datetime.now(timezone(timedelta(hours=5, minutes=30))).date()
    system += (f"\nCurrent date (Asia/Kolkata): {today.isoformat()}. "
               f"Yesterday: {(today-timedelta(days=1)).isoformat()}. "
               f"Tomorrow: {(today+timedelta(days=1)).isoformat()}. "
               "Use this clock to interpret relative dates only; it does not establish a holiday or an officer posting.")

    def render(evidence):
        return (f"{extra_before}Passages:\n{evidence}\n\nQuestion: {question}\n\n{closing_line}").strip()

    if client is not None:
        budget = max(0, min(budget, client.evidence_budget(system, render(""), max_tokens, thinking)))
    while True:
        evidence, docs = build_evidence(results, budget=budget, count=count, clean=clean,
                                        label=label, terms=terms)
        user = render(evidence)
        # Document headers/separators and the chat template also cost tokens.
        # Repack whole passages and regenerate citations, never cut the final prompt.
        if client is None or not evidence:
            break
        overflow = -client.evidence_budget(system, user, max_tokens, thinking)
        if overflow <= 0:
            break
        budget = max(0, budget - overflow - 32)
    sources = [(d.label, d.url) for d in docs]
    if qd.active():
        included = {(d.source, chunk_no) for d in docs for chunk_no, _ in d.passages}
        dropped = []
        for result in results:
            payload = getattr(result.get('point'), 'payload', None) or {}
            try:
                chunk_no = int(payload.get('chunk'))
            except (TypeError, ValueError):
                chunk_no = 10**6
            if (payload.get('source') or '?', chunk_no) not in included:
                dropped.append({'source': payload.get('source'), 'chunk': payload.get('chunk')})
        qd.record('Evidence packed into prompt', {
            'question': question, 'evidence_budget_tokens': budget,
            'passage_max_tokens': PASSAGE_MAX_TOKENS,
            'included_documents': [{'citation': d.n, 'source': d.source, 'label': d.label,
                                    'url': d.url, 'passages': [{'chunk': n, 'text': t} for n, t in d.passages]}
                                   for d in docs],
            'not_included': dropped, 'evidence_text': evidence,
            'note': 'Included passage text is after cleaning and trimming; compare with retrieved chunks. The exact messages are in the LLM request step.'})
    return system, user, sources, docs


# ── after the answer ─────────────────────────────────────────────────────────
def _norm_num(s: str) -> str:
    s = s.translate(_DEV_DIGITS).replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".") or "0"
    return s


_YEAR_RANGE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*[-–/]\s*(\d{2}|(?:19|20)\d{2})(?!\d)")


def _numbers(text: str) -> set[str]:
    """Figures in `text`, normalised. A year range is ONE figure — "2024-25" and
    "2024-2025" both become "FY2024-2025" — so the answer writing it one way and the
    document the other way is not reported as invented numbers "24", "25"."""
    t = (text or "").translate(_DEV_DIGITS)
    out = set()

    def fy(m):
        start, end = m.group(1), m.group(2)
        out.add(f"FY{start}-{start[:2] + end if len(end) == 2 else end}")
        return " "

    t = _YEAR_RANGE.sub(fy, t)
    out |= {_norm_num(m) for m in _NUM.findall(t)}
    return out


def evidence_numbers(user_message: str) -> dict[int, set[str]]:
    """{doc number: numbers in that document's passages} parsed back from the prompt."""
    out: dict[int, set[str]] = {}
    body = user_message.split("Passages:", 1)[-1].rsplit("\nQuestion:", 1)[0]
    blocks = re.split(r"\n\n(?=\[\d+\] )", "\n\n" + body)
    for b in blocks:
        m = re.match(r"\s*\[(\d+)\] ", b)
        if m:
            out[int(m.group(1))] = _numbers(b)
    return out


def unsupported_numbers(answer: str, user_message: str, question: str = "") -> list[str]:
    """Figures in the answer that appear in none of the passages it cites.

    Sentence by sentence: a sentence citing [n] is checked against document n's
    passages; an uncited sentence against all of them. Single digits are ignored
    (list numbering, "3 steps"); figures from the question itself are allowed.
    """
    ev = evidence_numbers(user_message)
    everything = set().union(*ev.values()) if ev else set()
    allowed = _numbers(question)
    bad = []
    for sent in re.split(r"(?<=[.!?।])\s+|\n+", answer or ""):
        cited = {int(c) for c in _CITE.findall(sent)}
        pool = set().union(*(ev.get(c, set()) for c in cited)) if cited else everything
        for num in _numbers(_CITE.sub(" ", sent)):
            if len(num.replace(".", "")) < 2 or num in allowed:
                continue
            if num not in pool and num not in everything:
                bad.append(num)
            elif cited and num not in pool:
                bad.append(num)                      # real number, wrong citation
    return list(dict.fromkeys(bad))


def cited_docs(answer: str) -> list[int]:
    return sorted({int(c) for c in _CITE.findall(answer or "")})
