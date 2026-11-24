"""Thinking mode — multi-hop retrieval with an inspectable trace.

WHY THIS EXISTS
Single-shot RAG retrieves once and answers. That is right for "what is the sanctioned
strength of X", and wrong for the questions this corpus actually gets asked:

  "Notice A says one thing and notice B says another — which applies to my case?"
  "Is this officer eligible under the current rules?"
  "What changed between the 2019 order and now?"

Those need more than one look at the corpus. You cannot retrieve for a question you
have not realised you need to ask yet: you first find the 2019 order, and only then
learn there is a 2023 amendment worth retrieving. So this module runs a loop —
retrieve, read what came back, decide what is still missing, retrieve again — for a
bounded number of hops, then answers from everything it gathered.

THE LOOP
    PLAN        break the question into sub-questions              (LLM)
    RETRIEVE    hybrid search for each, merged and deduped         (deterministic)
    ASSESS      what is answered, what is missing, what to ask next(LLM)
      ...repeat RETRIEVE/ASSESS while there are gaps and hops left...
    TIMELINE    order the evidence by date, flag supersession      (deterministic)
    SYNTHESIZE  answer, with precedence rules made explicit        (LLM)

EVERY STEP IS RECORDED. `deep_query` returns a `trace`: an ordered list of what it
did, why, what came back and how long it took. The UI renders it as an expandable
panel. That is deliberate — an answer about which government order applies is only
useful if you can check the reasoning, and a black box that says "notice B applies"
is worse than useless in an official setting.

WHAT IS DETERMINISTIC ON PURPOSE
Date extraction and supersession detection are regex, not LLM. They are the load
bearing part of a precedence answer, they are cheap, and — most importantly — they
are auditable: the trace shows the exact date string matched and the exact phrase
("in supersession of", "अधिक्रमण में") that triggered the ruling. An LLM asked to
"figure out which is newer" gives you a confident sentence with nothing to check.
The LLM is used where judgment is genuinely needed: decomposing the question,
noticing gaps, and writing the final answer.

USAGE
    from thinking_mode import deep_query
    out = deep_query("Which leave rules apply to a deputationist?")
    out["answer"]           # str
    out["trace"]            # [{step, kind, title, detail, elapsed_s}, ...]
    out["results"]          # the retrieved chunks, same shape as retrieve_context
"""

import json
import os
import re
import time
from datetime import date

import rag_pipeline as rp
import answer_prompt as ap          # same directory; on sys.path via rag_pipeline
import llm_client
import query_debug as qd

# ── Tunables ────────────────────────────────────────────────────────────────
# Hops are the expensive dimension: each one costs a retrieval plus an LLM call.
# 3 is enough for "find the rule -> find its amendment -> check the exception";
# beyond that the model tends to re-ask what it already has.
MAX_HOPS = 3
SUB_QUESTIONS_PER_HOP = 3      # asked per hop; kept small so each stays specific
CHUNKS_PER_SUB_QUESTION = 6    # retrieved per sub-question before dedupe
SNIPPET_CHARS = 400            # per-chunk text shown in the trace (not to the LLM)
# Evidence and answer sizes are TOKEN budgets now (answer_prompt.py): synthesis gets
# ANSWER_EVIDENCE_TOKENS of evidence, ASSESS gets ASSESS_EVIDENCE_TOKENS, and the
# answer SYNTH_MAX_TOKENS (+ THINK_MAX_TOKENS when the GPU model thinks). The old
# fixed 16 chunks / 8192 output tokens came to ~26k tokens of Hindi before the model
# had thought at all — more than a 32k window can carry with thinking on.
# Thinking mode lets the GPU model reason before answering (its "Thought process").
SYNTH_THINKING = os.getenv("THINKING_MODE_THINKING", "on").strip().lower() in ("1", "on", "true", "yes")

# Output cap for the PLAN/ASSESS JSON steps (thinking off, schema enforced): the JSON
# is a few hundred tokens; the cap is only a safety margin.
REASONING_TOKENS = 1500


# ── Supersession vocabulary ─────────────────────────────────────────────────
# Phrases that mark one order acting on another. Split by direction because they
# mean opposite things for precedence:
#   SUPERSEDES_*  this document REPLACES an earlier one   -> this one wins
#   AMENDS_*      this document MODIFIES an earlier one   -> read both together
#   CONTINUES_*   this document EXTENDS an earlier one    -> earlier still stands
# Hindi and English are both present because this corpus mixes them within a
# single order (an English body under a Hindi header is routine).
SUPERSEDES_PATTERNS = [
    r"in supersession of", r"supersession", r"supersede[sd]?", r"superseding",
    r"in cancellation of", r"hereby cancell?ed", r"stands? cancell?ed",
    r"rescind(?:ed|s)?", r"withdrawn with effect", r"repeal(?:ed|s)?",
    r"अधिक्रमण", r"निरस्त", r"निरस्तीकरण", r"रद्द",
]
AMENDS_PATTERNS = [
    r"in partial modification", r"partially modif(?:y|ied)", r"amendment to",
    r"amend(?:ed|ment|s)?", r"modif(?:y|ied|ication)", r"substituted? for",
    r"read as", r"आंशिक संशोधन", r"संशोधन", r"संशोधित",
]
CONTINUES_PATTERNS = [
    r"in continuation of", r"further to", r"in reference to", r"in pursuance of",
    r"क्रम में", r"के संदर्भ में", r"अनुक्रम में",
]

_REL_KINDS = (
    ("supersedes", SUPERSEDES_PATTERNS),
    ("amends", AMENDS_PATTERNS),
    ("continues", CONTINUES_PATTERNS),
)


# ── Date extraction ─────────────────────────────────────────────────────────
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_HI_MONTHS = {
    "जनवरी": 1, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6,
    "जुलाई": 7, "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10,
    "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12,
}

# Numeric d/m/y — govt documents are day-first, so 03/04/2024 is 3 April.
_RE_NUMERIC = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-]((?:19|20)\d{2})\b")
_RE_TEXT_DMY = re.compile(
    r"\b(\d{1,2})\s*(?:st|nd|rd|th)?[\s.,-]+"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s.,-]+((?:19|20)\d{2})\b", re.I)
_RE_TEXT_MDY = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s.,-]+"
    r"(\d{1,2})\s*(?:st|nd|rd|th)?[\s.,-]+((?:19|20)\d{2})\b", re.I)
_RE_HI = re.compile(r"\b(\d{1,2})\s+(" + "|".join(_HI_MONTHS) + r")\s+((?:19|20)\d{2})\b")
_RE_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")


def _safe_date(y, m, d):
    try:
        return date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None


def extract_dates(text):
    """Every date found in `text`, as (date, matched_string), newest last.

    Day-first for numeric forms (Indian government convention). A bare year is only
    used when nothing more precise is present — it becomes 1 Jan of that year and is
    marked as such by its short matched string, so the trace shows the imprecision
    rather than hiding it.
    """
    found = []
    for m in _RE_NUMERIC.finditer(text):
        d = _safe_date(m.group(3), m.group(2), m.group(1))
        if d:
            found.append((d, m.group(0)))
    for m in _RE_TEXT_DMY.finditer(text):
        d = _safe_date(m.group(3), _MONTHS[m.group(2).lower()[:3]], m.group(1))
        if d:
            found.append((d, m.group(0)))
    for m in _RE_TEXT_MDY.finditer(text):
        d = _safe_date(m.group(3), _MONTHS[m.group(1).lower()[:3]], m.group(2))
        if d:
            found.append((d, m.group(0)))
    for m in _RE_HI.finditer(text):
        d = _safe_date(m.group(3), _HI_MONTHS[m.group(2)], m.group(1))
        if d:
            found.append((d, m.group(0)))
    if not found:
        for m in _RE_YEAR.finditer(text):
            d = _safe_date(m.group(1), 1, 1)
            if d:
                found.append((d, m.group(1)))
    return sorted(set(found))


def find_relations(text):
    """Supersession/amendment phrases present in `text`.

    Returns [(kind, matched_phrase), ...] where kind is supersedes|amends|continues.
    The matched phrase is kept so the trace can quote the exact words that drove the
    precedence call instead of asserting it.
    """
    low = text.lower()
    out = []
    for kind, patterns in _REL_KINDS:
        for pat in patterns:
            m = re.search(pat, low, re.I)
            if m:
                out.append((kind, m.group(0)))
                break          # one hit per kind is enough to flag the document
    return out


# ── Trace ───────────────────────────────────────────────────────────────────
class Trace:
    """Ordered record of what thinking mode did. This is the product, not a log."""

    def __init__(self, on_step=None):
        self.steps = []
        self._on_step = on_step

    def add(self, kind, title, detail=None, started=None):
        step = {
            "step": len(self.steps) + 1,
            "kind": kind,
            "title": title,
            "detail": detail or {},
            "elapsed_s": round(time.time() - started, 2) if started else 0.0,
        }
        self.steps.append(step)
        qd.record('Thinking mode — ' + title, step)
        print(f"  🧠 [{step['step']}] {kind.upper()}: {title} ({step['elapsed_s']}s)")
        if self._on_step:
            try:
                self._on_step(step)
            except Exception:
                pass           # a broken listener must never break the query
        return step


# ── LLM plumbing ────────────────────────────────────────────────────────────
def _json_from_llm(text):
    """Parse JSON out of an LLM reply, tolerating prose and code fences.

    Models wrap JSON in ```json blocks, prefix it with "Here is...", or add a
    trailing note. Rather than demand perfection from the prompt, extract the first
    balanced {...} and parse that. Returns None if nothing parses — callers treat
    that as "no structured answer" and fall back.
    """
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


PLANNER_SYSTEM = """You plan how to search a corpus of Indian government documents
(orders, circulars, notices, office memoranda, meeting minutes) in Hindi and English.

Break the user's question into focused sub-questions that can each be answered by
retrieving a passage. Think about which ARCHETYPE the question is, because each needs
different sub-questions:

- PRECEDENCE ("which notice applies", "is this still in force"): retrieve the rule
  itself, AND anything that amends/supersedes/cancels it, AND its effective date.
- COMPARISON ("A vs B", "what changed"): retrieve each side separately.
- ELIGIBILITY ("does X qualify"): retrieve the criteria, the exceptions, and any
  clarification issued later.
- PROCEDURE ("how do I", "what is the process"): retrieve the steps and the
  authority competent to approve.
- AGGREGATION ("list all", "how many"): retrieve each category separately.
- FACTUAL (single lookup): one or two sub-questions is enough — do not pad.

Rules:
- Write sub-questions in the SAME language as the user's question.
- Each sub-question must stand alone (no "it", "that order").
- Prefer the document's own vocabulary over the user's paraphrase.
- Do not invent document numbers or dates you have not seen.

Reply with ONLY JSON:
{"archetype": "<one of the above>", "reasoning": "<one sentence>",
 "sub_questions": ["...", "..."]}"""

ASSESS_SYSTEM = """You are checking whether retrieved passages actually answer a
question, and deciding what to search for next.

Be strict. "Related to the topic" is not "answers the question". In particular:
- If the passages give a rule but nothing about whether it was later amended,
  superseded or cancelled, that is a GAP for any question about what currently applies.
- If they mention another order by number or date without its contents, that is a GAP.
- If they state a proposal but not the decision taken, that is a GAP.
- If a date or authority the question depends on is missing, that is a GAP.

Do NOT invent follow-up questions when the evidence is genuinely sufficient — say so
and stop. Unnecessary hops cost the user time.

Reply with ONLY JSON:
{"sufficient": true|false,
 "answered": ["what the evidence establishes"],
 "gaps": ["what is still missing"],
 "next_questions": ["specific retrievable question", "..."]}
next_questions must be [] when sufficient is true."""

# The model server enforces these (structured output), so PLAN/ASSESS always return
# valid JSON; _json_from_llm is the safety net for a server without schema support.
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "archetype": {"type": "string", "enum": ["PRECEDENCE", "COMPARISON", "ELIGIBILITY",
                                                 "PROCEDURE", "AGGREGATION", "FACTUAL"]},
        "reasoning": {"type": "string"},
        "sub_questions": {"type": "array", "items": {"type": "string"},
                          "minItems": 1, "maxItems": SUB_QUESTIONS_PER_HOP},
    },
    "required": ["archetype", "sub_questions"],
}
ASSESS_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "answered": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "next_questions": {"type": "array", "items": {"type": "string"},
                           "maxItems": SUB_QUESTIONS_PER_HOP},
    },
    "required": ["sufficient", "gaps", "next_questions"],
}


def _plan(question, trace):
    """Decompose the question. Falls back to the question itself on any failure."""
    t0 = time.time()
    try:
        # Structured JSON, not deep reasoning: thinking off, the schema enforced by
        # the server.
        raw, _ = rp.call_llm(PLANNER_SYSTEM, f"Question: {question}",
                             temperature=rp.ANSWER_TEMPERATURE, max_tokens=REASONING_TOKENS,
                             thinking=False, json_schema=PLAN_SCHEMA)
        parsed = _json_from_llm(raw) or {}
    except rp.LLMError as e:
        trace.add("plan", "Planner unavailable — searching the question as asked",
                  {"error": str(e), "sub_questions": [question]}, t0)
        return [question], "unknown"

    subs = [s for s in (parsed.get("sub_questions") or []) if isinstance(s, str) and s.strip()]
    subs = subs[:SUB_QUESTIONS_PER_HOP] or [question]
    archetype = parsed.get("archetype", "unknown")
    trace.add("plan", f"Recognised a {archetype} question, split into {len(subs)} searches",
              {"archetype": archetype, "reasoning": parsed.get("reasoning", ""),
               "sub_questions": subs}, t0)
    return subs, archetype


def _retrieve(questions, seen_ids, trace, hop, *, date_question=None):
    """Retrieve for each sub-question; return (fresh_chunks, n_retrieval_errors). An
    error (e.g. the local Qdrant lock is held) is distinct from an empty result — the
    caller must NOT answer from the reference note when retrieval actually FAILED, or a
    transient lock turns into a confident wrong answer."""
    t0 = time.time()
    fresh, per_q, n_err = [], [], 0
    for q in questions:
        try:
            hits = rp.retrieve_context(q, num_context=CHUNKS_PER_SUB_QUESTION, date_question=date_question)
        except Exception as e:
            per_q.append({"question": q, "found": 0, "error": str(e)})
            n_err += 1
            continue
        if hits is None:                     # retrieval errored (returned None), not empty
            per_q.append({"question": q, "found": 0, "error": "retrieval unavailable"})
            n_err += 1
            continue
        new = 0
        for h in hits:
            pid = getattr(h.get("point"), "id", None)
            if pid is None or pid in seen_ids:
                continue
            seen_ids.add(pid)
            fresh.append(h)
            new += 1
        per_q.append({"question": q, "found": len(hits), "new": new})

    trace.add("retrieve", f"Hop {hop}: searched {len(questions)} question(s), "
                          f"{len(fresh)} new passage(s)" + (f", {n_err} error(s)" if n_err else ""),
              {"queries": per_q, "new_chunks": [_describe(h) for h in fresh]}, t0)
    return fresh, n_err


def _describe(hit):
    """Compact, human-readable summary of one retrieved chunk for the trace."""
    p = hit.get("point")
    payload = getattr(p, "payload", {}) or {}
    source = payload.get("source", "")
    text = payload.get("text", "") or ""
    return {
        "source": rp.source_label(source)[0] if source else "?",
        "chunk": payload.get("chunk"),
        "score": round(float(hit.get("score", 0)), 4),
        "supporting": "neighbor_of" in hit,
        "snippet": text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS else ""),
    }


def _assess(question, evidence, trace, hop):
    """Ask whether we can answer yet, and what to retrieve next."""
    t0 = time.time()
    try:
        _, user, _, _ = ap.build_messages(
            question, evidence, rules=ASSESS_SYSTEM, language_rule="",
            closing_line="Can this question be answered fully from these passages?",
            budget=ap.ASSESS_EVIDENCE_TOKENS, count=rp._count_tokens_fn(),
            clean=rp.clean_chunk_text, label=rp.source_label, terms=rp.query_terms(question),
            client=llm_client.client(), max_tokens=REASONING_TOKENS, thinking=False)
        raw, _ = rp.call_llm(ASSESS_SYSTEM, user, temperature=rp.ANSWER_TEMPERATURE,
                             max_tokens=REASONING_TOKENS,
                             thinking=False, json_schema=ASSESS_SCHEMA)
        parsed = _json_from_llm(raw) or {}
    except (rp.LLMError, llm_client.LLMClientError) as e:
        trace.add("assess", "Could not assess coverage — answering with what we have",
                  {"error": str(e)}, t0)
        return True, []

    sufficient = bool(parsed.get("sufficient"))
    gaps = [g for g in (parsed.get("gaps") or []) if isinstance(g, str)]
    nxt = [q for q in (parsed.get("next_questions") or []) if isinstance(q, str) and q.strip()]
    nxt = [] if sufficient else nxt[:SUB_QUESTIONS_PER_HOP]

    title = ("Evidence is sufficient — ready to answer" if sufficient
             else f"Found {len(gaps)} gap(s), searching {len(nxt)} more")
    trace.add("assess", f"Hop {hop}: {title}",
              {"sufficient": sufficient, "answered": parsed.get("answered") or [],
               "gaps": gaps, "next_questions": nxt}, t0)
    return sufficient, nxt


def _timeline(evidence, trace):
    """Order the evidence documents by date and mark which replace which.

    From each document's metadata record first (core/01_preprocessing/doc_metadata.py: the issue date
    read from the order itself; the corpus-wide pass that marks an order replaced by
    a newer one), then from dates and supersession phrases in the passages for
    documents without a record. Deterministic and auditable: the trace shows the
    date and the phrase or record behind every ruling. Documents are named by the
    same title as in the evidence ("[n] title"), so the model can match the two.
    """
    t0 = time.time()
    meta = rp._meta_by_dir()
    docs = {}
    for hit in evidence:
        payload = getattr(hit.get("point"), "payload", {}) or {}
        source = payload.get("source", "")
        if not source:
            continue
        text = payload.get("text", "") or ""
        rec = docs.get(source)
        if rec is None:
            m = meta.get(source) or {}
            rec = docs[source] = {"source": rp.source_label(source)[0], "dates": [],
                                  "relations": [], "has_date": False}
            if m.get("issue_date"):
                try:
                    rec["dates"].append((date.fromisoformat(m["issue_date"]),
                                         f"issue date ({m.get('issue_date_source') or 'record'})"))
                    rec["has_date"] = True
                except ValueError:
                    pass
            if m.get("status") == "superseded":
                newer = rp.source_label(m["superseded_by"])[0] if m.get("superseded_by") else "a newer order"
                rec["relations"].append(("replaced by", newer))
            for kind in ("supersedes", "amends", "continues"):
                for link in m.get(kind) or []:
                    what = " ".join(x for x in (link.get("order_number"), link.get("date")) if x)
                    rec["relations"].append((kind, what or (link.get("evidence") or "")[:120]))
        if not rec["has_date"]:
            rec["dates"].extend(extract_dates(text))
        if not meta.get(source):
            rec["relations"].extend(find_relations(text))

    rows = []
    for rec in docs.values():
        dates = sorted(set(rec["dates"]))
        rels = sorted(set(rec["relations"]))
        rows.append({
            "source": rec["source"],
            "latest_date": dates[-1][0].isoformat() if dates else None,
            "date_evidence": dates[-1][1] if dates else None,
            "all_dates": [d.isoformat() for d, _ in dates][:6],
            "relations": [{"kind": k, "phrase": p} for k, p in rels],
        })

    # Newest first; undated documents last (they cannot win a precedence argument).
    rows.sort(key=lambda r: (r["latest_date"] is not None, r["latest_date"] or ""), reverse=True)

    flagged = [r for r in rows if r["relations"]]
    dated = [r for r in rows if r["latest_date"]]
    if flagged:
        title = f"{len(dated)} dated document(s); {len(flagged)} replace, amend or are replaced"
    elif dated:
        title = f"{len(dated)} dated document(s), newest first"
    else:
        title = "No dates found in the evidence — precedence cannot be established"
    trace.add("timeline", title, {"documents": rows}, t0)
    return rows


def _render_timeline(rows):
    """Timeline as text for the synthesis prompt."""
    if not rows:
        return "(no dated documents)"
    lines = []
    for r in rows:
        when = r["latest_date"] or "undated"
        rel = ""
        if r["relations"]:
            rel = "  [" + "; ".join(f'{x["kind"]}: "{x["phrase"]}"' for x in r["relations"]) + "]"
        lines.append(f"- {r['source']} — {when}{rel}")
    return "\n".join(lines)


SYNTH_SYSTEM = ap.SYNTH_RULES        # the answer rules + the timeline instruction


def _synthesize(question, evidence, timeline_rows, plan_archetype, trace, on_token=None):
    """Final answer over everything gathered -> (answer, display_results).

    Evidence goes through the same builder as the single-shot answer — a token
    budget, grouped per document, numbered [1], [2]… for citations — with the
    question type and the dated timeline in front, and the answer is finished the
    same way (rp.finish_answer). The model thinks first (SYNTH_THINKING) within
    THINK_MAX_TOKENS. Blocking by default; with `on_token` it streams to the UI.
    """
    t0 = time.time()
    extra = (f"Question type: {plan_archetype}\n\n"
             f"Document timeline (newest first):\n{_render_timeline(timeline_rows)}\n\n")
    budget = max(2000, ap.EVIDENCE_TOKENS - llm_client.estimate_tokens(extra))
    max_tokens = ap.SYNTH_ANSWER_TOKENS + (ap.THINK_TOKENS if SYNTH_THINKING else 0)
    truncated = False
    try:
        system, user, sources, lang = rp.build_answer_messages(
            question, evidence, rules=SYNTH_SYSTEM, budget=budget, extra_before=extra,
            max_tokens=max_tokens, thinking=SYNTH_THINKING)
        if on_token is not None:
            buf = []
            for delta in rp.call_llm_stream(system, user, temperature=rp.ANSWER_TEMPERATURE, max_tokens=max_tokens,
                                            thinking=SYNTH_THINKING):
                buf.append(delta)
                on_token(delta)
            raw = "".join(buf)
        else:
            raw, truncated = rp.call_llm(system, user, temperature=rp.ANSWER_TEMPERATURE, max_tokens=max_tokens,
                                         thinking=SYNTH_THINKING)
    except rp.LLMError as e:
        trace.add("synthesize", f"Answer generation failed: {e}", {"error": str(e)}, t0)
        return f"Error generating answer: {e}", []

    answer, tail, display = rp.finish_answer(raw, user=user, question=question, sources=sources,
                                             lang=lang, context_results=evidence, truncated=truncated)
    if on_token is not None and tail:
        on_token(tail)
    shown_ranks = {r.get('rank') for r in display}
    shown = [source for n, source in enumerate(sources, 1) if n in shown_ranks]
    trace.add("synthesize", f"Answered from {len(shown)} document(s)",
              {"sources": [lbl for lbl, _ in shown], "streamed": on_token is not None,
               "truncated": truncated,
               "unsupported_figures": ap.unsupported_numbers(llm_client.strip_thinking(answer),
                                                             user, question)}, t0)
    return answer, display


# ── Entry point ─────────────────────────────────────────────────────────────
def deep_query(question, max_hops=MAX_HOPS, on_step=None, on_token=None):
    """Answer `question` with multi-hop retrieval, returning the reasoning trace.

    Args:
        question: the user's question.
        max_hops: retrieve/assess rounds before answering with what is gathered.
        on_step: optional callable invoked with each trace step as it completes
                 (for streaming a live view of PLAN/RETRIEVE/ASSESS/… to the UI).
        on_token: optional callable invoked with each answer delta during synthesis
                  (for streaming the final answer token-by-token to the UI).

    Returns:
        {answer, results, trace, hops, evidence_count, timeline, elapsed_s}
    """
    started = time.time()
    trace = Trace(on_step=on_step)
    print(f"\n🧠 Thinking mode: {question}")

    sub_questions, archetype = _plan(question, trace)

    # Search the original question too: a planner's subquestions can omit the
    # exact role/date that an administrator corrected.
    evidence = rp.correction_context(question)
    seen_ids = {h['point'].id for h in evidence}
    hop = 0
    pending = sub_questions
    retrieval_errors = 0
    while hop < max_hops and pending:
        hop += 1
        fresh, n_err = _retrieve(pending, seen_ids, trace, hop, date_question=question)
        evidence.extend(fresh)
        retrieval_errors += n_err
        if not evidence:
            break
        if hop >= max_hops:
            trace.add("assess", f"Hop limit ({max_hops}) reached — answering with "
                                f"{len(evidence)} passage(s)", {"hop_limit": max_hops})
            break
        sufficient, pending = _assess(question, evidence, trace, hop)
        if sufficient or not pending:
            break

    if not evidence:
        # A failed search (e.g. the index is locked) is NOT "nothing found": say it is
        # temporarily unavailable, so the user retries instead of trusting a no.
        if retrieval_errors:
            trace.add("synthesize", "Retrieval unavailable (index busy) — not answering", {})
            answer = ("The document index is temporarily busy and I couldn't search it for "
                      "this question. Please try again in a moment.")
        else:
            trace.add("synthesize", "No passages retrieved — cannot answer", {})
            answer = "No relevant documents were found for this question."
        return {"answer": answer, "results": [], "trace": trace.steps, "hops": hop,
                "evidence_count": 0, "timeline": [], "elapsed_s": round(time.time() - started, 2)}

    # Best evidence first, so the synthesis budget keeps the strongest passages.
    evidence.sort(key=lambda h: float(h.get("score", 0)), reverse=True)
    timeline_rows = _timeline([h for h in evidence if not h['point'].payload.get('verified_correction')], trace)
    answer, display = _synthesize(question, evidence, timeline_rows, archetype, trace,
                                  on_token=on_token)

    total = round(time.time() - started, 2)
    print(f"🧠 Thinking mode done in {total}s over {hop} hop(s), {len(evidence)} passage(s)")
    return {
        "answer": answer,
        "results": display,              # the documents the answer was given, numbered
        "trace": trace.steps,
        "hops": hop,
        "evidence_count": len(display),
        "timeline": timeline_rows,
        "elapsed_s": total,
    }
