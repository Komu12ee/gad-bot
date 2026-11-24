"""Model metadata pass — METADATA_LAYER_PLAN.md step 3 (§2, §5).

One model call per document (the GPU VM's answer model, Qwen3-32B, through vLLM
with a JSON schema — never free text) fills what the rules (doc_metadata.py)
cannot: a clean title in Hindi and English, topics from the fixed list, a short
summary in Hindi and English, key facts, people with their posts, and the doc
type when the rules found none.

Guardrails — a wrong fact is worse than no fact:
  * the model sees page 1 of the document, its headings, the rules' facts and the
    website's listing label (marked as possibly wrong) — nothing else;
  * every number, date and name it writes must occur in the document's text, or
    that field is dropped (a summary is regenerated once, then dropped);
  * topics, doc type and scope come from fixed lists (the schema enforces them);
  * the document's own subject line / listing-confirmed title stays the title; the
    model's title is used only where the rules found nothing good, and is marked;
  * summaries are for FINDING and DISPLAYING a document (doc cards, sources) —
    the answer model is still given the document's own text.

Resumable: outputs are cached per document in data/<pipeline>/doc_model_meta.json,
keyed by the input, so a re-run only calls the model for new or changed documents.
doc_metadata.py merges the cache into doc_metadata.json on every build.

    PIPELINE=fg2 python core/02_optimization/doc_model_meta.py            # needs the chat model (LLM_BASE_URL)
    PIPELINE=fg2 python core/02_optimization/doc_model_meta.py --limit 20 --show
"""

from __future__ import annotations

from importlib import import_module
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "04_embeddings_and_kg" / "scripts"))
dm = import_module("01_preprocessing.doc_metadata")

PROMPT_VERSION = "model-meta v1"
EXCERPT_TOKENS = int(os.getenv("MODEL_META_EXCERPT_TOKENS", "2500"))
WORKERS = int(os.getenv("MODEL_META_WORKERS", "24"))

TOPICS = ["transfer", "promotion", "seniority", "deputation", "compassionate_appointment",
          "reservation_roster", "recruitment_rules", "leave_holidays", "pay_allowances",
          "pension", "discipline", "training", "awards_grants", "rti", "organisation",
          "budget", "scheme", "events", "elections", "audit_accounts", "other"]
TOPIC_HI = {"transfer": "स्थानांतरण", "promotion": "पदोन्नति", "seniority": "वरिष्ठता",
            "deputation": "प्रतिनियुक्ति", "compassionate_appointment": "अनुकंपा नियुक्ति",
            "reservation_roster": "आरक्षण / रोस्टर", "recruitment_rules": "भर्ती नियम",
            "leave_holidays": "अवकाश", "pay_allowances": "वेतन / भत्ते", "pension": "पेंशन",
            "discipline": "अनुशासन", "training": "प्रशिक्षण", "awards_grants": "पुरस्कार / अनुदान",
            "rti": "सूचना का अधिकार", "organisation": "संगठन / संरचना", "budget": "बजट",
            "scheme": "योजना", "events": "समारोह / आयोजन", "elections": "निर्वाचन",
            "audit_accounts": "लेखा / संपरीक्षा", "other": "अन्य"}

SCHEMA = {
    "type": "object",
    "properties": {
        "title_hi": {"type": "string", "maxLength": 220},
        "title_en": {"type": "string", "maxLength": 220},
        "doc_type": {"type": "string", "enum": list(dm.DOC_TYPES)},
        "topics": {"type": "array", "items": {"type": "string", "enum": TOPICS}, "minItems": 1, "maxItems": 3},
        "summary_hi": {"type": "string", "maxLength": 600},
        "summary_en": {"type": "string", "maxLength": 600},
        "applies_to": {"type": "string", "maxLength": 200},
        "effective": {"type": "string", "maxLength": 120},
        "deadline": {"type": "string", "maxLength": 120},
        "people": {"type": "array", "maxItems": 5, "items": {
            "type": "object",
            "properties": {"name": {"type": "string", "maxLength": 80},
                           "post": {"type": "string", "maxLength": 120}},
            "required": ["name", "post"]}},
        "scope": {"type": "string", "enum": ["state", "central", "other"]},
    },
    "required": ["title_hi", "title_en", "doc_type", "topics", "summary_hi", "summary_en",
                 "applies_to", "effective", "deadline", "people", "scope"],
}

SYSTEM = """You catalogue Chhattisgarh government documents (OCR text, may contain errors).
Fill the JSON fields from the document text ONLY.

- title_hi: what the document is, in Hindi, at most 20 words — like its subject line ("विषय"). title_en: the same in English.
- doc_type: the form of the document (circular, order, notification, …). topics: 1-3 from the list.
- summary_hi / summary_en: 2-3 sentences: what it decides or contains and for whom. No opinions.
- applies_to, effective, deadline: short phrases from the text, or "" if the text does not say.
- people: officials named with their post (the signatory, a minister presenting a budget, listed officeholders). [] if none.
- scope: "state" for Chhattisgarh's own documents, "central" for Government of India ones, "other" otherwise.

Every number, date, order number and name you write must appear in the text exactly. Never guess, calculate or translate a number.
The "website label" can be wrong — trust the document text."""


# ── the evidence check ───────────────────────────────────────────────────────
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    t = dm.prep(text)
    out = set()
    for m in _NUM.finditer(t):
        n = m.group(0).replace(",", "").rstrip(".")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        if len(n) >= 2:                     # single digits: list numbering, "3 days"
            out.add(n)
    return out


def unsupported_numbers(generated: str, source: str) -> list[str]:
    """Numbers in `generated` that the document text does not contain. Year ranges
    count both ways ("2025-26" vs "2025-2026")."""
    src = _numbers(source)
    for fy in dm.fiscal_years(dm.prep(source)):
        a, b = fy.split("-")
        src |= {a, b, b[-2:]}
    return sorted(n for n in _numbers(generated) if n not in src)


def name_supported(name: str, source: str) -> bool:
    """Every word of a person's name (2+ letters) occurs in the text."""
    words = [w for w in re.findall(r"[A-Za-zऀ-ॿ]{2,}", dm.prep(name))
             if w.lower() not in ("shri", "smt", "dr", "mr", "mrs", "ms", "श्री", "श्रीमती", "डॉ", "कु")]
    body = dm.prep(source).lower()
    return bool(words) and all(w.lower() in body for w in words)


def check(out: dict, source: str) -> tuple[dict, list[str]]:
    """(kept fields, problems). Fields that state facts the text does not hold are
    dropped; people whose names are not in the text are dropped."""
    problems, kept = [], dict(out)
    for f in ("title_hi", "title_en", "applies_to", "effective", "deadline"):
        bad = unsupported_numbers(kept.get(f) or "", source)
        if bad:
            problems.append(f"{f}: {bad}")
            kept[f] = ""
    for f in ("summary_hi", "summary_en"):
        bad = unsupported_numbers(kept.get(f) or "", source)
        if bad:
            problems.append(f"{f}: {bad}")
            kept[f] = None                   # None = failed the check (regenerate once)
    people = []
    for p in kept.get("people") or []:
        if p.get("name") and name_supported(p["name"], source):
            people.append({"name": dm.squash(p["name"], 80), "post": dm.squash(p.get("post") or "", 120) or None})
        elif p.get("name"):
            problems.append(f"person not in text: {p['name']}")
    kept["people"] = people
    for f in ("title_hi", "title_en"):
        kept[f] = dm.squash(kept.get(f) or "", 220) or None
    return kept, problems


# ── one document ─────────────────────────────────────────────────────────────
def excerpt(md: str, budget: int = EXCERPT_TOKENS) -> str:
    """Page 1 (then 2…) up to the budget, plus the later headings — what a person
    skims to catalogue a document."""
    from llm_client import estimate_tokens
    t = dm.prep(md)
    pages = dm.pages_of(t)
    out, used = [], 0
    for i, p in enumerate(pages):
        p = re.sub(r"\n{3,}", "\n\n", p).strip()
        c = estimate_tokens(p)
        if used + c > budget:
            if i == 0:                        # page 1 alone is too long: its start
                words = p.split()
                keep = max(50, int(len(words) * (budget - used) / max(c, 1)))
                out.append(" ".join(words[:keep]) + " …")
            break
        out.append(p)
        used += c
    heads = re.findall(r"^#{1,3}\s+(.+)$", "\n".join(pages[len(out):]), re.M)[:25]
    if heads:
        out.append("Later headings: " + " | ".join(dm.squash(h, 80) for h in heads))
    return "\n\n".join(out)


def user_message(rec: dict, text: str) -> str:
    known = [f"rules found — type: {rec.get('doc_type')}", f"order no.: {rec.get('order_number') or '—'}",
             f"date: {rec.get('issue_date') or '—'}", f"office: {rec.get('issuing_office') or '—'}",
             f"subject line: {rec.get('subject') or '—'}"]
    lst = (rec.get("listing") or {}).get("title")
    if lst:
        known.append(f"website label (may be wrong): {lst}")
    return "Known: " + "; ".join(known) + f"\n\nURL: {rec.get('source_url')}\n\nDocument text:\n{text}"


def input_key(rec: dict, text: str) -> str:
    return hashlib.sha1(json.dumps([PROMPT_VERSION, text, rec.get("subject"), rec.get("doc_type"),
                                    (rec.get("listing") or {}).get("title")],
                                   ensure_ascii=False).encode()).hexdigest()


def run_one(rec: dict, md: str, chat) -> dict:
    """Model output for one document, checked. `chat(system, user) -> str` (JSON)."""
    text = excerpt(md)
    if len(text.strip()) < 40:
        return {"status": "skipped", "reason": "no text"}
    user = user_message(rec, text)
    t0 = time.time()
    raw = chat(SYSTEM, user)
    out = json.loads(raw)
    kept, problems = check(out, text)
    if kept.get("summary_hi") is None or kept.get("summary_en") is None:
        # one retry, told what was wrong; then the summary is dropped
        retry = chat(SYSTEM, user + "\n\nYour previous summary contained numbers that are not in "
                     "the text (" + "; ".join(p for p in problems if p.startswith("summary")) +
                     "). Write the summaries again using only numbers from the text, or none.")
        again, problems2 = check(json.loads(retry), text)
        for f in ("summary_hi", "summary_en"):
            if kept.get(f) is None:
                kept[f] = again.get(f)
        problems += [f"retry: {p}" for p in problems2]
    kept = {k: v for k, v in kept.items() if v not in (None, "")} | {
        "summary_hi": kept.get("summary_hi") or None, "summary_en": kept.get("summary_en") or None}
    return {"status": "ok", "out": kept, "problems": problems, "seconds": round(time.time() - t0, 1)}


# ── merging into the rules record ────────────────────────────────────────────
WEAK_TITLES = (None, "heading", "lead_sentence", "filename")


def apply(rec: dict, m: dict | None) -> dict:
    """The rules record with the checked model fields merged in (doc_metadata calls
    this on every build). Rules facts are never overwritten."""
    # start from the rules' own values every time, so a newer model output replaces an
    # older one and a rebuild without the model gives back the rules record
    base = rec.setdefault("_rules", {k: rec.get(k) for k in ("title", "title_source", "doc_type",
                                                              "doc_type_source", "scope")})
    rec.update(base)
    rec.setdefault("entities", {})["people"] = [p for p in (rec["entities"].get("people") or [])
                                                if p.get("role") != "named"]
    if not m or m.get("status") != "ok":
        return rec
    o = m["out"]
    rec["title_en"] = o.get("title_en") or rec.get("title_en")
    if rec.get("title_source") in WEAK_TITLES and o.get("title_hi"):
        rec["title"], rec["title_source"] = o["title_hi"], "model"
    rec["title_model"] = o.get("title_hi")
    if rec.get("doc_type") in ("other", None) and o.get("doc_type") and o["doc_type"] != "other":
        rec["doc_type"], rec["doc_type_source"] = o["doc_type"], "model"
    rec["topics"] = [t for t in o.get("topics") or [] if t in TOPICS]
    rec["summary_hi"], rec["summary_en"] = o.get("summary_hi"), o.get("summary_en")
    facts = {k: o[k] for k in ("applies_to", "effective", "deadline") if o.get(k)}
    rec["key_facts"] = facts or None
    people = list((rec.get("entities") or {}).get("people") or [])
    have = {p.get("name") for p in people}
    for p in o.get("people") or []:
        if p["name"] not in have:
            people.append({"name": p["name"], "post": p.get("post"), "as_of": rec.get("issue_date"),
                           "role": "named"})
    rec.setdefault("entities", {})["people"] = people
    if o.get("scope") and rec.get("scope") is None:
        rec["scope"] = o["scope"]
    rec["generated_by"] = {**(rec.get("generated_by") or {}), "model": m.get("model")}
    return rec


def load_cache(data_root: Path) -> dict:
    p = data_root / "doc_model_meta.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("documents", {})
    except (OSError, ValueError):
        return {}


# ── the run ──────────────────────────────────────────────────────────────────
def run(data_root: Path, *, limit: int | None = None, workers: int = WORKERS, log=print) -> dict:
    import llm_client
    meta_path = data_root / "doc_metadata.json"
    recs = json.loads(meta_path.read_text(encoding="utf-8"))["documents"]
    cache = load_cache(data_root)
    llm = llm_client.client()
    model = llm.model

    def chat(system, user):
        # 2000: room for the longest answer the schema allows (titles, two 600-char
        # summaries, five people — ~1,700 tokens with Hindi). At 900 a long one was
        # cut off mid-JSON and the document counted as failed.
        text, _ = llm.chat(system, user, max_tokens=2000, temperature=0.1, thinking=False,
                           json_schema=SCHEMA, timeout=300)
        return text

    todo = []
    for doc_id, rec in recs.items():
        if rec.get("status") == "duplicate":
            continue
        md = data_root / "stage2" / (rec.get("doc_dir") or "") / "structured.md"
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8", errors="replace")
        key = input_key(rec, excerpt(text))
        if cache.get(doc_id, {}).get("key") == key and cache[doc_id].get("status") == "ok":
            continue
        todo.append((doc_id, rec, text, key))
    if limit:
        todo = todo[:limit]
    log(f"model metadata: {len(todo)} document(s) to read with {model} "
        f"({len(recs) - len(todo)} cached or skipped), {workers} at a time")

    lock = threading.Lock()
    done = failed = 0
    t0 = time.time()

    def work(item):
        doc_id, rec, text, key = item
        try:
            res = run_one(rec, text, chat)
        except Exception as e:  # noqa: BLE001 — one document must not stop the run
            res = {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:200]}"}
        res.update(key=key, model=model)
        return doc_id, res

    def save():
        tmp = data_root / "doc_model_meta.json.tmp"
        tmp.write_text(json.dumps({"prompt_version": PROMPT_VERSION, "documents": cache},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, data_root / "doc_model_meta.json")

    # One document alone first: a server that rejects the request (the JSON schema,
    # the model name) or is not there fails HERE, once, with its reason — instead of
    # every document failing quietly and the run still ending "done".
    rest = todo
    if todo:
        doc_id, res = work(todo[0])
        if res["status"] == "error":
            raise SystemExit(f"model metadata: the first document failed — {res.get('reason')}\n"
                             f"  not starting the other {len(todo) - 1}; fix the model server and run again")
        cache[doc_id] = res
        done += res["status"] == "ok"
        rest = todo[1:]
    reasons = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (doc_id, res) in enumerate(pool.map(work, rest), 2):
            with lock:
                cache[doc_id] = res
                done += res["status"] == "ok"
                if res["status"] == "error":
                    failed += 1
                    if len(reasons) < 3:
                        reasons.append(res.get("reason"))
                if i % 100 == 0 or i == len(todo):
                    save()
                    rate = i / max(time.time() - t0, 1e-6)
                    log(f"  {i}/{len(todo)} documents, {failed} failed, {rate * 60:.0f}/min")
    if todo:
        save()
    log(f"model metadata: {done} read, {failed} failed -> {data_root / 'doc_model_meta.json'}")
    for r in reasons:
        log(f"  e.g. {r}")
    return cache, [d for d, *_ in todo], failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", help="chips | fg | fg2 (default: $PIPELINE)")
    ap.add_argument("--limit", type=int, help="first N documents needing the model (trial)")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--show", action="store_true", help="print the outputs of this run")
    args = ap.parse_args(argv)
    if args.pipeline:
        os.environ["PIPELINE"] = args.pipeline
    import pipeline_config as pc
    cache, done, failed = run(pc.DATA_ROOT, limit=args.limit, workers=args.workers)
    # merge into doc_metadata.json (the rules records are rebuilt from their cache)
    dm.build_all(pc.DATA_ROOT, pc.DATA_BASE / "scrapped_data" / pc.PIPELINE)
    if args.show:
        for doc_id in done:
            print(json.dumps({"doc_id": doc_id, **cache[doc_id]}, ensure_ascii=False, indent=1)[:2500])
    if failed > max(10, len(done) // 5):
        # what did succeed is kept; a re-run reads only the documents that failed
        print(f"model metadata: {failed} of {len(done)} documents failed — stopping so this is "
              f"seen; run the same command again to retry them")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
