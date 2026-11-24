import os
import math
import json
import re
import time
import sys
from pathlib import Path

# Windows OpenMP and Torchvision fix
sys.modules['torchvision'] = None
sys.modules['torchvision.transforms'] = None
sys.modules['torchvision.io'] = None
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# NOTE: QdrantClient is deliberately NOT imported here. Every open goes through
# qdrant_store so the "one client per process, one process per store" invariant
# cannot be bypassed by accident. Only the payload/filter models are needed.
from qdrant_client.http import models as qmodels
from FlagEmbedding import BGEM3FlagModel   # the reranker: CrossEncoderReranker (plain transformers)

# Shared per-pipeline config (core/ is parents[2] of this script)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pipeline_config as pc
import qdrant_store                       # the one place the vector store is opened
import web_boilerplate                    # the scraped website's header/menus/footer
from qdrant_store import StoreUnavailable  # re-exported: callers catch it from here

# The GPU answer model's client and the prompts written for it (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_client                         # noqa: E402
import contextvars                        # noqa: E402
import answer_prompt as ap                # noqa: E402
import query_debug as qd                  # noqa: E402

# fp16 is a win on CUDA but a slowdown on CPU — pick it based on the actual device.
try:
    import torch
    _USE_FP16 = torch.cuda.is_available()
except Exception:
    _USE_FP16 = False

for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# ── Timing Utilities ───────────────────────────────────────────
_pipeline_start = None
_stage_times = {}

def _mark_time(stage_name):
    """Mark the current time for a stage and print elapsed time."""
    global _pipeline_start, _stage_times
    current_time = time.time()
    
    if _pipeline_start is None:
        _pipeline_start = current_time
        _stage_times[stage_name] = current_time
        print(f"⏱️ [{stage_name}] STARTED")
    else:
        if stage_name in _stage_times:
            elapsed = current_time - _stage_times[stage_name]
            print(f"⏱️ [{stage_name}] COMPLETED in {elapsed:.2f}s")
        _stage_times[stage_name] = current_time

def _print_timing_summary():
    """Print timing summary for entire pipeline."""
    global _pipeline_start
    if _pipeline_start is None:
        return
    total = time.time() - _pipeline_start
    print(f"\n{'='*70}")
    print(f"⏱️ TOTAL PIPELINE TIME: {total:.2f} seconds")
    print(f"{'='*70}\n")

# ── Configuration ──────────────────────────────────────────────
def _get_config():
    """Load configuration with environment variable overrides.

    Store connection settings (mode, path, host, lock budget) are NOT duplicated
    here — qdrant_store.config() is the single source for them. The keys mirrored
    below are read-only conveniences for logs and callers.
    """
    store = qdrant_store.config()
    return {
        "chunk_dir": Path(os.getenv("CHIPPY_CHUNK_DIR", str(pc.paths.chunks))),
        "collection": store.collection,
        "qdrant_mode": store.mode,
        "qdrant_local_path": store.path,
        "qdrant_host": store.host,
        "qdrant_port": store.port,
        "qdrant_target": store.target,
        "encode_batch_size": 8,
        "max_length": 1024,
    }

CFG = _get_config()
CHUNK_DIR = CFG["chunk_dir"]
COLLECTION_NAME = CFG["collection"]
ENCODE_BATCH_SIZE = CFG["encode_batch_size"]
MAX_LENGTH = CFG["max_length"]

# ── The answer model ───────────────────────────────────────────
# One model answers everything: Qwen3-32B served by vLLM on the GPU VM (:8000,
# OpenAI-compatible; deploy/gpu-vm.sh serve). llm_client.py talks to it — thinking
# switch, JSON schema, exact token counts, window fit — and answer_prompt.py holds
# the prompts. LLM_BASE_URL may point at any OpenAI-compatible server (local tests).
# Thinking for normal answers is off: Qwen3's reasoning costs seconds and the
# evidence is already selected. Thinking mode switches it on for its own answer.
ANSWER_THINKING = os.getenv("ANSWER_THINKING", "off").strip().lower() in ("1", "on", "true", "yes")
# Sampling temperature for every answer (single-shot, streaming, batch, thinking-mode
# steps, department bots). 0 = the same evidence gives the same answer: at 0.2 the
# same question could come back different, and with conflicting passages even pick a
# different fact. ANSWER_TEMPERATURE=0.2 restores the old behaviour.
ANSWER_TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0"))

# ── Retrieval Configuration ────────────────────────────────────
HYBRID_ALPHA = 0.6           # 0.0 = pure sparse, 1.0 = pure dense (0.6 = 60% dense, 40% sparse)
USE_MULTI_QUERY = True       # Enable multi-query retrieval for better coverage
MAX_QUERY_VARIATIONS = int(os.getenv("MAX_QUERY_VARIATIONS", "3"))  # incl. the original query
DENSE_CANDIDATE_LIMIT = 60   # Dense hits fetched per query variation (sparse re-scores only these)
MAX_CHUNKS_PER_SOURCE = 3    # Cap chunks contributed by any single document (0 = no cap)
MAX_DOC_CARDS = 2            # doc cards (chunk_builder) that may take a slot in one answer
# Each metadata feature can be switched off; each stays on only while the retrieval
# eval (scripts/eval_retrieval.py) shows it helps:
QUERY_BOOSTS = os.getenv("QUERY_BOOSTS", "1").strip().lower() not in ("0", "off", "false", "no")
TABLE_COUNT = os.getenv("TABLE_COUNT", "1").strip().lower() not in ("0", "off", "false", "no")

# ── Reranking (OFF by default; ON on the GPU VM) ───────────────
# A cross-encoder (bge-reranker-v2-m3) reads the question and each candidate chunk
# together and re-orders the hybrid search's candidates. On CPU it costs ~60s per
# query, so it is opt-in; on the GPU VM it takes well under a second and
# deploy/gpu-vm.sh serve switches it on. The 2.2GB model is only downloaded/loaded
# when enabled (see load_reranker).
#
# It only RE-ORDERS: the exact-identifier lexical arm, the per-document cap and the
# neighbour expansion in retrieve_context still apply to the reranked order.
#
#   RERANK_ENABLED=1 uv run python app.py     # turn it on, no code edit
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "100"))    # hybrid top-N the reranker reads
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.01"))   # 0-1; below = unrelated, dropped
# ...but never fewer than this many survive. It was 3: a question in another language
# than its passages (Hinglish / English against Hindi orders) gets cross-encoder scores
# near 0 for everything, so only 3 passages reached the answer and the right section
# was often not among them (live GAD traces, 2026-10-21: "child care leave ..." kept 3,
# best score 0.0007). 8 = the passages one answer uses; the answer rules (10, 11)
# already ignore a passage that does not answer.
RERANK_MIN_K = int(os.getenv("RERANK_MIN_K", "8"))
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "1024"))   # tokens per question+chunk pair
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    """bge-reranker-v2-m3 scored with plain transformers.

    FlagEmbedding 1.4's FlagReranker builds each question+chunk input with
    tokenizer.prepare_for_model, which transformers 5 removed: every rerank failed
    and the bot quietly kept the hybrid order (GAD trace, 2026-10-15). This is the same
    model and the same input — [CLS] question [SEP][SEP] chunk [SEP], the chunk cut
    to fit, the question to 3/4 of the length — scored the same way: one logit per
    pair, sigmoid -> 0..1 with normalize=True.
    """

    def __init__(self, name=RERANK_MODEL, use_fp16=False, max_length=1024, batch_size=32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if use_fp16 and self.device == "cuda":
            self.model.half()
        self.model.to(self.device).eval()
        self.max_length, self.batch_size = max_length, batch_size

    def compute_score(self, pairs, normalize=True):
        if pairs and isinstance(pairs[0], str):
            pairs = [pairs]
        import torch
        q_max = self.max_length * 3 // 4

        def cut(q):      # a question over 3/4 of the length is cut, as FlagReranker did
            ids = self.tokenizer(q, add_special_tokens=False)["input_ids"]
            return q if len(ids) <= q_max else self.tokenizer.decode(ids[:q_max])

        questions = [cut(q) for q, _ in pairs]          # the chunk takes the rest ('only_second')
        order = sorted(range(len(pairs)), key=lambda i: -len(pairs[i][1]))   # less padding
        scores = [0.0] * len(pairs)
        with torch.no_grad():
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                enc = self.tokenizer([questions[i] for i in idx], [pairs[i][1] for i in idx],
                                     padding=True, truncation="only_second",
                                     max_length=self.max_length, return_tensors="pt").to(self.device)
                logits = self.model(**enc, return_dict=True).logits.view(-1).float().cpu().tolist()
                for i, x in zip(idx, logits):
                    scores[i] = x
        if normalize:
            scores = [1.0 / (1.0 + math.exp(-x)) for x in scores]
        return scores

# ── Helper: Extract highlighted excerpt from chunk text ─────────
def extract_highlighted_excerpt(chunk_text, query_words, max_length=300):
    """Extract the most relevant part of chunk text containing query words.
    
    Args:
        chunk_text: Full chunk text
        query_words: List of important words from the query
        max_length: Max length of excerpt
    
    Returns:
        Highlighted excerpt with query words in context
    """
    sentences = chunk_text.split('. ')
    best_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(word.lower() in sentence_lower for word in query_words if len(word) > 3):
            best_sentences.append(sentence.strip())
    
    if best_sentences:
        excerpt = '. '.join(best_sentences[:2])  # Take first 2 matching sentences
    else:
        excerpt = chunk_text[:max_length]
    
    # Truncate if too long
    if len(excerpt) > max_length:
        excerpt = excerpt[:max_length].rsplit(' ', 1)[0] + '...'
    
    return excerpt.strip()

# Loaded on first retrieval, never at import time — importing this module must stay
# cheap so the Flask app boots fast and `--help`/health checks don't pay for a model.
model = None
reranker = None
_models_loaded = False

# One bge-m3 call at a time: its fast tokenizer is not safe to share between
# threads ("Already borrowed"), and the department bots (dept_service.py) embed
# documents in this process while the bots answer questions.
import threading                          # noqa: E402
ENCODE_LOCK = threading.Lock()


def ensure_models_loaded():
    """Load the embedding model on first use. Idempotent."""
    global model, _models_loaded
    if _models_loaded:
        return

    t0 = time.time()
    print("Loading embedding model bge-m3 (deferred)...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=_USE_FP16)
    model.return_sparse = True
    print(f"✓ Embedding model ready in {time.time() - t0:.1f}s (fp16={_USE_FP16})")
    load_reranker()

    _models_loaded = True


def load_reranker():
    """The cross-encoder, loaded only if RERANK_ENABLED — it is 2.2GB.

    Separate from the embedding model on purpose: that one is required for every
    query, the reranker is optional. Returns None when reranking is off, which is
    what makes the default path cost nothing."""
    global reranker
    if not RERANK_ENABLED:
        return None
    if reranker is None:
        t0 = time.time()
        print("Loading reranker bge-reranker-v2-m3 (RERANK_ENABLED)...")
        reranker = CrossEncoderReranker(RERANK_MODEL, use_fp16=_USE_FP16,
                                        max_length=RERANK_MAX_LENGTH, batch_size=32)
        print(f"✓ Reranker ready in {time.time() - t0:.1f}s (fp16={_USE_FP16})")
        if not _USE_FP16:
            print("  ⚠ reranker is on the CPU — expect about a minute per query")
    return reranker

# Kept as a module attribute because app.py's compatibility shims read
# `rag_module.client` directly. It mirrors qdrant_store's handle; qdrant_store, not
# this variable, is the authority on whether the store is open.
client = None


def ensure_qdrant_client():
    """Return this process's Qdrant client, opening it on first use.

    All the hard parts — one client per process even under concurrent Flask request
    threads, one process per embedded store, a bounded wait that costs no collection
    load, and a named holder in the error when someone else has it — live in
    qdrant_store. This is the thin, cached front door.

    Raises StoreUnavailable (not a bare RuntimeError, and never a silent None) so
    callers can tell "the index is busy" apart from "nothing matched".
    """
    global client
    if client is not None:
        return client
    client = qdrant_store.get_client()
    return client


def close_qdrant_client():
    """Release the store so another process can take it. Safe to call repeatedly."""
    global client
    client = None
    qdrant_store.close_client()


def store_status():
    """Non-invasive store diagnostics for /api/health — see qdrant_store.describe."""
    return qdrant_store.describe()


# qdrant_store registers its own atexit hook that closes the client and releases
# ownership, so there is nothing to unwind here.

# ── Helper: Sparse search ──────────────────────────────────────
def sparse_search(query_sparse, all_points, limit=5):
    """Score points based on sparse embeddings overlap."""
    scores = []
    for point in all_points:
        sparse_payload = point.payload.get("sparse_embedding", {})
        score = sum(sparse_payload.get(token, 0) * query_sparse.get(token, 0) 
                   for token in query_sparse if token in sparse_payload)
        scores.append((point.id, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:limit]

# ── Helper: Pull the chunk that follows a retrieved chunk ──────
def expand_with_neighbors(qdrant, results, max_extra=None):
    """Append the chunks immediately BEFORE and AFTER each retrieved chunk, same document.

    The chunker splits documents on headings, which separates a thing from its detail:
    "AGENDA # 7" from its "DECISION TAKEN" (detail AFTER), and equally a figure like
    "each mission gets ₹100 crore/year" from the list of missions it introduces (figure
    BEFORE). Retrieval lands on one and misses the other, so the answer is incomplete —
    or reports a deferred item as approved. Following the chunk pointer in BOTH
    directions costs one filtered lookup and repairs the pairing without re-chunking or
    re-embedding anything. Neighbours are added as low-weight supporting context.
    """
    # Expand every hit by default. Capping this made it first-come-first-served over
    # the result list, so the neighbour that actually carried the decision was often
    # skipped in favour of neighbours of higher-ranked but less relevant chunks. Each
    # hit can contribute two neighbours (previous + next).
    if max_extra is None:
        max_extra = 2 * len(results)

    have = {(r["point"].payload.get("source"), str(r["point"].payload.get("chunk"))) for r in results}

    # Work out every neighbour we want BEFORE touching the DB, then fetch them all in
    # one filtered scroll (an OR over the wanted source/chunk pairs). This used to be
    # one round trip per result — 8 sequential queries for a default 8-hit answer.
    wanted = []          # [(source, neighbour_chunk, parent_result)]
    for r in results:
        payload = r["point"].payload
        source = payload.get("source")
        try:
            n = int(payload.get("chunk"))
        except (TypeError, ValueError):
            continue
        # previous first (the setup/figure), then next (the detail/outcome)
        for neigh in (n - 1, n + 1):
            if len(wanted) >= max_extra:
                break
            if neigh < 0:
                continue
            neigh_chunk = str(neigh)
            if (source, neigh_chunk) in have:
                continue
            have.add((source, neigh_chunk))      # also dedupes within this batch
            wanted.append((source, neigh_chunk, r))

    if not wanted:
        return results

    _ensure_text_index(qdrant)      # source / chunk indexes: this lookup scanned every point without them
    try:
        found, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qmodels.Filter(should=[
                qmodels.Filter(must=[
                    qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=src)),
                    qmodels.FieldCondition(key="chunk", match=qmodels.MatchValue(value=chunk)),
                ])
                for src, chunk, _ in wanted
            ]),
            limit=len(wanted),
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        print(f"  ⚠ Neighbor lookup failed ({len(wanted)} wanted): {e}")
        return results

    # Match fetched points back to the hit that asked for them, so the supporting
    # chunk inherits a score derived from its parent.
    by_key = {(p.payload.get("source"), str(p.payload.get("chunk"))): p for p in found}
    extras = []
    for source, next_chunk, parent in wanted:
        point = by_key.get((source, next_chunk))
        if point is None:
            continue
        extras.append({
            "point": point,
            "score": parent["score"] * 0.5,   # supporting context, not a primary hit
            "rank": len(results) + len(extras) + 1,
            "neighbor_of": f"{source}#{parent['point'].payload.get('chunk')}",
        })

    if extras:
        print(f"  🔗 Added {len(extras)} adjacent chunk(s) for decision/outcome continuity "
              f"(1 batched lookup)")
    return results + extras


# ── Helper: Detect the language the answer should be written in ─
# Hindi written in Roman script ("mujhe chutti ke niyam batao") is extremely common
# from these users, and a Devanagari-vs-Latin character count calls it English —
# so the user asked in Hinglish and got a formal English answer back. These tokens
# are Hindi function/question words that are rare or absent in English prose, so
# their presence in a Latin-script query is a reliable Hinglish signal.
# Deliberately EXCLUDED as too ambiguous in English: the, me, par, or, is, do, to,
# he, we, so, ka/ki/ke (too short, appear inside other tokens).
_HINGLISH_TOKENS = {
    "kya", "kyu", "kyun", "kyon", "kaise", "kaisa", "kaisi", "kaun", "kaunsa", "konsa",
    "kaunsi", "konsi", "kab", "kahan", "kitna", "kitne", "kitni", "hai", "hain", "hoga",
    "hogi", "honge", "hona", "mujhe", "mera", "meri", "mere", "hume", "humein", "aap",
    "aapka", "aapko", "aapki", "chahiye", "batao", "bataye", "bataiye", "batana",
    "nahi", "nahin", "karna", "karne", "karta", "karti", "karte", "sakta", "sakti",
    "sakte", "raha", "rahi", "rahe", "gaya", "gayi", "diya", "kiya", "milega", "milegi",
    "milta", "jankari", "niyam", "adhikari", "karmchari", "chutti", "vetan", "aavedan",
    "prakriya", "yadi", "lekin", "aur", "wala", "wale", "wali", "hoti", "hota", "hote",
}
_HINGLISH_MIN_HITS = 2      # 2+ markers: one stray loanword should not flip an English query


def detect_query_language(query):
    """Return 'hi' (Devanagari), 'hinglish' (Roman-script Hindi), or 'en'.

    The corpus is mixed Hindi/English and bge-m3 retrieves across both, so the
    retrieved chunks are a poor signal for what language the *user* wants back.
    Left to itself the LLM mirrors the chunks, which means an English question
    about a Hindi document comes back in Hindi (and vice versa). Decide from the
    question instead.
    """
    devanagari = sum(1 for ch in query if 'ऀ' <= ch <= 'ॿ')
    latin = sum(1 for ch in query if ch.isascii() and ch.isalpha())
    if devanagari > latin:
        return "hi"
    words = set(re.findall(r"[a-z]+", query.lower()))
    if len(words & _HINGLISH_TOKENS) >= _HINGLISH_MIN_HITS:
        # A romanized-Hindi question ("2016 me konsa niyam add hua") IS asked in Hindi,
        # just typed in Roman. For a government audience answer in proper Devanagari
        # Hindi rather than Roman-script Hinglish — the demo wants clean Hindi/English,
        # not a mixed register. (Flip this to "hinglish" to restore Roman replies.)
        return "hi"
    return "en"


# How each detected language is enforced. The model reliably drifts into the language
# of the retrieved chunks, so the rule is stated THREE times per request: in the system
# prompt, as the last line of the user message, and as an explicit first-sentence
# constraint. Repetition is what actually holds it — a single rule buried above several
# thousand tokens of Hindi context loses.
_LANGUAGE_RULES = {
    "hi": {
        "label": "Hindi (Devanagari)",
        "rule": ("LANGUAGE (ABSOLUTE): The user wrote in Hindi. Write EVERY sentence of "
                 "your answer in Hindi using Devanagari script — including headings, "
                 "labels and explanations — even though most source documents are in "
                 "English. Translate what you quote into Hindi, keeping the original "
                 "English alongside exact quotes. Keep official names, order numbers, "
                 "department names and filenames in their original form. Do NOT answer "
                 "in English. Do NOT switch language part-way."),
        "closing": "उत्तर पूरी तरह हिंदी (देवनागरी) में दें, चाहे दस्तावेज़ किसी भी भाषा में हों।",
    },
    "hinglish": {
        "label": "Hinglish (Hindi written in Roman/English script)",
        "rule": ("LANGUAGE (ABSOLUTE): The user wrote in Hinglish — Hindi using Roman "
                 "letters, e.g. 'mujhe chutti ke niyam batao'. Reply in the SAME style: "
                 "Hindi sentences written in Roman script, NOT Devanagari and NOT formal "
                 "English. Example of the required style: 'Is order ke according, 15 din "
                 "ki casual leave allowed hai.' Keep technical terms, official names, "
                 "order numbers and filenames as they are. Do NOT reply in Devanagari. "
                 "Do NOT reply in pure English."),
        "closing": "Jawab Hinglish (Roman script) mein hi dijiye, Devanagari ya pure English mein nahi.",
    },
    "en": {
        "label": "English",
        "rule": ("LANGUAGE (ABSOLUTE): The user wrote in English. Write EVERY sentence of "
                 "your answer in English — including headings and explanations — even "
                 "though many source documents are in Hindi. Translate any Hindi you "
                 "quote into English, keeping the original Hindi alongside exact quotes. "
                 "Do NOT answer in Hindi. Do NOT switch language part-way."),
        "closing": "Answer entirely in English, regardless of the language of the source documents.",
    },
}


def answer_language_ok(answer, expected_lang):
    """Does `answer` appear to be written in `expected_lang`?

    Prompting alone does not hold. With a couple of dozen Hindi passages in context
    the model mirrors the sources and answers in Hindi even when the rule says
    English three times over — observed on a real query. So the result is CHECKED
    rather than trusted: finish_answer logs a mismatch (not retried — the streamed
    answer is already on the reader's screen).

    Majority script, not mere presence: an English answer legitimately quotes
    Devanagari, so a few Hindi words must not count as a violation.
    """
    if not answer:
        return True
    dev = sum(1 for ch in answer if 'ऀ' <= ch <= 'ॿ')
    latin = sum(1 for ch in answer if ch.isascii() and ch.isalpha())
    if expected_lang == "hi":
        return dev >= latin
    # en and hinglish are both Latin-script outputs; Hinglish is scored as Latin
    # here, so this only catches the "asked in English, answered in Devanagari" flip.
    return latin >= dev


def language_directive(query):
    """(lang, rule, closing_line, label) for the language the answer must use.

    One place decides this, so the single-shot answer and every thinking-mode step
    enforce language identically.
    """
    lang = detect_query_language(query)
    spec = _LANGUAGE_RULES.get(lang, _LANGUAGE_RULES["en"])
    return lang, spec["rule"], spec["closing"], spec["label"]


# ── Bilingual glossary for cross-lingual retrieval ─────────────
# The corpus mixes Hindi (Devanagari) and English documents. An English query must
# also reach the Hindi text and vice-versa: a "finance minister" question has to find
# "वित्त मंत्री श्री ओ.पी. चौधरी", a "guarantees" question has to find "प्रत्याभूति /
# गारंटी". bge-m3 is multilingual but does not bridge this reliably for a rare, dense
# gov corpus. This glossary augments BOTH the dense query (append translations so the
# embedding sits between the two languages) and the lexical arm (MatchText the other
# script's term). Terms are matched case-insensitively; multi-word keys first.
_GLOSSARY_EN_HI = {
    "finance minister": ["वित्त मंत्री"], "chief minister": ["मुख्यमंत्री"],
    "minister": ["मंत्री"], "finance department": ["वित्त विभाग"], "finance": ["वित्त", "वित्तीय"],
    "budget": ["बजट"], "fiscal deficit": ["राजकोषीय घाटा"], "fiscal": ["राजकोषीय"],
    "deficit": ["घाटा"], "surplus": ["अधिशेष"], "revenue": ["राजस्व"], "expenditure": ["व्यय"],
    "capital expenditure": ["पूंजीगत व्यय"], "capital": ["पूंजीगत"], "receipts": ["प्राप्तियां"],
    "guarantee": ["गारंटी", "प्रत्याभूति"], "guarantees": ["गारंटी", "प्रत्याभूति"],
    "social welfare": ["समाज कल्याण"], "scheme": ["योजना"],
    "schemes": ["योजनाओं", "योजना"], "rules": ["नियम"], "rule": ["नियम"], "act": ["अधिनियम"],
    "amendment": ["संशोधन"], "circular": ["परिपत्र"], "circulars": ["परिपत्र"],
    "notification": ["अधिसूचना"], "pension": ["पेंशन"], "grant": ["अनुदान"], "grants": ["अनुदान"],
    "demand": ["मांग"], "loan": ["ऋण"], "debt": ["ऋण"], "advance": ["अग्रिम"], "audit": ["लेखा परीक्षा"],
    "agriculture": ["कृषि"], "education": ["शिक्षा"], "health": ["स्वास्थ्य"], "employment": ["रोजगार"],
    "chhattisgarh": ["छत्तीसगढ़"],
    "financial year": ["वित्तीय वर्ष"], "provision": ["प्रावधान"], "allocation": ["आवंटन"],
    "vitta nirdesh": ["वित्त निर्देश"], "gsdp": ["सकल राज्य घरेलू उत्पाद"],
    "finance commission": ["वित्त आयोग"], "provident fund": ["भविष्य निधि"],
    "tax": ["कर"], "gst": ["जीएसटी"], "salary": ["वेतन"], "seniority": ["वरिष्ठता"],
    # administrative / GAD terms
    "compassionate appointment": ["अनुकंपा नियुक्ति"], "compassionate": ["अनुकंपा"],
    "appointment": ["नियुक्ति"], "transfer": ["स्थानांतरण"], "transfers": ["स्थानांतरण"],
    "reservation": ["आरक्षण"], "promotion": ["पदोन्नति"], "deputation": ["प्रतिनियुक्ति"],
    "holiday": ["सार्वजनिक अवकाश", "छुट्टी"], "holidays": ["सार्वजनिक अवकाश", "छुट्टी"], "establishment": ["स्थापना"],
    "gazette": ["राजपत्र"], "disabled": ["दिव्यांग"], "divyang": ["दिव्यांग"],
    "recruitment": ["भर्ती"], "committee": ["समिति"], "leave": ["अवकाश", "छुट्टी"], "general administration": ["सामान्य प्रशासन"],
    # finance / fund / accounts
    "consolidated fund": ["संचित निधि"], "contingency fund": ["आकस्मिकता निधि"],
    "public account": ["लोक लेखा"], "sinking fund": ["निक्षेप निधि", "ऋण शोधन निधि"],
    "treasury": ["कोषालय"], "sanction": ["स्वीकृति", "मंजूरी"], "allotment": ["आवंटन"],
    "outstanding liabilities": ["बकाया देयताएं"], "liability": ["देयता"], "liabilities": ["देयताएं"],
    "borrowing": ["उधार"], "interest": ["ब्याज"], "repayment": ["भुगतान", "चुकौती"],
    "arrears": ["बकाया"], "ways and means": ["साधन एवं उपाय"], "overdraft": ["अधिविकर्ष"],
    "disbursement": ["संवितरण"], "reimbursement": ["प्रतिपूर्ति"], "subsidy": ["अनुदान", "सब्सिडी"],
    "devolution": ["हस्तांतरण"], "transfer of funds": ["निधि अंतरण"], "budget estimate": ["बजट अनुमान"],
    "revised estimate": ["पुनरीक्षित अनुमान"], "actuals": ["वास्तविक"], "per capita income": ["प्रति व्यक्ति आय"],
    # administrative / service
    "cadre": ["संवर्ग"], "posting": ["पदस्थापना"], "suspension": ["निलंबन"],
    "disciplinary": ["अनुशासनात्मक"], "inquiry": ["जांच"], "retirement": ["सेवानिवृत्ति"],
    "gratuity": ["उपादान"], "increment": ["वेतनवृद्धि"], "pay scale": ["वेतनमान"],
    "dearness allowance": ["महंगाई भत्ता"], "allowance": ["भत्ता"], "civil services": ["सिविल सेवा"],
    "gazetted": ["राजपत्रित"], "seniority list": ["वरिष्ठता सूची"], "vacancy": ["रिक्ति"],
    # general government
    "notification": ["अधिसूचना"], "resolution": ["संकल्प"], "memorandum": ["ज्ञापन"],
    "proceedings": ["कार्यवाही"], "minutes": ["कार्यवृत्त"], "agenda": ["कार्यसूची"],
    "decision": ["निर्णय"], "proposal": ["प्रस्ताव"], "approval": ["अनुमोदन", "स्वीकृति"],
    "guideline": ["दिशानिर्देश"], "guidelines": ["दिशानिर्देश"], "policy": ["नीति"],
    "annual report": ["वार्षिक प्रतिवेदन", "प्रतिवेदन"], "report": ["प्रतिवेदन", "रिपोर्ट"],
    "director": ["संचालक", "निदेशक"], "secretary": ["सचिव"], "commissioner": ["आयुक्त"],
    "collector": ["कलेक्टर"], "panchayat": ["पंचायत"], "municipal": ["नगरीय"],
    "local body": ["स्थानीय निकाय"], "local bodies": ["स्थानीय निकायों"], "urban": ["शहरी", "नगरीय"],
    "rural": ["ग्रामीण"], "women": ["महिला"], "children": ["बच्चे", "बाल"], "farmer": ["किसान", "कृषक"],
    "tribal": ["आदिवासी", "जनजातीय"], "electricity": ["बिजली", "विद्युत"], "irrigation": ["सिंचाई"],
    "road": ["सड़क"], "water supply": ["जल आपूर्ति"], "electronics": ["इलेक्ट्रॉनिक्स"],
    "information technology": ["सूचना प्रौद्योगिकी"], "e-governance": ["ई-गवर्नेंस"],
    # missions / flagship programmes (CG budget "मुख्यमंत्री मिशन" set incl. AI Mission)
    "mission": ["मिशन"], "missions": ["मिशन"], "chief minister": ["मुख्यमंत्री"],
    "mukhyamantri": ["मुख्यमंत्री"], "cm": ["मुख्यमंत्री"],
    "artificial intelligence": ["आर्टिफिशियल इंटेलिजेंस", "एआई"],
    "startup": ["स्टार्ट-अप", "स्टार्टअप"], "start-up": ["स्टार्ट-अप", "स्टार्टअप"],
    "tourism": ["पर्यटन"], "sports": ["खेल"], "infrastructure": ["अधोसंरचना", "बुनियादी ढांचा"],
    "skill development": ["कौशल विकास"], "innovation": ["नवाचार"],
    # GAD: abbreviations, kinds of leave, service matters (an English word in the
    # question finds the Hindi document)
    "gad": ["सामान्य प्रशासन विभाग"], "general administration department": ["सामान्य प्रशासन विभाग"],
    "da": ["महंगाई भत्ता"], "rti": ["सूचना का अधिकार"], "right to information": ["सूचना का अधिकार"],
    "sc": ["अनुसूचित जाति"], "st": ["अनुसूचित जनजाति"], "obc": ["अन्य पिछड़ा वर्ग"],
    "scheduled caste": ["अनुसूचित जाति"], "scheduled tribe": ["अनुसूचित जनजाति"],
    "other backward classes": ["अन्य पिछड़ा वर्ग"],
    "earned leave": ["अर्जित अवकाश"], "casual leave": ["आकस्मिक अवकाश"],
    "medical leave": ["चिकित्सा अवकाश"], "maternity leave": ["प्रसूति अवकाश", "मातृत्व अवकाश"],
    "child care leave": ["संतान पालन अवकाश"], "public holiday": ["सार्वजनिक अवकाश"],
    "public holidays": ["सार्वजनिक अवकाश"], "age limit": ["आयु सीमा"], "maximum age": ["अधिकतम आयु"],
    "age relaxation": ["आयु सीमा में छूट"], "relaxation": ["छूट"], "eligibility": ["पात्रता"],
    "eligible": ["पात्र", "पात्रता"], "departmental enquiry": ["विभागीय जांच"],
    "departmental inquiry": ["विभागीय जांच"], "contractual": ["संविदा"], "contract employees": ["संविदा"],
    "regularisation": ["नियमितीकरण"], "regularization": ["नियमितीकरण"],
    "property return": ["अचल संपत्ति"], "property": ["संपत्ति"], "training": ["प्रशिक्षण"],
    "workshop": ["कार्यशाला"], "independence day": ["स्वतंत्रता दिवस"], "republic day": ["गणतंत्र दिवस"],
    "award": ["पुरस्कार", "सम्मान"], "awards": ["पुरस्कार", "सम्मान"],
    "persons with disabilities": ["दिव्यांग", "दिव्यांगजन"], "disability": ["दिव्यांग"],
    "ban": ["प्रतिबंध"], "restriction": ["प्रतिबंध"], "civil list": ["सिविल सूची", "पदक्रम सूची"],
    "gradation list": ["पदक्रम सूची"], "governor": ["राज्यपाल"],
    "governor's address": ["राज्यपाल का अभिभाषण", "अभिभाषण"],
    "governor address": ["राज्यपाल का अभिभाषण", "अभिभाषण"],
    "transfer policy": ["स्थानांतरण नीति"],
}
# Hindi typed in Roman letters (Hinglish) -> the Hindi the documents use. Kept apart from
# the English list: a Hindi question must not get these Roman spellings added back.
_GLOSSARY_HINGLISH = {
    "anukampa niyukti": ["अनुकंपा नियुक्ति"], "anukampa": ["अनुकंपा"], "niyukti": ["नियुक्ति"],
    "tabadla": ["स्थानांतरण", "तबादला"], "tabadle": ["स्थानांतरण", "तबादला"],
    "tabadlon": ["स्थानांतरण", "तबादला"], "sthanantaran": ["स्थानांतरण"],
    "chhutti": ["अवकाश", "छुट्टी"], "chutti": ["अवकाश", "छुट्टी"], "avkash": ["अवकाश"],
    "awkash": ["अवकाश"], "arjit avkash": ["अर्जित अवकाश"], "aakasmik avkash": ["आकस्मिक अवकाश"],
    "vetan": ["वेतन"], "bhatta": ["भत्ता"], "mahangai bhatta": ["महंगाई भत्ता"],
    "mehngai bhatta": ["महंगाई भत्ता"], "padonnati": ["पदोन्नति"], "nilamban": ["निलंबन"],
    "sevanivritti": ["सेवानिवृत्ति"], "sevanivrutti": ["सेवानिवृत्ति"], "aarakshan": ["आरक्षण"],
    "arakshan": ["आरक्षण"], "bharti": ["भर्ती"], "pratibandh": ["प्रतिबंध"],
    "paripatra": ["परिपत्र"], "adhisuchna": ["अधिसूचना"], "adhisuchana": ["अधिसूचना"],
    "niyam": ["नियम"], "adhiniyam": ["अधिनियम"], "patrata": ["पात्रता"],
    "aayu seema": ["आयु सीमा"], "ayu seema": ["आयु सीमा"], "aayu sima": ["आयु सीमा"],
    "chhoot": ["छूट"], "chhut": ["छूट"], "vibhagiya jaanch": ["विभागीय जांच"],
    "vibhagiya janch": ["विभागीय जांच"], "jaanch": ["जांच"], "janch": ["जांच"],
    "sampatti": ["संपत्ति"], "achal sampatti": ["अचल संपत्ति"], "vivran": ["विवरण"],
    "samvida": ["संविदा"], "niyamitikaran": ["नियमितीकरण"], "rajyapal": ["राज्यपाल"],
    "abhibhashan": ["अभिभाषण"], "samaroh": ["समारोह"], "prashikshan": ["प्रशिक्षण"],
    "sarkari naukri": ["शासकीय सेवा"], "puraskar": ["पुरस्कार"], "samman": ["सम्मान"],
    "varishthata suchi": ["वरिष्ठता सूची"], "varishthata": ["वरिष्ठता"], "karmayogi": ["कर्मयोगी"],
}
# reverse map (Hindi phrase -> English), built once: from the English list only (no
# Roman-Hindi spellings), and a plural only when its singular is not there already
_GLOSSARY_HI_EN = {}
for _en, _his in _GLOSSARY_EN_HI.items():
    if len(_en) <= 3 and _en.isalpha():          # "da", "sc", "gad": no abbreviation back
        continue
    for _hi in _his:
        have = _GLOSSARY_HI_EN.setdefault(_hi, [])
        if not (_en.endswith("s") and _en[:-1] in have) and _en + "s" not in have:
            have.append(_en)
# English and Hinglish keys, longest-first so multi-word phrases match before their parts
_GLOSSARY_EN_HI_ALL = {**_GLOSSARY_EN_HI, **_GLOSSARY_HINGLISH}
_GLOSSARY_EN_KEYS = sorted(_GLOSSARY_EN_HI_ALL, key=len, reverse=True)
_GLOSSARY_HI_KEYS = sorted(_GLOSSARY_HI_EN, key=len, reverse=True)
# English keys matched on WORD BOUNDARIES, not raw substrings, so short/embedded keys
# don't misfire — e.g. "mission" must not match inside "commission", "cm" not inside a
# longer token. Boundary = not flanked by another latin letter or digit (punctuation and
# hyphens are fine, so "start-up" and "e-governance" still match).
_GLOSSARY_EN_RE = [
    (re.compile(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])"), k)
    for k in _GLOSSARY_EN_KEYS
]


# Hindi keys as whole words, with the common plural/oblique endings: a substring match
# found "कर" (tax) inside "करना" and "नियम" inside "नियमितीकरण".
_HI_END = "(?:ओं|ों|ें|ो|े|ियों|ियाँ|ाओं)?"
_GLOSSARY_HI_RE = [
    (k, re.compile(r"(?<![\u0900-\u097F])" + re.escape(k) + _HI_END + r"(?![\u0900-\u097F])"))
    for k in _GLOSSARY_HI_KEYS
]


def glossary_translations(text):
    """Return translations of any glossary terms found in `text` (both directions),
    deduped, preserving order. Empty if none match. A matched phrase is used up:
    "compassionate appointment" adds अनुकंपा नियुक्ति once, not again for each word."""
    low = text.lower()
    rest = text
    out, seen = [], set()
    for rx, k in _GLOSSARY_EN_RE:
        if rx.search(low):
            low = rx.sub(" ", low)
            for tr in _GLOSSARY_EN_HI_ALL[k]:
                if tr not in seen:
                    seen.add(tr); out.append(tr)
    for k, rx in _GLOSSARY_HI_RE:
        if rx.search(rest):
            rest = rx.sub(" ", rest)
            for tr in _GLOSSARY_HI_EN[k]:
                if tr.lower() not in seen:
                    seen.add(tr.lower()); out.append(tr)
    return out


# ── Helper: Expand query into multiple perspectives ────────────
def expand_query(original_query):
    """Cross-lingual query expansion. The old version appended English keywords
    ("approval", "agenda") which did nothing for a bilingual gov corpus. Instead we
    add the query augmented with Hindi/English translations of its key terms, so the
    dense embedding sits between both languages and finds same-topic chunks written
    in the OTHER script (the #1 cause of "answers in Hindi but not English")."""
    variations = [original_query]
    tr = glossary_translations(original_query)
    if tr:
        # append translations so the embedding covers both languages
        variations.append(f"{original_query} {' '.join(tr)}")
    seen, unique = set(), []
    for v in variations:
        if v.lower() not in seen:
            seen.add(v.lower()); unique.append(v)
    return unique[:MAX_QUERY_VARIATIONS]

# ── Helper: Encode queries (one model call for the whole batch) ─
def encode_queries(queries):
    """Encode every query variation in ONE model.encode call.

    bge-m3 has a fixed per-call cost (tokenise, move to device, forward pass setup)
    that dwarfs the marginal cost of extra rows, so encoding 5 variations together
    is far cheaper than 5 separate calls. Returns [(dense_list, sparse_dict), ...]
    aligned with `queries`; entries that failed to encode are None.
    """
    ensure_models_loaded()
    if not queries:
        return []

    t0 = time.time()
    with ENCODE_LOCK:
        enc = model.encode(queries, batch_size=len(queries), max_length=MAX_LENGTH)
    if not enc or "dense_vecs" not in enc:
        return [None] * len(queries)

    dense_vecs = enc.get("dense_vecs")
    if dense_vecs is None or len(dense_vecs) == 0:
        return [None] * len(queries)

    # lexical_weights is a per-row list; older/edge builds hand back a bare dict for
    # a single row, so normalise both shapes into "one sparse dict per query".
    lex = enc.get("lexical_weights")
    if isinstance(lex, dict):
        lex = [lex]
    elif not isinstance(lex, list):
        lex = []

    out = []
    for i in range(len(queries)):
        if i >= len(dense_vecs):
            out.append(None)
            continue
        try:
            sparse = dict(lex[i]) if i < len(lex) and lex[i] is not None else {}
        except (TypeError, ValueError):
            sparse = {}
        out.append((dense_vecs[i].tolist(), sparse))

    print(f"  🧮 Encoded {len(queries)} query variation(s) in one batch ({time.time() - t0:.2f}s)")
    if qd.active():
        qd.record('Query embeddings', {'model': 'BAAI/bge-m3', 'max_length': MAX_LENGTH,
                  'fp16': _USE_FP16, 'queries': queries, 'vectors': out})
    return out


# ── Helper: Single query retrieval ─────────────────────────────
def perform_single_retrieval(query):
    """Retrieve for ONE query (the USE_MULTI_QUERY=False path)."""
    try:
        qdrant = ensure_qdrant_client()
        encoded = encode_queries([query])
        if not encoded or encoded[0] is None:
            return None
        query_dense, query_sparse = encoded[0]

        # Dense search. The sparse arm only re-scores points that dense already
        # returned, so this limit is also the ceiling on what sparse can ever
        # rescue — keep it well above num_context.
        dense_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_dense,
            limit=DENSE_CANDIDATE_LIMIT,
            with_payload=True,
        )
        if qd.active():
            qd.record('Dense search results', {'query': query, 'collection': COLLECTION_NAME,
                      'chunks': qd.chunks([{'point': p, 'score': p.score} for p in
                                          (getattr(dense_results, 'points', None) or [])])})

        if dense_results is None or not dense_results.points:
            return None

        return {
            "dense_results": dense_results,
            "dense_scores": [(p.id, p.score) for p in dense_results.points],
            "query_sparse": query_sparse
        }

    except Exception as e:
        print(f"  Single retrieval error: {e}")
        _RETRIEVAL_ERROR.set(f"{type(e).__name__}: {e}")
        return None

# ── Helper: Multi-query retrieval ──────────────────────────────
def multi_query_retrieval(query):
    """Retrieve results using multiple query variations and merge them.
    
    Benefits:
    - Captures different aspects of the query
    - Better coverage of semantically related documents
    - More robust to query phrasing variations
    """
    _mark_time("MULTI_QUERY_RETRIEVAL")
    ensure_models_loaded()
    
    if not USE_MULTI_QUERY:
        qd.record('Query expansion skipped', {'query': query, 'variations': [query], 'enabled': False})
        # Fall back to single query
        result = perform_single_retrieval(query)
        if result is None:
            return None, [], {}
        return result["dense_results"], result["dense_scores"], result["query_sparse"]
    
    # Expand query into multiple variations
    query_variations = expand_query(query)
    qd.record('Query expansion', {'query': query, 'variations': query_variations,
                                'method': 'bilingual glossary', 'collection': COLLECTION_NAME})
    print(f"  🔍 Searching with {len(query_variations)} query variations...")

    # ONE encode call for every variation, then ONE batched Qdrant round trip —
    # this used to be N encodes and N searches issued strictly in sequence.
    encoded = encode_queries(query_variations)
    live = [(i, e) for i, e in enumerate(encoded) if e is not None]
    if not live:
        print("  Error: No query variation could be encoded.")
        return None, [], {}

    qdrant = ensure_qdrant_client()
    try:
        t0 = time.time()
        responses = qdrant.query_batch_points(
            collection_name=COLLECTION_NAME,
            requests=[
                qmodels.QueryRequest(query=dense, limit=DENSE_CANDIDATE_LIMIT, with_payload=True)
                for _, (dense, _sparse) in live
            ],
        )
        print(f"  🔎 Batched {len(live)} dense search(es) into one call ({time.time() - t0:.2f}s)")
    except Exception as e:
        # Any batch-API incompatibility falls back to the original sequential path
        # rather than failing the query outright.
        print(f"  ⚠ Batched search unavailable ({e}); falling back to sequential")
        qd.record('Dense batch fallback', {'error': str(e), 'fallback': 'sequential search'})
        responses = None

    all_dense_results = {}   # {point_id: point}
    aggregated_scores = {}   # {point_id: weighted score sum}
    all_sparse_queries = {}

    def _absorb(i, points, scored, sparse):
        """Merge one variation's hits: each chunk keeps its BEST score. (Earlier
        variations used to weigh more, 1/(i+1): a Hindi document only the translated
        variation found came in at half its score, under unrelated English hits.)"""
        for point in points:
            all_dense_results[point.id] = point
        for point_id, score in scored:
            aggregated_scores[point_id] = max(aggregated_scores.get(point_id, 0.0), score)
        return sparse or all_sparse_queries

    if responses is not None:
        for (i, (_dense, sparse)), resp in zip(live, responses):
            points = getattr(resp, "points", None) or []
            if qd.active():
                qd.record('Dense search results', {'query': query_variations[i], 'variation': i + 1,
                          'collection': COLLECTION_NAME,
                          'chunks': qd.chunks([{'point': p, 'score': p.score} for p in points])})
            all_sparse_queries = _absorb(i, points, [(p.id, p.score) for p in points], sparse)
    else:
        for i, q_variant in enumerate(query_variations):
            r = perform_single_retrieval(q_variant)
            if r is None:
                continue
            all_sparse_queries = _absorb(i, r["dense_results"].points,
                                         r["dense_scores"], r["query_sparse"])

    if not all_dense_results:
        print("  Error: No results from multi-query retrieval.")
        return None, [], {}

    # Lightweight stand-in matching the .points shape the callers expect.
    class MergedResults:
        __slots__ = ("points",)

        def __init__(self, points):
            self.points = points

    dense_results = MergedResults(list(all_dense_results.values()))
    qd.ranking('Merged dense candidates', sorted(aggregated_scores.items(), key=lambda s: -s[1]),
               all_dense_results, weighting='best score across variations', query=query)

    _mark_time("MULTI_QUERY_RETRIEVAL")
    return dense_results, list(aggregated_scores.items()), all_sparse_queries

# ── Helper: Rerank the hybrid candidates ───────────────────────
def rerank_candidates(query, candidate_points, min_score=None, min_k=None, variations=None):
    """[(point_id, score)] best first, scored by the cross-encoder (0-1).

    The reranker reads the chunk's text without its stored header. Candidates below
    `min_score` are unrelated to the question and dropped — but at least `min_k`
    survive, so an unusual phrasing never leaves the answer with nothing.

    `variations`: the search's query variations (expand_query — the question plus its
    Hindi/English glossary translations). Each candidate keeps its BEST score over
    them: the cross-encoder judges "child care leave kisko milti hai" against a Hindi
    order badly and "... संतान पालन अवकाश" well — the same reason the search itself
    uses the variations.
    """
    min_score = RERANK_MIN_SCORE if min_score is None else min_score
    min_k = RERANK_MIN_K if min_k is None else min_k
    if not candidate_points:
        return []
    texts = [clean_chunk_text(readable_text(p.payload)) or p.payload.get("text", "") for p in candidate_points]
    queries = list(dict.fromkeys([query] + [v for v in (variations or []) if v]))
    pairs = [[q, t] for q in queries for t in texts]
    _mark_time("RERANKING")
    try:
        flat = load_reranker().compute_score(pairs, normalize=True)
    finally:
        _mark_time("RERANKING")
    if not isinstance(flat, list):
        flat = [flat]
    n = len(texts)
    scores = [max(flat[i + k * n] for k in range(len(queries))) for i in range(n)]
    ranked = sorted(zip((p.id for p in candidate_points), (float(x) for x in scores)),
                    key=lambda t: t[1], reverse=True)
    if qd.active():
        qd.ranking('All reranker scores before threshold filtering', ranked,
                   {p.id: p for p in candidate_points}, query=query,
                   model='BAAI/bge-reranker-v2-m3', min_score=min_score, min_k=min_k,
                   inputs=pairs)
    kept = [t for i, t in enumerate(ranked) if t[1] >= min_score or i < min_k]
    print(f"  📊 Reranked {len(pairs)} candidates: kept {len(kept)}, "
          f"top {', '.join(f'{x:.2f}' for _, x in ranked[:5])}")
    return kept

# ── Helper: Hybrid search using RRF ───────────────────────────
def hybrid_search(dense_scores, sparse_scores, alpha=0.5, k=60):
    """Combine dense and sparse scores using RRF (Reciprocal Rank Fusion)."""
    rrf_scores = {}
    
    # Add dense scores (RRF formula: 1 / (k + rank))
    for rank, (point_id, score) in enumerate(dense_scores):
        rrf_scores[point_id] = alpha / (k + rank + 1)
    
    # Add sparse scores
    for rank, (point_id, score) in enumerate(sparse_scores):
        if point_id not in rrf_scores:
            rrf_scores[point_id] = 0
        rrf_scores[point_id] += (1 - alpha) / (k + rank + 1)
    
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

# ── Lexical (keyword / exact-match) retrieval arm ──────────────
# Dense+sparse embeddings are great for "summarise X" but weak at "find this exact
# thing": a reference/order number, a document number (e.g. Vitta Nirdesh-12), a
# specific figure buried in a table. Those tokens ARE in the OCR text, so a literal
# full-text lookup finds them when semantic ranking does not. This arm runs a
# Qdrant full-text (MatchText) search over the chunk `text` and merges the hits into
# the candidate set so exact matches always reach the answer.
_TEXT_INDEX_READY = False
# words too common to be worth a literal lookup (EN + a few romanised-Hindi query words)
_LEX_STOP = set((
    "the a an of to in for on at by is are was were be been being and or not no "
    "what which who whom whose how when where why do does did done tell me about "
    "all list give show provide detail details regarding year years this that these "
    "those it its as with from into over under between please can could would should "
    "kya hai kaun kab kaha kaise ke ki ka bare batao btao mujhe kar karo").split())


def _ensure_text_index(qdrant):
    """The payload indexes the search filters on — full text on `text`, keywords on
    source / chunk / doc_id (qdrant_store.ensure_indexes) — once per process."""
    global _TEXT_INDEX_READY
    if _TEXT_INDEX_READY:
        return
    qdrant_store.ensure_indexes(qdrant, COLLECTION_NAME)
    _TEXT_INDEX_READY = True


# What may push a passage in AHEAD of the semantic ranking: an exact identifier —
# a number of 3+ digits that is not a bare year (an order / reference / phone number),
# a code with a digit and a separator ("F-4-1/2023", "07768-299002"), or an acronym of
# 3+ capitals as typed (GSDP, PMAY). An ordinary word never does, however rare: in a
# mostly-Hindi corpus every English or Hinglish word is "rare" ("child", "care",
# "kisko", "milti"), and the old rule (any term in <= LEX_RARE_FREQ chunks) put up to
# four unrelated English notices in front of the real evidence for such questions
# (live GAD traces, 2026-10-21). LEXICAL_FORCE=rare_words restores the old rule.
LEXICAL_FORCE = os.getenv("LEXICAL_FORCE", "identifiers").strip().lower()
# segments joined by - / . with a digit somewhere: F-4-1/2023, क्र.2/4/2026, 07768-299002
_CODE_RE = re.compile(r"(?<![A-Za-z0-9ऀ-ॿ])(?=[A-Za-z0-9ऀ-ॿ./-]*\d)[A-Za-z0-9ऀ-ॿ]+(?:[-/.][A-Za-z0-9ऀ-ॿ]+)+")


def is_identifier(term):
    t = str(term or "").strip(",")
    if any(ch.isdigit() for ch in t):
        if _CODE_RE.fullmatch(t):
            return True
        digits = t.replace(",", "")
        return digits.isdigit() and len(digits) >= 3 and not re.fullmatch(r"(?:19|20)\d{2}", digits)
    return len(t) >= 3 and t.isascii() and t.isalpha() and t.isupper()


def lexical_forces(hit):
    """May this lexical hit go in AHEAD of the semantic ranking? Only for a rare
    identifier it contains (LEXICAL_FORCE=rare_words: any rare term — the old rule)."""
    rarest = hit.get("min_freq") if LEXICAL_FORCE == "rare_words" else hit.get("ident_freq")
    return rarest is not None and rarest <= LEX_RARE_FREQ


def lexical_terms(query):
    """Distinctive tokens worth a literal lookup: content words (>=3 chars, not
    stopwords) and numeric / code tokens (reference numbers, demand numbers, years,
    document numbers). Handles Devanagari too. Single digits (e.g. "2+2") are dropped."""
    terms, seen = [], set()
    for tok in re.findall(r"[0-9][0-9,]*|[A-Za-zऀ-ॿ][A-Za-z0-9ऀ-ॿ]*", query):
        t = tok.strip(",")
        low = t.lower()
        if low in _LEX_STOP or low in seen:
            continue
        if t.replace(",", "").isdigit():
            if len(t.replace(",", "")) < 2:     # skip bare single digits
                continue
        elif len(t) < 3:
            # Most 2-char tokens are noise, but short ACRONYMS carry the whole query:
            # "AI", "IT", "HR", "5G". Keep a 2-char token when it is an acronym
            # (upper-case in the original) or alphanumeric (has a digit); drop the rest.
            if not (t.isupper() or any(ch.isdigit() for ch in t)):
                continue
        seen.add(low)
        terms.append(t)
    # Whole codes as typed ("F-4-1/2023"): the full-text index matches all of their parts.
    for code in _CODE_RE.findall(query or ""):
        if code.lower() not in seen:
            seen.add(code.lower())
            terms.insert(0, code)
    # Cross-lingual: also search the OTHER script's word for any glossary term, so an
    # English query literally matches Hindi chunks (e.g. "minister" also hunts "मंत्री",
    # "guarantee" also hunts "प्रत्याभूति"). These are single words for MatchText.
    for tr in glossary_translations(query):
        for w in tr.split():
            if w.lower() not in seen:
                seen.add(w.lower())
                terms.append(w)
    return terms[:16]


# A term rarer than this many chunks is "distinctive" — a reference number, a scheme
# code, "GSDP", a specific scheme name. A single distinctive term is enough to trust a
# lexical hit. Common words ("budget", "department", "IT") sit far above this, so on
# their own they cannot promote a chunk ahead of the semantic ranking (see the strong-
# match rule in retrieve_context).
LEX_RARE_FREQ = 40

# Master switch for the lexical (full-text exact-match) arm. On by default because it is
# the only thing that reliably surfaces exact reference numbers / scheme codes that the
# embedding ranks low. Set LEXICAL_ARM=0 to fall back to pure dense+sparse hybrid — kept
# switchable so it can be A/B'd or disabled instantly if it ever misbehaves in prod.
LEXICAL_ARM_ENABLED = os.getenv("LEXICAL_ARM", "1").strip().lower() not in ("0", "false", "no", "off")


def lexical_search(query, limit=12):
    """Keyword/exact-match retrieval via the full-text index. Ranks chunks by
    IDF-weighted term matches: a rare term (a reference number, "fiscal", "GSDP")
    is worth far more than a term that appears in thousands of chunks ("budget",
    "2026"), so the distinctive chunk surfaces even if it matches fewer total terms.

    Returns a list of dicts: {point, score, matched, min_freq} where `matched` is the
    number of DISTINCT query terms the chunk contains and `min_freq` is the corpus
    frequency of its rarest matched term. The caller uses those to reject weak hits
    (a chunk that matched only one common word) before injecting them ahead of the
    semantic arm."""
    try:
        qdrant = ensure_qdrant_client()
        _ensure_text_index(qdrant)
        terms = lexical_terms(query)
        if not terms:
            return []
        hits = {}   # point_id -> {score, point, matched, min_freq, ident_freq}
        for term in terms:
            ident = is_identifier(term)
            cond = qmodels.Filter(must=[
                qmodels.FieldCondition(key="text", match=qmodels.MatchText(text=term))])
            try:
                freq = qdrant.count(COLLECTION_NAME, count_filter=cond, exact=False).count
            except Exception:
                freq = 0
            if freq == 0:
                continue
            # IDF-ish weight: rare terms dominate. A reference number (freq≈1) ~= 3.0;
            # a term in ~thousands of chunks ~= 0.1. Skip near-ubiquitous terms.
            weight = 1.0 / math.log(2 + freq)
            fetch = 60 if freq <= 60 else 30      # pull more of a rare term's hits
            try:
                pts, _ = qdrant.scroll(COLLECTION_NAME, scroll_filter=cond,
                                       limit=fetch, with_payload=True)
            except Exception:
                continue
            for p in pts:
                h = hits.get(p.id)
                if h is None:
                    h = hits[p.id] = {"score": weight, "point": p,
                                      "matched": 1, "min_freq": freq, "ident_freq": None}
                else:
                    h["score"] += weight
                    h["matched"] += 1
                    h["min_freq"] = min(h["min_freq"], freq)
                if ident:   # the rarest IDENTIFIER this chunk contains (None: none)
                    h["ident_freq"] = freq if h["ident_freq"] is None else min(h["ident_freq"], freq)
        ranked = sorted(hits.values(), key=lambda h: h["score"], reverse=True)
        if ranked:
            top = ranked[0]
            print(f"  🔑 lexical arm: {len(terms)} term(s) {terms} → {len(ranked)} chunk(s), "
                  f"top score {top['score']:.2f} (matched {top['matched']}, "
                  f"rarest freq {top['min_freq']})")
        return ranked[:limit]
    except Exception as e:
        print(f"  lexical_search error: {e}")
        return []


# ── Helper: Retrieve context ───────────────────────────────────
_RETRIEVAL_ERROR = contextvars.ContextVar("retrieval_error", default=None)


def last_retrieval_error():
    """Why the last retrieve_context in this request failed — None when it ran (even if
    it found nothing). Lets the chat say "the search failed" instead of "no documents"."""
    return _RETRIEVAL_ERROR.get()


def correction_context(query, *, date_question=None):
    import verified_memory
    return verified_memory.lookup(query, encode_queries, date_question=date_question)


def retrieve_context(query, num_context=8, *, date_question=None):
    corrections = correction_context(query, date_question=date_question)
    documents = _retrieve_documents(query, num_context=num_context)
    # Keep the existing document search intact. Verified memory is evidence for
    # the answer model to assess, never an unconditional cached response.
    results = corrections + list(documents or [])
    for rank, hit in enumerate(results, 1):
        hit['rank'] = rank
    return results or documents


def _retrieve_documents(query, num_context=8):
    """Retrieve context chunks for a query via hybrid (dense+sparse) search.

    Args:
        query: Search query
        num_context: Number of context results to return

    Returns:
        List of result dicts with 'point', 'score', 'rank'
    """
    _mark_time("RETRIEVE_CONTEXT")
    _RETRIEVAL_ERROR.set(None)
    try:
        # Step 1: Perform embedding-based retrieval
        print("🔍 Retrieving context...")
        
        # Multi-query retrieval (or single-query fallback)
        dense_results, aggregated_scores, query_sparse = multi_query_retrieval(query)
        
        if dense_results is None or not dense_results.points:
            print("Notice: No dense points in Qdrant; querying Temporal Knowledge Graph retriever...")
            try:
                import retrievers
                tg_retriever = retrievers.TemporalGraphRetriever()
                tg_resp = tg_retriever.retrieve(query, top_k=num_context)
                if tg_resp and tg_resp.results:
                    class GraphPoint:
                        def __init__(self, id, payload):
                            self.id = id
                            self.payload = payload
                    tg_results = []
                    for idx, c in enumerate(tg_resp.results, 1):
                        payload = {
                            'source': c.doc_id,
                            'chunk': 1,
                            'text': c.content,
                            'status': c.effective_status or c.status,
                            'order_no': c.order_no,
                            'doc_date': c.doc_date,
                            'title': c.title,
                            'superseded_by': c.superseded_by,
                            'amended_by': c.amended_by,
                        }
                        tg_results.append({
                            'point': GraphPoint(c.chunk_id, payload),
                            'score': c.score,
                            'rank': idx
                        })
                    _mark_time("RETRIEVE_CONTEXT")
                    return tg_results
            except Exception as tg_err:
                print(f"Temporal graph retrieval error: {tg_err}")
            print("Error: No results from retrieval.")
            qd.record('Retrieval stopped', {'query': query, 'reason': 'No dense candidates returned'})
            return None
        
        # Sort aggregated scores
        dense_scores = sorted(aggregated_scores, key=lambda x: x[1], reverse=True)
        
        # Sparse search (if available)
        if query_sparse:
            sparse_scores = sparse_search(query_sparse, dense_results.points, limit=DENSE_CANDIDATE_LIMIT)
            # Use configurable HYBRID_ALPHA
            hybrid_scores = hybrid_search(dense_scores, sparse_scores, alpha=HYBRID_ALPHA)
            print(f"  ⚡ Using hybrid search (α={HYBRID_ALPHA}: {int(HYBRID_ALPHA*100)}% dense, {int((1-HYBRID_ALPHA)*100)}% sparse)")
        else:
            hybrid_scores = [(pid, score) for rank, (pid, score) in enumerate(dense_scores)]
            print(f"  ⚡ Using dense-only search (sparse embeddings unavailable)")
        
        # Index the points once by id. Both the candidate collection below and the
        # result assembly further down look points up by id inside a loop; doing that
        # with a linear scan over dense_results.points made retrieval O(hits x points)
        # — with multi-query merging several hundred points that is tens of thousands
        # of comparisons per query, for what is a dict lookup.
        points_by_id = {p.id: p for p in dense_results.points}
        if query_sparse:
            qd.ranking('Sparse scores', sparse_scores, points_by_id, query=query)
        qd.ranking('Hybrid ranking', hybrid_scores, points_by_id, query=query,
                   sparse_available=bool(query_sparse), dense_weight=HYBRID_ALPHA)

        # Optional cross-encoder rerank (RERANK_ENABLED; on for the GPU VM): it
        # replaces the hybrid ORDER; everything below (exact-identifier lexical hits,
        # per-document cap, neighbours) works on the reranked order. A reranker
        # failure (e.g. out of GPU memory) falls back to the hybrid order.
        if RERANK_ENABLED and load_reranker() is not None:
            candidates = [points_by_id[pid] for pid, _ in hybrid_scores[:RERANK_CANDIDATES]
                          if pid in points_by_id]
            try:
                hybrid_scores = rerank_candidates(query, candidates,
                                                  variations=expand_query(query) if USE_MULTI_QUERY else None)
                qd.ranking('Reranked candidates', hybrid_scores, points_by_id, query=query,
                           model='BAAI/bge-reranker-v2-m3', min_score=RERANK_MIN_SCORE,
                           min_k=RERANK_MIN_K, max_length=RERANK_MAX_LENGTH)
            except Exception as e:      # noqa: BLE001 — never fail a query over the reranker
                print(f"  ⚠ reranker failed ({type(e).__name__}: {e}); using the hybrid order")
                qd.record('Reranker fallback', {'query': query, 'error': str(e), 'fallback': 'hybrid order'})
        else:
            print("⚡ Using hybrid search results directly (reranking disabled)")
            qd.record('Reranking skipped', {'query': query, 'enabled': RERANK_ENABLED})

        # What the question STATES — years, document types, topics, districts —
        # lifts matching chunks a little; a superseded order sinks unless the question
        # asks about history; an exact duplicate of another document is dropped.
        # Ranking only: nothing else is excluded, so a stated value that matches no
        # document can never turn into "not in the documents" (core/query_plan.py).
        # QUERY_BOOSTS=0 turns the lifting/sinking off (duplicates are still dropped) —
        # kept on only while scripts/eval_retrieval.py shows it helps.
        try:
            import query_plan
            pl = query_plan.plan(query)
            hybrid_scores = sorted(
                ((pid, s * (query_plan.score_factor(pl, points_by_id[pid].payload) if QUERY_BOOSTS else 1.0))
                 for pid, s in hybrid_scores
                 if pid in points_by_id and points_by_id[pid].payload.get("status") != "duplicate"),
                key=lambda t: -t[1])
            qd.ranking('Ranking after metadata boosts and duplicate removal', hybrid_scores, points_by_id,
                       query=query, boosts_enabled=QUERY_BOOSTS, query_plan=vars(pl))
            if pl.intent != "general" or pl.years or pl.doc_types:
                print(f"  🧭 question: {pl.intent}" + (f", years {pl.years}" if pl.years else "")
                      + (f", types {pl.doc_types}" if pl.doc_types else ""))
        except Exception as e:      # noqa: BLE001 — ranking hints never fail a query
            print(f"  ⚠ question plan skipped ({type(e).__name__}: {e})")
            qd.record('Metadata ranking skipped', {'query': query, 'error': str(e)})

        # Convert hybrid scores to result format.
        # A per-source cap keeps one large document from monopolising the context
        # window: EB 56 alone is 319 of the 471 chunks in the chips collection, so
        # without this the top-5 is frequently 5/5 EB 56 and answers get built from
        # whichever document is biggest rather than whichever is relevant.
        results = []
        per_source_count = {}
        seen_ids = set()

        def _add(point, score):
            if point.id in seen_ids:
                return False
            seen_ids.add(point.id)
            src = point.payload.get("source", "")
            per_source_count[src] = per_source_count.get(src, 0) + 1
            results.append({"point": point, "score": score, "rank": len(results) + 1})
            return True

        # 1) LEXICAL / exact-match hits first — guarantees exact identifiers, reference
        #    numbers, document numbers and rare terms reach the answer even when the
        #    embedding ranks them low. A "tell me about document X" query wants several
        #    of X's chunks, so lexical gets a looser per-source cap. Capped overall so it
        #    never fully crowds out the semantic arm.
        #
        #    A lexical hit is only injected ahead of the semantic ranking when it is a
        #    STRONG match: it contains at least two distinct query terms, OR one
        #    distinctive (rare) term such as a reference number or scheme code. A chunk
        #    that merely happens to contain one common word ("budget", "department") is
        #    NOT promoted — that used to drag unrelated chunks to rank 1 and anchor the
        #    model on the wrong topic. Such chunks can still arrive via the semantic arm
        #    below if they are actually relevant.
        lexical_hits = lexical_search(query, limit=DENSE_CANDIDATE_LIMIT) if LEXICAL_ARM_ENABLED else []
        if qd.active():
            qd.record('Lexical candidates', {'query': query, 'enabled': LEXICAL_ARM_ENABLED,
                      'rare_frequency_limit': LEX_RARE_FREQ,
                      'hits': [{**{k: v for k, v in h.items() if k != 'point'},
                                'chunks': qd.chunks([h])} for h in lexical_hits]})
        lex_cap = max(2, num_context // 2)
        lex_source_cap = (MAX_CHUNKS_PER_SOURCE + 2) if MAX_CHUNKS_PER_SOURCE else 0
        lex_added = 0
        # ONLY distinctive lexical hits are force-injected — a hit whose rarest matched
        # term is below the rare cutoff (a reference number, a scheme code, a rare name).
        # That is the one thing this arm does that dense+sparse cannot: land an exact
        # identifier the embedding ranks low. Matches on ordinary words ("budget",
        # "department", "mission", "IT") are IGNORED here — promoting them used to seat a
        # distractor at rank 1 and anchor the model on the wrong topic. Ordinary relevance
        # is left entirely to the semantic ranking below.
        for h in lexical_hits:
            if lex_added >= lex_cap or len(results) >= num_context:
                break
            if not lexical_forces(h):
                continue
            p = h["point"]
            if p.payload.get("status") == "duplicate":    # same rule as the semantic arm
                continue
            src = p.payload.get("source", "")
            if lex_source_cap and per_source_count.get(src, 0) >= lex_source_cap:
                continue
            if _add(p, 1.0):
                lex_added += 1

        # 2) SEMANTIC hybrid hits fill the remaining slots (per-source capped).
        #    A doc card (chunk_builder) only brings a document in — its facts are the
        #    document's metadata, not its text: it is skipped when a real chunk of that
        #    document is already in, and at most MAX_DOC_CARDS cards take a slot.
        def _is_card(r_or_point):
            p = r_or_point["point"] if isinstance(r_or_point, dict) else r_or_point
            return p.payload.get("chunk_type") == "doc_card"

        def _fill():
            for point_id, score in hybrid_scores:
                if len(results) >= num_context:
                    break
                point = points_by_id.get(point_id)
                if not point or point.id in seen_ids:
                    continue
                src = point.payload.get("source", "")
                if MAX_CHUNKS_PER_SOURCE and per_source_count.get(src, 0) >= MAX_CHUNKS_PER_SOURCE:
                    continue
                if _is_card(point) and (sum(map(_is_card, results)) >= MAX_DOC_CARDS
                                        or any(r["point"].payload.get("source") == src for r in results)):
                    continue
                _add(point, score)

        _fill()
        # a card whose document then got a real chunk gives its slot back
        content_srcs = {r["point"].payload.get("source") for r in results if not _is_card(r)}
        dropped = [r for r in results if _is_card(r) and r["point"].payload.get("source") in content_srcs]
        if dropped:
            for r in dropped:
                results.remove(r)
                src = r["point"].payload.get("source", "")
                per_source_count[src] = per_source_count.get(src, 1) - 1
            _fill()                              # seen_ids keeps the dropped cards out
            for i, r in enumerate(results, 1):
                r["rank"] = i

        if qd.active():
            qd.record('Selected chunks before neighbors', {'query': query, 'requested': num_context,
                      'per_source_cap': MAX_CHUNKS_PER_SOURCE, 'chunks': qd.chunks(results)})
        results = expand_with_neighbors(ensure_qdrant_client(), results)
        if qd.active():
            qd.record('Retrieved chunks including neighbors', {'query': query, 'chunks': qd.chunks(results)})
        _mark_time("RETRIEVE_CONTEXT")
        return results
    
    except Exception as e:
        print(f"Retrieval error: {e}")
        _RETRIEVAL_ERROR.set(f"{type(e).__name__}: {e}")
        qd.record('Retrieval error', {'query': query, 'error': str(e)})
        import traceback
        traceback.print_exc()
        _mark_time("RETRIEVE_CONTEXT")
        return None

# ── Helper: One place that talks to the LLM ────────────────────
class LLMError(RuntimeError):
    """Raised when the LLM call fails. Callers decide whether that is fatal."""


def call_llm(system, user, temperature=0.2, max_tokens=None, timeout=300,
             thinking=None, json_schema=None):
    """One chat completion from the answer model -> (text, truncated).

    Shared by every caller: the single-shot answer, each thinking-mode step, table
    counts. `thinking` switches Qwen3's reasoning (None = ANSWER_THINKING);
    `json_schema` makes the server return JSON matching it. Raises LLMError rather
    than returning an error string, so a caller can degrade instead of showing
    "Error: ..." as an answer.
    """
    try:
        return llm_client.client().chat(
            system, user, max_tokens=max_tokens or ap.ANSWER_TOKENS, temperature=temperature,
            thinking=ANSWER_THINKING if thinking is None else thinking,
            json_schema=json_schema, timeout=timeout)
    except llm_client.LLMClientError as e:
        raise LLMError(str(e)) from e


def call_llm_stream(system, user, temperature=0.2, max_tokens=None, timeout=300, thinking=None):
    """Streaming twin of call_llm: yields text deltas as they arrive."""
    try:
        yield from llm_client.client().stream(
            system, user, max_tokens=max_tokens or ap.ANSWER_TOKENS, temperature=temperature,
            thinking=ANSWER_THINKING if thinking is None else thinking, timeout=timeout)
    except llm_client.LLMClientError as e:
        raise LLMError(str(e)) from e


# ── Readable source labels + clean chunk text ──────────────────
# The vector store keys a chunk by its doc_dir (a urlhash). On its own that is
# meaningless to a reader — "5f369c50ead865ab.pdf" tells an officer nothing — so we
# resolve it back to the document's real title/URL via the pipeline registry.
_reg_cache = {"mtime": None, "idx": {}}


def _registry_index():
    """{doc_dir: record} from this pipeline's documents.json, reloaded on change."""
    reg = pc.DATA_ROOT / "documents.json"
    try:
        m = reg.stat().st_mtime
    except OSError:
        return {}
    if _reg_cache["mtime"] != m:
        try:
            docs = json.loads(reg.read_text(encoding="utf-8"))
            docs = docs if isinstance(docs, list) else docs.get("documents", [])
            _reg_cache["idx"] = {d.get("doc_dir"): d for d in docs if d.get("doc_dir")}
            _reg_cache["mtime"] = m
        except Exception:
            pass
    return _reg_cache["idx"]


def _meta_by_dir():
    """doc_dir -> document record from doc_metadata.json (re-read when it changes)."""
    try:
        import catalogue
        docs = catalogue.records(pc.DATA_ROOT)
    except Exception:          # noqa: BLE001
        return {}
    cached = getattr(_meta_by_dir, "_cache", None)
    if cached is None or cached[0] is not docs:
        _meta_by_dir._cache = (docs, {r.get("doc_dir"): r for r in docs})
    return _meta_by_dir._cache[1]


CATALOGUE_SOURCE = "__catalogue__"


def with_catalogue(query, context_results):
    """For a "which / latest documents" question, put the matching documents from
    doc_metadata.json (core/catalogue.py: type, topic, year, newest first, links) in
    front of the passages, as one more piece of EVIDENCE. The answer model still
    writes the answer: a question misread as a catalogue question then carries an
    extra passage instead of being answered with a list."""
    try:
        import catalogue
        lang = detect_query_language(query)
        listing = catalogue.answer(query, pc.DATA_ROOT, "en" if lang == "en" else "hi")
        qd.record('Catalogue lookup', {'query': query, 'listing': listing, 'added': bool(listing)})
        if not listing:
            return context_results
        print("  🗂 document list added to the evidence (catalogue)")
        from types import SimpleNamespace
        text = ("# Chunk catalogue\nDocument: catalogue\nPages: Unknown\nTokens: 0\nHeadings: []\n"
                "Type: catalogue\n\n---\n\n" + listing)
        point = SimpleNamespace(id="catalogue", payload={"text": text, "source": CATALOGUE_SOURCE,
                                                         "chunk": "catalogue", "chunk_type": "catalogue"})
        return [{"point": point, "score": 1.0, "rank": 0}] + list(context_results or [])
    except Exception as e:     # noqa: BLE001 — the passages still answer
        print(f"  ⚠ catalogue skipped ({type(e).__name__}: {e})")
        qd.record('Catalogue lookup skipped', {'error': str(e)})
        return context_results


def source_label(chunk_source):
    """(display_name, url) for a chunk's source — human-readable, never a bare hash.

    Prefers the document's title; falls back to a decoded URL path segment; only then
    to the hash. This is what a reader sees in the Sources list and what the model is
    told to refer to, so it must read like a document name, not an id.
    """
    if chunk_source == CATALOGUE_SOURCE:
        return "विभाग के दस्तावेज़ों की सूची (document records)", None
    rec = _registry_index().get(chunk_source, {}) or {}
    url = rec.get("source_url") or None
    # the document record (doc_metadata.py): "title · क्र. … · 14.08.2026" — a reader
    # recognises that, not "202600323_001.pdf"
    meta = _meta_by_dir().get(chunk_source)
    if rec.get("source_type") == "web":
        # A web page is the live site as it was read: say so, with that date, so the
        # answer can tell current pages from old orders (answer rule 7). An order
        # number / date in its record comes from one item on the page (a news line on
        # the home page), not from the page — not shown.
        import catalogue
        title = (catalogue.display_title(meta) if meta else "") or (rec.get("original_name") or "").strip()
        if not title or title.lower().startswith(("http://", "https://")):
            from urllib.parse import unquote, urlparse
            parts = [x for x in urlparse(url or "").path.split("/") if x]
            title = unquote(parts[-1]) if parts else (urlparse(url or "").netloc or f"{chunk_source}")
        seen = str(rec.get("fetched_at") or rec.get("uploaded_at") or "")[:10]
        when = ".".join(reversed(seen.split("-"))) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", seen) else ""
        return f"{title} · वेब पेज{f' ({when} की स्थिति)' if when else ''}", url
    if meta:
        import catalogue
        bits = [catalogue.display_title(meta)]
        if meta.get("order_number"):
            bits.append(f"क्र. {catalogue.display_order(meta['order_number'])}")
        if meta.get("issue_date"):
            bits.append(".".join(reversed(meta["issue_date"].split("-"))))
        return " · ".join(bits), url or meta.get("source_url")
    name = (rec.get("original_name") or "").strip()
    if name and not name.lower().startswith(("http://", "https://")):
        return name, url
    if url:
        from urllib.parse import unquote, urlparse
        parts = [x for x in urlparse(url).path.split("/") if x]
        return (unquote(parts[-1]) if parts else urlparse(url).netloc), url
    return f"{chunk_source}.pdf", None


def _web_template():
    try:
        return web_boilerplate.for_registry(pc.DATA_ROOT / "documents.json", pc.paths.stage2)
    except Exception as e:         # noqa: BLE001 — the text is used as it is, but say why
        print(f"  ⚠ site template not read ({type(e).__name__}: {e}); web chunks used as stored")
        return frozenset()


def readable_text(payload):
    """A chunk's stored text; a web page's without the website's header/menus/footer
    (web_boilerplate). Chunks indexed before that change still carry them."""
    text = (payload or {}).get("text", "") or ""
    src = (payload or {}).get("source")
    if not text or not src or (_registry_index().get(src) or {}).get("source_type") != "web":
        return text
    try:
        return web_boilerplate.strip_chunk(text, _web_template(), payload.get("context_line"))
    except Exception as e:         # noqa: BLE001 — never fail an answer over this
        print(f"  ⚠ site template not removed ({type(e).__name__}: {e})")
        return text


def without_site_template(results):
    """The hits with each web page chunk's text as readable_text gives it; a chunk that
    was nothing but the site's menu is left empty, and the evidence skips it."""
    out = []
    for r in results or []:
        point = r.get("point")
        payload = getattr(point, "payload", None) or {}
        text = payload.get("text")
        new = readable_text(payload) if text else text
        if new == text:
            out.append(r)
            continue
        from types import SimpleNamespace
        out.append({**r, "point": SimpleNamespace(id=getattr(point, "id", None),
                                                  payload={**payload, "text": new})})
    return out


def clean_chunk_text(text):
    """Strip the chunk-file metadata header so only real document content reaches the
    LLM. Chunks are stored as `# Chunk N / Document: / Pages: / Tokens: / Headings:`
    then a `---` separator then the content. Left in, the model parrots that header
    into its answer ("Document: <hash> (Chunk 6)"). The header is short, so a `---`
    far into the text is a real horizontal rule and is left alone.
    """
    marker = '\n---\n'
    idx = text.find(marker)
    if 0 <= idx <= 400:
        return text[idx + len(marker):].strip()
    return (text or "").strip()


# A doc_dir is a 16-hex urlhash; the model sometimes echoes it as "<hash>.pdf".
# Match 8+ hex so shorter leaks are caught too. It is an internal id, not a title.
# NB: match only the token (and hugging parens) — never the surrounding spaces, or
# removal would fuse the words on either side ("document X lists" -> "documentlists").
_HASH_PDF = re.compile(r'\(?\b[0-9a-f]{8,}\.pdf\b\)?', re.I)

# A "Sources:" / "स्रोत:" style heading sitting on its own line — the start of a
# source list the model tacked on despite being told not to.
_SOURCES_HEADING = re.compile(
    r'(?:^|\n)[ \t]*(?:[*_#>\-]+[ \t]*)*'
    r'(?:sources?(?:\s+used)?|references?|source\s+documents?|स्रोत|स्रोतों|संदर्भ)'
    r'[ \t]*:?[ \t]*(?:\*\*|__)?[ \t]*(?:\n|$)',
    re.I)


def strip_model_source_refs(answer):
    """Keep document identifiers OUT of the answer prose — they belong only in the
    Sources list we append ourselves (what the UI shows under "View sources").

    A model sometimes ignores "do not write a list of sources": it (a) tacks its own
    "Sources:" list onto the end and (b) drops an internal id such as
    `be302bf58abc3a98.pdf` (a document with no title yet) into a sentence. Both read
    as gibberish to an official, so they are removed and our numbered Sources list
    is the only place a document is named. The [n] citation markers stay.
    """
    if not answer:
        return answer

    # 1. Cut a model-written source list, including plain numbered entries.
    #    Only headings outside thinking count. The heading pattern does not match
    #    legitimate prose such as "Source of funds:".
    visible_start = answer.rfind('</think>') + len('</think>') if '</think>' in answer else 0
    for m in _SOURCES_HEADING.finditer(answer):
        if m.start() >= visible_start:
            answer = answer[:m.start()]
            break

    # 2. Strip any bare urlhash filename left in the prose.
    answer = _HASH_PDF.sub('', answer)

    # 2b. Strip inline "[Source 3]", "(Source 3)", "Source 3:" citation tags. The
    #     synthesis prompt numbers the evidence as [Source N] and the model cites them
    #     back despite being told not to; a reader does not have that numbering.
    answer = re.sub(r'[\(\[]?\bSources?\s+\d+\b[\)\]]?\s*:?', '', answer, flags=re.I)

    # 3. Tidy the punctuation/whitespace the removals leave behind.
    answer = re.sub(r'\(\s*\)', '', answer)                       # empty ()
    answer = re.sub(r'\bdocuments?\s+(?=[,.;:)])', '', answer, flags=re.I)  # "document ." -> ""
    answer = re.sub(r'[ \t]{2,}', ' ', answer)                    # doubled spaces
    answer = re.sub(r'[ \t]+([,.;:)])', r'\1', answer)            # space before punctuation
    answer = re.sub(r'\n{3,}', '\n\n', answer)                    # collapse blank runs
    return answer.strip()


def _count_tokens_fn():
    """Token counter for evidence budgets: the answer model's own tokenizer
    (llm_client /tokenize; a calibrated estimate when the server cannot count)."""
    return lambda t: llm_client.client().count_tokens(text=t)


_TERM_RE = re.compile(r"[\wऀ-ॿ]+")


def query_terms(query):
    """Words to keep when a long passage is trimmed: the question's own terms, its
    rare lexical terms, and their glossary translations (so an English question
    still finds the lines of a Hindi passage)."""
    words = [w for w in _TERM_RE.findall(query or "") if len(w) >= 3]
    try:
        words += list(lexical_terms(query))
    except Exception:
        pass
    try:
        words += list(glossary_translations(query))
    except Exception:
        pass
    return list(dict.fromkeys(w for w in words if w))


def with_table_count(query, context_results):
    """For a COUNT question ("Durg domicile ke kitne Section Officers hain?"), put a
    database count over the table behind the best table chunks in front of the
    passages (core/table_query.py); anything that fails leaves the passages as
    they were."""
    if not context_results or not TABLE_COUNT:
        qd.record('Table count skipped', {'enabled': TABLE_COUNT, 'has_context': bool(context_results)})
        return context_results
    try:
        import query_plan
        if query_plan.plan(query).intent != "count":
            qd.record('Table count skipped', {'reason': 'Question not classified as a count'})
            return context_results
        import table_query

        def chat(system, user):
            text, _ = llm_client.client().chat(system, user, max_tokens=400, temperature=0.0,
                                               thinking=False, json_schema=table_query.SQL_SCHEMA)
            return text

        ev = table_query.evidence(query, context_results, pc.paths.stage2, chat, source_label)
        if qd.active():
            qd.record('Table count result', {'chunks': qd.chunks([ev] if ev else [])})
        return [ev] + list(context_results) if ev else context_results
    except Exception as e:     # noqa: BLE001 — the passages still answer
        print(f"  ⚠ table count skipped ({type(e).__name__}: {e})")
        qd.record('Table count fallback', {'error': str(e)})
        return context_results


# ── Follow-up questions ─────────────────────────────────────────
# "और इसकी आयु सीमा?" means nothing to the search on its own. The page sends the last
# few turns of the chat; the answer model rewrites a follow-up into a standalone
# question (what the search uses — the "condense question" step of chat RAG), and
# the answer prompt sees the recent turns, so "explain point 2" or "say that in
# English" work. Facts still come only from the passages. CHAT_CONTEXT=0 = off.
CHAT_CONTEXT = os.getenv("CHAT_CONTEXT", "1").strip().lower() not in ("0", "off", "false", "no")
CHAT_TURNS = 3                  # previous question/answer pairs used
_Q_CHARS, _A_CHARS, _LAST_A_CHARS = 500, 400, 1500

CONDENSE_SYSTEM = """You turn a follow-up question from a chat about Chhattisgarh government documents into a standalone search question.

Use the conversation only to resolve what the follow-up refers to: "इसकी", "उसके लिए", "this order", "and for 2024?", "the second point". Add just those names, schemes, posts, years or order numbers.
Keep the user's language, script and wording, and any instruction in it ("in short", "in English").
If the question is already complete, or asks about something new, return it unchanged.
Never answer the question and never add a fact that is not in the conversation."""

CONDENSE_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}},
                   "required": ["question"], "additionalProperties": False}

_SOURCES_TAIL = re.compile(r"\n+\*\*(?:Sources|स्रोत):\*\*.*$", re.S)


def clean_history(history):
    """Up to CHAT_TURNS previous turns from the page as [{question, answer}]: the
    Sources list and the model's thinking cut from the answers, lengths capped.
    Anything malformed is dropped — the page's history is a hint, not trusted input."""
    if not CHAT_CONTEXT or not isinstance(history, list):
        return []
    turns = []
    for t in history[-CHAT_TURNS:]:
        if not isinstance(t, dict):
            continue
        q = str(t.get("question") or "").strip()[:_Q_CHARS]
        a = _SOURCES_TAIL.sub("", llm_client.strip_thinking(str(t.get("answer") or ""))).strip()
        if q:
            turns.append({"question": q, "answer": a})
    for i, t in enumerate(turns):                    # the last answer may be needed whole-ish
        cap = _LAST_A_CHARS if i == len(turns) - 1 else _A_CHARS
        if len(t["answer"]) > cap:
            t["answer"] = t["answer"][:cap].rsplit(" ", 1)[0] + " …"
    return turns


def _turns_text(history):
    return "\n\n".join(f"Q: {t['question']}\nA: {t['answer'] or '—'}" for t in history)


def standalone_question(question, history):
    """The follow-up rewritten as a standalone question, or the question itself (no
    history, a model failure, or an implausible rewrite)."""
    if not history:
        qd.record('Follow-up rewrite skipped', {'reason': 'No previous conversation turns', 'question': question})
        return question
    user = f"Conversation:\n{_turns_text(history)}\n\nFollow-up question: {question}"
    try:
        raw, _ = call_llm(CONDENSE_SYSTEM, user, temperature=0.0, max_tokens=300,
                          thinking=False, json_schema=CONDENSE_SCHEMA)
        out = str((json.loads(raw) or {}).get("question") or "").strip()
    except (LLMError, ValueError, AttributeError) as e:
        print(f"  ⚠ follow-up not rewritten ({type(e).__name__}: {e})")
        qd.record('Follow-up rewrite fallback', {'error': str(e), 'question_kept': question})
        return question
    if not out or len(out) > 3 * len(question) + 300:
        qd.record('Follow-up rewrite rejected', {'model_output': out, 'question_kept': question})
        return question
    if out != question:
        print(f"  💬 follow-up: {question!r} -> {out!r}")
    return out


def history_block(history):
    """The recent turns for the answer prompt (placed before the passages)."""
    if not history:
        return ""
    return ("Earlier in this conversation — use it to understand what the question refers "
            "to; every fact in your answer must still come from the passages below:\n"
            f"{_turns_text(history)}\n\n")


def build_answer_messages(query, context_results, *, rules=None, budget=None, extra_before="",
                          history=None, asked=None, max_tokens=None, thinking=None):
    """Build (system, user, sources, lang) for one answer.

    Written for the GPU answer model's 32k window (METADATA_LAYER_PLAN.md §7.1):
    evidence is added in rank order until a TOKEN budget is spent (not a fixed chunk
    count), grouped per document and numbered [1], [2]… — the numbers the model cites
    and the Sources list shows. Rules are short (answer_prompt.RULES). The blocking
    and streaming paths share this, so there is one prompt.

    In a chat, `query` is the standalone question (standalone_question), `asked` the
    user's own words — they decide the answer's language — and `history` the recent
    turns, shown before the passages; their tokens come out of the evidence budget.
    """
    lang, language_rule_text, closing_line, answer_in = language_directive(asked or query)
    qd.record('Answer language selected', {'original_question': asked or query, 'search_question': query,
              'language': lang, 'language_rule': language_rule_text, 'closing_instruction': closing_line})
    context_results = with_catalogue(query, with_table_count(query, without_site_template(context_results)))
    if qd.active():
        qd.record('Evidence after catalogue and table-count enrichment', {
            'question': query, 'chunks': qd.chunks(context_results)})
    earlier = history_block(history)
    budget = budget or ap.EVIDENCE_TOKENS
    if earlier:
        budget = max(2000, budget - llm_client.estimate_tokens(earlier))
    try:
        system, user, sources, docs = ap.build_messages(
            query, context_results,
            rules=rules or ap.RULES,
            language_rule=language_rule_text,
            closing_line=f"{closing_line} Answer in {answer_in}.",
            budget=budget,
            count=_count_tokens_fn(),
            clean=clean_chunk_text,
            label=source_label,
            terms=query_terms(query),
            extra_before=earlier + extra_before,
            client=llm_client.client(), max_tokens=max_tokens or ap.ANSWER_TOKENS,
            thinking=ANSWER_THINKING if thinking is None else thinking)
    except llm_client.LLMClientError as e:
        raise LLMError(str(e)) from e
    return system, user, Sources(sources, [d.source for d in docs]), lang


def check_answer(answer, user_message, question=""):
    """Figures in the answer that its cited passages do not contain (logged).

    A cheap guard against the one failure that matters most in this domain: a
    number stated with confidence that no document gives.
    """
    bad = ap.unsupported_numbers(llm_client.strip_thinking(answer), user_message, question)
    qd.record('Answer number check', {'unsupported_numbers': bad,
              'note': 'Checks numbers against cited evidence; this is not a semantic correctness verdict.'})
    if bad:
        print(f"  ⚠ answer states figures not found in its cited passages: {', '.join(bad[:10])}",
              flush=True)
    return bad


_NOTFOUND_RE = re.compile(
    r"do(?:es)?\s+not\s+(?:contain|mention|provide|specify|include|state|describe|cover)"
    r"|no\s+(?:information|mention|details?|data|reference|specific)"
    r"|not\s+(?:found|available|provided|mentioned|covered|present|specified)\b"
    r"|documents?\s+do\s+not"
    r"|could\s*n'?t\s+find|cannot\s+find|could\s+not\s+find"
    r"|नहीं\s+(?:है|हैं|मिली|मिला|दी\s+गई|दिया\s+गया|दिये\s+गये|दिए\s+गए|किया\s+गया|मिलता|करते)"
    r"|(?:कोई\s+)?उत्तर\s+नहीं|उपलब्ध\s+नहीं|स्पष्ट\s+(?:रूप\s+से\s+)?(?:उल्लेख|नहीं)"
    r"|जानकारी\s+नहीं|उल्लेख\s+नहीं|शामिल\s+नहीं|विवरण\s+नहीं|प्रदान\s+नहीं|दर्ज\s+नहीं",
    re.IGNORECASE)


def answer_is_ungrounded(answer):
    """True when the answer is essentially "the passages do not contain this" — the
    sources list and panel are then suppressed (sources under a not-found reply only
    confuse a reader).

    The answer prompt makes the model put a passage number after every fact and a
    "not in the passages" sentence at the END, never first (answer_prompt.RULES 4, 11).
    So an answer that cites a passage ([2]) is grounded, whatever else it says: a
    correct answer that also notes a missing detail ("…the date is not given") keeps
    its sources. Without a citation, it is ungrounded when it says nothing was found —
    in its opening words, or anywhere in a short reply.
    """
    a = llm_client.strip_thinking(answer or "")
    if not a:
        return True
    if ap.cited_docs(a):
        return False
    if _NOTFOUND_RE.search(" ".join(a.split()[:24])):
        return True
    return bool(_NOTFOUND_RE.search(a)) and len(a.split()) < 70


class Sources(list):
    """[(label, url), ...] numbered like the answer's [n] markers; `.ids` holds the
    documents' source ids in the same order (labels can repeat: several budget pages
    are all called "Main Budget")."""

    def __init__(self, items=(), ids=()):
        super().__init__(items)
        self.ids = list(ids)


def results_for_sources(context_results, sources, cited=None):
    """The retrieval hits behind the documents the answer was given: one per
    document, in the order of the numbered Sources list, `rank` = its number.

    The sources panel is built from this, so it shows the documents the model saw —
    the evidence budget can leave lower-ranked ones out — under the same numbers as
    the [n] citations and the Sources list below the answer. With `cited`, only the
    documents the answer cites."""
    ids = getattr(sources, "ids", None) or []
    by_id = {sid: n for n, sid in enumerate(ids, 1)}
    by_label = {lbl: n for n, (lbl, _u) in enumerate(sources or [], 1)}
    best = {}
    for r in context_results or []:
        src = (getattr(r.get("point"), "payload", None) or {}).get("source", "")
        if not src or src == CATALOGUE_SOURCE:
            continue
        n = by_id.get(src) if by_id else by_label.get(source_label(src)[0])
        if n is not None and n not in best and (cited is None or n in cited):
            best[n] = dict(r, rank=n)
    return [best[n] for n in sorted(best)]


def sources_block(sources, lang, cited=None):
    """The one clean Sources list appended below every answer (the model is told
    not to write its own), numbered like the [n] markers in the answer.

    With `cited`, only the documents the answer cites — under their own numbers, so
    [5] still means the fifth. Listing every passage the model was GIVEN put
    "Training Program — Call for Nominations" under an answer about compassionate
    appointment (seen on the live GAD bot, 2026-10-12)."""
    if not sources:
        return ""
    heading = "स्रोत" if lang == "hi" else "Sources"
    lines = [f"[{i}] [{lbl}]({u})" if u else f"[{i}] {lbl}"
             for i, (lbl, u) in enumerate(sources, 1) if cited is None or i in cited]
    return (f"\n\n**{heading}:**\n" + "\n".join(lines)) if lines else ""


def finish_answer(answer, *, user, question, sources, lang, context_results, truncated=False):
    """Everything after the model has answered, the same for every path (blocking,
    streaming, thinking mode): document ids and a self-written source list out of
    the prose, figures the passages lack logged, a truncation note, and our numbered
    Sources list — only for valid passage citations in the visible answer body.

    Returns (answer, tail, display): `tail` is what was appended (the streaming path
    sends it as the last token), `display` the hits for the sources panel.
    """
    answer = strip_model_source_refs(answer)
    check_answer(answer, user, question)
    visible = llm_client.strip_thinking(answer)
    qd.record('Answer cleanup and citation selection', {
        'visible_answer': visible, 'language': lang, 'truncated': truncated,
        'cited_passage_numbers': sorted(set(ap.cited_docs(visible))),
        'available_sources': sources, 'ungrounded': answer_is_ungrounded(visible)})
    if visible and not answer_language_ok(visible, lang):
        print(f"  ⚠ answer is not in the question's language ({lang})", flush=True)
    tail = "\n\n_(Answer truncated — output token limit reached.)_" if truncated else ""
    if answer_is_ungrounded(visible):
        return answer + tail, tail, []
    cited = set(ap.cited_docs(visible)) & set(range(1, len(sources or []) + 1))
    if not cited:
        return answer + tail, tail, []
    tail += sources_block(sources, lang, cited)
    return answer + tail, tail, results_for_sources(context_results, sources, cited)


def answer_question(query, context_results, *, history=None, asked=None):
    """(answer, display_results): the single-shot answer, blocking. Raises LLMError.
    In a chat: `query` the standalone question, `asked` the user's words, `history`
    the recent turns (see build_answer_messages)."""
    _mark_time("ANSWER_GENERATION")
    try:
        system, user, sources, lang = build_answer_messages(query, context_results,
                                                            history=history, asked=asked)
        text, truncated = call_llm(system, user, temperature=ANSWER_TEMPERATURE)
        final, _tail, display = finish_answer(text, user=user, question=query, sources=sources,
                                              lang=lang, context_results=context_results,
                                              truncated=truncated)
        return final, display
    finally:
        _mark_time("ANSWER_GENERATION")


def generate_answer(query, context_results):
    """The answer text alone (scripts, tests); a model failure comes back as a message."""
    if not context_results:
        return "No context found to generate an answer."
    try:
        return answer_question(query, context_results)[0]
    except LLMError as e:
        return f"Error generating answer: {e}"


if __name__ == "__main__":
    # From the shell:  PIPELINE=fg2 python rag_pipeline.py "अनुकंपा नियुक्ति की पात्रता क्या है?"
    q = " ".join(sys.argv[1:]).strip() or input("Question: ").strip()
    hits = retrieve_context(q)
    if not hits:
        sys.exit("No passages found.")
    print(generate_answer(q, hits))
