"""Garbage gate: is this text real language that belongs in the index, or junk?

Entry points, one set of signals:

  clean_page(text)            repair what is safely repairable -> (text, repairs)
  assess_page(text, ...)      one OCR'd page -> PageVerdict(ok|blank|suspect|reject),
                              carrying the repaired text that may be indexed
  assess_document(verdicts)   the document-level outcome
  text_layer_quality(text, n) a PDF's embedded text layer -> (keep?, reason)

Why a gate at all: anything written to stage2/<doc>/structured.md is chunked,
embedded and retrieved. Garbage there is not "lower quality" — it is actively
harmful. Measured on this corpus: a correctly spelled Hindi question cannot match
mangled stored Hindi, so the bot answers "not mentioned in the documents" for a
fact that IS in the corpus. Every rejection carries a human-readable reason so the
decision can be audited.

REPAIR BEFORE JUDGING. Calibrating on ~13k real OCR'd pages showed that most bad
pages are good pages with a bad SPAN in them — even Sarvam pasted a Tamil couplet
into English pages, looped ": BUSOPAR" 448 times, and read Krutidev glyphs as
Latin gibberish next to clean Hindi. Rejecting such a page throws away the good
text with the bad, so clean_page() cuts the span and keeps the rest:

  danda misread        EasyOCR reads "।" as l/I/| ("हैl")       -> fixed
  foreign-script words VLM drift into Tamil/Arabic/Gurmukhi…   -> removed
  back-to-back loops   "A N D A N D …", repeated lines           -> one copy kept
  legacy-font lines    Krutidev/DevLys rows ("'kklu fnukad")     -> line dropped

Only BACK-TO-BACK repetition is a loop. Real documents repeat legitimately — a
distribution list ("Director, IIIT Lucknow / Director, IIIT Kottayam …"), a
running header, budget tables full of "1,98,00,00" — and an earlier draft that
counted repeats anywhere on the page (or measured compressibility) rejected
hundreds of good budget pages. Numeric-only repeats are never touched.

What is judged on the repaired text (each signal was seen on the real CG corpus
or is a known failure mode of the engine that produced the text):

  control / PUA / U+FFFD broken font encodings, legacy-font private-use glyphs
  orphaned matras       Devanagari whose conjunct glyphs were dropped
  illegal sequences     sign orders no real Hindi word can have (EasyOCR emits them)
  dictionary hit rate   share of words that are real Hindi / English words
  trigram language model letter phonotactics — script-valid mojibake fails it
  symbol soup           OCR noise reading lines and stamps as ~|{}^ characters
  heavy repair          a page that needed most of its text cut is not trusted

Pure functions: no model, no network, milliseconds per page. The language models
are built once from wordfreq's word lists and cached.
"""

from __future__ import annotations

import math
import re
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

# ── thresholds ───────────────────────────────────────────────────────────────
# Calibrated on the existing fg/fg2 OCR output (Sarvam pages as the clean
# reference, EasyOCR pages as the noisy one) — scripts/calibrate_text_gate.py.
# Move them only with a calibration run.

MIN_CONTENT_CHARS = 25          # letters+digits below this: nothing to index
INK_CONTENT_RATIO = 0.005       # ink share (vlm_ocr.render_page) above which a page HAS
                                # content: text pages measure >= 1% (p1), blank <= 0.07%.
                                # Of 40 pages EasyOCR read as empty, 16 were above this —
                                # photos, maps, and a budget table it had missed.

# repair
LOOP_MAX_PERIOD = 12            # token period of a loop ("A N D" … up to a looped clause)
LOOP_MIN_REPEATS = 4            # back-to-back copies of a multi-token block = loop
LOOP_MIN_REPEATS_SINGLE = 6     # … of a single token ("श्री श्री श्री …")
LINE_MIN_RUN = 3                # back-to-back identical lines collapsed to one
MOJIBAKE_LINE_MIN_WORDS = 4     # judge a line as legacy-font junk only with this many words
MOJIBAKE_LINE_MAX_DICT = 0.30   # … and fewer than this share of them real English
FOREIGN_TOKEN_SHARE = 0.5       # a word is "foreign" when most of its letters are

# verdict on the repaired page
MAX_REPAIRED_SHARE = 0.60       # junk cuts took most of the page: not trusted
SUSPECT_REPAIRED_SHARE = 0.20
LOOP_SUSPECT_SHARE = 0.30       # repeats collapsed: no text lost, but re-read the page
MAX_CTRL_RATIO = 0.0            # any C0/C1 control char is damage
MAX_PUA_RATIO = 0.005           # legacy-font private-use glyphs
MAX_REPLACEMENT_RATIO = 0.005   # U+FFFD
MAX_SYMBOL_RATIO = 0.08         # ~{}^\ etc. after markdown/emails/form lines are stripped
SUSPECT_SYMBOL_RATIO = 0.04

DEV_MIN_WORDS = 12              # too few Devanagari words to judge below this
MAX_ORPHAN_MATRA = 0.15         # dropped-conjunct text layers run 20-40%
SUSPECT_ORPHAN_MATRA = 0.04
MAX_ILLEGAL_SEQ = 0.12          # illegal sign sequences per Devanagari word
SUSPECT_ILLEGAL_SEQ = 0.04
MIN_HI_DICT = 0.25              # share of Devanagari words found in the Hindi lexicon
SUSPECT_HI_DICT = 0.40
TRIGRAM_HI_FLOOR = -5.0         # real Hindi >= -4.3; below the floor is garbage
SUSPECT_TRIGRAM_HI = -4.6

EN_MIN_WORDS = 12
TRIGRAM_EN_MIN = -2.75          # real English >= -2.63, mojibake <= -2.85 (measured)
MAX_VOWELLESS = 0.15            # real English ~3% vowel-less words; mojibake 20-40%
MIN_EN_DICT = 0.35
SUSPECT_EN_DICT = 0.55
MAX_INTRAWORD_PUNCT = 0.08      # Krutidev signature: "dk;Z", "'kklu", "fu;e"

MAX_MIXED_SCRIPT = 0.15         # words mixing Devanagari and Latin letters
SUSPECT_MIXED_SCRIPT = 0.05

# text-layer gate (PDF embedded text; see text_layer_quality)
MIN_CHARS_PER_PAGE = 40
DEVANAGARI_RATIO = 0.15
MAX_JUNK_RATIO = 0.04
FUNCTION_WORD_RATIO = 0.06
MAX_ORPHAN_MATRA_RATIO = 0.02
NGRAM_TOP_WORDS = 40000

# ── character classes ────────────────────────────────────────────────────────
_DEV_LO, _DEV_HI = "ऀ", "ॿ"
# Dependent vowel signs: must follow a consonant (optionally a nukta).
_VOWEL_SIGNS = set("ऺऻािीुूृॄॅ"
                   "ॆेैॉॊोौॎॏ"
                   "ॕॖॗॢॣ")
_VIRAMA = "्"
_NUKTA = "़"
_MODIFIERS = set("ऀँंः")          # candrabindu, anusvara, visarga
_DEV_MATRAS = _VOWEL_SIGNS | _MODIFIERS | {_VIRAMA, _NUKTA}
_DEV_CONSONANT = re.compile(r"[क-हक़-य़ॸ-ॿ]")
_DEV_WORD = re.compile(r"[ऀ-ॣॱ-ॿ]+")   # no danda, no digits
_JUNK = set("~{}<>|\\^`_[]¤¦§¨©®¯°±²³´µ¶·¸¹º»¼½¾¿×÷")
_VOWELS = set("aeiou")
_FUNCTION_WORDS = {"the", "of", "and", "for", "to", "in", "is", "by", "on",
                   "with", "as", "shall", "be", "are", "that", "from"}
_NUMERIC_TOKEN = re.compile(r"^[\d०-९.,:;%₹()\[\]/\\+\-–—=*#|]+$")
# EasyOCR reads the danda "।" glued to a Hindi word as l / I / | ("हैl").
_DANDA_MISREAD = re.compile(r"(?<=[ऀ-ॣॱ-ॿ])[lI|](?=[\s,;:)\]]|$)", re.M)
_TABLE_LINE = re.compile(r"^\s*(\||<t[rdh]|<table|</t)", re.I)

# Markdown/HTML scaffolding the OCR engine adds around real text. It is layout,
# not content: a table's pipes are not "symbol soup", a heading's # is not junk,
# and an e-mail address is not a misspelled English word.
#
# Every pattern here runs on raw model output, which can be 60,000 characters of a
# loop — so none may backtrack super-linearly. On 2026-10-10 the |---| pattern
# (then `^\s*\|?\s*…` with re.M: two \s* side by side that also crossed newlines)
# took hours on a page that looped into blank lines — cubic: 2,000 newlines 36 s —
# and, holding the GIL, froze the whole OCR run for ~5 h. Hence: table rules are
# found line by line (is_table_rule), whitespace runs never sit next to each other
# or span lines, and open-ended runs are bounded. tests/test_text_gate.py times
# every pattern on such inputs.
TABLE_RULE = re.compile(r"\|?\s*:?-{3,}:?(?:\s*\|\s*:?-{3,}:?)*\s*\|?")


def is_table_rule(line: str) -> bool:
    """A Markdown table's header rule: |---|:---:|, | --- | --- |, or ---."""
    return bool(TABLE_RULE.fullmatch(line.strip()))


def drop_table_rules(text: str) -> str:
    return "\n".join(" " if is_table_rule(ln) else ln for ln in text.split("\n"))


_MD_STRIP = [
    (re.compile(r"<!--.{0,1000}?-->", re.S), " "),                # comments, page markers
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), " "),                    # images
    (re.compile(r"</?(table|thead|tbody|tr|td|th|br|p|div|span|b|i|u|sup|sub|strong|em|caption)"
                r"(\s[^>]{0,300})?/?>", re.I), " "),               # HTML table/inline tags
    (None, drop_table_rules),                                      # |---| (line by line)
    (re.compile(r"^[ \t]{0,3}#{1,6}\s+", re.M), ""),               # headings
    (re.compile(r"^[ \t]*(?:[-*+•]|\d+[.)])\s+", re.M), ""),       # list bullets
    (re.compile(r"\*\*|__|`"), ""),                                # emphasis/code
    (re.compile(r"\|"), " "),                                      # table cell pipes
    (re.compile(r"&(amp|lt|gt|nbsp|quot|#\d+);"), " "),            # entities
    (re.compile(r"<?[\w.+-]{1,64}\s?@\s?[\w-]{1,63}(\.[\w-]{1,63})*\s?>?"), " "),  # e-mail addresses
    (re.compile(r"(https?://|www\.)\S+"), " "),                    # URLs
    # e-mail printouts: recipient lists of quoted aliases ("ash-mohwf" , "cs-goa")
    # are not English words — they made whole circular cover pages read as junk.
    # Quoted phrases WITH spaces (programme titles) are content and stay.
    (re.compile(r'"[\w.\-]{2,40}"\s*(<[^>\n]{0,60}>?)?'), " "),
    # comparison operators in benchmark tables ("<=50, >25") are not stray symbols
    (re.compile(r"[<>≤≥]\s?=?(?=\s{0,3}[\d.])"), " "),
    (re.compile(r"_{2,}|\.{3,}|…{2,}|-{4,}"), " "),                # form blanks, leaders
]


def normalize_for_scoring(text: str) -> str:
    """The text a reader would see: scaffolding, e-mails, form lines removed; NFC."""
    t = unicodedata.normalize("NFC", text or "")
    for pat, rep in _MD_STRIP:
        t = rep(t) if pat is None else pat.sub(rep, t)
    return t


def _content_chars(t: str) -> int:
    return sum(1 for c in t if c.isalnum())


# ── language models (built once from wordfreq) ───────────────────────────────
# Built ONCE, under a lock. Without it the OCR's 64 worker threads all built the
# same 200,000-word lists at the same moment on start-up (2026-10-11): 55 copies of
# a CPU-bound job sharing one GIL — no page finished for over 15 minutes — and a
# thread could read a half-filled cache. vlm_ocr.DocumentRunner calls warm_up()
# before any worker starts.
_lex_cache: dict = {}
_ngram_cache: dict = {}
_build_lock = threading.Lock()


def warm_up() -> None:
    """Build the word lists and n-gram models now (idempotent, thread-safe)."""
    _lexicons()
    _ngram_models()


def _lexicons():
    """({hindi words}, {english words}) or (None, None) without wordfreq."""
    if not _lex_cache:
        with _build_lock:
            if not _lex_cache:
                try:
                    from wordfreq import top_n_list
                    hi = {unicodedata.normalize("NFC", w) for w in top_n_list("hi", 200000)}
                    en = {w for w in top_n_list("en", 200000) if w.isalpha()}
                except Exception:
                    hi = en = None
                _lex_cache.update(hi=hi, en=en)          # both keys at once
    return _lex_cache["hi"], _lex_cache["en"]


def _build_ngram(lang: str, lo: str, hi: str) -> dict:
    """Char-trigram log-prob model over the language's top words (add-1 smoothed)."""
    from collections import defaultdict
    from wordfreq import top_n_list
    tri: dict = defaultdict(lambda: defaultdict(int))
    charset: set = set()
    for w in top_n_list(lang, NGRAM_TOP_WORDS):
        w = w.lower()
        if len(w) < 2 or not all(lo <= c <= hi for c in w):
            continue
        charset |= set(w)
        s = "  " + w + " "
        for i in range(len(s) - 2):
            tri[s[i:i + 2]][s[i + 2]] += 1
    v = len(charset) + 2
    lp = {}
    for k, d in tri.items():
        tot = sum(d.values())
        lp[k] = {c: math.log((n + 1) / (tot + v)) for c, n in d.items()}
        lp[k]["_d"] = math.log(1 / (tot + v))
    return {"lp": lp, "floor": math.log(1.0 / NGRAM_TOP_WORDS)}


def _ngram_models():
    if not _ngram_cache:
        with _build_lock:
            if not _ngram_cache:
                try:
                    en, hi = _build_ngram("en", "a", "z"), _build_ngram("hi", _DEV_LO, _DEV_HI)
                except Exception:   # wordfreq missing/unloadable -> heuristic fallback
                    en = hi = None
                _ngram_cache.update(en=en, hi=hi)        # both keys at once
    return _ngram_cache["en"], _ngram_cache["hi"]


def _ngram_score(model, words, min_words: int = 5) -> float | None:
    """Mean char-trigram log-prob of `words` under `model` (higher = more real)."""
    if model is None or len(words) < min_words:
        return None
    lp, floor = model["lp"], model["floor"]
    s = n = 0
    for w in words:
        t = "  " + w + " "
        for i in range(len(t) - 2):
            k = t[i:i + 2]
            s += lp[k].get(t[i + 2], lp[k]["_d"]) if k in lp else floor
            n += 1
    return s / n if n else None


# ── individual signals ───────────────────────────────────────────────────────
def _orphan_matra_ratio(text: str) -> float:
    """Share of Devanagari tokens that BEGIN with a combining mark.

    Structurally impossible in real Devanagari, and the signature of an extractor
    (or OCR engine) that dropped conjunct glyphs. 0.0 when there is too little
    Devanagari to judge, so non-Hindi text is unaffected.
    """
    tokens = [t for t in text.split() if t]
    dev_tokens = [t for t in tokens if any(_DEV_LO <= c <= _DEV_HI for c in t)]
    if len(dev_tokens) < 10:
        return 0.0
    orphans = sum(1 for t in dev_tokens if t[0] in _DEV_MATRAS)
    return orphans / len(dev_tokens)


def _illegal_sequences(words: list[str]) -> int:
    """Count sign orders no real Hindi word has.

    - a vowel sign not preceded by a consonant / nukta (e.g. two vowel signs)
    - a virama followed by a modifier (anusvara/visarga/candrabindu)
    - a nukta not preceded by a consonant
    These are what EasyOCR emits when it misreads a conjunct; a VLM or a clean
    text layer essentially never does.
    """
    bad = 0
    for w in words:
        prev = ""
        for c in w:
            if c in _VOWEL_SIGNS and not (_DEV_CONSONANT.match(prev or " ") or prev == _NUKTA):
                bad += 1
            elif c in _MODIFIERS and prev == _VIRAMA:
                bad += 1
            elif c == _NUKTA and not _DEV_CONSONANT.match(prev or " "):
                bad += 1
            prev = c
    return bad


def _is_foreign_letter(c: str) -> bool:
    """A letter that is neither Devanagari nor (possibly accented) Latin."""
    if not c.isalpha() or _DEV_LO <= c <= _DEV_HI or c.isascii():
        return False
    return not unicodedata.name(c, "").startswith("LATIN")


_EDGE_PUNCT = ".,:;!?()[]{}\"|<>*_-–—/\\"
# Legacy-font (Krutidev/DevLys) text decodes Hindi glyphs to Latin keys, so its
# "words" carry punctuation INSIDE them: "U;ln", "I[;h", "'kklu", "fo'ks", "gS]".
# A possessive "State's" is ordinary English and is not counted.
_LEGACY_MARK = re.compile(r"[A-Za-z][;\[\]{}+=&~^`][A-Za-z]|[A-Za-z]'(?!s\b)[A-Za-z]|^'[A-Za-z]")
LEGACY_MIN_TOKENS = 8           # Latin words needed before a page is judged legacy-font
LEGACY_MARK_SHARE = 0.15        # … and the share of them carrying in-word punctuation
NOISE_LINE_MIN_TOKENS = 6       # speckle lines: "'-' f 1 - - a r ( t"


def _latin_tokens(t: str) -> list[str]:
    """Lower-cased Latin tokens for language checks, split on WHITESPACE only.

    Splitting on whitespace (not on every non-letter) is what makes legacy-font
    junk visible: "U;ln" stays one token — not a dictionary word — instead of
    breaking into "u" + "ln" fragments that look like abbreviations. Edge
    punctuation is stripped; short ALL-CAPS tokens ("CGST", "HRA", "GPF") are
    skipped as abbreviations — real content that fails every English signal.
    """
    out = []
    for raw in t.split():
        tok = raw.strip(_EDGE_PUNCT)
        if not tok or any(_DEV_LO <= c <= _DEV_HI for c in tok):
            continue
        if sum(1 for c in tok if c.isascii() and c.isalpha()) < 2:
            continue
        if tok.isalpha() and tok.isupper() and len(tok) <= 6:
            continue
        out.append(tok.lower())
    return out


def _english_signals(tokens: list[str]) -> dict:
    """Dictionary / trigram / vowel-less / in-word-punctuation for Latin tokens.

    A token with punctuation inside it is never a dictionary hit, so legacy-font
    junk scores low on the dictionary AND high on in-word punctuation.
    """
    en_model, _ = _ngram_models()
    _, en_lex = _lexicons()
    judged = [w for w in tokens if len(w) >= 3]
    alpha = [w for w in judged if w.isalpha()]
    return {
        "dict": (sum(1 for w in judged if w.isalpha() and w in en_lex) / len(judged)
                 if en_lex is not None and judged else None),
        "trigram": _ngram_score(en_model, alpha, min_words=3),
        "vowelless": (sum(1 for w in alpha if not (set(w) & _VOWELS)) / len(alpha)) if alpha else 0.0,
        "intraword": (sum(1 for w in tokens if _LEGACY_MARK.search(w)) / len(tokens)) if tokens else 0.0,
    }


def legacy_font_signature(text: str) -> bool:
    """Does this page carry legacy-font (Krutidev-type) text as Latin junk?

    Happens when a PDF uses a non-Unicode Hindi font that is not embedded: the
    page RENDERS as Latin letters, so any OCR engine — the VLM included — reads
    the junk faithfully. Such pages are candidates for a Krutidev->Unicode
    conversion later; for now their junk words are removed (clean_page).
    """
    toks = _latin_tokens(normalize_for_scoring(text))
    if len(toks) < LEGACY_MIN_TOKENS:
        return False
    s = _english_signals(toks)
    return (s["intraword"] >= LEGACY_MARK_SHARE
            or (s["dict"] is not None and s["dict"] < 0.3
                and s["trigram"] is not None and s["trigram"] < -3.2))


# ── repair ───────────────────────────────────────────────────────────────────
def _drop_foreign_tokens(line: str) -> tuple[str, int]:
    parts = re.split(r"(\s+)", line)
    kept, dropped = [], 0
    for p in parts:
        letters = [c for c in p if c.isalpha()]
        if letters and sum(_is_foreign_letter(c) for c in letters) / len(letters) >= FOREIGN_TOKEN_SHARE:
            dropped += 1
            continue
        kept.append(p)
    return ("".join(kept) if dropped else line), dropped


def _strip_legacy_tokens(line: str) -> tuple[str, int]:
    """On a legacy-font page: drop Latin tokens that are not real English words.

    Numbers, Devanagari and abbreviations stay; so does any dictionary word.
    """
    _, en_lex = _lexicons()
    parts = re.split(r"(\s+)", line)
    out, removed = [], 0
    for p in parts:
        core = p.strip(_EDGE_PUNCT)
        if (core and any(c.isascii() and c.isalpha() for c in core)
                and not any(_DEV_LO <= c <= _DEV_HI for c in core)
                and not (core.isalpha() and core.isupper() and len(core) <= 6)
                and not (core.isalpha() and len(core) >= 3 and en_lex is not None
                         and core.lower() in en_lex)):
            removed += 1
            continue
        out.append(p)
    if not removed:
        return line, 0
    return re.sub(r"[ \t]{2,}", " ", "".join(out)), removed


def _is_filler_block(block: list[str]) -> bool:
    """Numeric/punctuation-only blocks repeat legitimately (table cells, leaders)."""
    return all(_NUMERIC_TOKEN.match(t) or not any(ch.isalnum() for ch in t) for t in block)


def _collapse_token_loops(line: str) -> tuple[str, int]:
    """Keep one copy of any block of 1..LOOP_MAX_PERIOD words repeated back-to-back."""
    toks = line.split()
    n = len(toks)
    if n < LOOP_MIN_REPEATS * 2:
        return line, 0
    out, removed, i = [], 0, 0
    while i < n:
        hit = False
        for p in range(1, min(LOOP_MAX_PERIOD, (n - i) // 2) + 1):
            block = toks[i:i + p]
            k = 1
            while toks[i + k * p:i + (k + 1) * p] == block:
                k += 1
            need = LOOP_MIN_REPEATS_SINGLE if p == 1 else LOOP_MIN_REPEATS
            if k >= need:
                if _is_filler_block(block):
                    # kept as it is — but stepped over in one go: re-counting the run
                    # from every token was quadratic (a 30,000-token "@ @ @" loop = hours)
                    out.extend(toks[i:i + p * (k - 1)])
                    i += p * (k - 1)
                else:
                    out.extend(block)
                    removed += p * (k - 1)
                    i += p * k
                hit = True
                break
        if not hit:
            out.append(toks[i])
            i += 1
    if not removed:
        return line, 0
    indent = line[:len(line) - len(line.lstrip())]
    return indent + " ".join(out), removed


def _is_mojibake_line(line: str) -> bool:
    """A Latin-script line that is legacy-font junk ("fooldk lhu dj", "dk;Z")."""
    t = normalize_for_scoring(line)
    toks = _latin_tokens(t)
    if len(toks) < MOJIBAKE_LINE_MIN_WORDS:
        return False
    letters = [c for c in t if c.isalpha()]
    if sum(1 for c in letters if c.isascii()) < 0.6 * len(letters):
        return False
    s = _english_signals(toks)
    if s["dict"] is None or s["dict"] >= MOJIBAKE_LINE_MAX_DICT:
        return False
    return (s["intraword"] >= 0.10
            or (s["trigram"] is not None and s["trigram"] < -3.2)
            or s["vowelless"] > 0.30)


def _is_noise_line(line: str) -> bool:
    """OCR speckle: a line of 1-2 character fragments with no real word in it.

    "'-' f 1 - - a r ( t" — what EasyOCR makes of stamps, rules and scan dirt.
    Numbers are ignored (a row of column numbers is content), and any dictionary
    word (English >= 4 letters, Hindi >= 3 characters) keeps the line.
    """
    if _TABLE_LINE.match(line):
        return False
    toks = [w for w in normalize_for_scoring(line).split() if not _NUMERIC_TOKEN.match(w)]
    if len(toks) < NOISE_LINE_MIN_TOKENS:
        return False
    if sum(1 for w in toks if sum(ch.isalnum() for ch in w) <= 2) / len(toks) < 0.7:
        return False
    hi_lex, en_lex = _lexicons()
    for w in toks:
        core = w.strip(_EDGE_PUNCT)
        if en_lex is not None and len(core) >= 4 and core.isalpha() and core.lower() in en_lex:
            return False
        if hi_lex is not None and len(core) >= 3 and core in hi_lex:
            return False
    return True


def clean_cell(text: str, strip_legacy: bool = False) -> str:
    """Repair one table cell the way clean_page repairs a line.

    Applied to cells BEFORE a table is rendered, so the rendered table is already in
    its final form and clean_page leaves it untouched — keeping tables.json and
    structured.md in agreement. `strip_legacy` must match what clean_page is told.
    """
    t = unicodedata.normalize("NFC", text or "")
    t = _DANDA_MISREAD.sub("।", t)
    t = _drop_foreign_tokens(t)[0]
    if strip_legacy:
        t = _strip_legacy_tokens(t)[0]
    return t.strip()


def is_mojibake_line(text: str) -> bool:
    """Public form of the legacy-font line check (used on table rows)."""
    return _is_mojibake_line(text)


def clean_page(text: str, strip_legacy: bool | None = None) -> tuple[str, dict]:
    """(repaired text, repair counts). Never invents text — only fixes or cuts.

    strip_legacy: remove non-dictionary Latin tokens (legacy-font junk). None =
    decide from legacy_font_signature(text); vlm_ocr decides once per page and
    passes the same decision to the table cleaner.

    Table rows (markdown "|…" / HTML "<tr>…") are exempt from loop collapsing,
    speckle removal and duplicate-line collapsing: header cells legitimately
    repeat across column groups ("बजट अनुमान | बजट अनुमान | …"). Only rows that
    are wholesale legacy-font junk are dropped.
    """
    t = unicodedata.normalize("NFC", text or "")
    if strip_legacy is None:
        strip_legacy = legacy_font_signature(t)
    rep = Counter()
    t, rep["danda_fixed"] = _DANDA_MISREAD.subn("।", t)

    # Two kinds of cut, accounted separately (in content characters):
    #   junk  foreign-script words, legacy-font/speckle lines and words — text that
    #         was never real; a page that is mostly junk is not trusted
    #   loop  back-to-back repeats collapsed to one copy — the engine repeating
    #         itself; nothing real is lost, the page only deserves a second read
    def cc(x: str) -> int:
        return sum(1 for ch in x if ch.isalnum())

    before = cc(t)
    junk = loop = 0
    lines = t.split("\n")
    keep = [True] * len(lines)
    for i, ln in enumerate(lines):
        c0 = cc(ln)
        ln, nf = _drop_foreign_tokens(ln)
        rep["foreign_words_removed"] += nf
        c1 = cc(ln)
        table = bool(_TABLE_LINE.match(ln))
        if not table:
            ln, nl = _collapse_token_loops(ln)
            rep["loop_words_removed"] += nl
        c2 = cc(ln)
        if _is_mojibake_line(ln):
            keep[i] = False
            rep["mojibake_lines_dropped"] += 1
        elif not table and _is_noise_line(ln):
            keep[i] = False
            rep["noise_lines_dropped"] += 1
        elif strip_legacy:
            ln, nl = _strip_legacy_tokens(ln)
            rep["legacy_font_words_removed"] += nl
        c3 = cc(ln) if keep[i] else 0
        junk += (c0 - c1) + (c2 - c3)
        loop += c1 - c2
        lines[i] = ln

    # back-to-back identical lines (blank lines between them don't break the run).
    # Table rows are exempt: identical rows are rare but real, and tables.json must
    # keep agreeing with the Markdown.
    idx = [i for i, ln in enumerate(lines) if keep[i] and ln.strip() and not _TABLE_LINE.match(ln)]
    j = 0
    while j < len(idx):
        key = lines[idx[j]].strip()
        k = j + 1
        while k < len(idx) and lines[idx[k]].strip() == key:
            k += 1
        if k - j >= LINE_MIN_RUN and not _is_filler_block(key.split()):
            for m in idx[j + 1:k]:
                keep[m] = False
                loop += cc(lines[m])
            rep["repeated_lines_removed"] += k - j - 1
        j = k

    out = "\n".join(ln for ln, k in zip(lines, keep) if k)
    rep = {k: v for k, v in rep.items() if v}
    if strip_legacy:
        rep["legacy_font"] = True
    rep["removed_share"] = round(junk / before, 3) if before else 0.0
    if loop:
        rep["loop_share"] = round(loop / before, 3) if before else 0.0
    return out, rep


# ── the page verdict ─────────────────────────────────────────────────────────
TRUNCATED = "output truncated"          # reason prefix; assess_document looks for it

@dataclass
class PageVerdict:
    status: str                    # ok | blank | suspect | reject
    score: float                   # 0..1, higher is better — picks between attempts
    reasons: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    text: str = ""                 # the repaired text; what may be indexed

    @property
    def indexable(self) -> bool:
        """Text that may be written to structured.md (and so reach the index)."""
        return self.status in ("ok", "suspect")

    def to_dict(self) -> dict:
        return {"status": self.status, "score": round(self.score, 3),
                "reasons": self.reasons, "stats": self.stats}


def assess_page(text: str, *, ink_ratio: float | None = None,
                finish_reason: str | None = None,
                mean_logprob: float | None = None,
                repair: bool = True,
                strip_legacy: bool | None = None) -> PageVerdict:
    """Repair, then classify one page of OCR output.

    ink_ratio      share of dark pixels in the rendered page (vlm_ocr.ink_ratio).
                   Distinguishes a genuinely blank page from one whose text the
                   engine failed to read.
    finish_reason  the engine's stop reason; "length" means it hit the token cap,
                   which for OCR almost always means a loop ate the rest of the page.
    mean_logprob   mean token log-probability of the engine's output — recorded
                   and lightly weighted (a hint, not a verdict).
    """
    original = unicodedata.normalize("NFC", text or "")
    cleaned, repairs = clean_page(original, strip_legacy) if repair else (original, {})
    t = normalize_for_scoring(cleaned)
    rejects: list[str] = []
    suspects: list[str] = []
    stats: dict = {"repairs": repairs} if repairs else {}
    penalty = 0.0

    def reject(why, p=0.6):
        nonlocal penalty
        rejects.append(why)
        penalty += p

    def suspect(why, p=0.2):
        nonlocal penalty
        suspects.append(why)
        penalty += p

    # ── 0. is there anything here at all? ────────────────────────────────────
    had = _content_chars(normalize_for_scoring(original))
    alnum = _content_chars(t)
    stats["content_chars"] = alnum
    if ink_ratio is not None:
        stats["ink_ratio"] = round(ink_ratio, 4)
    if had < MIN_CONTENT_CHARS:
        if ink_ratio is not None and ink_ratio >= INK_CONTENT_RATIO:
            # Not blank: the engine failed to read it. Reject (nothing indexable)
            # so the caller escalates to the fallback engine.
            return PageVerdict("reject", 0.1,
                               [f"page has ink ({ink_ratio:.0%}) but only {had} characters were read"],
                               stats, cleaned)
        return PageVerdict("blank", 1.0, ["no text on the page"], stats, cleaned)
    if alnum < MIN_CONTENT_CHARS:
        return PageVerdict("reject", 0.0,
                           [f"everything on the page was garbage ({_repair_summary(repairs)})"],
                           stats, cleaned)

    n = max(len(t), 1)

    # ── 1. engine-level signals ──────────────────────────────────────────────
    if finish_reason == "length":
        suspect(TRUNCATED + " — the engine hit its length cap (usually a loop)", 0.3)
    if mean_logprob is not None:
        stats["mean_logprob"] = round(mean_logprob, 3)
        if mean_logprob < -1.0:
            suspect(f"engine was unsure (mean logprob {mean_logprob:.2f})", 0.1)

    # ── 2. how much had to be cut ────────────────────────────────────────────
    removed = repairs.get("removed_share", 0.0)
    if removed > MAX_REPAIRED_SHARE:
        reject(f"{removed:.0%} of the page was junk and had to be cut ({_repair_summary(repairs)})")
    elif removed > SUSPECT_REPAIRED_SHARE:
        suspect(f"{removed:.0%} of the page was junk and was cut ({_repair_summary(repairs)})", 0.15)
    looped = repairs.get("loop_share", 0.0)
    if looped > LOOP_SUSPECT_SHARE:
        # Collapsing kept one copy, so nothing real was lost — but an engine that
        # looped may also have skipped part of the page: worth a second read.
        suspect(f"the engine repeated itself ({looped:.0%} of the output was repeats, collapsed)", 0.15)

    # ── 3. encoding damage ───────────────────────────────────────────────────
    ctrl = sum(1 for c in t if (ord(c) < 0x20 and c not in "\t\n\r") or 0x7F <= ord(c) <= 0x9F)
    pua = sum(1 for c in t if "" <= c <= "")
    repl = t.count("�")
    if ctrl or pua or repl:
        stats.update(ctrl=ctrl, pua=pua, replacement=repl)
    if ctrl / n > MAX_CTRL_RATIO:
        reject(f"{ctrl} control character(s) — broken font encoding")
    if pua / n > MAX_PUA_RATIO:
        reject(f"{pua} private-use glyph(s) — legacy font, not Unicode")
    if repl / n > MAX_REPLACEMENT_RATIO:
        reject(f"{repl} replacement character(s) (U+FFFD)")

    # ── 4. symbols and script mixing ─────────────────────────────────────────
    nonspace = [c for c in t if not c.isspace()]
    junk = sum(1 for c in nonspace if c in _JUNK) / max(len(nonspace), 1)
    stats["symbol_ratio"] = round(junk, 4)
    if junk > MAX_SYMBOL_RATIO:
        reject(f"symbol soup ({junk:.0%} of characters are stray symbols)")
    elif junk > SUSPECT_SYMBOL_RATIO:
        suspect(f"many stray symbols ({junk:.0%})", 0.1)

    letters = [c for c in t if c.isalpha()]
    n_letters = max(len(letters), 1)
    dev_share = sum(1 for c in letters if _DEV_LO <= c <= _DEV_HI) / n_letters
    latin_share = sum(1 for c in letters if c.isascii()) / n_letters
    stats.update(dev_share=round(dev_share, 3), latin_share=round(latin_share, 3))

    words = t.split()
    tokens = [w for w in words if any(ch.isalpha() for ch in w)]
    if len(tokens) >= 20:
        mixed = sum(1 for w in tokens
                    if any(_DEV_LO <= ch <= _DEV_HI for ch in w)
                    and any(ch.isascii() and ch.isalpha() for ch in w)) / len(tokens)
        stats["mixed_script_words"] = round(mixed, 3)
        if mixed > MAX_MIXED_SCRIPT:
            reject(f"{mixed:.0%} of words mix Devanagari and Latin letters")
        elif mixed > SUSPECT_MIXED_SCRIPT:
            suspect(f"{mixed:.0%} of words mix Devanagari and Latin letters", 0.1)

    # ── 5. Devanagari quality ────────────────────────────────────────────────
    _, hi_model = _ngram_models()
    hi_lex, _ = _lexicons()
    dev_words = [w for w in _DEV_WORD.findall(t) if len(w) >= 2]
    if dev_share >= 0.15 and len(dev_words) >= DEV_MIN_WORDS:
        orphan = _orphan_matra_ratio(t)
        illegal = _illegal_sequences(dev_words) / len(dev_words)
        stats.update(orphan_matra=round(orphan, 3), illegal_seq=round(illegal, 3))
        if orphan > MAX_ORPHAN_MATRA:
            reject(f"{orphan:.0%} of Hindi words start with a vowel sign (dropped conjuncts)")
        elif orphan > SUSPECT_ORPHAN_MATRA:
            suspect(f"{orphan:.0%} of Hindi words start with a vowel sign")
        if illegal > MAX_ILLEGAL_SEQ:
            reject(f"illegal Devanagari sign order in {illegal:.0%} of words (misread conjuncts)")
        elif illegal > SUSPECT_ILLEGAL_SEQ:
            suspect(f"illegal Devanagari sign order in {illegal:.0%} of words")
        if hi_lex is not None:
            hit = sum(1 for w in dev_words if w in hi_lex) / len(dev_words)
            stats["hi_dict"] = round(hit, 3)
            if hit < MIN_HI_DICT:
                reject(f"only {hit:.0%} of Hindi words are real words")
            elif hit < SUSPECT_HI_DICT:
                suspect(f"only {hit:.0%} of Hindi words are real words")
        hs = _ngram_score(hi_model, dev_words)
        if hs is not None:
            stats["hi_trigram"] = round(hs, 3)
            if hs < TRIGRAM_HI_FLOOR:
                reject(f"garbled Devanagari (trigram {hs:.2f})")
            elif hs < SUSPECT_TRIGRAM_HI:
                suspect(f"unusual Devanagari (trigram {hs:.2f})")

    # ── 6. Latin quality (where legacy-font mojibake lives) ──────────────────
    lat_words = _latin_tokens(t)
    if latin_share >= 0.5 and len(lat_words) >= EN_MIN_WORDS:
        s = _english_signals(lat_words)
        es, hit = s["trigram"], s["dict"]
        stats.update(en_dict=None if hit is None else round(hit, 3),
                     en_trigram=None if es is None else round(es, 3),
                     vowelless=round(s["vowelless"], 3), intraword_punct=round(s["intraword"], 3))
        other_bad = ((es is not None and es < TRIGRAM_EN_MIN) or s["vowelless"] > MAX_VOWELLESS
                     or s["intraword"] > MAX_INTRAWORD_PUNCT)
        # Govt English is full of names, codes and abbreviations, so no single
        # signal is trusted alone: rejection needs a low dictionary rate AND a
        # second signal agreeing.
        if hit is not None and hit < MIN_EN_DICT and other_bad:
            reject(f"not readable English (dict {hit:.0%}, trigram {es if es is None else round(es, 2)}, "
                   f"vowel-less {s['vowelless']:.0%}, in-word punctuation {s['intraword']:.0%})")
        elif hit is not None and (hit < SUSPECT_EN_DICT or (hit < 0.7 and other_bad)):
            suspect(f"English reads poorly (dict {hit:.0%}, trigram {es if es is None else round(es, 2)})", 0.1)

    score = max(0.0, min(1.0, 1.0 - penalty))
    if rejects:
        return PageVerdict("reject", score, rejects + suspects, stats, cleaned)
    if suspects:
        return PageVerdict("suspect", score, suspects, stats, cleaned)
    return PageVerdict("ok", score, [], stats, cleaned)


def _repair_summary(repairs: dict) -> str:
    names = {"foreign_words_removed": "foreign-script words", "loop_words_removed": "looped words",
             "mojibake_lines_dropped": "legacy-font lines", "repeated_lines_removed": "repeated lines",
             "noise_lines_dropped": "speckle lines", "legacy_font_words_removed": "legacy-font words",
             "danda_fixed": "danda fixes"}
    parts = [f"{v} {names[k]}" for k, v in repairs.items() if k in names]
    return ", ".join(parts) or "no repairs"


def assess_document(verdicts: list[PageVerdict]) -> tuple[str, str]:
    """(status, summary) for a whole document from its final page verdicts.

    status: ok       every content page is indexable
            partial  some content pages were dropped, or kept only up to where the
                     engine looped and hit its length cap — still worth indexing,
                     and the next pass (--retry-flagged) reads them again
            failed   no content page survived; nothing should be indexed
            blank    the document has no text at all
    """
    content = [v for v in verdicts if v.status != "blank"]
    kept = [v for v in content if v.indexable]
    dropped = len(content) - len(kept)
    # A looped page keeps only the text before the loop: measured on a clean GAD
    # letter, 8% of the page survived. "Accepted" would hide that loss for good.
    cut = sum(1 for v in kept if any(r.startswith(TRUNCATED) for r in v.reasons))
    if not content:
        return "blank", "no text on any page"
    if not kept:
        reasons = Counter(re.sub(r"\s*\(.*", "", v.reasons[0]) for v in content if v.reasons)
        top = "; ".join(f"{r} ×{k}" for r, k in reasons.most_common(3))
        return "failed", f"all {len(content)} content page(s) rejected: {top}"
    if dropped or cut:
        parts = ([f"{dropped} of {len(content)} content page(s) rejected"] if dropped else []) + \
                ([f"{cut} page(s) cut short by an engine loop"] if cut else [])
        return "partial", "; ".join(parts)
    return "ok", f"{len(kept)} content page(s) accepted"


# ── text-layer gate (moved from ocr_scanned; behaviour unchanged) ────────────
def text_layer_quality(text: str, page_count: int) -> tuple[bool, str]:
    """(keep_embedded_text, reason). Conservative: anything unclear goes to OCR."""
    text = unicodedata.normalize("NFC", text or "")
    letters = sum(1 for c in text if c.isalpha())
    if page_count > 0 and len(text.strip()) < MIN_CHARS_PER_PAGE * page_count:
        return False, "image-only (too little text for its page count)"
    if letters < 50:
        return False, "no meaningful text"

    # C0 control characters mean the PDF's font maps glyphs to codepoints that are
    # not text at all: "छ\x02ीसगढ़ शासन" — every conjunct replaced by 0x02/0x0F/0x19.
    # Decisive, so it runs before the script checks.
    ctrl = sum(1 for c in text if (ord(c) < 0x20 and c not in "\t\n\r") or ord(c) == 0x7F)
    if ctrl:
        return False, (f"broken font encoding ({ctrl} control char(s) — conjuncts map to "
                       f"non-text codepoints)")

    # The same broken-cmap PDF read by poppler drops the glyphs silently instead:
    # "छ ीसगढ़" — no control chars, but matras stranded at token starts.
    orphan = _orphan_matra_ratio(text)
    if orphan > MAX_ORPHAN_MATRA_RATIO:
        return False, (f"dropped conjuncts ({orphan:.0%} of tokens start with a matra — "
                       f"text layer lost characters)")

    dev = sum(1 for c in text if c.isalpha() and _DEV_LO <= c <= _DEV_HI) / max(letters, 1)
    en_model, hi_model = _ngram_models()

    if dev > DEVANAGARI_RATIO:
        hs = _ngram_score(hi_model, re.findall(r"[ऀ-ॿ]{2,}", text))
        if hs is not None and hs < TRIGRAM_HI_FLOOR:
            return False, f"garbled Devanagari (trigram {hs:.2f} < {TRIGRAM_HI_FLOOR})"
        return True, f"clean Devanagari ({dev:.0%})"

    low = text.lower()
    words = [w for w in re.findall(r"[a-z']+", low) if len(w) > 1]
    junk_chars = set("~{}<>|\\^`_=[]/*")
    junk = sum(1 for c in text if c in junk_chars) / max(len(text), 1)
    vowelless = sum(1 for w in words if not (set(w) & _VOWELS)) / max(len(words), 1)
    es = _ngram_score(en_model, re.findall(r"[a-z]{2,}", low))

    if es is None:
        fnr = sum(1 for w in words if w in _FUNCTION_WORDS) / max(len(words), 1)
        if fnr > FUNCTION_WORD_RATIO and junk < MAX_JUNK_RATIO and vowelless < MAX_VOWELLESS:
            return True, f"clean English (heuristic: function words {fnr:.0%}, vowel-less {vowelless:.0%})"
        return False, f"mojibake/unreadable (heuristic: function words {fnr:.0%}, vowel-less {vowelless:.0%})"

    if es >= TRIGRAM_EN_MIN and vowelless < MAX_VOWELLESS and junk < MAX_JUNK_RATIO:
        return True, f"clean English (trigram {es:.2f}, vowel-less {vowelless:.0%}, junk {junk:.0%})"
    return False, f"mojibake/unreadable (trigram {es:.2f}, vowel-less {vowelless:.0%}, junk {junk:.0%})"
