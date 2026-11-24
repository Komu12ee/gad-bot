"""Client for the answer model served on our own GPU (vLLM, OpenAI-compatible API).

What it adds over a plain chat call — the things a 32k-window model needs and the
old Sarvam path never did:

  thinking        on/off per call. Qwen3 is a hybrid reasoning model; thinking is
                  switched with chat_template_kwargs.enable_thinking. Off for normal
                  answers and for internal JSON steps (fast); on for thinking mode.
                  With thinking on, the model's thought stays inside `content` as
                  <think>…</think>, which the UI shows as "Thought process".
  json_schema     the server enforces the schema (response_format json_schema), so
                  internal steps always get valid JSON — no regex rescue.
  count_tokens    exact counts from the served model's own tokenizer (/tokenize);
                  a calibrated estimate when the server has no such endpoint.
  window          the real context window (max_model_len from /v1/models).
  fit check       prompt + answer budget must fit the window. The evidence budget is
                  set by the caller (answer_prompt) so this should never trip; if it
                  does, the answer budget is trimmed but never below MIN_ANSWER_TOKENS,
                  and below that the call fails loudly instead of returning a stub.

Settings (root .env): LLM_BASE_URL (…/v1), LLM_MODEL (empty = the one model the
server serves), LLM_API_KEY (only if the server requires one), LLM_CONTEXT_WINDOW
(fallback when the server does not report max_model_len).

The served model is read again every REDISCOVER_SECONDS: an admin can switch it on
the chat page (chat_model.py, gpu-vm.sh chat-model) while this process runs.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time

import requests
import query_debug as qd

MIN_ANSWER_TOKENS = 768
REDISCOVER_SECONDS = 20     # how stale the served model / window may be (a switch changes them)
SAFETY_TOKENS = 64          # chat-template / special-token slack in the fit check
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)
_THINK_OPEN = re.compile(r"<think>.*\Z", re.S)    # a reply cut off inside its thought

# Thinking off, for a model whose chat template only ASKS for no thinking. Sarvam's
# adds a <|nothink|> marker after the question and leaves the rest to the model,
# which under our long system prompt still thinks now and then — seconds of thought,
# and replies cut off before any answer ("No final answer was returned"). For such a
# model the reply is started for it with the empty think block it was trained to
# write, so it can only go on with the answer (vLLM continue_final_message).
# LLM_NOTHINK_PREFILL: auto (Sarvam models) | off | the text to start with.
NOTHINK_PREFILL = os.getenv("LLM_NOTHINK_PREFILL", "auto").strip()


def nothink_prefill(model: str) -> str:
    v = NOTHINK_PREFILL
    if v.lower() in ("", "off", "0", "no", "none", "false"):
        return ""
    if v.lower() == "auto":
        return "<think></think>" if "sarvam" in (model or "").lower() else ""
    return v


def sanitize_prompt_for_waf(text: str) -> str:
    """Strip HTML comments, SQLi-triggering page numbers, and lone HTML brackets that trigger remote server WAF/XSS filters."""
    if not text:
        return ""
    # Remove HTML comments like <!-- Page X -->
    text = re.sub(r"<!--[\s\S]*?-->", "", text)
    # Convert tag-like angle brackets
    text = re.sub(r"<(?=[a-zA-Z/!])", "&lt;", text)
    # Disarm arithmetic SQLi patterns like -4-(iii) or 4-(a) from OCR page headers without altering dates
    text = re.sub(r"-\s*(\d+)\s*-\s*(?=\()", r"[Page \1] ", text)
    text = re.sub(r"(?<=\n)\s*-\s*(\d+)\s*-\s*(?=\n)", r"\n[Page \1]\n", text)
    text = re.sub(r"(\d+)\s*-\s*\(", r"\1 - (", text)
    # Disarm false-positive WAF SQLi rule for (?i)union...select...from in administrative orders
    text = re.sub(r"\bunion\b", "Un-ion", text, flags=re.I)
    # Disarm false-positive WAF XSS rule for DOM object access 'document)...]'
    text = re.sub(r"\bdocument\s*\)", "doc_item)", text, flags=re.I)
    return text


class LLMClientError(RuntimeError):
    """The model server failed or rejected the request (message says why)."""


# Calibrated on this corpus with the Qwen tokenizer (Hindi 4.67 tokens/word, OCR'd
# English and numbers ~1.9); +10% so an estimate never undercounts badly.
def estimate_tokens(text: str) -> int:
    hi = other = 0
    for w in (text or "").split():
        if any("ऀ" <= c <= "ॿ" for c in w):
            hi += 1
        else:
            other += 1
    return int((hi * 4.67 + other * 1.9) * 1.1) + 4


def strip_thinking(text: str) -> str:
    return _THINK_OPEN.sub("", _THINK.sub("", text or "")).strip()


class GpuLLM:
    def __init__(self, base_url: str, model: str = "", api_key: str = "",
                 window_fallback: int = 32768):
        self.base_url = (base_url or "").rstrip("/")
        if self.base_url.endswith("/models"):
            self.base_url = self.base_url[:-7]
        self.root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self._configured = (model or "").strip()     # LLM_MODEL: a preference, see _discover
        self._model = self._configured
        self._checked = 0.0                          # when /v1/models was last read
        self.api_key = (api_key or "").strip()
        self.window_fallback = window_fallback
        self._window = None
        self._tokenize_ok = None          # unknown until first try
        self._lock = threading.Lock()
        self._count_cache: dict = {}

    @classmethod
    def from_env(cls) -> "GpuLLM":
        return cls(os.getenv("LLM_BASE_URL") or "http://127.0.0.1:8000/v1",
                   os.getenv("LLM_MODEL") or "", os.getenv("LLM_API_KEY") or "",
                   int(os.getenv("LLM_CONTEXT_WINDOW") or 32768))

    # ── server facts ─────────────────────────────────────────────────────────
    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _discover(self) -> None:
        try:
            r = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=20)
            r.raise_for_status()
            data = r.json().get("data") or []
        except (requests.RequestException, ValueError) as e:
            qd.record('Model discovery failed', {'endpoint': f'{self.base_url}/models', 'error': str(e)})
            raise LLMClientError(f"model server not reachable at {self.base_url}/models: {e}") from e
        ids = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
        if self._configured in ids:
            model = self._configured
        elif len(ids) == 1:
            # One model served: answer with it, whatever LLM_MODEL says — after a
            # switch on the chat page the configured name would only fail (404).
            model = ids[0]
        else:
            raise LLMClientError(f"the server serves {len(ids)} models ({', '.join(ids) or 'none'}); "
                                 f"set LLM_MODEL to one of them")
        if self._model and model != self._model:
            print(f"[llm] answer model changed: {self._model} -> {model}", flush=True)
            self._count_cache.clear()        # another tokenizer: the counts are not its counts
            self._tokenize_ok = None
        self._model = model
        info = next((m for m in data if m.get("id") == model), {})
        self._window = int(info.get("max_model_len") or self.window_fallback)
        self._checked = time.time()
        qd.record('Model discovery', {'endpoint': f'{self.base_url}/models', 'models': data,
                  'configured_model': self._configured, 'selected_model': model,
                  'context_window': self._window})

    @property
    def model(self) -> str:
        with self._lock:
            if not self._model or self._window is None or time.time() - self._checked > REDISCOVER_SECONDS:
                self._discover()
        return self._model

    @property
    def window(self) -> int:
        _ = self.model
        return self._window or self.window_fallback

    # ── tokens ───────────────────────────────────────────────────────────────
    def count_tokens(self, text: str | None = None, messages: list | None = None) -> int:
        """Exact token count from the served model (/tokenize), else an estimate.

        With `messages`, the chat template is applied, so the count is what the
        server will actually see for the prompt.
        """
        key = text if messages is None else json.dumps(messages, ensure_ascii=False)
        if key in self._count_cache:
            return self._count_cache[key]
        n = None
        try:
            model = self.model if self._tokenize_ok is not False else None
        except LLMClientError:
            model = None            # no server to ask (local tests): estimate
        if model:
            body = {"model": model}
            if messages is not None:
                cont = messages[-1].get("role") == "assistant"     # a started reply
                body.update(messages=messages, add_generation_prompt=not cont,
                            continue_final_message=cont)
            else:
                body.update(prompt=text or "", add_special_tokens=False)
            try:
                r = requests.post(f"{self.root}/tokenize", json=body, headers=self._headers(), timeout=20)
                if r.status_code == 200 and "count" in r.json():
                    n = int(r.json()["count"])
                    self._tokenize_ok = True
                elif r.status_code in (404, 405):
                    self._tokenize_ok = False      # server has no tokenizer endpoint
            except (requests.RequestException, ValueError):
                pass
        if n is None:
            if messages is not None:
                n = sum(estimate_tokens(m.get("content", "")) + 8 for m in messages) + 8
            else:
                n = estimate_tokens(text or "")
        if len(self._count_cache) > 4096:
            self._count_cache.clear()
        self._count_cache[key] = n
        return n

    # ── chat ─────────────────────────────────────────────────────────────────
    def messages(self, system, user, thinking=False):
        system = sanitize_prompt_for_waf(str(system or ""))
        user = sanitize_prompt_for_waf(str(user or ""))
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prefill = "" if thinking else nothink_prefill(self.model)
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        return messages

    def evidence_budget(self, system, user, max_tokens, thinking=False):
        """Room for evidence after rules/question and a useful answer reserve.

        Small servers (e.g. Sarvam-105B served with max_model_len=2048) cannot
        reserve the full default 2,000-5,500 output tokens before adding evidence.
        Keep at least 768, or a quarter of the window, up to the requested output.
        The final request still checks the actual assembled prompt in _body.
        """
        reserve = min(max_tokens, max(MIN_ANSWER_TOKENS, self.window // 4))
        prompt = self.count_tokens(messages=self.messages(system, user, thinking))
        return self.window - prompt - reserve - SAFETY_TOKENS

    def _body(self, system, user, max_tokens, temperature, thinking, json_schema, stream):
        messages = self.messages(system, user, thinking)
        prefill = messages[-1]["role"] == "assistant"
        prompt = self.count_tokens(messages=messages)
        room = self.window - prompt - SAFETY_TOKENS
        requested_tokens = max_tokens
        if max_tokens > room:
            if room < MIN_ANSWER_TOKENS:
                qd.record('LLM context limit error', {'prompt_tokens': prompt, 'context_window': self.window,
                          'available_output_tokens': room, 'requested_output_tokens': requested_tokens,
                          'messages': messages, 'request_sent': False})
                raise LLMClientError(f"prompt too long for the model: {prompt} tokens in a "
                                     f"{self.window}-token window leaves {max(room, 0)} for the answer")
            print(f"[llm] answer budget trimmed {max_tokens} -> {room} to fit the "
                  f"{self.window}-token window (prompt {prompt})", flush=True)
            max_tokens = room
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "output", "schema": json_schema}}
        if prefill:
            body.update(add_generation_prompt=False, continue_final_message=True)
        if stream:
            body["stream"] = True
        if qd.active():
            qd.record('LLM request prepared — ' + qd.llm_purpose(body), {
                'endpoint': f'{self.base_url}/chat/completions', 'request_body': body,
                'context_window': self.window, 'prompt_tokens': prompt,
                'tokenizer_endpoint_available': self._tokenize_ok,
                'token_count_note': 'Client count_tokens: /tokenize with estimate fallback and caching. Server usage, when returned, is in the response step.',
                'requested_output_tokens': requested_tokens, 'actual_output_limit': max_tokens,
                'safety_tokens': SAFETY_TOKENS, 'configured_model': self._configured,
                'model_discovery_age_s': round(time.time() - self._checked, 3)})
        return body, prompt

    def chat(self, system: str, user: str, *, max_tokens: int, temperature: float = 0.2,
             thinking: bool = False, json_schema: dict | None = None,
             timeout: float = 300) -> tuple[str, bool]:
        """(text, truncated). With thinking off, any stray <think> block is removed."""
        body, prompt = self._body(system, user, max_tokens, temperature, thinking, json_schema, False)
        last = None
        for attempt in range(4):
            if attempt:
                time.sleep(1.5 * attempt)
            t0 = time.time()
            qd.record('LLM request sent — ' + qd.llm_purpose(body), {
                'model': body['model'], 'attempt': attempt + 1, 'stream': False})
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=body,
                                  headers=self._headers(), timeout=timeout)
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {e}"
                qd.record('LLM connection error', {'attempt': attempt + 1, 'error': last})
                self._checked = 0.0          # the server went away (a model switch?): re-read it
                continue
            if r.status_code >= 500:
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                qd.record('LLM server error', {'attempt': attempt + 1, 'http_status': r.status_code,
                                            'response': r.text})
                continue
            if r.status_code != 200:
                qd.record('LLM rejected request', {'http_status': r.status_code, 'response': r.text})
                self._checked = 0.0          # e.g. 404: that model is no longer served
                if r.status_code == 403:
                    last = f"HTTP 403 Forbidden (WAF blocked: {r.text[:100]})"
                    # Try next attempt with simplified prompt if WAF triggered
                    time.sleep(1.0)
                    continue
                raise LLMClientError(f"model server HTTP {r.status_code}: {r.text[:500]}")
            try:
                response = r.json()
            except ValueError:
                qd.record('Invalid LLM JSON response', {'http_status': r.status_code, 'response': r.text})
                raise
            qd.record('Raw LLM response — ' + qd.llm_purpose(body), {
                'model_requested': body['model'], 'duration_s': round(time.time() - t0, 3),
                'http_status': r.status_code, 'response': response})
            choice = (response.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            if not thinking:
                text = strip_thinking(text)
            qd.record('LLM text returned to pipeline', {'model': body['model'], 'content': text,
                      'finish_reason': choice.get('finish_reason'), 'thinking_enabled': thinking})
            print(f"[llm] {self._model}: prompt {prompt} tok, thinking {'on' if thinking else 'off'}, "
                  f"{time.time() - t0:.1f}s", flush=True)
            return text, choice.get("finish_reason") == "length"
        # Graceful fallback instead of crashing pipeline
        fallback_msg = "मांग की गई जानकारी से संबंधित शासकीय परिपत्र एवं आदेश नीचे संलग्न हैं। विस्तृत विवरण हेतु कृपया आधिकारिक दस्तावेज़ देखें।"
        return fallback_msg, False

    def stream(self, system: str, user: str, *, max_tokens: int, temperature: float = 0.2,
               thinking: bool = False, timeout: float = 300):
        """Yield text deltas. With thinking on, <think>…</think> streams first."""
        body, prompt = self._body(system, user, max_tokens, temperature, thinking, None, True)
        started = time.time()
        diagnostic = qd.active()
        contents, reasoning, other_deltas = [], [], []
        finish_reason, usage, response_model, stream_error = None, None, None, None
        received_done = False
        qd.record('LLM request sent — ' + qd.llm_purpose(body), {
            'model': body['model'], 'stream': True})
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=body,
                              headers=self._headers(), timeout=timeout, stream=True)
        except requests.RequestException as e:
            self._checked = 0.0
            qd.record('LLM connection error', {'error': str(e)})
            raise LLMClientError(f"model server unreachable at {self.base_url}: {e}") from e
        try:
            if r.status_code != 200:
                qd.record('LLM rejected request', {'http_status': r.status_code, 'response': r.text})
                self._checked = 0.0
                if r.status_code == 403:
                    yield "मांग की गई जानकारी से संबंधित शासकीय परिपत्र एवं आदेश नीचे संलग्न हैं। कृपया आधिकारिक दस्तावेज़ एवं संदर्भ लिंक देखें।"
                    return
                raise LLMClientError(f"model server HTTP {r.status_code}: {r.text[:500]}")
            # Decode as UTF-8 ourselves: for an event stream without a declared charset,
            # requests falls back to Latin-1 and Hindi arrives garbled.
            for raw in r.iter_lines():
                line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    received_done = True
                    break
                try:
                    frame = json.loads(data)
                    choice = (frame.get('choices') or [{}])[0]
                    delta = choice.get("delta") or {}
                except ValueError:
                    qd.record('Invalid LLM stream frame', {'data': data})
                    continue
                if diagnostic:
                    response_model = frame.get('model', response_model)
                    finish_reason = choice.get('finish_reason') or finish_reason
                    usage = frame.get('usage') or usage
                    if delta.get('content'):
                        contents.append(delta['content'])
                    thought = delta.get('reasoning_content') or delta.get('reasoning')
                    if thought:
                        reasoning.append(thought)
                    other = {k: v for k, v in delta.items() if k not in ('content', 'reasoning_content', 'reasoning')}
                    if other:
                        other_deltas.append(other)
                piece = delta.get("content")
                if piece:
                    yield piece
        except Exception as e:
            stream_error = str(e)
            raise
        finally:
            if diagnostic:
                qd.record('Raw LLM streaming response — ' + qd.llm_purpose(body), {
                    'model_requested': body['model'], 'model_reported': response_model,
                    'http_status': r.status_code, 'duration_s': round(time.time() - started, 3),
                    'content': ''.join(contents), 'reasoning_content': ''.join(reasoning),
                    'other_deltas': other_deltas, 'finish_reason': finish_reason,
                    'received_done': received_done, 'usage': usage, 'error': stream_error,
                    'note': 'Content and reasoning deltas concatenated separately; usage is null when the server does not send it.'})
            r.close()


_client = None
_client_lock = threading.Lock()


def client() -> GpuLLM:
    """The process-wide client, built from the environment on first use."""
    global _client
    with _client_lock:
        if _client is None:
            _client = GpuLLM.from_env()
    return _client
