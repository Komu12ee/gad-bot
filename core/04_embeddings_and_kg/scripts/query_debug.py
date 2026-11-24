"""Request-local diagnostics for authenticated admins; no extra model calls.

The Flask worker activates a recorder explicitly. Non-admin queries pay only a
ContextVar lookup. Large traces live beside pipeline data, not in chat payloads.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import uuid

_current = ContextVar("query_debug", default=None)
MAX_BYTES = 32 * 1024 * 1024


def active():
    return _current.get() is not None


@contextmanager
def capture(recorder):
    token = _current.set(recorder)
    try:
        yield recorder
    finally:
        _current.reset(token)


def record(title, output):
    recorder = _current.get()
    return recorder.add(title, output) if recorder else None


def chunks(hits):
    """Keep full stored text, identifiers, payload metadata and stage scores."""
    rows = []
    for rank, hit in enumerate(hits or [], 1):
        point = hit.get("point")
        rows.append({"rank": hit.get("rank", rank), "id": str(getattr(point, "id", "")),
                     "score": hit.get("score"), "neighbor_of": hit.get("neighbor_of"),
                     "payload": getattr(point, "payload", None) or {}})
    return rows


def ranking(title, scores, points, **extra):
    if active():
        record(title, {**extra, "chunks": chunks([
            {"point": points[pid], "score": score, "rank": i}
            for i, (pid, score) in enumerate(scores, 1) if pid in points])})


def llm_purpose(body):
    props = body.get("response_format", {}).get("json_schema", {}).get("schema", {}).get("properties", {})
    if "sub_questions" in props:
        return "Search planning"
    if "sufficient" in props:
        return "Evidence coverage assessment"
    if "question" in props:
        return "Follow-up question rewrite"
    return "Structured model call" if props else "Answer generation"


def trace_path(root, owner, trace_id):
    if str(uuid.UUID(trace_id)) != trace_id:
        raise ValueError("Invalid trace ID")
    return Path(root) / "query_debug" / hashlib.sha256(owner.encode()).hexdigest() / (trace_id + ".json")


class Recorder:
    def __init__(self, root, owner, pipeline, request_data):
        self.root, self.owner = root, owner
        self.started = time.perf_counter()
        self.bytes = 0
        self.data = {"id": str(uuid.uuid4()), "pipeline": pipeline,
                     "created_at": datetime.now(timezone.utc).isoformat(), "steps": [],
                     "limit_reached": False}
        self.add("Question received", request_data)

    def add(self, title, output):
        # Diagnostics must never change a successful chatbot request into a failure.
        try:
            encoded = json.dumps(output, ensure_ascii=False, default=str)
            size = len(encoded.encode("utf-8"))
            if self.bytes + size > MAX_BYTES:
                self.data["limit_reached"] = True
                output = {"omitted": True, "reason": "32 MiB diagnostic limit reached",
                          "original_bytes": size}
            else:
                output = json.loads(encoded)
                self.bytes += size
            step = {"number": len(self.data["steps"]) + 1, "title": title,
                    "elapsed_s": round(time.perf_counter() - self.started, 3), "output": output}
            self.data["steps"].append(step)
            return step["number"]
        except Exception:
            logging.exception("Could not capture query diagnostic step")
            return None

    def save(self):
        tmp = None
        try:
            self.data["elapsed_s"] = round(time.perf_counter() - self.started, 3)
            target = trace_path(self.root, self.owner, self.data["id"])
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            tmp = target.with_suffix(".tmp")
            with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as file:
                json.dump(self.data, file, ensure_ascii=False)
            os.replace(tmp, target)
            return self.data["id"]
        except Exception:
            logging.exception("Could not save query diagnostics")
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            return None
