"""Read approved corrections from this bot's Node service, with existing bge-m3.

No persistent answer cache: the DB rechecks publication and dates on each lookup.
This module never indexes arbitrary feedback or writes admin records.
"""
import hashlib
import hmac
import json
import os
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pipeline_config as pc
import query_debug as qd


def enabled():
    return os.getenv("VERIFIED_CORRECTIONS", "1").strip().lower() not in {"0", "off", "false", "no"}


def lookup(question, encode, *, date_question=None):
    if not enabled():
        return []
    try:
        secret = os.getenv("GAD_SESSION_SECRET") or os.getenv("SESSION_SECRET") or os.getenv("JWT_SECRET") or ""
        if not secret:
            return []
        encoded = encode([question])
        if not encoded or not encoded[0]:
            return []
        key = hmac.new(secret.encode(), b"dept-internal-v1", hashlib.sha256).hexdigest()
        body = json.dumps({"question": question, "dateQuestion": date_question or question, "dense": encoded[0][0]}).encode()
        req = Request(f"http://127.0.0.1:{pc.NODE_PORT}/_internal/verified-corrections/search", data=body,
                      headers={"Content-Type": "application/json", "X-Corrections-Internal": key})
        with urlopen(req, timeout=5) as response:
            data = json.loads(response.read(100000))
        matches = data.get("matches", []) if data.get("success") else []
        hits = []
        for item in matches:
            source = f"verified-correction:{item['id']}:v{item['revision']}"
            text = (f"Admin-verified correction: {item['topic']}\nApplicable question: {item['question']}\n"
                    f"Effective: {item['effectiveFrom']} to {item.get('effectiveUntil') or 'until superseded'}; "
                    f"review by {item['reviewBy']}.\nOfficial source: {item['sourceTitle']}\n"
                    f"Approved answer: {item['answer']}\nSupporting source excerpt:\n{item['sourceExcerpt']}")
            point = SimpleNamespace(id=source, payload={"source": source, "text": text,
                "chunk": 0, "chunk_type": "verified_correction", "verified_correction": item})
            hits.append({"point": point, "score": item["score"], "rank": len(hits) + 1})
        qd.record("Verified correction lookup", {"query": question, "model": "BAAI/bge-m3",
            "scope": pc.PIPELINE, "matches": matches, "note": "Approval, review date and query-date applicability checked in PostgreSQL."})
        return hits
    except Exception as exc:
        # The ordinary corpus continues to answer. Never replace unavailable memory
        # with a previous request's corrections, or claim no documents exist.
        qd.record("Verified correction lookup unavailable", {"query": question, "error": str(exc)})
        print(f"  [corrections] lookup skipped: {type(exc).__name__}")
        return []
