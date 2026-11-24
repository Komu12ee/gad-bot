import urllib.request
import urllib.error
import json

def test_p(p):
    payload = {"model": "ocr-32b", "messages": [{"role": "user", "content": p}], "max_tokens": 10}
    req = urllib.request.Request(
        "http://103.51.8.88/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code

lines = [
    "has an id):",
    "{{numbered_candidates_with_context}}",
    "If none is the document's own number, answer null.",
    "EXAMPLE\nCANDIDATES:",
    '[1] (line 3, top) "क्रमांक: सामान्य-2021/प्र.क्र.45/18"',
    "[2] (line 12,",
    "{{numbered_candidates_with_context}}\n\nIf none is the document's own number, answer null.",
    "If none is the document's own number, answer null.\n\nEXAMPLE",
    'EXAMPLE\nCANDIDATES:\n[1] (line 3, top) "क्रमांक: सामान्य-2021/प्र.क्र.45/18"',
]

for l in lines:
    print(repr(l), "->", test_p(l))
