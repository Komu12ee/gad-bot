import urllib.request
import urllib.error
import json

def test_text(txt):
    payload = {"model": "ocr-32b", "messages": [{"role": "user", "content": txt}]}
    req = urllib.request.Request(
        "http://103.51.8.88/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200
    except urllib.error.HTTPError as e:
        return e.code

with open("gad_kg/tests/debug_r2.py") as f:
    text = f.read()

# Test various parts
test_strings = [
    "OUTPUT SCHEMA: {\"quote\": string|null, \"choice\": integer|null}",
    "| 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9 |",
    "क्रमांक एफ 7-61/2022/1/एक-सीएम",
    "string|null",
    "integer|null",
    "| कुल राशि रु. | कुल राशि रु. |",
    "| 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9 | कोरबा / जांजगीर-चांम्पा / रायगढ़ / जशपुर / सूरजपुर / दुर्ग / रायपुर / बस्तर / धमतरी | 700000.00 / 160000.00 / 65",
]

for s in test_strings:
    code = test_text(s)
    print(f"'{s[:50]}' -> {code}")
