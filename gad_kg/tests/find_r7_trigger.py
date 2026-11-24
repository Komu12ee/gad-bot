import urllib.request
import urllib.error
import json

def test_p(p):
    payload = {"model": "ocr-32b", "messages": [{"role": "user", "content": p}], "max_tokens": 5}
    req = urllib.request.Request(
        "http://103.51.8.88/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:
        return "TIMEOUT"

with open("gad_kg/tests/r7_prompt_f675.txt", encoding="utf-8") as f:
    text = f.read()

# Binary search
low = 0
high = len(text)

while high - low > 30:
    mid = (low + high) // 2
    code = test_p(text[:mid])
    print(f"Slice 0:{mid} -> {code}", flush=True)
    if code == 403:
        high = mid
    else:
        low = mid

print("Transition between", low, "and", high)
print("Text at transition:")
print(repr(text[max(0, low - 50):high + 50]))
