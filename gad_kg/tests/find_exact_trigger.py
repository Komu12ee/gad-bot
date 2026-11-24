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
            return 200
    except urllib.error.HTTPError as e:
        return e.code

with open("gad_kg/prompts/R2.txt", encoding="utf-8") as f:
    text = f.read()

replaced = text.replace("{{numbered_candidates_with_context}}", "[1] (line 3, top) Order No. 123")

# Binary search the string
start = 0
end = len(replaced)

# Test prefixes
print("Full length:", end)
for length in range(50, end + 50, 50):
    sub = replaced[:length]
    code = test_p(sub)
    print(f"Length {length:3d}: {code} (ends with {repr(sub[-30:])})")
    if code == 403:
        # zoom in
        print("Found 403 transition between", length - 50, "and", length)
        for sub_len in range(length - 50, min(length + 1, end + 1), 5):
            print(f"  Sublen {sub_len:3d}: {test_p(replaced[:sub_len])} -> {repr(replaced[sub_len-15:sub_len])}")
        break
