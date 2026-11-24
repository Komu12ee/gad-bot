import urllib.request
import urllib.error
import json

def test_line(line):
    payload = {"model": "ocr-32b", "messages": [{"role": "user", "content": line}], "max_tokens": 10}
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
    except Exception as e:
        return str(e)

with open("gad_kg/prompts/R2.txt", encoding="utf-8") as f:
    lines = f.readlines()

for idx, l in enumerate(lines):
    l_str = l.strip()
    if l_str:
        print(f"Line {idx+1}: {l_str[:50]} -> {test_line(l_str)}", flush=True)
