import urllib.request
import urllib.error
import json

with open("gad_kg/tests/r7_prompt_f675.txt", encoding="utf-8") as f:
    text = f.read()

# Replace [string, ...] with valid JSON example
text_mod = text.replace("[string, ...]", '["topic_1", "topic_2"]')

payload = {"model": "ocr-32b", "messages": [{"role": "user", "content": text_mod}], "max_tokens": 10}
req = urllib.request.Request(
    "http://103.51.8.88/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        print("Success! Status:", resp.status)
except urllib.error.HTTPError as e:
    print("HTTPError:", e.code)
