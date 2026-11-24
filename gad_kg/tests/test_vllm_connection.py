import json
import urllib.request

payload = {
    "model": "ocr-32b",
    "messages": [
        {"role": "system", "content": "You output valid JSON."},
        {"role": "user", "content": 'Return a JSON object with key "test" and value "ok".'}
    ],
    "temperature": 0.0,
    "max_tokens": 50,
    "response_format": {"type": "json_object"}
}

req = urllib.request.Request(
    "http://103.51.8.88/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        res = json.loads(resp.read().decode())
        print("Success! Response:")
        print(res["choices"][0]["message"]["content"])
except Exception as e:
    print("Error:", e)
