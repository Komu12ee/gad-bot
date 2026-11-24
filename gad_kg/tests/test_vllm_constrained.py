import json
import urllib.request

schema = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ACTIVE", "EXPIRED"]},
        "code": {"type": "integer"}
    },
    "required": ["status", "code"],
    "additionalProperties": False
}

# Test guided_json parameter
payload = {
    "model": "ocr-32b",
    "messages": [
        {"role": "user", "content": "Return status and code for active system with code 200."}
    ],
    "temperature": 0.0,
    "max_tokens": 50,
    "guided_json": schema
}

req = urllib.request.Request(
    "http://103.51.8.88/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        res = json.loads(resp.read().decode())
        print("Success guided_json! Response:")
        print(res["choices"][0]["message"]["content"])
except Exception as e:
    print("guided_json error:", e)

# Test response_format json_schema parameter
payload2 = {
    "model": "ocr-32b",
    "messages": [
        {"role": "user", "content": "Return status and code for active system with code 200."}
    ],
    "temperature": 0.0,
    "max_tokens": 50,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "system_status",
            "schema": schema,
            "strict": True
        }
    }
}

req2 = urllib.request.Request(
    "http://103.51.8.88/v1/chat/completions",
    data=json.dumps(payload2).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req2, timeout=10) as resp:
        res = json.loads(resp.read().decode())
        print("Success response_format json_schema! Response:")
        print(res["choices"][0]["message"]["content"])
except Exception as e:
    print("json_schema error:", e)
