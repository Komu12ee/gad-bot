import sys
sys.path.insert(0, "gad_kg")
import json
import urllib.request
import urllib.error
from src.p04_header import load_prompt_template
from src.common.io import read_jsonl
from schemas.p04_header_schemas import R2_SCHEMA

feats = next(r for r in read_jsonl("gad_kg/data/work/features.jsonl") if r["file_id"] == "deef7607b75f22b9")
number_cands = [c for c in feats.get("number_candidates", []) if not c.get("in_reference_block", False)]
cand_lines = []
for i, c in enumerate(number_cands[:12], 1):
    cand_lines.append(f"[{i}] (line {c['line_no']}, top={c['in_top']}) \"{c['context']}\"")
cand_str = "\n".join(cand_lines)
r2_prompt = load_prompt_template("R2").replace("{{numbered_candidates_with_context}}", cand_str)
print("--- R2 PROMPT ---")
print(r2_prompt)
print("-----------------")

with open("gad_kg/prompts/SYSTEM_BASE.txt", encoding="utf-8") as f:
    sys_prompt = f.read().strip()

payload = {
    "model": "ocr-32b",
    "messages": [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": r2_prompt}
    ],
    "temperature": 0.0,
    "max_tokens": 600,
    "response_format": {
        "type": "json_schema",
        "json_schema": {"name": "r2_doc_number", "schema": R2_SCHEMA, "strict": True}
    }
}
req = urllib.request.Request("http://103.51.8.88/v1/chat/completions", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        print("SUCCESS! Response:", resp.read().decode())
except urllib.error.HTTPError as e:
    print("HTTPError code:", e.code)
    print("Error content:", e.read().decode())
