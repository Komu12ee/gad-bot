import sys
sys.path.insert(0, "gad_kg")
import urllib.request
import urllib.error
import json
from src.p04_header import load_prompt_template
from src.common.io import read_jsonl

feats = next(r for r in read_jsonl("gad_kg/data/work/features.jsonl") if r["file_id"] == "deef7607b75f22b9")
number_cands = [c for c in feats.get("number_candidates", []) if not c.get("in_reference_block", False)]
cand_lines = [f"[{i}] (line {c['line_no']}, top={c['in_top']}) \"{c['context']}\"" for i, c in enumerate(number_cands[:12], 1)]

def check_content(text):
    payload = {"model": "ocr-32b", "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(
        "http://103.51.8.88/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return "OK"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:
        return f"ERR: {e}"

tmpl = load_prompt_template("R2").replace("{{numbered_candidates_with_context}}", "none")
print("Template alone:", check_content(tmpl))

for i, line in enumerate(cand_lines):
    res = check_content(line)
    print(f"Cand {i}: {line[:50]} -> {res}")
