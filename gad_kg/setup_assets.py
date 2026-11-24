import re
from pathlib import Path

spec_path = Path("GAD_TKG_RAG_MASTER_PROMPT.md")
content = spec_path.read_text(encoding="utf-8")

root = Path("gad_kg")

# 1. Write config.yaml
config_yaml = """# config.yaml
paths:
  raw_md: data/raw_md
  clean: data/clean
  work: data/work
llm:
  base_url: http://103.51.8.88/v1
  model: ocr-32b
  temperature_greedy: 0.0
  temperature_vote: 0.4
  votes: 3                  # 1 greedy + 2 sampled
  max_concurrency: 16
  max_tokens_default: 600
  structured_output: true
embedding:
  model: BAAI/bge-m3        # dense (1024-d) + sparse
reranker:
  model: BAAI/bge-reranker-v2-m3
stores:
  graph: neo4j://localhost:7687
  vectors: qdrant           # local or docker
  tables: duckdb file data/work/tables.duckdb
thresholds:
  ocr_quality_min: 0.80
  dedup_jaccard: 0.90
  fuzzy_number_match: 0.92
  subject_embed_sim: 0.80
"""
(root / "config.yaml").write_text(config_yaml, encoding="utf-8")
print("Wrote config.yaml")

# 2. Extract prompts from Section 6
prompts_dir = root / "prompts"
prompts_dir.mkdir(parents=True, exist_ok=True)

# Find Section 6
sec6_match = re.search(r"## 6\. RUNTIME PROMPT LIBRARY.*?(?=## 7\. Lexicons)", content, re.DOTALL)
if sec6_match:
    sec6 = sec6_match.group(0)
    # Find all subsections: ### SYSTEM_BASE, ### R1 — ..., etc.
    blocks = re.findall(r"###\s+([A-Za-z0-9_]+)[^\n]*\n+```text\n(.*?)```", sec6, re.DOTALL)
    for p_id, p_text in blocks:
        out_file = prompts_dir / f"{p_id}.txt"
        out_file.write_text(p_text.strip() + "\n", encoding="utf-8")
        print(f"Extracted prompt {p_id} -> {out_file}")

# 3. Extract lexicons from Section 7
lex_dir = root / "lexicons"
lex_dir.mkdir(parents=True, exist_ok=True)

# 7.1 relation cues
rel_cues_match = re.search(r"### 7\.1 Relation cues\s+```yaml\n(.*?)```", content, re.DOTALL)
if rel_cues_match:
    (lex_dir / "relation_cues.yaml").write_text(rel_cues_match.group(1).strip() + "\n", encoding="utf-8")
    print("Extracted relation_cues.yaml")

# 7.2 effective date cues
eff_cues_match = re.search(r"### 7\.2 Effective-date cues\s+```yaml\n(.*?)```", content, re.DOTALL)
if eff_cues_match:
    (lex_dir / "effective_cues.yaml").write_text(eff_cues_match.group(1).strip() + "\n", encoding="utf-8")
    print("Extracted effective_cues.yaml")

print("Setup completed successfully.")
