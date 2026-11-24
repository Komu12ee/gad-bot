"""
Phase 11: Entities and Controlled Ontology
Spec: Section 5 P11, Section 6 Prompt R13
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.io import read_jsonl, write_jsonl, ensure_dir
from src.common.llm import call_llm
from schemas.p11_entity_schemas import R13_SCHEMA


def load_prompt_template(name: str) -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / f"{name}.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def extract_entity_mentions_from_clause(
    clause_text: str,
    r13_tmpl: str,
) -> List[Dict[str, Any]]:
    """Extract entity mentions from a clause using Prompt R13."""
    if len(clause_text.strip()) < 15:
        return []

    prompt = r13_tmpl.replace("{{clause_text}}", clause_text[:800])
    try:
        res = call_llm(prompt, schema=R13_SCHEMA, schema_name="r13_entities", temperature=0.0, max_tokens=400)
        return res.get("mentions", [])
    except Exception:
        return []


def canonicalize_entity(mention_text: str, kind: str) -> Tuple[str, str]:
    """Map mention text to a canonical ID and canonical display name."""
    clean = re.sub(r"[\s\-_/.]+", "_", mention_text.strip().lower())
    clean_id = f"ENT_{kind}_{clean[:30]}".strip("_")
    return clean_id, mention_text.strip()
