"""
LLM Call Cache
Spec: Rule 2: "cache every LLM call by sha256(prompt + model + params)"
"""

import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

from src.common.io import ensure_dir


class LLMCache:
    def __init__(self, db_path: str = "gad_kg/data/work/llm_cache.sqlite"):
        self.db_path = db_path
        ensure_dir(Path(db_path).parent)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    prompt_hash TEXT,
                    model TEXT,
                    params TEXT,
                    response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    @staticmethod
    def compute_key(prompt: str, model: str, params: Dict[str, Any]) -> str:
        param_str = json.dumps(params, sort_keys=True)
        key_content = f"{prompt}||{model}||{param_str}"
        return hashlib.sha256(key_content.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> Optional[str]:
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT response FROM llm_cache WHERE cache_key = ?", (cache_key,))
            row = cursor.fetchone()
            if row:
                return row[0]
        return None

    def set(self, cache_key: str, model: str, params: Dict[str, Any], response: str):
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache (cache_key, prompt_hash, model, params, response)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cache_key, cache_key[:16], model, json.dumps(params, sort_keys=True), response)
            )
            conn.commit()
