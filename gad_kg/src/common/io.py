import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Generator
import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Path = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = get_project_root() / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_file_id(file_bytes: bytes) -> str:
    """file_id = sha1(raw file bytes)[:16]"""
    return hashlib.sha1(file_bytes).hexdigest()[:16]


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: Path | str) -> Generator[Dict[str, Any], None, None]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path | str, records: List[Dict[str, Any]], append: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
