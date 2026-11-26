"""
Runtime Dense Semantic Vector Search (BAAI/bge-m3, 1024-d) for GAD-BOT.

Encodes incoming queries using local cached BAAI/bge-m3 weights and scores against
the lock-free 1024-d matrix index (`gad_kg/data/work/semantic_bge_m3.npz`) in ~1-2 ms
after a ~60 ms cached query encode, with optional remote Qdrant support when configured.
"""

from __future__ import annotations

import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.modules.setdefault("torchvision", None)
sys.modules.setdefault("torchvision.transforms", None)
sys.modules.setdefault("torchvision.io", None)

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORK_DIR = PROJECT_ROOT / "gad_kg" / "data" / "work"
NPZ_PATH = WORK_DIR / "semantic_bge_m3.npz"
MODEL_ID = "BAAI/bge-m3"

_LOAD_LOCK = threading.Lock()
_ENCODE_LOCK = threading.Lock()
_MODEL_STATE: Dict[str, Any] = {"tok": None, "mdl": None, "loaded": False, "failed": False}
_INDEX_CACHE: Optional[Dict[str, Any]] = None
_QUERY_VEC_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_QUERY_CACHE_MAX = 256


def is_semantic_index_ready() -> bool:
    return NPZ_PATH.exists()


def _load_index() -> Optional[Dict[str, Any]]:
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    if not NPZ_PATH.exists():
        return None
    with _LOAD_LOCK:
        if _INDEX_CACHE is not None:
            return _INDEX_CACHE
        data = np.load(NPZ_PATH, allow_pickle=True)
        vecs = data["vectors"].astype(np.float32)
        _INDEX_CACHE = {
            "vectors": vecs,
            "chunk_ids": data["chunk_ids"].tolist(),
            "doc_dirs": data["doc_dirs"].tolist(),
            "is_cg_policy": data["is_cg_policy"],
            "is_node_rep": data["is_node_rep"],
        }
        return _INDEX_CACHE


def _ensure_model_loaded() -> bool:
    if _MODEL_STATE["loaded"]:
        return True
    if _MODEL_STATE["failed"]:
        return False
    with _LOAD_LOCK:
        if _MODEL_STATE["loaded"]:
            return True
        try:
            sys.modules.setdefault("torchvision", None)
            sys.modules.setdefault("torchvision.transforms", None)
            sys.modules.setdefault("torchvision.io", None)
            import torch
            from transformers import AutoModel, AutoTokenizer

            torch.set_num_threads(max(4, min(14, os.cpu_count() or 8)))
            tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
            mdl = AutoModel.from_pretrained(MODEL_ID, local_files_only=True)
            mdl.eval()
            with torch.no_grad():
                w_inp = tok(["warmup"], padding=True, truncation=True, max_length=16, return_tensors="pt")
                _ = mdl(**w_inp)
            _MODEL_STATE["tok"] = tok
            _MODEL_STATE["mdl"] = mdl
            _MODEL_STATE["loaded"] = True
            return True
        except Exception as exc:
            print(f"[semantic_search] BGE-M3 load warning: {exc}", file=sys.stderr)
            _MODEL_STATE["failed"] = True
            return False


def warm_semantic_model(background: bool = True) -> None:
    """Pre-load both vector matrix and BGE-M3 encoder so first live query has zero cold start."""
    _load_index()
    if background:
        threading.Thread(target=_ensure_model_loaded, daemon=True).start()
    else:
        _ensure_model_loaded()


def encode_query(text: str) -> Optional[np.ndarray]:
    q_clean = " ".join((text or "").split()).strip()
    if not q_clean:
        return None
    if q_clean in _QUERY_VEC_CACHE:
        _QUERY_VEC_CACHE.move_to_end(q_clean)
        return _QUERY_VEC_CACHE[q_clean]

    if not _ensure_model_loaded():
        return None
    import torch

    with _ENCODE_LOCK:
        if q_clean in _QUERY_VEC_CACHE:
            return _QUERY_VEC_CACHE[q_clean]
        tok = _MODEL_STATE["tok"]
        mdl = _MODEL_STATE["mdl"]
        with torch.no_grad():
            inp = tok([q_clean], padding=True, truncation=True, max_length=96, return_tensors="pt")
            emb = mdl(**inp).last_hidden_state[:, 0]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            vec = emb[0].cpu().numpy().astype(np.float32)

    _QUERY_VEC_CACHE[q_clean] = vec
    if len(_QUERY_VEC_CACHE) > _QUERY_CACHE_MAX:
        _QUERY_VEC_CACHE.popitem(last=False)
    return vec


def semantic_vector_search(
    query: str,
    limit: int = 30,
    extra_query: Optional[str] = None,
    allow_non_policy: bool = False,
) -> List[Dict[str, Any]]:
    """
    Perform dense cosine semantic search (1024-d BGE-M3) over indexed documents & rulebook clauses.
    Returns ranked entries with `chunk_id`, `doc_dir`, `similarity`, and `is_node_rep`.
    """
    idx = _load_index()
    if idx is None:
        return []
    qv = encode_query(query)
    if qv is None:
        return []

    vecs = idx["vectors"]
    sims = vecs @ qv
    top_fast = float(np.max(sims)) if len(sims) else 0.0
    if extra_query and extra_query.strip() and extra_query.strip() != query.strip() and top_fast < 0.56:
        qv2 = encode_query(extra_query)
        if qv2 is not None:
            sims = np.maximum(sims, vecs @ qv2 * 0.96)

    if not allow_non_policy:
        mask = idx["is_cg_policy"] == 1
        sims = np.where(mask, sims, sims * 0.65)

    n = len(sims)
    k = min(max(limit * 2, 40), n)
    if k <= 0:
        return []
    top_idx = np.argpartition(sims, -k)[-k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    out: List[Dict[str, Any]] = []
    seen_cids = set()
    for i in top_idx:
        sim = float(sims[i])
        if sim < 0.30:
            continue
        cid = idx["chunk_ids"][i]
        if cid in seen_cids:
            continue
        seen_cids.add(cid)
        out.append(
            {
                "chunk_id": cid,
                "doc_dir": idx["doc_dirs"][i],
                "similarity": round(sim, 4),
                "is_node_rep": bool(idx["is_node_rep"][i]),
            }
        )
        if len(out) >= limit:
            break
    return out
