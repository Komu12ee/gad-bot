#!/usr/bin/env python3
"""
Build 1024-d BAAI/bge-m3 Semantic Vector Index for GAD-BOT.

Indexes all unique document nodes in data/fg2/temporal_graph.db plus substantive
clause chunks for core multi-chapter rulebooks into BOTH:
  1. Qdrant local collection (`data/fg2/db`, collection `db3`, size=1024, COSINE)
  2. Lock-free numpy matrix index (`gad_kg/data/work/semantic_bge_m3.npz` + meta JSON)

Usage:
    python core/04_embeddings_and_kg/scripts/build_semantic_index.py
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Guard against broken torchvision in mixed Torch/Transformers environments
sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "gad_kg"))

import pipeline_config as pc
import qdrant_store
from qdrant_client.models import Distance, PointStruct, VectorParams
from src.gad_retriever import (
    DOMAIN_PROFILES,
    _is_krutidev_toc_chunk,
    _substantive_clause_score,
)

DB_PATH = ROOT / "data" / "fg2" / "temporal_graph.db"
NPZ_PATH = ROOT / "gad_kg" / "data" / "work" / "semantic_bge_m3.npz"
META_PATH = ROOT / "gad_kg" / "data" / "work" / "semantic_bge_m3_meta.json"
MODEL_ID = "BAAI/bge-m3"
BATCH_SIZE = 32
MAX_LEN = 128

# Anchors from DOMAIN_PROFILES whose substantive clauses should ALL be indexed individually
CORE_RULEBOOK_DIRS: Set[str] = {
    anc for prof in DOMAIN_PROFILES for anc in prof.get("anchors", [])
} | {"4f672641e687a86e", "3b2fd23e74466e9b", "c9c88208e9d47d6f", "ac820304db0fff00"}


def _clean_passage_body(content: str) -> str:
    body = re.sub(
        r"^Document:[^\n]*(?:\nMinistry:[^\n]*)?(?:\nOffice:[^\n]*)?(?:\nStatus:[^\n]*)?(?:\nSection:[^\n]*)?\s*",
        "",
        (content or "").strip(),
    )
    body = re.sub(r"\|\s*uke\s*\|\s*1\s*\|\s*iwB\s*/\s*1\s*\|", " ", body)
    return re.sub(r"\s+", " ", body).strip()


def collect_index_items() -> List[Dict[str, Any]]:
    """Collect document-level passages for every unique node + key clause passages."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    nodes = cur.execute(
        """
        SELECT id, doc_id, doc_dir, order_no, doc_date, status,
               title_hi, title_en, summary, category, category_id, is_cg_policy, chunk_count
        FROM nodes
        WHERE doc_dir != '' AND status != 'duplicate'
        ORDER BY is_cg_policy DESC, doc_date DESC, doc_dir ASC
        """
    ).fetchall()

    # Pre-fetch chunk candidates grouped by doc_id in one scan
    chunks_by_doc: Dict[str, List[Tuple[str, str, int]]] = {}
    for cr in cur.execute(
        "SELECT chunk_id, doc_id, content, chunk_index FROM chunks ORDER BY doc_id, chunk_index ASC"
    ):
        chunks_by_doc.setdefault(cr["doc_id"], []).append(
            (cr["chunk_id"], cr["content"] or "", int(cr["chunk_index"] or 0))
        )
    conn.close()

    items: List[Dict[str, Any]] = []
    seen_chunk_ids: Set[str] = set()

    for n in nodes:
        dd = n["doc_dir"]
        cands = chunks_by_doc.get(dd, [])
        # Score non-TOC substantive chunks for this document
        scored: List[Tuple[float, int, str, str]] = []
        for idx, (cid, raw_txt, _) in enumerate(cands):
            if _is_krutidev_toc_chunk(raw_txt):
                continue
            sc = _substantive_clause_score(raw_txt, [])
            if dd == "4f672641e687a86e" and (idx < 5 or idx >= 47):
                sc -= 40.0
            scored.append((sc, idx, cid, raw_txt))
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

        title_hi = (n["title_hi"] or "").strip()
        title_en = (n["title_en"] or "").strip()
        cat = (n["category"] or "").strip()
        summary = (n["summary"] or "").strip()

        best_cid = scored[0][2] if scored else (cands[0][0] if cands else f"{dd}_c001")
        best_body = _clean_passage_body(scored[0][3]) if scored else (
            _clean_passage_body(cands[0][1]) if cands else summary
        )
        lead_piece = summary[:160] if summary and summary != title_hi else best_body[:220]
        doc_text = " | ".join(
            p for p in (title_hi, title_en, n["order_no"] or "", cat.split("(")[0].strip(), lead_piece) if p
        )[:520]
        if not doc_text:
            doc_text = best_body[:400] or dd

        if doc_text and best_cid not in seen_chunk_ids:
            seen_chunk_ids.add(best_cid)
            items.append(
                {
                    "chunk_id": best_cid,
                    "doc_id": dd,
                    "doc_dir": dd,
                    "canonical_doc_id": n["doc_id"] or dd,
                    "order_no": n["order_no"] or "",
                    "doc_date": n["doc_date"] or "",
                    "status": n["status"] or "current",
                    "title": title_hi or title_en,
                    "category": cat,
                    "category_id": n["category_id"] or "",
                    "is_cg_policy": int(n["is_cg_policy"] if n["is_cg_policy"] is not None else 1),
                    "is_node_rep": True,
                    "embed_text": doc_text,
                    "payload_text": f"{title_hi}\n{best_body[:900]}".strip(),
                }
            )

        # For core anchors and large multi-clause policy rulebooks, also index top distinct clauses
        extra_cap = 32 if dd in CORE_RULEBOOK_DIRS else (4 if (int(n["chunk_count"] or 0) >= 12 and int(n["is_cg_policy"] or 0) == 1) else 0)
        if extra_cap > 0 and len(scored) > 1:
            added = 0
            for sc_v, idx, cid, raw_txt in scored[1:]:
                if cid in seen_chunk_ids:
                    continue
                if sc_v < -5.0 and dd not in CORE_RULEBOOK_DIRS:
                    continue
                if dd == "4f672641e687a86e" and (idx < 5 or idx >= 47):
                    continue
                clean_b = _clean_passage_body(raw_txt)
                if len(clean_b) < 120:
                    continue
                seen_chunk_ids.add(cid)
                items.append(
                    {
                        "chunk_id": cid,
                        "doc_id": dd,
                        "doc_dir": dd,
                        "canonical_doc_id": n["doc_id"] or dd,
                        "order_no": n["order_no"] or "",
                        "doc_date": n["doc_date"] or "",
                        "status": n["status"] or "current",
                        "title": title_hi or title_en,
                        "category": cat,
                        "category_id": n["category_id"] or "",
                        "is_cg_policy": int(n["is_cg_policy"] if n["is_cg_policy"] is not None else 1),
                        "is_node_rep": False,
                        "embed_text": f"{title_hi[:120]} | {clean_b[:420]}",
                        "payload_text": clean_b[:1000],
                    }
                )
                added += 1
                if added >= extra_cap:
                    break

    return items


def encode_items(items: List[Dict[str, Any]]) -> np.ndarray:
    """Encode all items sorted by length for optimal CPU speed using BAAI/bge-m3 FP32."""
    torch.set_num_threads(max(4, min(14, os.cpu_count() or 8)))
    print(f"Loading {MODEL_ID} (local_files_only=True)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    mdl = AutoModel.from_pretrained(MODEL_ID, local_files_only=True)
    mdl.eval()

    # Sort indices by text length so padding overhead inside each batch is minimized
    order = sorted(range(len(items)), key=lambda i: len(items[i]["embed_text"]))
    dim = int(mdl.config.hidden_size)
    out_vecs = np.zeros((len(items), dim), dtype=np.float16)

    t0 = time.time()
    done = 0
    with torch.no_grad():
        for start in range(0, len(order), BATCH_SIZE):
            batch_idx = order[start:start + BATCH_SIZE]
            batch_txt = [items[i]["embed_text"] for i in batch_idx]
            inp = tok(batch_txt, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            cls_emb = mdl(**inp).last_hidden_state[:, 0]
            cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
            out_vecs[batch_idx] = cls_emb.cpu().numpy().astype(np.float16)
            done += len(batch_idx)
            if done % 640 == 0 or done == len(items):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 0.1)
                eta = (len(items) - done) / max(rate, 0.1) / 60.0
                print(f"  Encoded {done}/{len(items)} ({done*100/len(items):.1f}%) | {rate:.1f} vec/s | ETA {eta:.1f}m", flush=True)

    return out_vecs


def write_outputs(items: List[Dict[str, Any]], vectors: np.ndarray) -> None:
    """Persist both lock-free NPZ index and Qdrant db3 collection."""
    NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        NPZ_PATH,
        vectors=vectors,
        chunk_ids=np.array([it["chunk_id"] for it in items], dtype=object),
        doc_dirs=np.array([it["doc_dir"] for it in items], dtype=object),
        is_cg_policy=np.array([it["is_cg_policy"] for it in items], dtype=np.int8),
        is_node_rep=np.array([1 if it["is_node_rep"] else 0 for it in items], dtype=np.int8),
    )
    meta_map = {
        it["chunk_id"]: {
            "doc_dir": it["doc_dir"],
            "order_no": it["order_no"],
            "doc_date": it["doc_date"],
            "status": it["status"],
            "title": it["title"],
            "category": it["category"],
            "category_id": it["category_id"],
            "is_cg_policy": it["is_cg_policy"],
        }
        for it in items
    }
    META_PATH.write_text(json.dumps(meta_map, ensure_ascii=False), encoding="utf-8")
    print(f"Saved lock-free semantic matrix: {NPZ_PATH} ({NPZ_PATH.stat().st_size/1e6:.2f} MB, shape={vectors.shape})", flush=True)

    # Also populate Qdrant local collection (pc.COLLECTION = db3, 1024-d COSINE)
    client = qdrant_store.get_client(role="writer")
    coll = pc.COLLECTION
    if client.collection_exists(coll):
        client.delete_collection(coll)
    client.create_collection(
        collection_name=coll,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )
    upsert_bs = 200
    for start in range(0, len(items), upsert_bs):
        pts = []
        for idx in range(start, min(len(items), start + upsert_bs)):
            it = items[idx]
            pts.append(
                PointStruct(
                    id=idx + 1,
                    vector=vectors[idx].astype(np.float32).tolist(),
                    payload={
                        "chunk_id": it["chunk_id"],
                        "chunk": it["chunk_id"],
                        "doc_id": it["canonical_doc_id"],
                        "doc_dir": it["doc_dir"],
                        "source": it["doc_dir"],
                        "order_no": it["order_no"],
                        "doc_date": it["doc_date"],
                        "status": it["status"],
                        "title": it["title"],
                        "category": it["category"],
                        "category_id": it["category_id"],
                        "is_cg_policy": it["is_cg_policy"],
                        "text": it["payload_text"],
                    },
                )
            )
        client.upsert(collection_name=coll, points=pts)
    qdrant_store.ensure_indexes(client, coll)
    cnt = client.count(coll).count
    qdrant_store.close_client()
    print(f"Populated Qdrant collection '{coll}' with {cnt} 1024-d BGE-M3 vectors.", flush=True)


def main() -> None:
    t_all = time.time()
    items = collect_index_items()
    print(f"Collected {len(items)} items for BGE-M3 embedding.", flush=True)
    vectors = encode_items(items)
    write_outputs(items, vectors)
    print(f"Done in {(time.time() - t_all)/60.0:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
