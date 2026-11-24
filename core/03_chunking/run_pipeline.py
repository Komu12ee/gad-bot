"""CLI driver for upgraded document chunking, metadata extraction, and indexing.

Usage:
  python run_pipeline.py --input <dir> --output <dir> [--no-llm] [--index-qdrant] [--config <path>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root and core are in path
_SCRIPT_DIR = Path(__file__).resolve().parent
_CORE_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _CORE_DIR.parent

# If executed via system Python (e.g. `python3 run_pipeline.py`) outside the virtualenv,
# automatically delegate to the project's .venv python where dependencies are installed.
if sys.prefix == sys.base_prefix:
    _venv_py = _REPO_ROOT / ".venv" / "bin" / "python"
    if not _venv_py.exists():
        _venv_py = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if _venv_py.exists() and str(_venv_py) != sys.executable:
        os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)

for p in [str(_REPO_ROOT), str(_CORE_DIR), str(_SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml
from chunker import build_document_chunks
from extract_metadata import call_llm_json_extractor
from models import ChunkMetadata, ChunkRecord, DocumentCard
from qdrant_io import ensure_chunk_payload_indexes
from supersedes import link_supersedes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chunk_pipeline")


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    cfg_file = Path(config_path) if config_path else _SCRIPT_DIR / "config.yaml"
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def format_chunk_file_text(record: ChunkRecord) -> str:
    """Format chunk text with YAML-like metadata above '---' and embedded text below."""
    meta = record.metadata.model_dump(exclude_none=True)
    yaml_lines = []
    for k, v in meta.items():
        if isinstance(v, list):
            yaml_lines.append(f"{k}:")
            for item in v:
                yaml_lines.append(f"  - {item}")
        else:
            yaml_lines.append(f"{k}: {v}")
    
    yaml_header = "\n".join(yaml_lines)
    return f"{yaml_header}\n\n---\n\n{record.embedded_text}\n"


def process_corpus(
    input_dir: Path,
    output_dir: Path,
    use_llm: bool = True,
    index_qdrant: bool = False,
    collection_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Process all documents in input_dir and output structured chunks and cards."""
    config = config or {}
    chunk_cfg = config.get("chunking", {})
    llm_cfg = config.get("llm", {})

    target_tokens = chunk_cfg.get("target_tokens", 400)
    max_tokens = chunk_cfg.get("max_tokens", 480)
    overlap_tokens = chunk_cfg.get("overlap_tokens", 50)

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_jsonl_path = output_dir / "chunks.jsonl"
    cards_jsonl_path = output_dir / "doc_cards.jsonl"

    all_chunk_records: List[Dict[str, Any]] = []
    all_doc_cards: List[Dict[str, Any]] = []

    # Find all markdown/text/document files
    files = list(input_dir.glob("**/*.md")) + list(input_dir.glob("**/*.txt"))
    # Filter out already generated chunk files if re-running in same tree
    files = [f for f in files if "structured_chunk_" not in f.name and "chunks.jsonl" not in f.name]
    logger.info(f"Found {len(files)} documents to chunk in {input_dir}")

    # Prepare LLM callback if enabled
    llm_callback = None
    if use_llm and llm_cfg.get("enabled", True):
        llm_callback = lambda prompt, system_prompt: call_llm_json_extractor(
            prompt=prompt,
            system_prompt=system_prompt,
            llm_url=llm_cfg.get("base_url"),
            model_name=llm_cfg.get("model"),
        )

    for fpath in files:
        raw_text = fpath.read_text(encoding="utf-8", errors="replace")
        if not raw_text.strip():
            continue

        # Deterministic document ID from content hash or folder name
        doc_hash = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()[:16]
        doc_dir_name = fpath.parent.name if fpath.parent != input_dir else fpath.stem
        doc_id = doc_dir_name if len(doc_dir_name) == 16 else doc_hash

        file_info = {
            "filename": fpath.name,
            "source_type": "web" if fpath.suffix == ".html" or "web" in str(fpath).lower() else "pdf",
            "file_modified_at": datetime.fromtimestamp(fpath.stat().st_mtime).strftime("%Y-%m-%d"),
        }

        # Build chunks and document card
        chunks, card = build_document_chunks(
            text=raw_text,
            doc_id=doc_id,
            file_info=file_info,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            llm_extractor=llm_callback,
        )

        # Write human-readable structured_chunk_NNN.txt
        doc_out_dir = output_dir / doc_id
        doc_out_dir.mkdir(parents=True, exist_ok=True)
        for c in chunks:
            chunk_file_name = f"structured_chunk_{c.metadata.chunk_index:03d}.txt"
            (doc_out_dir / chunk_file_name).write_text(format_chunk_file_text(c), encoding="utf-8")
            all_chunk_records.append(c.model_dump())

        # Write document card txt
        card_file_name = f"structured_chunk_card.txt"
        card_txt = (
            f"Document Card: {card.title}\n"
            f"Doc ID: {card.doc_id}\n"
            f"Order No: {card.order_no or 'N/A'}\n"
            f"Date: {card.doc_date or 'N/A'} (Confidence: {card.date_confidence})\n"
            f"Status: {card.status}\n"
            f"Topic Key: {card.topic_key}\n"
            f"Summary: {card.summary}\n"
        )
        (doc_out_dir / card_file_name).write_text(card_txt, encoding="utf-8")
        all_doc_cards.append(card.model_dump())

    # Write chunks.jsonl
    with open(chunks_jsonl_path, "w", encoding="utf-8") as f:
        for cr in all_chunk_records:
            f.write(json.dumps(cr, ensure_ascii=False) + "\n")

    # Write doc_cards.jsonl
    with open(cards_jsonl_path, "w", encoding="utf-8") as f:
        for dc in all_doc_cards:
            f.write(json.dumps(dc, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(all_chunk_records)} chunks to {chunks_jsonl_path}")
    logger.info(f"Wrote {len(all_doc_cards)} document cards to {cards_jsonl_path}")

    # Run post-ingestion supersedes resolution
    logger.info("Executing post-ingestion supersedes linking...")
    supersedes_stats = link_supersedes(
        doc_cards=all_doc_cards,
        chunk_records=all_chunk_records,
        output_dir=output_dir,
    )
    logger.info(f"Supersedes resolution stats: {supersedes_stats}")

    # Optional: Qdrant Indexing
    if index_qdrant:
        try:
            from qdrant_client import QdrantClient
            q_cfg = config.get("qdrant", {})
            target_coll = collection_name
            if not target_coll:
                for b in ["chips", "fg2", "fg"]:
                    norm_in = f"/{str(input_dir).replace('\\', '/')}/"
                    norm_out = f"/{str(output_dir).replace('\\', '/')}/"
                    if f"/{b}/" in norm_in or f"/{b}/" in norm_out:
                        target_coll = f"{b}__db3"
                        break
            if not target_coll:
                target_coll = q_cfg.get("chunks_collection", "chips_chunks")

            client = QdrantClient(host=q_cfg.get("host", "127.0.0.1"), port=q_cfg.get("port", 6333))
            vector_size = q_cfg.get("vector_size", 1024)
            ensure_chunk_payload_indexes(client, target_coll, vector_size=vector_size)
            logger.info(f"Qdrant payload indexes ensured for collection '{target_coll}'.")
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant for indexing: {e}")

    return {
        "total_documents": len(files),
        "total_chunks": len(all_chunk_records),
        "total_cards": len(all_doc_cards),
        "supersedes_stats": supersedes_stats,
    }


def print_pipeline_status(config: Optional[Dict[str, Any]] = None) -> None:
    """Print comprehensive status of chunks, document cards, supersedes, and Qdrant collections."""
    config = config or {}
    q_cfg = config.get("qdrant", {})
    host = q_cfg.get("host", "127.0.0.1")
    port = q_cfg.get("port", 6333)

    print("\n" + "=" * 65)
    print("            CHiPS-RAG PIPELINE & VECTOR STORE STATUS")
    print("=" * 65)

    # 1. Check Qdrant
    print(f"\n[1] Qdrant Vector Store: http://{host}:{port}")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=host, port=port, timeout=5)
        colls_resp = client.get_collections()
        colls = [c.name for c in colls_resp.collections]
        print(f"    Server Status: ONLINE")
        print(f"    Collections Found ({len(colls)}):")
        for cname in colls:
            try:
                info = client.get_collection(cname)
                pts = info.points_count or 0
                vec_size = info.config.params.vectors.size if info.config and info.config.params and info.config.params.vectors else "N/A"
                print(f"      • {cname:<24} -> Points: {pts:<8} Vector Dim: {vec_size}")
            except Exception:
                print(f"      • {cname}")
        if not colls:
            print("      (No collections created yet)")
    except Exception as e:
        print(f"    Server Status: OFFLINE / UNREACHABLE ({e})")

    # 2. Check Department Chunks on Disk
    print(f"\n[2] Document Chunks & Metadata on Disk:")
    for bot in ["chips", "fg", "fg2"]:
        bot_chunks_dir = _REPO_ROOT / "data" / bot / "chunks"
        chunks_file = bot_chunks_dir / "chunks.jsonl"
        cards_file = bot_chunks_dir / "doc_cards.jsonl"
        unres_file = bot_chunks_dir / "unresolved_supersedes.csv"
        review_file = bot_chunks_dir / "needs_review.csv"

        if bot_chunks_dir.exists():
            n_chunks = sum(1 for _ in open(chunks_file, encoding="utf-8")) if chunks_file.exists() else 0
            n_cards = sum(1 for _ in open(cards_file, encoding="utf-8")) if cards_file.exists() else 0
            n_unres = max(0, sum(1 for _ in open(unres_file, encoding="utf-8")) - 1) if unres_file.exists() else 0
            n_review = max(0, sum(1 for _ in open(review_file, encoding="utf-8")) - 1) if review_file.exists() else 0

            print(f"    • {bot.upper():<5} (data/{bot}/chunks):")
            print(f"        Chunks: {n_chunks:,} | Doc Cards: {n_cards:,} | Needs Review: {n_review} | Unresolved: {n_unres}")
        else:
            print(f"    • {bot.upper():<5}: (no chunks generated yet)")

    print("\n" + "=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Upgrade document chunking pipeline for CHiPS-RAG.")
    parser.add_argument("--status", action="store_true", help="Print live status of chunks, cards, and Qdrant collections")
    parser.add_argument("--input", required=False, help="Input directory containing source Markdown/PDF text")
    parser.add_argument("--output", required=False, help="Output directory for chunks and JSONL files")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM fallback calls")
    parser.add_argument("--index-qdrant", action="store_true", help="Push and index payloads in Qdrant")
    parser.add_argument("--collection", default=None, help="Target Qdrant collection name (defaults to auto-detected <bot>__db3 or config)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.status:
        print_pipeline_status(config)
        return

    if not args.input or not args.output:
        parser.error("--input and --output are required unless --status is specified")

    process_corpus(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        use_llm=not args.no_llm,
        index_qdrant=args.index_qdrant,
        collection_name=args.collection,
        config=config,
    )


if __name__ == "__main__":
    main()
