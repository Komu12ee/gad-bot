"""Organize raw and processed documents into a hierarchical directory structure matching the official website URL hierarchy.

Reads data/<pipeline>/documents.json and builds:
  data/<pipeline>/site_structure/
  └── <domain>/
      └── <section>/
          └── <subsection>/
              ├── <document_name>.pdf         (Symlink or copy of raw PDF)
              ├── <document_name>.md          (Symlink or copy of OCR structured markdown)
              └── <document_name>_meta.json   (Document metadata & provenance)

Also generates `data/<pipeline>/site_structure/SITE_MAP.md` listing the entire tree.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Auto-delegate to .venv python if invoked via system Python
if sys.prefix == sys.base_prefix:
    _venv_py = _REPO_ROOT / ".venv" / "bin" / "python"
    if not _venv_py.exists():
        _venv_py = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if _venv_py.exists() and str(_venv_py) != sys.executable:
        os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("organize_by_url")


def sanitize_folder_name(name: str) -> str:
    """Sanitize directory name for Windows and Linux filesystems."""
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', '_', name).strip('. _')
    return name[:64] or "unnamed"


def url_to_hierarchy(url: str, default_domain: str = "site") -> tuple[str, list[str], str]:
    """Parse URL into domain, list of subfolder segments, and filename.

    Example:
      https://gad.cg.gov.in/circulars/2024/order_123.pdf
      -> ("gad.cg.gov.in", ["circulars", "2024"], "order_123.pdf")
    """
    if not url:
        return default_domain, ["unlinked"], "document"

    parsed = urlparse(url)
    domain = sanitize_folder_name(parsed.netloc or default_domain)
    raw_path = unquote(parsed.path).strip("/")

    if not raw_path:
        return domain, ["home"], "index.html"

    parts = [p for p in raw_path.split("/") if p]
    if len(parts) > 1:
        folders = [sanitize_folder_name(p) for p in parts[:-1]]
        filename = sanitize_folder_name(parts[-1])
    else:
        folders = ["root"]
        filename = sanitize_folder_name(parts[0])

    # If query parameters exist (e.g. ?year=2024), append sanitized query as subfolder
    if parsed.query:
        query_sanitized = sanitize_folder_name(parsed.query)
        folders.append(query_sanitized)

    return domain, folders, filename


def link_or_copy(src: Path, dst: Path, mode: str = "symlink") -> bool:
    """Link or copy src file to dst."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True

    if mode == "symlink":
        try:
            # Use relative symlink where possible
            rel_src = os.path.relpath(src, dst.parent)
            os.symlink(rel_src, dst)
            return True
        except (OSError, NotImplementedError):
            pass  # Fall back to hardlink or copy

    if mode in ("symlink", "hardlink"):
        try:
            os.link(src, dst)
            return True
        except (OSError, NotImplementedError):
            pass

    # Copy fallback
    shutil.copy2(src, dst)
    return True


def organize_pipeline_documents(
    pipeline: str,
    output_dir: Path | None = None,
    mode: str = "symlink",
) -> dict[str, int]:
    """Organize all documents for a pipeline according to website URL structure."""
    data_root = _REPO_ROOT / "data" / pipeline
    reg_path = data_root / "documents.json"
    if not reg_path.exists():
        logger.warning(f"No documents.json found at {reg_path}")
        return {"total": 0, "organized": 0}

    try:
        registry = json.loads(reg_path.read_text(encoding="utf-8"))
        docs = registry.get("documents", [])
    except Exception as e:
        logger.error(f"Failed to read {reg_path}: {e}")
        return {"total": 0, "organized": 0}

    out_base = output_dir or (data_root / "site_structure")
    out_base.mkdir(parents=True, exist_ok=True)

    tree_hierarchy: dict[str, Any] = {}
    stats = {"total": len(docs), "organized": 0, "pdfs_linked": 0, "mds_linked": 0}

    logger.info(f"Organizing {len(docs)} documents for {pipeline} into {out_base}...")

    for d in docs:
        doc_id = d.get("id") or d.get("doc_dir")
        source_url = d.get("source_url") or ""
        domain, folders, filename = url_to_hierarchy(source_url, default_domain=f"{pipeline}.cg.gov.in")

        # Destination folder
        dest_dir = out_base / domain
        for f in folders:
            dest_dir = dest_dir / f
        dest_dir.mkdir(parents=True, exist_ok=True)

        base_stem = Path(filename).stem or doc_id
        
        # Link raw PDF
        pdf_path = None
        if d.get("pdf_path"):
            pdf_path = Path(d["pdf_path"])
        elif d.get("doc_dir"):
            cand = data_root / "input_pdfs" / f"{d['doc_dir']}.pdf"
            if cand.exists():
                pdf_path = cand

        if pdf_path and pdf_path.exists():
            target_pdf = dest_dir / f"{base_stem}.pdf"
            if link_or_copy(pdf_path, target_pdf, mode=mode):
                stats["pdfs_linked"] += 1

        # Link OCR Markdown
        md_path = None
        if d.get("md_path"):
            md_path = Path(d["md_path"])
        elif d.get("doc_dir"):
            cand = data_root / "stage2" / d["doc_dir"] / "structured.md"
            if cand.exists():
                md_path = cand

        if md_path and md_path.exists():
            target_md = dest_dir / f"{base_stem}.md"
            if link_or_copy(md_path, target_md, mode=mode):
                stats["mds_linked"] += 1

        # Save metadata summary card
        meta_file = dest_dir / f"{base_stem}_meta.json"
        meta_record = {
            "id": doc_id,
            "original_name": d.get("original_name"),
            "source_url": source_url,
            "doc_dir": d.get("doc_dir"),
            "status": d.get("status"),
            "verified": d.get("verified"),
            "indexed": d.get("indexed"),
            "chunk_count": d.get("chunk_count"),
            "site_path": f"{domain}/{'/'.join(folders)}/{base_stem}",
        }
        meta_file.write_text(json.dumps(meta_record, indent=2, ensure_ascii=False), encoding="utf-8")

        # Track in tree hierarchy
        curr = tree_hierarchy
        path_keys = [domain] + folders
        for k in path_keys:
            curr = curr.setdefault(k, {})
        curr[base_stem] = {
            "status": d.get("status"),
            "verified": d.get("verified"),
            "indexed": d.get("indexed"),
            "url": source_url,
        }

        stats["organized"] += 1

    # Generate SITE_MAP.md
    sitemap_lines = [
        f"# Website Document Structure: {pipeline.upper()}",
        f"",
        f"Generated from `{len(docs)}` documents registered in `documents.json`.",
        f"Mirroring official site hierarchy for direct tracing and navigation.",
        f"",
    ]

    def render_tree(node: dict, indent: int = 0):
        for k, v in sorted(node.items()):
            if isinstance(v, dict) and "url" not in v:
                sitemap_lines.append("  " * indent + f"- **📁 {k}/**")
                render_tree(v, indent + 1)
            else:
                status_icon = "✅" if v.get("status") == "processed" else "⏳"
                idx_icon = "🔍" if v.get("indexed") else "⚪"
                sitemap_lines.append("  " * indent + f"- {status_icon} {idx_icon} `{k}` [URL]({v.get('url')})")

    render_tree(tree_hierarchy)
    sitemap_path = out_base / "SITE_MAP.md"
    sitemap_path.write_text("\n".join(sitemap_lines), encoding="utf-8")
    logger.info(f"Generated site map at {sitemap_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Organize documents into website directory structure.")
    parser.add_argument("--pipeline", default="fg2", choices=["chips", "fg", "fg2"], help="Target pipeline (default: fg2 / GAD)")
    parser.add_argument("--output", default=None, help="Custom output directory")
    parser.add_argument("--mode", default="symlink", choices=["symlink", "hardlink", "copy"], help="File linkage mode")

    args = parser.parse_args()
    out = Path(args.output) if args.output else None
    stats = organize_pipeline_documents(pipeline=args.pipeline, output_dir=out, mode=args.mode)
    print("\n" + "=" * 50)
    print(f"ORGANIZATION COMPLETE ({args.pipeline.upper()})")
    print("=" * 50)
    print(f"Total documents registered: {stats['total']}")
    print(f"Documents organized:        {stats['organized']}")
    print(f"PDF files linked:           {stats['pdfs_linked']}")
    print(f"Markdown files linked:      {stats['mds_linked']}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
