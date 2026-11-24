"""
Phase 1: Cleaning and Normalisation
Spec: Section 5 P1, Section 7.3 & Section 7.4

Transforms raw_md/<file_id>.md into:
  - clean/<file_id>.md
  - clean/<file_id>.search.txt (1:1 length guarantee, Devanagari digits mapped to ASCII)
"""

import os
import sys
import glob
import difflib
import random
from pathlib import Path
from typing import Dict, List, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.io import read_jsonl, write_jsonl, ensure_dir
from src.common.normalize import clean_document_text


def run_p01_clean(
    raw_dir: str = "gad_kg/data/raw_md",
    clean_dir: str = "gad_kg/data/clean",
    review_dir: str = "gad_kg/data/work/review",
    num_diff_samples: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    ensure_dir(clean_dir)
    ensure_dir(review_dir)

    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.md")))
    total_files = len(raw_files)
    print(f"[*] Found {total_files} raw files in {raw_dir}")

    broken_tables = []
    length_mismatches = []
    processed_count = 0

    diff_candidates = []

    for idx, fpath in enumerate(raw_files):
        file_id = Path(fpath).stem
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        clean_text, search_text, flags = clean_document_text(raw_text)

        if len(clean_text) != len(search_text):
            length_mismatches.append({
                "file_id": file_id,
                "clean_len": len(clean_text),
                "search_len": len(search_text),
            })

        if "broken_table" in flags:
            broken_tables.append({"file_id": file_id})

        # Save files
        clean_path = os.path.join(clean_dir, f"{file_id}.md")
        search_path = os.path.join(clean_dir, f"{file_id}.search.txt")

        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
        with open(search_path, "w", encoding="utf-8") as f:
            f.write(search_text)

        processed_count += 1
        if raw_text != clean_text:
            diff_candidates.append((file_id, raw_text, clean_text))

        if (idx + 1) % 1000 == 0 or (idx + 1) == total_files:
            print(f"    Progress: {idx + 1}/{total_files} processed...")

    # Write review table if any broken tables
    if broken_tables:
        csv_path = os.path.join(review_dir, "p1_broken_tables.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("file_id\n")
            for item in broken_tables:
                f.write(f"{item['file_id']}\n")
        print(f"[*] Flagged {len(broken_tables)} files with broken tables -> {csv_path}")

    # Acceptance check: 100% length invariant
    assert len(length_mismatches) == 0, f"FATAL: {len(length_mismatches)} files violated len(clean) == len(search)!"

    # Sample diffs for inspection
    rng = random.Random(seed)
    sampled_diffs = []
    if diff_candidates:
        samples = rng.sample(diff_candidates, min(num_diff_samples, len(diff_candidates)))
        for file_id, raw_t, clean_t in samples:
            raw_lines = raw_t.splitlines(keepends=True)
            clean_lines = clean_t.splitlines(keepends=True)
            diff = list(difflib.unified_diff(
                raw_lines[:30],
                clean_lines[:30],
                fromfile=f"raw/{file_id}.md",
                tofile=f"clean/{file_id}.md",
                n=2
            ))
            sampled_diffs.append((file_id, "".join(diff[:25])))

    metrics = {
        "total_files": total_files,
        "processed": processed_count,
        "broken_tables_count": len(broken_tables),
        "length_mismatches": len(length_mismatches),
        "files_modified_count": len(diff_candidates),
        "sampled_diffs": sampled_diffs,
    }
    return metrics


if __name__ == "__main__":
    metrics = run_p01_clean()
    print("\n" + "=" * 60)
    print("PHASE 1 ACCEPTANCE RESULTS:")
    print("=" * 60)
    print(f"Total files processed: {metrics['processed']}")
    print(f"Files modified:        {metrics['files_modified_count']} / {metrics['total_files']}")
    print(f"Length mismatch count: {metrics['length_mismatches']} (Requirement: 0)")
    print(f"Broken tables flagged: {metrics['broken_tables_count']}")
    print("=" * 60)
    print("\nSample Before/After Diffs (Showing up to 20):")
    for idx, (fid, diff_str) in enumerate(metrics["sampled_diffs"], 1):
        print(f"\n--- Sample {idx}: {fid} ---")
        if diff_str.strip():
            print(diff_str.strip())
        else:
            print("(Whitespace / formatting normalization only)")
