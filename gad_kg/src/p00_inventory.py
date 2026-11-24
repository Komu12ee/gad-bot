"""
P0: Inventory and OCR Quality Assessment
Spec: Section 5 P0 & Section 4.2

Computes:
- file_id = sha1(raw file bytes)[:16]
- n_chars
- devanagari_ratio
- lang_guess (hi, en, mixed)
- ocr_quality in [0, 1]
- flags (needs_reocr, empty_file, corrupted_chars, high_isolated_chars, invalid_matras, etc.)

Outputs:
- work/inventory.jsonl
- work/review/p0_bad_ocr.csv (files with ocr_quality < ocr_quality_min)
"""

import csv
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Ensure src in path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.io import load_config, compute_file_id, read_jsonl, write_jsonl


def compute_ocr_quality(text: str) -> Tuple[float, str, float, List[str]]:
    """Compute Devanagari ratio, language guess, OCR quality score in [0, 1], and defect flags.
    
    Formula based on Section 5 P0:
    - share of tokens that look valid
    - absence of replacement character and stray unprintable symbols
    - ratio of isolated single characters
    - repeated-character runs
    - (for Hindi) valid Devanagari matra positioning
    """
    n_chars = len(text)
    if n_chars == 0:
        return 0.0, "en", 0.0, ["empty_file", "needs_reocr"]

    n_dev = len(re.findall(r"[\u0900-\u097F]", text))
    n_lat = len(re.findall(r"[A-Za-z]", text))
    total_letters = n_dev + n_lat
    dev_ratio = n_dev / total_letters if total_letters > 0 else 0.0

    if dev_ratio >= 0.70:
        lang_guess = "hi"
    elif dev_ratio <= 0.30:
        lang_guess = "en"
    else:
        lang_guess = "mixed"

    flags = []

    # 1. Clean characters (absence of replacement character \ufffd and unprintable control chars)
    bad_chars = len(re.findall(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]", text))
    s_clean = max(0.0, 1.0 - (bad_chars / max(1, n_chars * 0.01)))
    if s_clean < 0.85:
        flags.append("corrupted_chars")

    # 2. Token validity and isolated single-character ratio
    words = re.findall(r"[\w\u0900-\u097F]+", text)
    total_words = len(words)
    if total_words == 0:
        return 0.0, lang_guess, dev_ratio, ["no_words", "needs_reocr"]

    # Standalone 1-character words that are not normal particles (like English 'a', 'I' or Hindi 'व', 'य', or numbers)
    single_chars = sum(
        1 for w in words
        if len(w) == 1 and w not in {"a", "A", "I", "व", "य", "या", "न", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
    )
    single_ratio = single_chars / total_words
    s_connected = max(0.0, 1.0 - (2.5 * single_ratio))
    if s_connected < 0.75:
        flags.append("high_isolated_chars")

    # 3. Repeated character runs (e.g. ........ or ---------- or xxxxxx)
    runs = re.findall(r"(.)\1{4,}", text)
    s_runs = max(0.0, 1.0 - (len(runs) / max(1, total_words * 0.04)))
    if s_runs < 0.80:
        flags.append("character_runs")

    # 4. Valid word tokens: Hindi words must begin with consonant or independent vowel, not matra
    valid_tokens = sum(
        1 for w in words
        if re.match(r"^[A-Za-z0-9]+$|^[\u0904-\u0939][\u0901-\u097F0-9]*$", w)
    )
    s_valid = valid_tokens / total_words
    if s_valid < 0.75:
        flags.append("low_valid_tokens")

    # 5. Hindi matra placement validity
    if lang_guess in ("hi", "mixed") and n_dev > 0:
        # Invalid matra: starts with dependent vowel, or 2+ consecutive dependent vowels
        invalid_matras = len(re.findall(r"(?:^|\s)[\u093E-\u094D]|[\u093E-\u094C]{2,}", text))
        matra_ratio = invalid_matras / max(1, n_dev)
        s_matras = max(0.0, 1.0 - (8.0 * matra_ratio))
        if s_matras < 0.85:
            flags.append("invalid_matras")

        quality = (
            0.30 * s_valid +
            0.20 * s_clean +
            0.15 * s_connected +
            0.10 * s_runs +
            0.25 * s_matras
        )
    else:
        quality = (
            0.35 * s_valid +
            0.30 * s_clean +
            0.25 * s_connected +
            0.10 * s_runs
        )

    final_score = round(max(0.0, min(1.0, quality)), 4)
    return final_score, lang_guess, round(dev_ratio, 4), flags


def setup_raw_md_files(raw_md_dir: Path, source_dir: Path):
    """Ensure raw_md directory is populated with structured.md files named by <file_id>.md."""
    raw_md_dir.mkdir(parents=True, exist_ok=True)
    existing = list(raw_md_dir.glob("*.md"))
    if len(existing) >= 5000:
        print(f"raw_md directory already populated with {len(existing)} files.")
        return

    print(f"Syncing raw files from {source_dir} to {raw_md_dir}...")
    source_files = list(source_dir.glob("*/structured.md"))
    copied = 0
    for sf in source_files:
        data = sf.read_bytes()
        fid = compute_file_id(data)
        target = raw_md_dir / f"{fid}.md"
        if not target.exists():
            target.write_bytes(data)
            copied += 1
    print(f"Copied {copied} files to {raw_md_dir} (total: {len(list(raw_md_dir.glob('*.md')))}).")


def run_inventory():
    config = load_config()
    raw_dir = ROOT / config["paths"]["raw_md"]
    work_dir = ROOT / config["paths"]["work"]
    review_dir = work_dir / "review"
    inventory_file = work_dir / "inventory.jsonl"
    bad_ocr_csv = review_dir / "p0_bad_ocr.csv"
    ocr_min = float(config.get("thresholds", {}).get("ocr_quality_min", 0.80))

    review_dir.mkdir(parents=True, exist_ok=True)

    # Ensure raw_md populated from existing stage2 if needed
    source_stage2 = ROOT.parent / "data" / "fg2" / "stage2"
    if source_stage2.exists():
        setup_raw_md_files(raw_dir, source_stage2)

    # Resumable: read already processed file_ids
    existing_records = {}
    if inventory_file.exists():
        for rec in read_jsonl(inventory_file):
            existing_records[rec["file_id"]] = rec
        print(f"Resuming P0: {len(existing_records)} files already inventoried.")

    raw_files = sorted(raw_dir.glob("*.md"))
    print(f"Total raw files to inspect: {len(raw_files)}")

    new_records = []
    bad_records = []

    for idx, fpath in enumerate(raw_files, start=1):
        raw_bytes = fpath.read_bytes()
        fid = compute_file_id(raw_bytes)

        if fid in existing_records:
            rec = existing_records[fid]
        else:
            text = raw_bytes.decode("utf-8", errors="replace")
            q_score, lang, dev_ratio, flags = compute_ocr_quality(text)
            if q_score < ocr_min:
                flags.append("needs_reocr")

            rec = {
                "file_id": fid,
                "path": str(fpath.relative_to(ROOT)).replace("\\", "/"),
                "n_chars": len(text),
                "devanagari_ratio": dev_ratio,
                "lang_guess": lang,
                "ocr_quality": q_score,
                "flags": flags
            }
            new_records.append(rec)
            existing_records[fid] = rec

        if "needs_reocr" in rec.get("flags", []) or rec["ocr_quality"] < ocr_min:
            bad_records.append(rec)

        if idx % 1000 == 0 or idx == len(raw_files):
            print(f"Progress: {idx}/{len(raw_files)} files evaluated...")

    # Write all records
    if new_records:
        write_jsonl(inventory_file, list(existing_records.values()), append=False)
        print(f"Saved complete inventory to {inventory_file} ({len(existing_records)} total records).")

    # Write review/p0_bad_ocr.csv
    with open(bad_ocr_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id", "path", "ocr_quality", "lang_guess", "n_chars", "flags"])
        for r in sorted(bad_records, key=lambda x: x["ocr_quality"]):
            writer.writerow([r["file_id"], r["path"], r["ocr_quality"], r["lang_guess"], r["n_chars"], ";".join(r["flags"])])
    print(f"Saved {len(bad_records)} bad OCR records (< {ocr_min}) to {bad_ocr_csv}")

    # Print ACCEPTANCE REPORT
    print("\n" + "=" * 70)
    print("PHASE 0 ACCEPTANCE TEST RESULTS")
    print("=" * 70)
    all_recs = list(existing_records.values())
    total_docs = len(all_recs)
    lang_counts = {}
    for r in all_recs:
        lang_counts[r["lang_guess"]] = lang_counts.get(r["lang_guess"], 0) + 1

    print(f"Total Documents: {total_docs}")
    print("Counts by Language:")
    for lang, cnt in sorted(lang_counts.items()):
        pct = (cnt / total_docs) * 100
        print(f"  - {lang.upper():5}: {cnt:5} ({pct:.1f}%)")

    avg_qual = sum(r["ocr_quality"] for r in all_recs) / total_docs if total_docs else 0.0
    good_docs = sum(1 for r in all_recs if r["ocr_quality"] >= ocr_min)
    reocr_docs = total_docs - good_docs

    print(f"\nAverage OCR Quality: {avg_qual:.4f}")
    print(f"High Quality (≥ {ocr_min}): {good_docs} ({(good_docs/total_docs)*100:.1f}%)")
    print(f"Flagged needs_reocr (< {ocr_min}): {reocr_docs} ({(reocr_docs/total_docs)*100:.1f}%)")

    # List of 30 worst files
    sorted_worst = sorted(all_recs, key=lambda x: x["ocr_quality"])[:30]
    print("\nList of 30 Worst Files (Lowest OCR Quality):")
    print(f"{'#':2} | {'File ID':16} | {'Lang':5} | {'Score':6} | {'Chars':7} | Flags")
    print("-" * 70)
    for i, r in enumerate(sorted_worst, start=1):
        flags_str = ", ".join(r["flags"])
        print(f"{i:2} | {r['file_id']:16} | {r['lang_guess']:5} | {r['ocr_quality']:0.4f} | {r['n_chars']:7} | {flags_str}")
    print("=" * 70)


if __name__ == "__main__":
    run_inventory()
