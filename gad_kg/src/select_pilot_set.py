"""
Select Stratified Pilot Set (200 canonical files)
Spec: Rule 3
  - ~40% Hindi (80 docs)
  - ~40% English (80 docs)
  - ~20% Mixed (40 docs)
  - Includes >= 20 table-heavy docs
  - Includes >= 20 docs with reference blocks
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.io import read_jsonl, ensure_dir


def select_pilot_set(
    inventory_path: str = "gad_kg/data/work/inventory.jsonl",
    features_path: str = "gad_kg/data/work/features.jsonl",
    output_path: str = "gad_kg/data/work/pilot_files.json",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    # Read inventory
    inv_map = {}
    for r in read_jsonl(Path(inventory_path)):
        inv_map[r["file_id"]] = r

    # Read features (canonical only)
    feat_map = {}
    for r in read_jsonl(Path(features_path)):
        feat_map[r["file_id"]] = r

    canonical_fids = list(feat_map.keys())
    print(f"[*] Available canonical files: {len(canonical_fids)}")

    # Group by language and features
    hi_files = []
    en_files = []
    mixed_files = []
    table_files = []
    ref_files = []

    for fid in canonical_fids:
        inv = inv_map.get(fid, {})
        feat = feat_map.get(fid, {})
        lang = inv.get("lang_guess", "hi").upper()
        n_tables = len(feat.get("tables", []))
        ref_items = len(feat.get("reference_block", {}).get("items", []))

        item = {
            "file_id": fid,
            "lang": lang,
            "n_tables": n_tables,
            "ref_items": ref_items,
            "ocr_quality": inv.get("ocr_quality", 0.9),
        }

        if n_tables >= 2:
            table_files.append(item)
        if ref_items >= 2:
            ref_files.append(item)

        if lang == "HI":
            hi_files.append(item)
        elif lang == "EN":
            en_files.append(item)
        else:
            mixed_files.append(item)

    print(f"    Language pools: HI={len(hi_files)}, EN={len(en_files)}, MIXED={len(mixed_files)}")
    print(f"    Special pools: Table-heavy={len(table_files)}, Reference-heavy={len(ref_files)}")

    rng = random.Random(seed)
    rng.shuffle(table_files)
    rng.shuffle(ref_files)
    rng.shuffle(hi_files)
    rng.shuffle(en_files)
    rng.shuffle(mixed_files)

    selected = {}

    # 1. Pick 20 table-heavy files
    for item in table_files[:20]:
        selected[item["file_id"]] = item

    # 2. Pick 20 reference-heavy files
    for item in ref_files:
        if len([x for x in selected.values() if x["ref_items"] >= 2]) >= 20:
            break
        selected[item["file_id"]] = item

    # 3. Fill up to ~80 HI, ~80 EN, ~40 MIXED (total 200)
    target_hi = 80
    target_en = 80
    target_mixed = 40

    # Count existing
    current_hi = len([x for x in selected.values() if x["lang"] == "HI"])
    current_en = len([x for x in selected.values() if x["lang"] == "EN"])
    current_mixed = len([x for x in selected.values() if x["lang"] == "MIXED"])

    for item in hi_files:
        if current_hi >= target_hi:
            break
        if item["file_id"] not in selected:
            selected[item["file_id"]] = item
            current_hi += 1

    for item in en_files:
        if current_en >= target_en:
            break
        if item["file_id"] not in selected:
            selected[item["file_id"]] = item
            current_en += 1

    for item in mixed_files:
        if current_mixed >= target_mixed:
            break
        if item["file_id"] not in selected:
            selected[item["file_id"]] = item
            current_mixed += 1

    # If any slack remains, fill from any remaining canonical files
    remaining = [inv_map[fid] for fid in canonical_fids if fid not in selected]
    rng.shuffle(remaining)
    for rem in remaining:
        if len(selected) >= 200:
            break
        fid = rem["file_id"]
        feat = feat_map.get(fid, {})
        selected[fid] = {
            "file_id": fid,
            "lang": rem.get("lang_guess", "HI"),
            "n_tables": len(feat.get("tables", [])),
            "ref_items": len(feat.get("reference_block", {}).get("items", [])),
            "ocr_quality": rem.get("ocr_quality", 0.9),
        }

    pilot_list = list(selected.values())[:200]
    ensure_dir(Path(output_path).parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pilot_list, f, indent=2, ensure_ascii=False)

    print(f"[*] Stratified Pilot Set of {len(pilot_list)} files saved to {output_path}")
    counts = {
        "HI": len([x for x in pilot_list if x["lang"] == "HI"]),
        "EN": len([x for x in pilot_list if x["lang"] == "EN"]),
        "MIXED": len([x for x in pilot_list if x["lang"] == "MIXED"]),
        "table_heavy": len([x for x in pilot_list if x["n_tables"] >= 2]),
        "ref_heavy": len([x for x in pilot_list if x["ref_items"] >= 2]),
    }
    print(f"    Counts: {counts}")
    return pilot_list


if __name__ == "__main__":
    select_pilot_set()
