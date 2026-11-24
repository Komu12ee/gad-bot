"""
Phase 2: Deduplication
Spec: Section 5 P2

Algorithms:
  - Exact: SHA-1 of clean text.
  - Near-duplicate: MinHash LSH on 5-word shingles, Jaccard >= dedup_jaccard (0.90).
  - Canonical selection: highest ocr_quality, then longest text.
Outputs:
  - work/dedup.jsonl: file_id, cluster_id, canonical_file_id, is_duplicate
"""

import os
import sys
import glob
import hashlib
import difflib
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict
from datasketch import MinHash, MinHashLSH

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.io import read_jsonl, write_jsonl, load_config, ensure_dir


class DisjointSet:
    def __init__(self):
        self.parent = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def get_5word_shingles(text: str) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) < 5:
        return [" ".join(words)]
    return [" ".join(words[i:i+5]) for i in range(len(words) - 4)]


def compute_minhash(shingles: List[str], num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def exact_jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def cluster_documents(
    docs: List[Dict[str, Any]],
    threshold: float = 0.90,
    num_perm: int = 128
) -> List[Dict[str, Any]]:
    """Cluster documents using Exact SHA-1 + MinHash LSH with exact Jaccard verification."""
    dsu = DisjointSet()
    doc_map = {d["file_id"]: d for d in docs}

    # 1. Exact Deduplication via SHA-1 of clean_text
    exact_hash_to_files = defaultdict(list)
    for d in docs:
        h = hashlib.sha1(d["clean_text"].encode("utf-8")).hexdigest()
        exact_hash_to_files[h].append(d["file_id"])

    # Union exact duplicates
    distinct_rep_ids = []
    for h, fids in exact_hash_to_files.items():
        first = fids[0]
        distinct_rep_ids.append(first)
        for other in fids[1:]:
            dsu.union(first, other)

    # 2. MinHash LSH over distinct representatives
    if len(distinct_rep_ids) > 1:
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        rep_minhashes = {}
        rep_shingle_sets = {}

        for fid in distinct_rep_ids:
            shingles = get_5word_shingles(doc_map[fid]["clean_text"])
            m = compute_minhash(shingles, num_perm=num_perm)
            rep_minhashes[fid] = m
            rep_shingle_sets[fid] = set(shingles)
            lsh.insert(fid, m)

        checked_pairs = set()
        for fid in distinct_rep_ids:
            candidates = lsh.query(rep_minhashes[fid])
            for cand_id in candidates:
                if fid == cand_id:
                    continue
                pair = tuple(sorted((fid, cand_id)))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                # Exact Jaccard verification
                sim = exact_jaccard(rep_shingle_sets[fid], rep_shingle_sets[cand_id])
                if sim >= threshold:
                    dsu.union(fid, cand_id)

    # 3. Group by connected component root
    clusters_by_root = defaultdict(list)
    for d in docs:
        root = dsu.find(d["file_id"])
        clusters_by_root[root].append(d)

    # 4. Canonical file selection within each cluster
    clusters = []
    for root, members in clusters_by_root.items():
        # Rank by ocr_quality desc, then clean_text length desc
        members.sort(
            key=lambda x: (x.get("ocr_quality", 0.0), len(x.get("clean_text", ""))),
            reverse=True
        )
        canonical = members[0]
        canonical_fid = canonical["file_id"]
        cluster_id = f"C_{canonical_fid}"

        clusters.append({
            "cluster_id": cluster_id,
            "canonical_file_id": canonical_fid,
            "file_ids": [m["file_id"] for m in members],
            "members": members,
            "cluster_size": len(members),
        })

    return clusters


def run_p02_dedup(
    clean_dir: str = "gad_kg/data/clean",
    inventory_path: str = "gad_kg/data/work/inventory.jsonl",
    output_path: str = "gad_kg/data/work/dedup.jsonl",
    config_path: str = "gad_kg/config.yaml",
) -> Dict[str, Any]:
    ensure_dir(Path(output_path).parent)
    cfg = load_config(Path(config_path))
    threshold = float(cfg.get("thresholds", {}).get("dedup_jaccard", 0.90))

    # Read inventory to get ocr_quality
    ocr_qualities = {}
    if os.path.exists(inventory_path):
        for rec in read_jsonl(Path(inventory_path)):
            ocr_qualities[rec["file_id"]] = rec.get("ocr_quality", 0.90)

    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.md")))
    print(f"[*] Reading {len(clean_files)} clean documents for deduplication...")

    docs = []
    for idx, fpath in enumerate(clean_files):
        fid = Path(fpath).stem
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        docs.append({
            "file_id": fid,
            "clean_text": text,
            "ocr_quality": ocr_qualities.get(fid, 0.90),
        })

    print(f"[*] Clustering with Jaccard threshold >= {threshold}...")
    clusters = cluster_documents(docs, threshold=threshold)

    # Prepare records for work/dedup.jsonl
    dedup_records = []
    histogram = defaultdict(int)

    for c in clusters:
        size = c["cluster_size"]
        histogram[size] += 1
        canon_id = c["canonical_file_id"]
        cid = c["cluster_id"]

        for m in c["members"]:
            fid = m["file_id"]
            is_dup = (fid != canon_id)
            dedup_records.append({
                "file_id": fid,
                "cluster_id": cid,
                "canonical_file_id": canon_id,
                "is_duplicate": is_dup,
                "cluster_size": size,
                "ocr_quality": m["ocr_quality"],
            })

    write_jsonl(Path(output_path), dedup_records)
    print(f"[*] Written {len(dedup_records)} records to {output_path}")

    # Sample clusters with >1 members for acceptance review
    multi_clusters = [c for c in clusters if c["cluster_size"] > 1]
    multi_clusters.sort(key=lambda x: x["cluster_size"], reverse=True)
    sampled_multi_clusters = multi_clusters[:20]

    cluster_diff_summaries = []
    for c in sampled_multi_clusters:
        canon_fid = c["canonical_file_id"]
        canon_text = next(m["clean_text"] for m in c["members"] if m["file_id"] == canon_fid)
        diff_info = []
        for m in c["members"]:
            if m["file_id"] == canon_fid:
                continue
            other_fid = m["file_id"]
            other_text = m["clean_text"]
            diff_lines = list(difflib.unified_diff(
                canon_text.splitlines(),
                other_text.splitlines(),
                fromfile=f"canonical:{canon_fid}",
                tofile=f"dup:{other_fid}",
                n=1
            ))
            diff_info.append({
                "dup_id": other_fid,
                "diff_sample": "\n".join(diff_lines[:15]) if diff_lines else "(identical clean text)",
            })
        cluster_diff_summaries.append({
            "cluster_id": c["cluster_id"],
            "size": c["cluster_size"],
            "canonical_file_id": canon_fid,
            "diffs": diff_info,
        })

    total_canonical = len(clusters)
    total_duplicates = len(docs) - total_canonical

    return {
        "total_documents": len(docs),
        "total_clusters": len(clusters),
        "total_canonical": total_canonical,
        "total_duplicates": total_duplicates,
        "histogram": dict(sorted(histogram.items())),
        "multi_clusters_count": len(multi_clusters),
        "sampled_cluster_diffs": cluster_diff_summaries,
    }


if __name__ == "__main__":
    results = run_p02_dedup()
    print("\n" + "=" * 60)
    print("PHASE 2 ACCEPTANCE RESULTS: DEDUPLICATION")
    print("=" * 60)
    print(f"Total documents:   {results['total_documents']}")
    print(f"Total clusters:    {results['total_clusters']}")
    print(f"Canonical files:   {results['total_canonical']} (Unique documents)")
    print(f"Duplicate files:   {results['total_duplicates']}")
    print(f"Multi-doc clusters:{results['multi_clusters_count']}")
    print("\nCluster Size Histogram:")
    for size, count in results["histogram"].items():
        print(f"  Size {size:2d}: {count:5d} clusters ({count * size} total files)")
    print("=" * 60)
    print(f"\nShowing {len(results['sampled_cluster_diffs'])} Sampled Multi-Document Clusters with Diffs:")
    for idx, c in enumerate(results["sampled_cluster_diffs"], 1):
        print(f"\n--- Cluster {idx}: {c['cluster_id']} (Size {c['size']}) ---")
        print(f"Canonical: {c['canonical_file_id']}")
        for d in c["diffs"]:
            print(f"  Duplicate: {d['dup_id']}")
            print(f"  Diff:\n{d['diff_sample']}")
