"""
Phase 8: Temporal Knowledge Graph & Deterministic Status Algorithm
Spec: Section 8.1 & 8.2
"""

from typing import Dict, Any, List, Tuple, Optional
from datetime import date

WHOLE = {"SUPERSEDES", "RESCINDS"}


class Edge:
    def __init__(
        self,
        edge_id: str,
        type: str,
        source_doc_id: str,
        target_doc_id: str,
        scope: str = "whole",
        scope_text: Optional[str] = None,
        effective_date: Optional[str] = None,
        source_issue_date: Optional[str] = None,
        source_effective_from: Optional[str] = None,
        tier: str = "Tier A",
        quote: Optional[str] = None,
    ):
        self.edge_id = edge_id
        self.type = type
        self.source_doc_id = source_doc_id
        self.target_doc_id = target_doc_id
        self.scope = scope
        self.scope_text = scope_text
        self.effective_date = effective_date
        self.source_issue_date = source_issue_date
        self.source_effective_from = source_effective_from
        self.tier = tier
        self.quote = quote

    @property
    def edge_date(self) -> Optional[str]:
        """When does this edge take effect?"""
        return self.effective_date or self.source_effective_from or self.source_issue_date


class EdgeStore:
    def __init__(self, edges: Optional[List[Edge]] = None):
        self.edges = edges or []
        self._incoming = {}
        for e in self.edges:
            self._incoming.setdefault(e.target_doc_id, []).append(e)

    def incoming(self, doc_id: str) -> List[Edge]:
        return self._incoming.get(doc_id, [])

    def add(self, e: Edge):
        self.edges.append(e)
        self._incoming.setdefault(e.target_doc_id, []).append(e)


def status_as_of(
    d: Dict[str, Any],
    T: str,
    edges: EdgeStore,
) -> Tuple[str, Any]:
    """Deterministic status algorithm (Spec Section 8.2).
    
    Returns (status, evidence)
    Status values:
      - NOT_YET_EFFECTIVE
      - EXPIRED
      - RESCINDED
      - SUPERSEDED
      - IN_FORCE_AMENDED
      - LIKELY_SUPERSEDED_DATE_UNKNOWN
      - IN_FORCE
    """
    doc_id = d.get("doc_id") or f"D_{d.get('file_id')}"
    eff_from = d.get("effective_from")
    eff_to = d.get("effective_to")

    if eff_from and T < eff_from:
        return "NOT_YET_EFFECTIVE", None

    if eff_to and T > eff_to:
        return "EXPIRED", None

    inc = [
        e for e in edges.incoming(doc_id)
        if e.tier in ("Tier A", "Tier B", "A", "B") and e.edge_date and e.edge_date <= T
    ]

    whole = [e for e in inc if e.type in WHOLE and e.scope == "whole"]
    if whole:
        e = min(whole, key=lambda x: x.edge_date)  # first replacement
        return ("RESCINDED" if e.type == "RESCINDS" else "SUPERSEDED"), e

    partial = [e for e in inc if e.type == "AMENDS" or (e.type in WHOLE and e.scope in ("partial", "unknown"))]
    corr = [e for e in inc if e.type == "CORRECTS"]
    ext = [e for e in inc if e.type == "EXTENDS"]
    if partial or corr or ext:
        return "IN_FORCE_AMENDED", partial + corr + ext

    # edges whose date is unknown and tier A/B -> be honest
    undated = [
        e for e in edges.incoming(doc_id)
        if e.tier in ("Tier A", "Tier B", "A", "B") and not e.edge_date and e.type in WHOLE
    ]
    if undated:
        return "LIKELY_SUPERSEDED_DATE_UNKNOWN", undated

    return "IN_FORCE", None


def compute_status_pipeline(
    header_path: str = "gad_kg/data/work/header.jsonl",
    edges_path: str = "gad_kg/data/work/edges.jsonl",
    output_path: str = "gad_kg/data/work/status.jsonl",
    as_of_date: Optional[str] = None,
) -> Dict[str, Any]:
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.common.io import read_jsonl, write_jsonl, ensure_dir

    if not as_of_date:
        as_of_date = date.today().isoformat()

    ensure_dir(Path(output_path).parent)

    headers = list(read_jsonl(Path(header_path))) if os.path.exists(header_path) else []
    headers_map = {h.get("doc_id", f"D_{h['file_id']}"): h for h in headers}

    raw_edges = list(read_jsonl(Path(edges_path))) if os.path.exists(edges_path) else []
    edge_objs = []
    for e in raw_edges:
        s_id = e.get("source_doc_id", "")
        s_header = headers_map.get(s_id, {})
        edge_objs.append(
            Edge(
                edge_id=e.get("edge_id", ""),
                type=e.get("relation", e.get("type", "")),
                source_doc_id=s_id,
                target_doc_id=e.get("target_doc_id", e.get("target", "")),
                scope=e.get("scope", "whole"),
                scope_text=e.get("scope_text"),
                effective_date=e.get("effective_date"),
                source_issue_date=s_header.get("issue_date"),
                source_effective_from=s_header.get("effective_from"),
                tier=e.get("tier", "Tier A"),
                quote=e.get("quote"),
            )
        )

    edge_store = EdgeStore(edge_objs)

    status_records = []
    status_counts = {}

    for h in headers:
        doc_id = h.get("doc_id", f"D_{h['file_id']}")
        status, evidence = status_as_of(h, as_of_date, edge_store)
        status_counts[status] = status_counts.get(status, 0) + 1

        ev_info = None
        if evidence:
            if isinstance(evidence, list):
                ev_info = [{"edge_id": ev.edge_id, "type": ev.type, "source": ev.source_doc_id, "date": ev.edge_date} for ev in evidence]
            elif isinstance(evidence, Edge):
                ev_info = {"edge_id": evidence.edge_id, "type": evidence.type, "source": evidence.source_doc_id, "date": evidence.edge_date}

        status_records.append({
            "doc_id": doc_id,
            "file_id": h["file_id"],
            "status_today": status,
            "as_of_date": as_of_date,
            "evidence": ev_info,
        })

    write_jsonl(Path(output_path), status_records)

    return {
        "total_docs": len(headers),
        "status_counts": status_counts,
        "as_of_date": as_of_date,
        "output_path": output_path,
    }


if __name__ == "__main__":
    res = compute_status_pipeline()
    print("\n" + "=" * 60)
    print("PHASE 8 DETERMINISTIC STATUS COMPUTATION RESULTS")
    print("=" * 60)
    print(f"Total Evaluated Documents: {res['total_docs']}")
    print(f"Status as of Date:         {res['as_of_date']}")
    print(f"Status Counts:             {res['status_counts']}")
    print(f"Status Output:             {res['output_path']}")
    print("=" * 60)
