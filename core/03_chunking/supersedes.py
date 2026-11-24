"""Post-ingestion supersedes linking and status update job.

Iterates over ingested batches and:
- Resolves referenced orders in `supersedes`
- Sets target document's `status = "superseded"` and `superseded_by = <new_doc_id>`
- Updates both document cards and all corresponding chunks in Qdrant
- Logs unresolved references to `unresolved_supersedes.csv`
- Logs documents with `doc_date = null` to `needs_review.csv` (they never supersede/get superseded automatically)
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def link_supersedes(
    doc_cards: List[Dict[str, Any]],
    chunk_records: List[Dict[str, Any]],
    output_dir: Path,
    qdrant_client: Optional[Any] = None,
    chunks_collection: Optional[str] = None,
    cards_collection: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute supersedes cross-linking across documents and update Qdrant payloads."""
    output_dir.mkdir(parents=True, exist_ok=True)
    unresolved_csv_path = output_dir / "unresolved_supersedes.csv"
    needs_review_csv_path = output_dir / "needs_review.csv"

    # Index cards by order_no and doc_id
    order_to_card: Dict[str, Dict[str, Any]] = {}
    doc_to_card: Dict[str, Dict[str, Any]] = {}
    
    needs_review_docs: List[Dict[str, Any]] = []

    for card in doc_cards:
        doc_id = card.get("doc_id")
        order_no = card.get("order_no")
        doc_date = card.get("doc_date")
        
        if doc_id:
            doc_to_card[doc_id] = card
        if order_no:
            # Normalize order_no for lookup (lowercase, stripped punctuation)
            norm_order = order_no.strip().lower()
            order_to_card[norm_order] = card

        # Rule: documents with doc_date=null never supersede or get superseded automatically
        if not doc_date or card.get("needs_review"):
            needs_review_docs.append({
                "doc_id": doc_id,
                "order_no": order_no or "",
                "title": card.get("title") or "",
                "reason": "Missing doc_date (null) or flagged for human review",
                "doc_date_raw": card.get("doc_date_raw") or "",
            })

    unresolved_records: List[Dict[str, Any]] = []
    superseded_doc_ids: set[str] = set()

    for card in doc_cards:
        new_doc_id = card.get("doc_id")
        new_doc_date = card.get("doc_date")
        supersedes_refs = card.get("supersedes") or []

        # If this document has no valid date, it cannot supersede other orders
        if not new_doc_date:
            continue

        for ref_order in supersedes_refs:
            norm_ref = ref_order.strip().lower()
            target_card = order_to_card.get(norm_ref)

            if target_card:
                target_doc_id = target_card.get("doc_id")
                # Rule: do not supersede if target has no date
                if not target_card.get("doc_date"):
                    needs_review_docs.append({
                        "doc_id": target_doc_id,
                        "order_no": target_card.get("order_no") or "",
                        "title": target_card.get("title") or "",
                        "reason": f"Referenced by {new_doc_id} but target doc_date is null; supersession blocked",
                        "doc_date_raw": target_card.get("doc_date_raw") or "",
                    })
                    continue

                # Update target card in-memory
                target_card["status"] = "superseded"
                target_card["superseded_by"] = new_doc_id
                if target_doc_id:
                    superseded_doc_ids.add(target_doc_id)

                # Update Qdrant if client is provided
                if qdrant_client and chunks_collection:
                    try:
                        from qdrant_client.models import Filter, FieldCondition, MatchValue
                        qdrant_client.set_payload(
                            collection_name=chunks_collection,
                            payload={"status": "superseded", "superseded_by": new_doc_id},
                            points=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=target_doc_id))])
                        )
                        if cards_collection:
                            qdrant_client.set_payload(
                                collection_name=cards_collection,
                                payload={"status": "superseded", "superseded_by": new_doc_id},
                                points=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=target_doc_id))])
                            )
                    except Exception as e:
                        logger.warning(f"Failed to update Qdrant payload for {target_doc_id}: {e}")
            else:
                unresolved_records.append({
                    "referencing_doc_id": new_doc_id,
                    "referencing_order_no": card.get("order_no") or "",
                    "unresolved_order_ref": ref_order,
                    "doc_date": new_doc_date,
                })

    # Update chunk_records in-memory
    for chunk in chunk_records:
        meta = chunk.get("metadata") or chunk
        c_doc_id = meta.get("doc_id")
        if c_doc_id in superseded_doc_ids:
            meta["status"] = "superseded"
            target_c = doc_to_card.get(c_doc_id)
            if target_c:
                meta["superseded_by"] = target_c.get("superseded_by")

    # Write unresolved_supersedes.csv
    with open(unresolved_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["referencing_doc_id", "referencing_order_no", "unresolved_order_ref", "doc_date"])
        writer.writeheader()
        writer.writerows(unresolved_records)

    # Write needs_review.csv
    with open(needs_review_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "order_no", "title", "reason", "doc_date_raw"])
        writer.writeheader()
        writer.writerows(needs_review_docs)

    return {
        "total_superseded": len(superseded_doc_ids),
        "unresolved_count": len(unresolved_records),
        "needs_review_count": len(needs_review_docs),
        "unresolved_csv": str(unresolved_csv_path),
        "needs_review_csv": str(needs_review_csv_path),
    }
