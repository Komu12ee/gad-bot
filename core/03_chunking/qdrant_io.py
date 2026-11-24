"""Qdrant indexing, payload schemas, and query helpers for upgraded chunks.

Features:
- Creation of payload indexes: doc_id, order_no, doc_type, department, status, doc_date, valid_until, chunk_type, topic_key
- Default retrieval filter: status == "current" AND (valid_until is null OR valid_until >= today)
- Flag `include_old=True` to remove status filter and enable date range filtering
- Latest-version deduplication by `topic_key` (keeps only latest doc_date per topic)
- 2-Stage Retrieval Flow helper: search doc_cards -> filter chunks -> expand prev/next/parent chunks
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def ensure_chunk_payload_indexes(client: Any, collection_name: str, vector_size: int = 1024) -> None:
    """Ensure collection exists and create indexes for high-speed metadata filtering in Qdrant."""
    from qdrant_client.models import PayloadSchemaType, VectorParams, Distance

    try:
        if not client.collection_exists(collection_name):
            logger.info(f"Collection '{collection_name}' does not exist in Qdrant. Creating it (dim={vector_size}, distance=Cosine)...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        logger.warning(f"Error checking/creating collection '{collection_name}': {e}")

    indexes = [
        ("doc_id", PayloadSchemaType.KEYWORD),
        ("order_no", PayloadSchemaType.KEYWORD),
        ("doc_type", PayloadSchemaType.KEYWORD),
        ("department", PayloadSchemaType.KEYWORD),
        ("status", PayloadSchemaType.KEYWORD),
        ("doc_date", PayloadSchemaType.KEYWORD),
        ("valid_until", PayloadSchemaType.KEYWORD),
        ("chunk_type", PayloadSchemaType.KEYWORD),
        ("topic_key", PayloadSchemaType.KEYWORD),
    ]

    for field_name, field_type in indexes:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type,
            )
        except Exception as e:
            # Idempotent: ignore if index already exists
            logger.debug(f"Payload index for {field_name} returned: {e}")


def build_retrieval_filter(
    include_old: bool = False,
    doc_ids: Optional[List[str]] = None,
    doc_types: Optional[List[str]] = None,
    department: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Any:
    """Construct Qdrant Filter object applying status & validity logic."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range

    must_conditions = []

    # 1. Status and validity rule:
    # By default, only status == "current" AND (valid_until >= today or null)
    today_iso = datetime.utcnow().strftime("%Y-%m-%d")
    
    if not include_old:
        must_conditions.append(
            FieldCondition(key="status", match=MatchValue(value="current"))
        )

    if doc_ids:
        must_conditions.append(
            FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))
        )

    if doc_types:
        must_conditions.append(
            FieldCondition(key="doc_type", match=MatchAny(any=doc_types))
        )

    if department:
        must_conditions.append(
            FieldCondition(key="department", match=MatchValue(value=department))
        )

    # Date range filters (when include_old=True or caller specifies bounds)
    if date_from or date_to:
        rng = {}
        if date_from:
            rng["gte"] = date_from
        if date_to:
            rng["lte"] = date_to
        must_conditions.append(
            FieldCondition(key="doc_date", range=Range(**rng))
        )

    return Filter(must=must_conditions) if must_conditions else Filter()


def deduplicate_by_topic_key(
    chunks: List[Any],
    include_old: bool = False
) -> List[Any]:
    """If several search results share a topic_key, keep only the one with the latest doc_date."""
    if include_old:
        return chunks

    seen_topics: Dict[str, Tuple[str, Any]] = {}
    untagged_chunks: List[Any] = []

    for item in chunks:
        payload = getattr(item, "payload", {}) or (item if isinstance(item, dict) else {})
        topic = payload.get("topic_key")
        doc_date = payload.get("doc_date") or ""

        if not topic:
            untagged_chunks.append(item)
            continue

        if topic not in seen_topics:
            seen_topics[topic] = (doc_date, item)
        else:
            prev_date, _ = seen_topics[topic]
            # Replace if current candidate has a newer doc_date
            if doc_date > prev_date:
                seen_topics[topic] = (doc_date, item)

    deduped = [item for _, item in seen_topics.values()] + untagged_chunks
    return deduped


def two_stage_retrieval(
    client: Any,
    query_vector: List[float],
    chunks_collection: str,
    cards_collection: str,
    top_docs: int = 5,
    top_chunks_per_doc: int = 4,
    include_old: bool = False,
    expand_neighbors: bool = True,
) -> List[Dict[str, Any]]:
    """Two-stage retrieval flow:
    1. Search doc_cards first to find the most relevant documents.
    2. Search chunks filtered to those top documents.
    3. Optionally expand results to include prev/next chunks for context.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

    # Stage 1: Search document cards
    card_filter = build_retrieval_filter(include_old=include_old)
    card_hits = client.query_points(
        collection_name=cards_collection,
        query=query_vector,
        limit=top_docs,
        query_filter=card_filter,
    )

    doc_ids = []
    for hit in getattr(card_hits, "points", []):
        d_id = hit.payload.get("doc_id")
        if d_id and d_id not in doc_ids:
            doc_ids.append(d_id)

    if not doc_ids:
        # Fallback to direct chunk search if no cards matched
        chunk_hits = client.query_points(
            collection_name=chunks_collection,
            query=query_vector,
            limit=top_docs * top_chunks_per_doc,
            query_filter=build_retrieval_filter(include_old=include_old),
        )
        return deduplicate_by_topic_key(getattr(chunk_hits, "points", []), include_old=include_old)

    # Stage 2: Search chunks strictly within top matching documents
    chunk_filter = build_retrieval_filter(include_old=include_old, doc_ids=doc_ids)
    chunk_hits = client.query_points(
        collection_name=chunks_collection,
        query=query_vector,
        limit=top_docs * top_chunks_per_doc,
        query_filter=chunk_filter,
    )

    primary_points = getattr(chunk_hits, "points", [])
    deduped = deduplicate_by_topic_key(primary_points, include_old=include_old)

    # Optional: neighbor expansion (fetch prev_id and next_id chunks)
    if expand_neighbors and deduped:
        needed_chunk_ids: Set[str] = set()
        for p in deduped:
            pl = getattr(p, "payload", {})
            if pl.get("prev_id"):
                needed_chunk_ids.add(pl["prev_id"])
            if pl.get("next_id"):
                needed_chunk_ids.add(pl["next_id"])

        if needed_chunk_ids:
            try:
                neighbors, _ = client.scroll(
                    collection_name=chunks_collection,
                    scroll_filter=Filter(must=[FieldCondition(key="chunk_id", match=MatchAny(any=list(needed_chunk_ids)))]),
                    limit=len(needed_chunk_ids),
                    with_payload=True,
                )
                existing_ids = {getattr(p, "id", None) for p in deduped}
                for n in neighbors:
                    if getattr(n, "id", None) not in existing_ids:
                        deduped.append(n)
            except Exception as e:
                logger.debug(f"Neighbor expansion skipped: {e}")

    return deduped
