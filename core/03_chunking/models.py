"""Chunk metadata models and schemas for CHiPS-RAG.

Defines Pydantic models for chunks, tables, and document cards to be stored in Qdrant
and serialized to JSONL / structured text files.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class TableChunkInfo(BaseModel):
    """Metadata specific to tabular chunks."""
    table_id: Optional[str] = None
    table_caption: Optional[str] = None
    fiscal_unit: Optional[str] = None
    column_headers: Optional[List[str]] = None
    row_range: Optional[List[int]] = None
    has_total_row: Optional[bool] = False
    continued_from_previous_page: Optional[bool] = False


class ReferencedDocument(BaseModel):
    """A referenced prior order, gazette, or act mentioned in body text."""
    ref_no: Optional[str] = None
    ref_date: Optional[str] = None
    ref_type: Optional[str] = None
    raw_mention: Optional[str] = None


class ChunkMetadata(BaseModel):
    """Complete metadata payload for a single chunk stored in Qdrant."""
    chunk_id: str = Field(description="Deterministic ID: <doc_id>_c001 or <doc_id>_card")
    doc_id: str = Field(description="Stable hash/identifier of the document")
    chunk_index: int = Field(default=1, ge=0)
    prev_id: Optional[str] = None
    next_id: Optional[str] = None
    parent_id: Optional[str] = None

    doc_type: Literal[
        "vacancy_circular",
        "order",
        "circular",
        "notification",
        "budget_table",
        "act_rule",
        "web_page",
        "other"
    ] = "other"
    order_no: Optional[str] = None
    title: Optional[str] = None
    ministry: Optional[str] = None
    department: Optional[str] = None
    office: Optional[str] = None
    language: Literal["en", "hi", "mixed"] = "mixed"

    doc_date: Optional[str] = Field(default=None, description="ISO format YYYY-MM-DD")
    doc_date_raw: Optional[str] = Field(default=None, description="Exact date string from document")
    doc_date_zone: Optional[Literal["header", "signature", "web", "file_metadata"]] = None
    date_confidence: Literal["high", "medium", "low", "none"] = "none"
    needs_review: bool = False

    effective_from: Optional[str] = None
    valid_until: Optional[str] = None
    date_source: Literal["text", "llm", "file_metadata"] = "text"
    status: Literal["current", "superseded", "expired"] = "current"
    supersedes: List[str] = Field(default_factory=list)
    superseded_by: Optional[str] = None

    topic_key: Optional[str] = Field(default=None, description="department:doc_type:normalized_subject")

    source_type: Literal["pdf", "web"] = "pdf"
    source_file: Optional[str] = None
    source_url: Optional[str] = None
    scraped_at: Optional[str] = None
    ingested_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    pages: List[int] = Field(default_factory=list)
    headings: List[str] = Field(default_factory=list)
    chunk_type: Literal["text", "table", "table_group", "card"] = "text"
    tokens: int = 0
    extraction_notes: List[str] = Field(default_factory=list)

    # Table-specific metadata fields
    table_id: Optional[str] = None
    table_caption: Optional[str] = None
    fiscal_unit: Optional[str] = None
    column_headers: Optional[List[str]] = None
    row_range: Optional[List[int]] = None
    has_total_row: Optional[bool] = None
    continued_from_previous_page: Optional[bool] = None


class DocumentCard(BaseModel):
    """Document-level card containing summary and verified facts."""
    doc_id: str
    order_no: Optional[str] = None
    title: Optional[str] = None
    doc_type: str = "other"
    ministry: Optional[str] = None
    department: Optional[str] = None
    office: Optional[str] = None
    doc_date: Optional[str] = None
    doc_date_raw: Optional[str] = None
    date_confidence: str = "none"
    needs_review: bool = False
    effective_from: Optional[str] = None
    valid_until: Optional[str] = None
    status: str = "current"
    supersedes: List[str] = Field(default_factory=list)
    superseded_by: Optional[str] = None
    topic_key: Optional[str] = None
    summary: str = Field(default="", description="2-3 line verified summary")
    total_chunks: int = 0
    total_pages: int = 0
    source_file: Optional[str] = None
    source_url: Optional[str] = None
    ingested_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))


class ChunkRecord(BaseModel):
    """Complete chunk record with metadata, body, and full embedded text."""
    metadata: ChunkMetadata
    body: str
    embedded_text: str
