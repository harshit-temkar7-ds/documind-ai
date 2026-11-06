"""
DocuMind AI — API Schemas
Pydantic models for all API request and response bodies.
These enforce type safety and auto-generate OpenAPI docs.
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


# ── Upload ─────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    success:      bool
    doc_id:       str
    filename:     str
    total_pages:  int
    total_chunks: int
    file_size_kb: float
    message:      str
    job_id:       Optional[str] = None
    status:       Optional[str] = None


# ── Documents ──────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    doc_id:       str
    filename:     str
    total_pages:  int
    total_chunks: int
    file_size_kb: float


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total:     int


class DeleteResponse(BaseModel):
    success: bool
    message: str


# ── Query ──────────────────────────────────────────────────────────────────────

class ChatTurn(BaseModel):
    """One earlier chat message, used only to resolve follow-up questions."""
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        examples=["What are the main findings of this document?"]
    )
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Restrict search to specific document IDs. None = search all."
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of chunks to retrieve. Defaults to setting in .env"
    )
    history: Optional[List[ChatTurn]] = Field(
        default=None,
        max_length=10,
        description="Up to 10 recent chat turns used to resolve follow-up questions.",
    )


class SourceInfo(BaseModel):
    doc_name:    str
    page_number: int
    text:        str
    similarity:  float


class QueryResponse(BaseModel):
    answer:           str
    sources:          List[SourceInfo]
    query:            str
    model_used:       str
    latency_ms:       float
    chunks_retrieved: int
    is_grounded:      bool
    confidence:       str


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:          str
    app_name:        str
    version:         str
    model:           str
    embedding_model: str
    documents_count: int
    groq_configured: bool
    vector_backend:  str = "chroma"
