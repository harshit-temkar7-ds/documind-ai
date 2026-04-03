"""
DocuMind AI — API Routes
─────────────────────────
Endpoints:
  POST /api/upload          → Upload + index a PDF
  GET  /api/documents       → List all indexed documents
  DELETE /api/documents/{id}→ Remove a document
  POST /api/query           → Ask a question (RAG pipeline)
  GET  /api/health          → Health check
  GET  /                    → Serve the HTML frontend
"""

import os
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse

from backend.core import settings, get_logger
from backend.services import (
    PDFProcessor, get_vector_store, get_rag_engine
)
from backend.api.schemas import (
    UploadResponse, DocumentInfo, DocumentListResponse,
    DeleteResponse, QueryRequest, QueryResponse,
    SourceInfo, HealthResponse
)

logger = get_logger(__name__)
router = APIRouter()

# Lazy singletons
def _processor():  return PDFProcessor()
def _store():      return get_vector_store()
def _engine():     return get_rag_engine()


# ── Health Check ───────────────────────────────────────────────────────────────

@router.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API status, model availability, and document count."""
    store = _store()
    docs  = store.list_documents()

    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        model=settings.llm_model,
        embedding_model=settings.embedding_model,
        documents_count=len(docs),
        groq_configured=bool(settings.GROQ_API_KEY),    )


# ── Document Upload ────────────────────────────────────────────────────────────

@router.post("/api/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, extract text, chunk it, embed it, and store in vector DB.

    Process:
      1. Validate file type (PDF only)
      2. Save to uploads folder
      3. Extract text via PyMuPDF
      4. Split into overlapping chunks
      5. Embed all chunks (sentence-transformers)
      6. Store in ChromaDB
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file."
        )

    # Validate file size (max 50MB)
    MAX_SIZE = 50 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 50MB."
        )

    # Save file to uploads folder (use basename to prevent path traversal)
    safe_filename = Path(file.filename).name
    upload_path = Path(settings.upload_folder) / safe_filename
    with open(upload_path, "wb") as f:
        f.write(content)

    logger.info(f"File saved: {upload_path}")

    try:
        # Process PDF
        processor = _processor()
        processed_doc = processor.process(str(upload_path))

        # Check if already indexed
        store = _store()
        if store.document_exists(processed_doc.doc_id):
            logger.info(f"Document already indexed: {file.filename}")
            return UploadResponse(
                success=True,
                doc_id=processed_doc.doc_id,
                filename=processed_doc.filename,
                total_pages=processed_doc.total_pages,
                total_chunks=processed_doc.total_chunks,
                file_size_kb=processed_doc.file_size_kb,
                message=f"'{file.filename}' was already indexed. Ready to query!"
            )

        # Index in vector store
        store.index_document(processed_doc)

        return UploadResponse(
            success=True,
            doc_id=processed_doc.doc_id,
            filename=processed_doc.filename,
            total_pages=processed_doc.total_pages,
            total_chunks=processed_doc.total_chunks,
            file_size_kb=processed_doc.file_size_kb,
            message=(
                f"'{file.filename}' uploaded successfully! "
                f"{processed_doc.total_chunks} chunks indexed across "
                f"{processed_doc.total_pages} pages."
            )
        )

    except ValueError as e:
        # PDF has no extractable text (scanned/image PDF)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ── List Documents ─────────────────────────────────────────────────────────────

@router.get("/api/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """Return all indexed documents with metadata."""
    store = _store()
    docs  = store.list_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs)
    )


# ── Delete Document ────────────────────────────────────────────────────────────

@router.delete("/api/documents/{doc_id}", response_model=DeleteResponse, tags=["Documents"])
async def delete_document(doc_id: str):
    """Remove a document and all its embeddings from the vector store."""
    store = _store()

    # Fetch filename BEFORE deleting so we can clean up the physical file
    all_docs = store.list_documents()
    doc_meta = next((d for d in all_docs if d["doc_id"] == doc_id), None)

    success = store.delete_document(doc_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found."
        )

    # Remove the physical PDF file from uploads if it exists
    if doc_meta:
        file_path = Path(settings.upload_folder) / doc_meta["filename"]
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Deleted file from disk: {file_path}")
            except Exception as e:
                logger.warning(f"Could not delete file {file_path}: {e}")

    return DeleteResponse(
        success=True,
        message=f"Document '{doc_id}' removed successfully."
    )


# ── RAG Query ──────────────────────────────────────────────────────────────────

@router.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Ask a question about your uploaded documents.

    The RAG pipeline:
      1. Embeds your question
      2. Retrieves top-K most similar chunks from vector store
      3. Builds an augmented prompt (question + retrieved context)
      4. Sends to Groq LLM for grounded answer generation
      5. Returns answer + cited sources + confidence metrics
    """
    engine = _engine()
    store  = _store()

    # Verify documents exist
    all_docs = store.list_documents()
    if not all_docs:
        raise HTTPException(
            status_code=404,
            detail="No documents indexed yet. Please upload a PDF first."
        )

    # Validate requested doc_ids
    if request.doc_ids:
        all_ids = {d["doc_id"] for d in all_docs}
        invalid = set(request.doc_ids) - all_ids
        if invalid:
            raise HTTPException(
                status_code=404,
                detail=f"Documents not found: {list(invalid)}"
            )

    # Run RAG pipeline
    rag_response = engine.query(
        question=request.question,
        doc_ids=request.doc_ids,
        top_k=request.top_k,
    )

    return QueryResponse(
        answer=rag_response.answer,
        sources=[
            SourceInfo(
                doc_name=s.doc_name,
                page_number=s.page_number,
                text=s.text,
                similarity=s.similarity,
            )
            for s in rag_response.sources
        ],
        query=rag_response.query,
        model_used=rag_response.model_used,
        latency_ms=rag_response.latency_ms,
        chunks_retrieved=rag_response.chunks_retrieved,
        is_grounded=rag_response.is_grounded,
        confidence=rag_response.confidence,
    )
