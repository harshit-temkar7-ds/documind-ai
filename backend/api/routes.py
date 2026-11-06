"""Public API endpoints; response schemas remain backwards compatible."""
import hashlib
import json
import os
import re
import secrets
import time
import uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from backend.core import settings, get_logger
from backend.services import PDFProcessor, get_vector_store, get_rag_engine
from backend.services.rag_engine import excerpt, normalize_citations, select_cited_chunks
from backend.services.rag_graph import NO_EVIDENCE_ANSWER, SYSTEM_PROMPT, is_refusal
from backend.api.schemas import (UploadResponse, DocumentInfo, DocumentListResponse,
    DeleteResponse, QueryRequest, QueryResponse, SourceInfo, HealthResponse)

logger = get_logger(__name__)
router = APIRouter()
MAX_UPLOAD = 50 * 1024 * 1024
BACKGROUND_THRESHOLD = 10 * 1024 * 1024  # larger PDFs are indexed in the background
jobs = {}
JOB_TTL_SECONDS = 3600  # finished jobs are forgotten after one hour


def _prune_jobs() -> None:
    """Drop old finished jobs so the in-memory registry cannot grow forever."""
    cutoff = time.time() - JOB_TTL_SECONDS
    for job_id in [j for j, v in jobs.items()
                   if v.get("status") in ("done", "failed") and v.get("created", 0) < cutoff]:
        jobs.pop(job_id, None)


def _history(request: "QueryRequest") -> list:
    return [turn.model_dump() for turn in (request.history or [])]

def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> Optional[str]:
    if not settings.api_key:
        return None
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return hashlib.sha256(x_api_key.encode()).hexdigest()

def _store(): return get_vector_store()
def _safe_name(name: str) -> str:
    name = Path(name or "document.pdf").name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return (name or "document.pdf")[:180]

def _ingest_file(path: Path, filename: str, owner: Optional[str]):
    """Process + index a PDF. Returns (processed_document, already_indexed)."""
    processed = PDFProcessor().process(str(path))
    processed.filename = filename
    for chunk in processed.chunks:
        chunk.doc_name = filename
        chunk.metadata["source"] = filename
    store = _store()
    if store.document_exists(processed.doc_id, owner):
        return processed, True
    if owner is not None and store.document_exists(processed.doc_id):
        raise HTTPException(409, "This PDF content is already indexed.")
    destination = Path(settings.upload_folder) / f"{processed.doc_id}.pdf"
    os.replace(path, destination)
    try:
        if not store.index_document(processed, owner or ""):
            destination.unlink(missing_ok=True)
            raise ValueError("PDF contains no indexable text.")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return processed, False

async def _run_ingest_job(job_id: str, path: Path, filename: str, owner: str):
    jobs[job_id]["status"] = "processing"
    try:
        processed, _ = await run_in_threadpool(_ingest_file, path, filename, owner)
        jobs[job_id].update(status="done", doc_id=processed.doc_id,
            total_pages=processed.total_pages, total_chunks=processed.total_chunks)
    except ValueError as exc:
        jobs[job_id].update(status="failed", error=str(exc))
    except Exception:
        logger.exception("Background ingestion failed | job_id=%s", job_id)
        jobs[job_id].update(status="failed", error="Ingestion failed. See server logs.")
    finally:
        path.unlink(missing_ok=True)

@router.get("/api/health", response_model=HealthResponse, tags=["System"])
def health_check():
    store = _store()
    docs = store.list_documents()
    return HealthResponse(status="healthy", app_name=settings.app_name, version=settings.app_version,
        model=settings.llm_model, embedding_model=settings.embedding_model,
        documents_count=len(docs), groq_configured=bool(settings.GROQ_API_KEY),
        vector_backend=getattr(store, "backend", "chroma"))

@router.post("/api/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                     owner: str = Depends(require_api_key)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    folder = Path(settings.upload_folder); folder.mkdir(parents=True, exist_ok=True)
    staging = folder / f".{uuid.uuid4().hex}.tmp"
    size = 0
    try:
        with staging.open("wb") as output:
            first = await file.read(5)
            if not first: raise HTTPException(400, "Empty files are not accepted.")
            if first != b"%PDF-": raise HTTPException(400, "Invalid PDF file signature.")
            output.write(first); size += len(first)
            while block := await file.read(1024 * 1024):
                size += len(block)
                if size > MAX_UPLOAD: raise HTTPException(400, "File too large. Maximum size is 50MB.")
                output.write(block)
        if size <= 5: raise HTTPException(400, "Empty PDF files are not accepted.")
        safe_filename = _safe_name(file.filename)
        if size >= BACKGROUND_THRESHOLD:
            job_id = uuid.uuid4().hex
            pending = folder / f".{job_id}.pending"
            os.replace(staging, pending)
            staging = None
            _prune_jobs()
            jobs[job_id] = {"status": "queued", "owner_id": owner, "created": time.time()}
            background_tasks.add_task(_run_ingest_job, job_id, pending, safe_filename, owner)
            return UploadResponse(success=True, doc_id="", filename=safe_filename,
                total_pages=0, total_chunks=0, file_size_kb=round(size/1024, 2),
                message="Upload accepted for background processing.", job_id=job_id, status="queued")
        processed, already_indexed = await run_in_threadpool(
            _ingest_file, staging, safe_filename, owner)
        if already_indexed:
            existing = next((d for d in _store().list_documents(owner)
                             if d["doc_id"] == processed.doc_id), None)
            return UploadResponse(success=True, doc_id=processed.doc_id,
                filename=existing["filename"] if existing else processed.filename,
                total_pages=processed.total_pages, total_chunks=processed.total_chunks,
                file_size_kb=processed.file_size_kb, message="Document already indexed.")
        return UploadResponse(success=True, doc_id=processed.doc_id, filename=processed.filename,
            total_pages=processed.total_pages, total_chunks=processed.total_chunks,
            file_size_kb=processed.file_size_kb, message=f"'{processed.filename}' uploaded successfully!")
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception:
        logger.exception("Upload processing failed")
        raise HTTPException(500, "Upload processing failed. See server logs for details.") from None
    finally:
        if staging is not None: staging.unlink(missing_ok=True)
        await file.close()

@router.get("/api/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents(owner: str = Depends(require_api_key)):
    docs = _store().list_documents(owner)
    return DocumentListResponse(documents=[DocumentInfo(**d) for d in docs], total=len(docs))

@router.delete("/api/documents/{doc_id}", response_model=DeleteResponse, tags=["Documents"])
def delete_document(doc_id: str, owner: str = Depends(require_api_key)):
    store = _store()
    doc = next((d for d in store.list_documents(owner) if d["doc_id"] == doc_id), None)
    if not doc or not store.delete_document(doc_id, owner): raise HTTPException(404, f"Document '{doc_id}' not found.")
    (Path(settings.upload_folder) / f"{doc_id}.pdf").unlink(missing_ok=True)
    return DeleteResponse(success=True, message=f"Document '{doc_id}' removed successfully.")

@router.get("/api/jobs/{job_id}", tags=["Documents"])
def get_job_status(job_id: str, owner: str = Depends(require_api_key)):
    job = jobs.get(job_id)
    if not job or job.get("owner_id") != owner:
        raise HTTPException(404, "Job not found.")
    return {key: value for key, value in job.items() if key not in ("owner_id", "created")}

@router.post("/api/query", response_model=QueryResponse, tags=["Query"])
def query_documents(request: QueryRequest, owner: str = Depends(require_api_key)):
    store = _store(); docs = store.list_documents(owner)
    if not docs: raise HTTPException(404, "No documents indexed yet. Please upload a PDF first.")
    allowed = {d["doc_id"] for d in docs}
    if request.doc_ids and set(request.doc_ids) - allowed:
        raise HTTPException(404, f"Documents not found: {list(set(request.doc_ids)-allowed)}")
    result = get_rag_engine().query(request.question, request.doc_ids, request.top_k,
                                    history=_history(request), owner_id=owner)
    return QueryResponse(answer=result.answer, sources=[SourceInfo(doc_name=s.doc_name,
        page_number=s.page_number, text=s.text, similarity=s.similarity) for s in result.sources],
        query=result.query, model_used=result.model_used, latency_ms=result.latency_ms,
        chunks_retrieved=result.chunks_retrieved, is_grounded=result.is_grounded, confidence=result.confidence)

@router.post("/api/query/stream", tags=["Query"])
def query_stream(request: QueryRequest, owner: str = Depends(require_api_key)):
    """Stream Groq output tokens as SSE, followed by the regular response fields."""
    store = _store()
    docs = store.list_documents(owner)
    if not docs: raise HTTPException(404, "No documents indexed yet. Please upload a PDF first.")
    allowed = {d["doc_id"] for d in docs}
    if request.doc_ids and set(request.doc_ids) - allowed:
        raise HTTPException(404, "One or more requested documents were not found.")
    graph = get_rag_engine().graph
    history = _history(request)
    question = request.question
    if history:  # resolve follow-ups BEFORE retrieval so the right chunks are found
        question = graph.condense_question(
            {"question": request.question, "history": history}).get("question", request.question)
    chunks = store.search(question, request.doc_ids,
        request.top_k or settings.top_k_results, owner_id=owner)
    start = time.perf_counter()

    def sse(event: str, payload: dict) -> str:
        return f"event: {event}\ndata: " + json.dumps(payload) + "\n\n"

    if not chunks or max(c["similarity"] for c in chunks) < settings.retrieval_threshold:
        def no_evidence_events():
            yield sse("token", {"token": NO_EVIDENCE_ANSWER})
            yield sse("final", {"answer": NO_EVIDENCE_ANSWER, "sources": [],
                "query": request.question, "model_used": settings.llm_model,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "chunks_retrieved": 0, "is_grounded": False, "confidence": "LOW"})
        return StreamingResponse(no_evidence_events(), media_type="text/event-stream")

    context = "\n\n---\n\n".join(
        f"[Source {i}: {c['doc_name']}, Page {c['page_number']}]\n{c['text']}"
        for i, c in enumerate(chunks, 1))

    def events():
        from langchain_core.messages import HumanMessage, SystemMessage
        answer = ""
        if graph.llm is None:
            answer = "GROQ_API_KEY is not configured. Add your key to the .env file."
            yield sse("token", {"token": answer})
        else:
            messages = [SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION: {question}")]
            try:
                for part in graph.llm.stream(messages):
                    token = part.content if isinstance(part.content, str) else ""
                    if token:
                        answer += token
                        yield sse("token", {"token": token})
            except Exception:
                logger.exception("Streaming generation failed")
                answer = "Answer generation failed. Please try again."
                yield sse("token", {"token": answer})
        answer = normalize_citations(answer)
        grounded = graph.check_evidence({"chunks": chunks, "answer": answer}).get("is_grounded", False)
        confidence = get_rag_engine()._assess_confidence(chunks)
        if is_refusal(answer):
            sources, confidence = [], "LOW"  # a refusal is not backed by any source
        else:
            sources = select_cited_chunks(answer, chunks)
        yield sse("final", {"answer": answer,
            "sources": [{"doc_name": c["doc_name"], "page_number": c["page_number"],
                         "text": excerpt(c["text"]), "similarity": c["similarity"]} for c in sources],
            "query": request.question, "model_used": settings.llm_model,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "chunks_retrieved": len(sources), "is_grounded": grounded,
            "confidence": confidence})
    return StreamingResponse(events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
