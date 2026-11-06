
"""RAG engine adapter that preserves DocuMind AI's existing API contract."""

import time
import re
from dataclasses import dataclass
from typing import List, Optional

from backend.core import settings, get_logger
from backend.services.rag_graph import get_rag_graph, is_refusal

logger = get_logger(__name__)


@dataclass
class Source:
    """A cited source chunk in the answer."""
    doc_name: str
    page_number: int
    text: str
    similarity: float


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""
    answer: str
    sources: List[Source]
    query: str
    model_used: str
    latency_ms: float
    chunks_retrieved: int
    is_grounded: bool
    confidence: str


# Models write citations in different styles: [Source 2], 【Source 2】, *Source 2*, (Source 2)
_CITATION_NUMBER = re.compile(r"\bSource\s+(\d+)\b", flags=re.I)
_FANCY_BRACKETS = re.compile(r"[\[【［]\s*(Source\s+\d+)\s*[\]】］]", flags=re.I)


def normalize_citations(answer: str) -> str:
    """Rewrite 【Source 2】-style markers to the standard [Source 2]."""
    return _FANCY_BRACKETS.sub(r"[\1]", answer or "")


def select_cited_chunks(answer: str, chunks: list) -> list:
    """Return chunks the answer cites as 'Source N'; fall back to all chunks.

    The fallback also covers answers that cite no source or cite a number that
    does not exist, so a real answer never ends up with zero sources.
    """
    cited = {int(n) for n in _CITATION_NUMBER.findall(answer or "")}
    picked = [chunk for index, chunk in enumerate(chunks, 1) if index in cited]
    return picked or list(chunks)


def excerpt(text: str, limit: int = 300) -> str:
    """Shorten a chunk for display in the sources list."""
    return text[:limit] + "..." if len(text) > limit else text


class RAGEngine:
    """Adapt LangGraph output to the existing RAGResponse contract."""

    def __init__(self):
        self.graph = get_rag_graph()
        logger.info(
            "RAGEngine initialized with LangGraph | "
            f"model={settings.llm_model} | "
            f"top_k={settings.top_k_results}"
        )
    @staticmethod
    def _assess_confidence(chunks: list[dict]) -> str:
        """Estimate confidence from top and average retrieval similarity."""
        if not chunks:
            return "LOW"

        similarities = [
            max(0.0, min(1.0, float(chunk.get("similarity", 0.0))))
            for chunk in chunks
        ]

        top_similarity = max(similarities)
        average_similarity = sum(similarities) / len(similarities)

        score = (0.7 * top_similarity) + (0.3 * average_similarity)

        if score >= 0.70:
            return "HIGH"
        if score >= 0.40:
            return "MEDIUM"
        return "LOW"

    def query(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        history: Optional[list] = None,
        owner_id: Optional[str] = None,
    ) -> RAGResponse:
        """Execute LangGraph and return the existing response structure."""
        start_time = time.perf_counter()

        logger.info("Processing query through LangGraph")

        try:
            result = self.graph.invoke(
                question=question,
                doc_ids=doc_ids,
                top_k=top_k,
                history=history or [],
                owner_id=owner_id,
            )
        except Exception:
            logger.exception("LangGraph query failed")
            raise

        chunks = result.get("chunks", [])
        answer = normalize_citations(result.get("answer", ""))
        error = result.get("error")
        # Confidence reflects retrieval quality of everything retrieved, while the
        # sources shown to the user are limited to the ones the answer cites.
        confidence = self._assess_confidence(chunks)
        retrieved_count = len(chunks)
        if is_refusal(answer):
            chunks, confidence = [], "LOW"  # a refusal is not backed by any source
        else:
            chunks = select_cited_chunks(answer, chunks)

        if not chunks and not answer:
            answer = (
                "I couldn't find relevant information in the uploaded "
                "documents. Please try rephrasing your question."
            )

        sources = [
            Source(
                doc_name=chunk.get("doc_name", ""),
                page_number=chunk.get("page_number", 0),
                text=excerpt(chunk.get("text", "")),
                similarity=float(chunk.get("similarity", 0.0)),
            )
            for chunk in chunks
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000

        response = RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            model_used=settings.llm_model,
            latency_ms=round(latency_ms, 1),
            chunks_retrieved=len(chunks),
            is_grounded=bool(result.get("is_grounded", False)),
            confidence=confidence,
        )

        logger.info(
            f"LangGraph query complete | latency={latency_ms:.0f}ms | "
            f"chunks={len(chunks)}/{retrieved_count} | grounded={response.is_grounded} | "
            f"error={error or 'none'}"
        )

        return response


_rag_engine_instance = None


def get_rag_engine() -> RAGEngine:
    """Return the shared RAG engine instance."""
    global _rag_engine_instance

    if _rag_engine_instance is None:
        _rag_engine_instance = RAGEngine()

    return _rag_engine_instance
