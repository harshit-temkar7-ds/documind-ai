"""
DocuMind AI — RAG Engine
─────────────────────────
The heart of the system. Orchestrates:
  1. Retrieval — find relevant chunks from vector store
  2. Prompt building — construct a grounded prompt with context
  3. Generation — call Groq LLM for the answer
  4. Hallucination check — flag if answer isn't grounded in sources
  5. Response formatting — return answer + cited sources

RAG = Retrieval Augmented Generation
  Without RAG: LLM answers from training data (can hallucinate)
  With RAG:    LLM answers ONLY from your documents (grounded, cited)
"""

import time
from typing import List, Optional, Dict, Any
from groq import Groq
from dataclasses import dataclass, field

from backend.core import settings, get_logger
from backend.services.vector_store import get_vector_store

logger = get_logger(__name__)


@dataclass
class Source:
    """A cited source chunk in the answer."""
    doc_name:    str
    page_number: int
    text:        str
    similarity:  float


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""
    answer:          str
    sources:         List[Source]
    query:           str
    model_used:      str
    latency_ms:      float
    chunks_retrieved: int
    is_grounded:     bool       # True if answer is based on retrieved context
    confidence:      str        # HIGH / MEDIUM / LOW


class RAGEngine:
    """
    Orchestrates the full RAG pipeline:
    Query → Retrieve → Augment → Generate → Validate → Return
    """

    SYSTEM_PROMPT = """You are DocuMind AI, an expert document analyst.

Your job is to answer questions strictly based on the provided document excerpts.

Rules:
1. Answer ONLY using information from the provided context
2. If the context doesn't contain enough information, say so clearly
3. Always be precise, factual, and concise
4. Reference specific parts of the context when relevant
5. Never make up information not present in the context
6. If asked about something outside the document, politely redirect

Format your response in clear, readable paragraphs."""

    def __init__(self):
        if not settings.GROQ_API_KEY:
            logger.warning(
                "GROQ_API_KEY not set! Add it to your .env file.\n"
                "Get a free key at: https://console.groq.com"
            )
            self.client = None
        else:
            self.client = Groq(api_key=settings.GROQ_API_KEY)

        self.vector_store = get_vector_store()
        logger.info(
            f"RAGEngine initialized | model={settings.llm_model} | "
            f"top_k={settings.top_k_results}"
        )

    def _build_context_prompt(
        self,
        query: str,
        chunks: List[Dict]
    ) -> str:
        """
        Build the augmented prompt:
          [System prompt] + [Document context] + [User question]

        This is the core of RAG — the model sees both the question
        AND the retrieved evidence before generating an answer.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['doc_name']}, Page {chunk['page_number']}]\n"
                f"{chunk['text']}"
            )

        context_text = "\n\n---\n\n".join(context_parts)

        return f"""Based on the following document excerpts, please answer the question.

DOCUMENT CONTEXT:
{context_text}

---

QUESTION: {query}

ANSWER:"""

    def _check_grounding(self, answer: str, chunks: List[Dict]) -> bool:
        """
        Simple hallucination check:
        Verify the answer references concepts actually present in the chunks.

        Strategy: check if key answer words appear in at least one chunk.
        (Production would use a cross-encoder reranker for this.)
        """
        if not chunks:
            return False

        # Get all context text
        context_text = " ".join(
            chunk["text"].lower() for chunk in chunks
        )

        # Check if substantive answer words appear in context
        answer_words = set(
            w.lower() for w in answer.split()
            if len(w) > 4  # skip stop words
        )
        context_words = set(context_text.split())

        if not answer_words:
            return False

        overlap = answer_words & context_words
        overlap_ratio = len(overlap) / len(answer_words)

        # If >30% of answer words appear in context, it's grounded
        return overlap_ratio > 0.30

    def _assess_confidence(self, chunks: List[Dict]) -> str:
        """
        Assess answer confidence based on retrieval quality.
        HIGH:   top chunk similarity > 0.75
        MEDIUM: top chunk similarity > 0.50
        LOW:    top chunk similarity ≤ 0.50
        """
        if not chunks:
            return "LOW"
        top_sim = chunks[0]["similarity"]
        if top_sim > 0.75:
            return "HIGH"
        elif top_sim > 0.50:
            return "MEDIUM"
        return "LOW"

    def query(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        Main RAG pipeline — processes a user question end-to-end.

        Args:
            question: Natural language question from the user
            doc_ids:  Restrict search to specific documents (None = all)
            top_k:    Override default number of chunks to retrieve

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        start_time = time.perf_counter()
        top_k = top_k or settings.top_k_results

        logger.info(f"Processing query: '{question[:80]}{'...' if len(question) > 80 else ''}'")

        # ── Step 1: Retrieve relevant chunks ──────────────────────────────────
        retrieved_chunks = self.vector_store.search(
            query=question,
            doc_ids=doc_ids,
            top_k=top_k,
        )

        if not retrieved_chunks:
            return RAGResponse(
                answer="I couldn't find any relevant information in the uploaded documents. "
                       "Please make sure you've uploaded a PDF and try rephrasing your question.",
                sources=[],
                query=question,
                model_used=settings.llm_model,
                latency_ms=0,
                chunks_retrieved=0,
                is_grounded=False,
                confidence="LOW",
            )

        # ── Step 2: Build augmented prompt ────────────────────────────────────
        augmented_prompt = self._build_context_prompt(question, retrieved_chunks)

        # ── Step 3: Generate answer via Groq LLM ─────────────────────────────
        if self.client is None:
            answer = (
                "⚠️ GROQ_API_KEY is not configured. "
                "Please add your Groq API key to the .env file.\n\n"
                "Get a FREE key at: https://console.groq.com\n\n"
                f"Retrieved {len(retrieved_chunks)} relevant chunks — "
                "once the API key is set, you'll get full AI-generated answers."
            )
        else:
            try:
                response = self.client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system",  "content": self.SYSTEM_PROMPT},
                        {"role": "user",    "content": augmented_prompt},
                    ],
                    temperature=0.1,    # low temperature = more factual
                    max_tokens=1024,
                    top_p=0.9,
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Groq API error: {e}")
                answer = f"LLM generation failed: {str(e)}"

        # ── Step 4: Build source citations ────────────────────────────────────
        sources = [
            Source(
                doc_name=chunk["doc_name"],
                page_number=chunk["page_number"],
                text=chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"],
                similarity=chunk["similarity"],
            )
            for chunk in retrieved_chunks
        ]

        # ── Step 5: Validate grounding ────────────────────────────────────────
        is_grounded = self._check_grounding(answer, retrieved_chunks)
        confidence  = self._assess_confidence(retrieved_chunks)

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Query complete | latency={latency_ms:.0f}ms | "
            f"chunks={len(retrieved_chunks)} | confidence={confidence} | "
            f"grounded={is_grounded}"
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            model_used=settings.llm_model,
            latency_ms=round(latency_ms, 1),
            chunks_retrieved=len(retrieved_chunks),
            is_grounded=is_grounded,
            confidence=confidence,
        )


# Singleton
_rag_engine_instance = None

def get_rag_engine() -> RAGEngine:
    global _rag_engine_instance
    if _rag_engine_instance is None:
        _rag_engine_instance = RAGEngine()
    return _rag_engine_instance
