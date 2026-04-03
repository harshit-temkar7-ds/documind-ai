"""
DocuMind AI — Embedding Service
─────────────────────────────────
Uses sentence-transformers to convert text → dense vectors.

Model: all-MiniLM-L6-v2
  - 384-dimensional vectors
  - Runs fully locally (no API key needed)
  - 80MB model, ~5ms per sentence on CPU
  - Excellent balance of speed vs quality for RAG

Why local embeddings?
  - Free (no API cost per query)
  - Fast and deterministic
  - Privacy-safe (documents never leave your machine)
  - Same model for both indexing and querying → consistent vector space
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from functools import lru_cache

from backend.core import settings, get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Wraps sentence-transformers for document and query embedding.
    Uses a singleton pattern — model is loaded once and reused.
    """

    _instance = None  # singleton

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self._initialized = True
        logger.info(
            f"Embedding model loaded | dimension={self.dimension} | "
            f"model={settings.embedding_model}"
        )

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Embed a list of texts in batches.

        Args:
            texts:      list of strings to embed
            batch_size: how many texts to process at once (GPU/CPU memory)

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([])

        logger.info(f"Embedding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,   # L2 norm → cosine sim = dot product
            convert_to_numpy=True,
        )
        logger.info(f"Embedding complete | shape={embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        Same model → compatible vector space as document embeddings.
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0]  # return 1D array

    @property
    def vector_dimension(self) -> int:
        return self.dimension


# Convenience singleton accessor
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
