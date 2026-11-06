"""
DocuMind AI - FAISS similarity index
────────────────────────────────────
Exact cosine-similarity search with FAISS.

How it works
  * Every vector is L2-normalised, so the inner product (IndexFlatIP) equals cosine similarity.
  * IndexIDMap lets us delete one document's vectors without rebuilding the index.
  * FAISS only stores vectors. Chunk text, metadata and the document list stay in ChromaDB
    (the source of truth); `VectorStoreService` rebuilds this index from Chroma whenever the
    two disagree, so they can never drift apart.
  * FAISS cannot filter by metadata, so we keep a tiny (doc_id, owner_id) record per vector and
    over-fetch until enough results pass the filter.

Why a flat (exact) index? At this scale (thousands of chunks) it is faster than building an
approximate index and gives perfect recall. Switch to IVF/HNSW only past ~1M vectors.
"""
import threading
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from backend.core import get_logger

logger = get_logger(__name__)


def _as_matrix(vectors) -> np.ndarray:
    """float32, C-contiguous, shape (n, dim), L2-normalised (zero vectors are left as zeros)."""
    matrix = np.ascontiguousarray(np.asarray(vectors, dtype="float32"))
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(matrix / norms)


class FaissIndex:
    """Thread-safe FAISS index addressed by string chunk ids."""

    def __init__(self) -> None:
        import faiss  # imported lazily so the app still starts (on Chroma) if faiss-cpu is missing
        self._faiss = faiss
        self._lock = threading.RLock()
        self.dim: Optional[int] = None
        self.index = None
        self._next_id = 0
        self._id_to_chunk: Dict[int, str] = {}
        self._chunk_to_id: Dict[str, int] = {}
        self._meta: Dict[int, Tuple[str, str]] = {}  # internal id -> (doc_id, owner_id)

    # ── state ────────────────────────────────────────────────────────────────
    @property
    def ntotal(self) -> int:
        return len(self._id_to_chunk)

    def clear(self) -> None:
        with self._lock:
            self.dim = None
            self.index = None
            self._next_id = 0
            self._id_to_chunk.clear()
            self._chunk_to_id.clear()
            self._meta.clear()

    # ── writes ───────────────────────────────────────────────────────────────
    def add(self, chunk_ids: Sequence[str], vectors, doc_ids: Sequence[str],
            owner_ids: Optional[Sequence[str]] = None) -> None:
        """Add (or replace) vectors. Re-adding an existing chunk id overwrites it."""
        if len(chunk_ids) == 0:
            return
        matrix = _as_matrix(vectors)
        if not (len(chunk_ids) == len(doc_ids) == matrix.shape[0]):
            raise ValueError("chunk_ids, doc_ids and vectors must have the same length")
        owners = list(owner_ids) if owner_ids is not None else [""] * len(chunk_ids)

        with self._lock:
            if self.index is None:
                self.dim = matrix.shape[1]
                self.index = self._faiss.IndexIDMap(self._faiss.IndexFlatIP(self.dim))
            elif matrix.shape[1] != self.dim:
                raise ValueError(f"Vector dimension {matrix.shape[1]} does not match index ({self.dim})")

            self._remove_chunks([c for c in chunk_ids if c in self._chunk_to_id])  # upsert semantics
            ids = np.arange(self._next_id, self._next_id + len(chunk_ids), dtype="int64")
            self._next_id += len(chunk_ids)
            self.index.add_with_ids(matrix, ids)
            for internal, chunk, doc, owner in zip(ids.tolist(), chunk_ids, doc_ids, owners):
                self._id_to_chunk[internal] = chunk
                self._chunk_to_id[chunk] = internal
                self._meta[internal] = (doc, owner or "")

    def remove_doc(self, doc_id: str) -> int:
        """Remove every vector that belongs to a document. Returns how many were removed."""
        with self._lock:
            chunks = [self._id_to_chunk[i] for i, (doc, _) in self._meta.items() if doc == doc_id]
            self._remove_chunks(chunks)
            return len(chunks)

    def _remove_chunks(self, chunk_ids: Iterable[str]) -> None:
        internal = [self._chunk_to_id[c] for c in chunk_ids if c in self._chunk_to_id]
        if not internal or self.index is None:
            return
        self.index.remove_ids(np.asarray(internal, dtype="int64"))
        for i in internal:
            self._chunk_to_id.pop(self._id_to_chunk.pop(i), None)
            self._meta.pop(i, None)

    # ── reads ────────────────────────────────────────────────────────────────
    def search(self, query, top_k: int, doc_ids: Optional[set] = None,
               owner_id: Optional[str] = None) -> List[Tuple[str, float]]:
        """Return [(chunk_id, cosine_similarity)] best-first, honouring doc/owner filters."""
        with self._lock:
            total = self.ntotal
            if self.index is None or total == 0 or top_k <= 0:
                return []
            q = _as_matrix(query)
            if q.shape[1] != self.dim:
                raise ValueError(f"Query dimension {q.shape[1]} does not match index ({self.dim})")

            k = min(total, max(top_k * 4, 32))
            while True:
                scores, ids = self.index.search(q, k)
                hits: List[Tuple[str, float]] = []
                for score, internal in zip(scores[0], ids[0]):
                    if internal < 0:
                        continue
                    doc, owner = self._meta[int(internal)]
                    if doc_ids is not None and doc not in doc_ids:
                        continue
                    if owner_id is not None and owner != owner_id:
                        continue
                    hits.append((self._id_to_chunk[int(internal)], float(score)))
                    if len(hits) == top_k:
                        return hits
                if k >= total:  # scanned everything
                    return hits
                k = min(total, k * 4)  # filter was strict: look deeper
