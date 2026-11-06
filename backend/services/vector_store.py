"""Persistent vector storage: ChromaDB (text + metadata) with FAISS or Chroma similarity search."""
from typing import Dict, List, Optional
import time
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from backend.core import settings, get_logger
from backend.services.embeddings import get_embedding_service
from backend.services.pdf_processor import ProcessedDocument

logger = get_logger(__name__)


class VectorStoreService:
    # Class-level defaults keep partially constructed instances (e.g. in tests) on the Chroma path.
    backend = "chroma"
    _faiss = None

    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.vectorstore_path,
            settings=ChromaSettings(anonymized_telemetry=False))
        self.embedding_service = None
        # embedding_function=None: we always pass our own vectors. Without this, Chroma
        # would silently download a second (ONNX) embedding model when none is supplied.
        self.collection = self.client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}, embedding_function=None)
        self.registry = self.client.get_or_create_collection(
            name="document_registry", metadata={"description": "Document metadata"},
            embedding_function=None)
        self._documents_cache = (0.0, [])
        if settings.vector_backend == "faiss":
            try:
                from backend.services.faiss_index import FaissIndex
                self._faiss = FaissIndex()
                self.backend = "faiss"
            except ImportError:
                logger.warning("VECTOR_BACKEND=faiss but faiss-cpu is not installed; "
                               "falling back to ChromaDB search. Fix: pip install faiss-cpu")
        logger.info(f"Vector search backend: {self.backend}")

    # ── FAISS <-> Chroma synchronisation ─────────────────────────────────────
    def _ensure_faiss(self) -> None:
        """Chroma is the source of truth: rebuild FAISS whenever the vector counts differ."""
        if self._faiss is not None and self._faiss.ntotal != self.collection.count():
            self._rebuild_faiss()

    def _rebuild_faiss(self, page_size: int = 2000) -> None:
        started = time.perf_counter()
        self._faiss.clear()
        offset = 0
        while True:
            page = self.collection.get(limit=page_size, offset=offset,
                                       include=["embeddings", "metadatas"])
            ids = page["ids"]
            if len(ids) == 0:
                break
            metas = page["metadatas"]
            self._faiss.add(ids, np.asarray(page["embeddings"], dtype="float32"),
                            [(m or {}).get("doc_id", "") for m in metas],
                            [(m or {}).get("owner_id", "") for m in metas])
            offset += len(ids)
        logger.info(f"FAISS index rebuilt from ChromaDB | vectors={self._faiss.ntotal} "
                    f"| {(time.perf_counter() - started) * 1000:.0f} ms")

    def index_document(self, processed_doc: ProcessedDocument, owner_id: str = "") -> bool:
        if not processed_doc.chunks:
            return False
        chunks = processed_doc.chunks
        try:
            if self.embedding_service is None:
                self.embedding_service = get_embedding_service()
            vectors = self.embedding_service.embed_texts([c.text for c in chunks])
            if self._faiss is not None:
                self._ensure_faiss()  # bring FAISS level with Chroma before adding to both
            self.collection.upsert(ids=[c.chunk_id for c in chunks],
                embeddings=vectors.tolist(), documents=[c.text for c in chunks],
                metadatas=[{"doc_id": c.doc_id, "doc_name": c.doc_name,
                    "page_number": c.page_number, "chunk_index": c.chunk_index,
                    "char_count": c.char_count, "owner_id": owner_id} for c in chunks])
            if self._faiss is not None:
                self._faiss.add([c.chunk_id for c in chunks], vectors,
                                [c.doc_id for c in chunks], [owner_id] * len(chunks))
            # Registry is the commit marker; write it only after every vector succeeds.
            # The registry only stores metadata; a constant 1-d vector satisfies Chroma.
            self.registry.upsert(ids=[processed_doc.doc_id], embeddings=[[0.0]],
                documents=[processed_doc.filename],
                metadatas=[{"doc_id": processed_doc.doc_id, "filename": processed_doc.filename,
                    "total_pages": processed_doc.total_pages, "total_chunks": len(chunks),
                    "file_size_kb": processed_doc.file_size_kb, "owner_id": owner_id}])
            self.invalidate_documents_cache()
            return True
        except Exception:
            self.collection.delete(where={"doc_id": processed_doc.doc_id})
            if self._faiss is not None:
                self._faiss.remove_doc(processed_doc.doc_id)
            raise

    def list_documents(self, owner_id: Optional[str] = None) -> List[Dict]:
        try:
            cached_at, cached_docs = self._documents_cache
            if owner_id is None and time.monotonic() - cached_at < 5:
                return list(cached_docs)
            result = self.registry.get(include=["metadatas"])
            docs = []
            for doc_id, meta in zip(result.get("ids", []), result.get("metadatas", [])):
                meta = meta or {}
                if owner_id is not None and meta.get("owner_id", "") != owner_id:
                    continue
                docs.append({k: meta.get(k, default) for k, default in {
                    "doc_id": doc_id, "filename": "", "total_pages": 0,
                    "total_chunks": 0, "file_size_kb": 0, "owner_id": ""}.items()})
            if owner_id is None:
                self._documents_cache = (time.monotonic(), docs)
            return docs
        except Exception:
            # Do not hide storage failures: returning [] would look like "no documents".
            logger.exception("Failed to list documents")
            raise

    def search(self, query: str, doc_ids: Optional[List[str]] = None,
               top_k: int = None, owner_id: Optional[str] = None) -> List[Dict]:
        top_k = top_k or settings.top_k_results
        docs = self.list_documents(owner_id)
        selected = [d["doc_id"] for d in docs if doc_ids is None or d["doc_id"] in doc_ids]
        count = self.collection.count()
        if not selected or top_k <= 0 or count == 0:
            return []
        if self.embedding_service is None:
            self.embedding_service = get_embedding_service()
        n_candidates = min(max(top_k, 20 if settings.hybrid_search or settings.enable_rerank else top_k), count)
        query_vector = self.embedding_service.embed_query(query)
        if self._faiss is not None:
            rows = self._faiss_candidates(query_vector, n_candidates, selected, owner_id)
        else:
            rows = self._chroma_candidates(query_vector, n_candidates, selected, owner_id)
        if settings.hybrid_search:
            from rank_bm25 import BM25Okapi
            corpus = [r["text"].lower().split() for r in rows]
            if corpus:
                scores = BM25Okapi(corpus).get_scores(query.lower().split())
                lexical = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
                vec = sorted(range(len(rows)), key=lambda i: rows[i]["similarity"], reverse=True)
                rrf = {}
                for ranking in (vec, lexical):
                    for rank, ix in enumerate(ranking, 1): rrf[ix] = rrf.get(ix, 0) + 1/(60+rank)
                rows = [rows[i] for i in sorted(rrf, key=rrf.get, reverse=True)]
        if settings.enable_rerank and rows:
            from sentence_transformers import CrossEncoder
            if not hasattr(self, "reranker"):
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            scores = self.reranker.predict([(query, row["text"]) for row in rows])
            rows = [row for _, row in sorted(zip(scores, rows), reverse=True, key=lambda p: p[0])]
        return rows[:top_k]

    def _chroma_candidates(self, query_vector, n: int, selected: List[str],
                           owner_id: Optional[str]) -> List[Dict]:
        filters = {"doc_id": {"$in": selected}}
        if owner_id is not None:
            filters = {"$and": [filters, {"owner_id": owner_id}]}
        result = self.collection.query(query_embeddings=[query_vector.tolist()], n_results=n,
                                       where=filters, include=["documents", "metadatas", "distances"])
        rows = []
        for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            if owner_id is not None and meta.get("owner_id", "") != owner_id:
                continue
            rows.append({"text": text, "metadata": meta,
                "similarity": round(max(0.0, min(1.0, 1.0 - float(distance))), 4),
                "doc_name": meta.get("doc_name", ""), "page_number": meta.get("page_number", 0)})
        return rows

    def _faiss_candidates(self, query_vector, n: int, selected: List[str],
                          owner_id: Optional[str]) -> List[Dict]:
        """FAISS finds the nearest chunk ids; Chroma supplies their text and metadata."""
        self._ensure_faiss()
        hits = self._faiss.search(query_vector, n, doc_ids=set(selected), owner_id=owner_id)
        if not hits:
            return []
        got = self.collection.get(ids=[chunk_id for chunk_id, _ in hits], include=["documents", "metadatas"])
        by_id = dict(zip(got["ids"], zip(got["documents"], got["metadatas"])))
        rows = []
        for chunk_id, score in hits:
            if chunk_id not in by_id:  # removed between the FAISS search and the Chroma lookup
                continue
            text, meta = by_id[chunk_id]
            rows.append({"text": text, "metadata": meta,
                "similarity": round(max(0.0, min(1.0, score)), 4),
                "doc_name": meta.get("doc_name", ""), "page_number": meta.get("page_number", 0)})
        return rows

    def delete_document(self, doc_id: str, owner_id: Optional[str] = None) -> bool:
        docs = self.list_documents(owner_id)
        if not any(d["doc_id"] == doc_id for d in docs): return False
        self.collection.delete(where={"doc_id": doc_id})
        if self._faiss is not None:
            self._faiss.remove_doc(doc_id)
        self.registry.delete(ids=[doc_id])
        self.invalidate_documents_cache()
        return True

    def invalidate_documents_cache(self) -> None:
        self._documents_cache = (0.0, [])

    def document_exists(self, doc_id: str, owner_id: Optional[str] = None) -> bool:
        return any(d["doc_id"] == doc_id for d in self.list_documents(owner_id))


_vector_store_instance = None
def get_vector_store() -> VectorStoreService:
    global _vector_store_instance
    if _vector_store_instance is None: _vector_store_instance = VectorStoreService()
    return _vector_store_instance
