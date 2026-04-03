"""
DocuMind AI — Vector Store Service
────────────────────────────────────
Manages document storage and similarity search using ChromaDB.

Architecture:
  - ChromaDB: persistent vector database (stores embeddings + metadata)
  - FAISS: used for fast in-memory similarity search on retrieval
  - Each document gets its own "collection" for isolation

Why ChromaDB?
  - Persistent (survives server restarts)
  - Stores both vectors AND metadata (filename, page number, etc.)
  - Simple Python API — no external server needed
  - Free and open source
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.core import settings, get_logger
from backend.services.embeddings import get_embedding_service
from backend.services.pdf_processor import DocumentChunk, ProcessedDocument

logger = get_logger(__name__)


class VectorStoreService:
    """
    Handles:
      1. Indexing — storing document chunks + embeddings in ChromaDB
      2. Retrieval — finding top-K similar chunks for a query
      3. Document management — list, delete, metadata
    """

    def __init__(self):
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=settings.vectorstore_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.embedding_service = get_embedding_service()

        # Master collection to track all indexed documents
        self.registry = self.client.get_or_create_collection(
            name="document_registry",
            metadata={"description": "Tracks all indexed documents"}
        )
        logger.info(
            f"VectorStore initialized | path={settings.vectorstore_path}"
        )

    def _get_collection_name(self, doc_id: str) -> str:
        """Each document gets its own ChromaDB collection."""
        return f"doc_{doc_id}"

    def index_document(self, processed_doc: ProcessedDocument) -> bool:
        """
        Index all chunks of a processed document into ChromaDB.

        Steps:
          1. Embed all chunk texts
          2. Store embeddings + texts + metadata in ChromaDB collection
          3. Register document in the master registry

        Returns:
            True if successful, False otherwise
        """
        try:
            doc_id = processed_doc.doc_id
            chunks = processed_doc.chunks

            if not chunks:
                logger.warning(f"No chunks to index for {processed_doc.filename}")
                return False

            logger.info(
                f"Indexing {len(chunks)} chunks for '{processed_doc.filename}'..."
            )

            # Step 1: Create or get collection for this document
            collection = self.client.get_or_create_collection(
                name=self._get_collection_name(doc_id),
                metadata={
                    "doc_id":   doc_id,
                    "filename": processed_doc.filename,
                    "pages":    processed_doc.total_pages,
                }
            )

            # Step 2: Embed all chunk texts at once (batched for speed)
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(texts)

            # Step 3: Store in ChromaDB
            collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[{
                    "doc_id":      chunk.doc_id,
                    "doc_name":    chunk.doc_name,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "char_count":  chunk.char_count,
                } for chunk in chunks]
            )

            # Step 4: Register in master registry
            self.registry.upsert(
                ids=[doc_id],
                documents=[processed_doc.filename],
                metadatas=[{
                    "doc_id":       doc_id,
                    "filename":     processed_doc.filename,
                    "total_pages":  processed_doc.total_pages,
                    "total_chunks": processed_doc.total_chunks,
                    "file_size_kb": processed_doc.file_size_kb,
                }]
            )

            logger.info(
                f"Indexing complete | doc_id={doc_id} | "
                f"{len(chunks)} chunks stored"
            )
            return True

        except Exception as e:
            logger.error(f"Indexing failed for {processed_doc.filename}: {e}")
            raise

    def search(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = None
    ) -> List[Dict]:
        """
        Find the most relevant chunks for a query using cosine similarity.

        Args:
            query:   Natural language question
            doc_ids: Search only within these documents (None = all)
            top_k:   Number of results to return

        Returns:
            List of dicts with text, metadata, and similarity score
        """
        top_k = top_k or settings.top_k_results

        # Embed the query
        query_embedding = self.embedding_service.embed_query(query)

        # Determine which collections to search
        if doc_ids:
            collections_to_search = [
                self._get_collection_name(did) for did in doc_ids
            ]
        else:
            # Search all registered documents
            all_docs = self.list_documents()
            collections_to_search = [
                self._get_collection_name(d["doc_id"]) for d in all_docs
            ]

        if not collections_to_search:
            logger.warning("No documents indexed yet.")
            return []

        # Search each collection and merge results
        all_results = []
        for col_name in collections_to_search:
            try:
                collection = self.client.get_collection(col_name)
                count = collection.count()
                if count == 0:
                    continue  # skip empty collections — n_results=0 crashes ChromaDB
                results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=min(top_k, count),
                    include=["documents", "metadatas", "distances"]
                )

                for i in range(len(results["ids"][0])):
                    # Convert distance to similarity score (0-1, higher = better)
                    distance = results["distances"][0][i]
                    similarity = 1 - (distance / 2)  # cosine: dist in [0,2]

                    all_results.append({
                        "text":        results["documents"][0][i],
                        "metadata":    results["metadatas"][0][i],
                        "similarity":  round(float(similarity), 4),
                        "doc_name":    results["metadatas"][0][i].get("doc_name", ""),
                        "page_number": results["metadatas"][0][i].get("page_number", 0),
                    })
            except Exception as e:
                logger.warning(f"Search failed for collection {col_name}: {e}")
                continue

        # Sort by similarity (highest first) and return top_k
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = all_results[:top_k]

        logger.info(
            f"Search complete | query='{query[:50]}{'...' if len(query) > 50 else ''}' | "
            f"returned {len(top_results)} chunks"
        )
        return top_results

    def list_documents(self) -> List[Dict]:
        """Return metadata for all indexed documents."""
        try:
            results = self.registry.get(include=["metadatas", "documents"])
            if not results["ids"]:
                return []

            docs = []
            for i, doc_id in enumerate(results["ids"]):
                meta = results["metadatas"][i]
                docs.append({
                    "doc_id":       doc_id,
                    "filename":     meta.get("filename", ""),
                    "total_pages":  meta.get("total_pages", 0),
                    "total_chunks": meta.get("total_chunks", 0),
                    "file_size_kb": meta.get("file_size_kb", 0),
                })
            return docs
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document and all its chunks from the vector store."""
        try:
            col_name = self._get_collection_name(doc_id)
            self.client.delete_collection(col_name)
            self.registry.delete(ids=[doc_id])
            logger.info(f"Document deleted | doc_id={doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document is already indexed."""
        try:
            results = self.registry.get(ids=[doc_id])
            return len(results["ids"]) > 0
        except:
            return False


# Singleton
_vector_store_instance = None

def get_vector_store() -> VectorStoreService:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreService()
    return _vector_store_instance
