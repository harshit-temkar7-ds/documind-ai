import numpy as np
from backend.services.pdf_processor import DocumentChunk, ProcessedDocument
from backend.services.vector_store import VectorStoreService

class FakeCollection:
    def __init__(self): self.rows = {}; self.last_where = None
    def count(self): return len(self.rows)
    def upsert(self, ids, embeddings, documents, metadatas):
        for row in zip(ids, embeddings, documents, metadatas): self.rows[row[0]] = row[1:]
    def query(self, query_embeddings, n_results, where, include):
        self.last_where = where
        matches = [(key, *value) for key, value in self.rows.items()
                   if self._matches(value[2], where)]
        return {"ids": [[row[0] for row in matches]], "documents": [[row[2] for row in matches]],
            "metadatas": [[row[3] for row in matches]], "distances": [[0.0 for row in matches]]}
    @staticmethod
    def _matches(meta, where):
        terms = where.get("$and", [where])
        return all(meta.get("doc_id") in term.get("doc_id", {}).get("$in", [meta.get("doc_id")])
            and ("owner_id" not in term or meta.get("owner_id") == term["owner_id"])
            for term in terms)
    def delete(self, where):
        self.rows = {key: value for key, value in self.rows.items()
            if not all(value[2].get(k) == v for k, v in where.items())}

class FakeRegistry:
    def __init__(self): self.rows = {}
    def upsert(self, ids, documents, metadatas, embeddings=None):
        for row in zip(ids, metadatas): self.rows[row[0]] = row[1]
    def get(self, include):
        return {"ids": list(self.rows), "metadatas": list(self.rows.values())}
    def delete(self, ids):
        for key in ids: self.rows.pop(key, None)

class FakeEmbeddings:
    def embed_texts(self, texts): return np.array([[1.0, 0.0] for _ in texts])
    def embed_query(self, query): return np.array([1.0, 0.0])

def test_index_cosine_search_filter_and_delete():
    service = object.__new__(VectorStoreService)
    service.collection = FakeCollection(); service.registry = FakeRegistry()
    service.embedding_service = FakeEmbeddings(); service._documents_cache = (0.0, [])
    chunk = DocumentChunk("chunk-a", "doc-a", "a.pdf", "identical text", 1, 0, 14)
    doc = ProcessedDocument("doc-a", "a.pdf", 1, 1, [chunk], 1.0)
    assert service.index_document(doc, "owner-a") is True
    result = service.search("identical text", ["doc-a"], top_k=1, owner_id="owner-a")
    assert result[0]["similarity"] == 1.0
    assert service.collection.last_where["$and"][0]["doc_id"]["$in"] == ["doc-a"]
    assert service.delete_document("doc-a", "owner-a") is True
    assert service.list_documents("owner-a") == []
