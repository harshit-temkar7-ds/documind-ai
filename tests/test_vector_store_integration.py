"""Real ChromaDB, fake (deterministic) embeddings: index -> search -> isolate -> delete."""
import hashlib

import fitz
import numpy as np
import pytest

from backend.core import settings
from backend.services.pdf_processor import PDFProcessor
from backend.services.vector_store import VectorStoreService


class FakeEmbedder:
    """Bag-of-words hashing into 64 dims, L2-normalised (like the real model's output)."""
    DIM = 64

    def _vec(self, text):
        v = np.zeros(self.DIM, dtype="float32")
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_texts(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def make_doc(tmp_path, name, text):
    path = tmp_path / name
    pdf = fitz.open(); pdf.new_page().insert_text((72, 72), text); pdf.save(path); pdf.close()
    return PDFProcessor().process(str(path))


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "vectorstore_path", str(tmp_path / "chroma"))
    service = VectorStoreService()
    service.embedding_service = FakeEmbedder()
    return service


def test_index_search_filter_isolate_and_delete(store, tmp_path):
    finance = make_doc(tmp_path, "finance.pdf", "Quarterly revenue increased twelve percent due to subscription growth")
    biology = make_doc(tmp_path, "biology.pdf", "Mitochondria produce cellular energy through oxidative phosphorylation")
    assert store.index_document(finance, "alice") and store.index_document(biology, "bob")

    # exact text -> cosine similarity ~ 1, and the right document wins
    hits = store.search("Quarterly revenue increased twelve percent due to subscription growth", top_k=3)
    assert hits[0]["doc_name"] == "finance.pdf"
    assert hits[0]["similarity"] == pytest.approx(1.0, abs=1e-3)
    assert hits[0]["page_number"] == 1

    # owner isolation
    assert {d["doc_id"] for d in store.list_documents("alice")} == {finance.doc_id}
    only_alice = store.search("mitochondria energy", top_k=5, owner_id="alice")
    assert all(h["doc_name"] == "finance.pdf" for h in only_alice)

    # doc_id filter
    only_bio = store.search("revenue growth", doc_ids=[biology.doc_id], top_k=5)
    assert only_bio and all(h["doc_name"] == "biology.pdf" for h in only_bio)

    # existence + delete are owner-aware
    assert store.document_exists(finance.doc_id, "alice")
    assert not store.document_exists(finance.doc_id, "bob")
    assert not store.delete_document(finance.doc_id, "bob")
    assert store.delete_document(finance.doc_id, "alice")
    assert not store.document_exists(finance.doc_id)
    assert all(h["doc_name"] != "finance.pdf" for h in store.search("revenue", top_k=5))


def test_search_on_empty_store_returns_empty_list(store):
    assert store.search("anything at all") == []
