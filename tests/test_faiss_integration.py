"""FAISS + real ChromaDB through VectorStoreService, compared with the Chroma-only backend."""
import hashlib
import sys

import fitz
import numpy as np
import pytest

from backend.core import settings
from backend.services.pdf_processor import PDFProcessor
from backend.services.vector_store import VectorStoreService


class FakeEmbedder:
    """Deterministic bag-of-words hashing into 64 dims (stand-in for MiniLM)."""
    DIM = 64

    def _vec(self, text):
        v = np.zeros(self.DIM, dtype="float32")
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_texts(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


TEXTS = {
    "finance.pdf": "Quarterly revenue increased twelve percent due to subscription growth",
    "biology.pdf": "Mitochondria produce cellular energy through oxidative phosphorylation",
    "travel.pdf": "The ferry to the island departs every morning from the northern harbour",
}


def make_doc(tmp_path, name):
    path = tmp_path / name
    pdf = fitz.open(); pdf.new_page().insert_text((72, 72), TEXTS[name]); pdf.save(path); pdf.close()
    return PDFProcessor().process(str(path))


def make_store(path, backend, monkeypatch):
    monkeypatch.setattr(settings, "vectorstore_path", str(path))
    monkeypatch.setattr(settings, "vector_backend", backend)
    store = VectorStoreService()
    store.embedding_service = FakeEmbedder()
    return store


@pytest.fixture(params=["faiss", "chroma"])
def backend(request):
    return request.param


def test_full_lifecycle_on_both_backends(tmp_path, monkeypatch, backend):
    store = make_store(tmp_path / "db", backend, monkeypatch)
    assert store.backend == backend
    finance, biology = make_doc(tmp_path, "finance.pdf"), make_doc(tmp_path, "biology.pdf")
    assert store.index_document(finance, "alice") and store.index_document(biology, "bob")

    top = store.search(TEXTS["finance.pdf"], top_k=3)[0]
    assert top["doc_name"] == "finance.pdf" and top["page_number"] == 1
    assert top["similarity"] == pytest.approx(1.0, abs=1e-3)

    assert all(h["doc_name"] == "finance.pdf" for h in store.search("energy", top_k=5, owner_id="alice"))
    only_bio = store.search("revenue growth", doc_ids=[biology.doc_id], top_k=5)
    assert only_bio and all(h["doc_name"] == "biology.pdf" for h in only_bio)

    assert not store.delete_document(finance.doc_id, "bob")
    assert store.delete_document(finance.doc_id, "alice")
    assert all(h["doc_name"] != "finance.pdf" for h in store.search("revenue", top_k=5))
    assert store.search("anything", doc_ids=["unknown-id"]) == []


def test_faiss_and_chroma_return_the_same_ranking(tmp_path, monkeypatch):
    docs = {n: make_doc(tmp_path, n) for n in TEXTS}
    results = {}
    for name in ("faiss", "chroma"):
        store = make_store(tmp_path / f"db_{name}", name, monkeypatch)
        for doc in docs.values():
            store.index_document(doc, "")
        results[name] = {q: [(h["doc_name"], h["similarity"]) for h in store.search(q, top_k=3)]
                         for q in ("subscription revenue", "cellular energy", "ferry harbour morning")}
    for query in results["faiss"]:
        f, c = results["faiss"][query], results["chroma"][query]
        assert [d for d, _ in f] == [d for d, _ in c]
        assert [s for _, s in f] == pytest.approx([s for _, s in c], abs=2e-3)


def test_faiss_is_rebuilt_from_chroma_after_restart(tmp_path, monkeypatch):
    first = make_store(tmp_path / "db", "faiss", monkeypatch)
    doc = make_doc(tmp_path, "biology.pdf")
    first.index_document(doc, "")
    assert first._faiss.ntotal == doc.total_chunks

    restarted = make_store(tmp_path / "db", "faiss", monkeypatch)   # new process: FAISS starts empty
    assert restarted._faiss.ntotal == 0
    hits = restarted.search("mitochondria energy", top_k=3)
    assert hits and hits[0]["doc_name"] == "biology.pdf"
    assert restarted._faiss.ntotal == doc.total_chunks


def test_faiss_heals_itself_when_it_drifts_from_chroma(tmp_path, monkeypatch):
    store = make_store(tmp_path / "db", "faiss", monkeypatch)
    store.index_document(make_doc(tmp_path, "finance.pdf"), "")
    store._faiss.clear()                                            # simulate a lost / stale index
    assert store.search("subscription revenue", top_k=3)[0]["doc_name"] == "finance.pdf"


def test_new_upload_after_restart_keeps_old_documents_searchable(tmp_path, monkeypatch):
    first = make_store(tmp_path / "db", "faiss", monkeypatch)
    first.index_document(make_doc(tmp_path, "finance.pdf"), "")
    second = make_store(tmp_path / "db", "faiss", monkeypatch)
    second.index_document(make_doc(tmp_path, "travel.pdf"), "")     # FAISS empty before this upload
    assert second.search("subscription revenue", top_k=3)[0]["doc_name"] == "finance.pdf"
    assert second.search("ferry harbour", top_k=3)[0]["doc_name"] == "travel.pdf"


def test_falls_back_to_chroma_when_faiss_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", None)                 # makes `import faiss` fail
    store = make_store(tmp_path / "db", "faiss", monkeypatch)
    assert store.backend == "chroma" and store._faiss is None
    store.index_document(make_doc(tmp_path, "finance.pdf"), "")
    assert store.search("subscription revenue", top_k=1)[0]["doc_name"] == "finance.pdf"


def test_invalid_backend_setting_is_rejected():
    from backend.core.config import Settings
    with pytest.raises(ValueError):
        Settings(vector_backend="pinecone")
    assert Settings(vector_backend=" FAISS ").vector_backend == "faiss"


def test_health_endpoint_reports_the_vector_backend():
    from fastapi.testclient import TestClient
    import main
    assert TestClient(main.app).get("/api/health").json()["vector_backend"] in {"faiss", "chroma"}
