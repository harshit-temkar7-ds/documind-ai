from fastapi.testclient import TestClient
from main import app
from backend.core import settings
from types import SimpleNamespace
import fitz

client = TestClient(app)

def test_authentication_gate_and_public_health():
    previous = settings.api_key
    settings.api_key = "test-secret"
    try:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/documents").status_code == 401
        assert client.get("/api/documents", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/api/documents", headers={"X-API-Key": "test-secret"}).status_code == 200
    finally:
        settings.api_key = previous

def test_upload_rejects_empty_and_bad_magic():
    empty = client.post("/api/upload", files={"file": ("empty.pdf", b"", "application/pdf")})
    invalid = client.post("/api/upload", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")})
    assert empty.status_code == 400
    assert invalid.status_code == 400

def test_sample_pdf_upload_then_query_with_mocked_llm(tmp_path, monkeypatch):
    import backend.api.routes as routes
    from backend.services.rag_engine import Source

    pdf = fitz.open(); page = pdf.new_page()
    page.insert_text((72, 72), "The sample report states that revenue increased by twelve percent this year.")
    content_path = tmp_path / "fixture.pdf"; pdf.save(content_path); pdf.close()
    content = content_path.read_bytes()

    class FakeStore:
        docs = {}
        def list_documents(self, owner_id=None): return list(self.docs.values())
        def document_exists(self, doc_id, owner_id=None): return doc_id in self.docs
        def index_document(self, processed, owner_id=""):
            self.docs[processed.doc_id] = {"doc_id": processed.doc_id, "filename": processed.filename,
                "total_pages": processed.total_pages, "total_chunks": processed.total_chunks,
                "file_size_kb": processed.file_size_kb, "owner_id": owner_id}
            return True

    fake = FakeStore()
    monkeypatch.setattr(routes, "get_vector_store", lambda: fake)
    monkeypatch.setattr(settings, "upload_folder", str(tmp_path / "uploads"))
    monkeypatch.setattr(routes, "get_rag_engine", lambda: SimpleNamespace(query=lambda *a, **k: SimpleNamespace(
        answer="Revenue increased by twelve percent [Source 1].", sources=[Source("fixture.pdf", 1,
        "The sample report states that revenue increased by twelve percent this year.", 0.91)],
        query="What happened to revenue?", model_used="mock", latency_ms=3.0,
        chunks_retrieved=1, is_grounded=True, confidence="HIGH")))
    upload = client.post("/api/upload", files={"file": ("fixture.pdf", content, "application/pdf")})
    assert upload.status_code == 200
    query = client.post("/api/query", json={"question": "What happened to revenue?"})
    assert query.status_code == 200
    assert query.json()["sources"][0]["page_number"] == 1
    assert query.json()["is_grounded"] is True
