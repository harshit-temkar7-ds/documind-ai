"""
DocuMind AI — API Integration Tests
Run: pytest tests/ -v
"""

import pytest
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Set test environment
os.environ["GROQ_API_KEY"]        = "test_key"
os.environ["VECTORSTORE_PATH"]    = "./data/test_vectorstore"
os.environ["UPLOAD_FOLDER"]       = "./data/test_uploads"

from main import app

client = TestClient(app)


class TestHealth:
    def test_health_returns_200(self):
        res = client.get("/api/health")
        assert res.status_code == 200

    def test_health_fields(self):
        res = client.get("/api/health")
        data = res.json()
        assert "status"   in data
        assert "app_name" in data
        assert "version"  in data
        assert data["status"] == "healthy"


class TestDocuments:
    def test_list_documents_empty(self):
        res = client.get("/api/documents")
        assert res.status_code == 200
        data = res.json()
        assert "documents" in data
        assert "total"     in data
        assert isinstance(data["documents"], list)

    def test_upload_non_pdf_rejected(self):
        res = client.post(
            "/api/upload",
            files={"file": ("test.txt", b"hello world", "text/plain")}
        )
        assert res.status_code == 400
        assert "PDF" in res.json()["detail"]

    def test_upload_empty_pdf_handled(self):
        # Minimal valid PDF bytes
        minimal_pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF"
        res = client.post(
            "/api/upload",
            files={"file": ("empty.pdf", minimal_pdf, "application/pdf")}
        )
        # Should either succeed or return 422 (no text) — not 500
        assert res.status_code in [200, 422]

    def test_delete_nonexistent_document(self):
        res = client.delete("/api/documents/nonexistent_id")
        assert res.status_code in [404, 400]


class TestQuery:
    def test_query_no_documents(self):
        res = client.post(
            "/api/query",
            json={"question": "What is this about?"}
        )
        # Should get 404 (no docs) or 200 (with "no docs" message)
        assert res.status_code in [200, 404]

    def test_query_short_question_rejected(self):
        res = client.post(
            "/api/query",
            json={"question": "hi"}
        )
        assert res.status_code == 422  # Pydantic min_length validation

    def test_query_schema_validation(self):
        res = client.post(
            "/api/query",
            json={"question": "Valid question here", "top_k": 999}
        )
        # top_k max is 20
        assert res.status_code == 422


class TestFrontend:
    def test_root_returns_html(self):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers.get("content-type", "")
