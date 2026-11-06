"""Regression tests for model-specific citation styles and refusal handling."""
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
import backend.api.routes as routes
from backend.services.rag_engine import RAGEngine, normalize_citations, select_cited_chunks
from backend.services.rag_graph import is_refusal

client = TestClient(main.app)

CHUNKS = [{"doc_name": "g.pdf", "page_number": n, "text": f"chunk {n}", "similarity": 0.8}
          for n in range(1, 6)]


def test_normalize_citations_converts_fancy_brackets():
    # gpt-oss models answer with 【Source 1】 instead of [Source 1]
    assert normalize_citations("Google Docs 【Source 1】 and Canva 【 Source 3 】.") == \
        "Google Docs [Source 1] and Canva [Source 3]."
    assert normalize_citations("Plain [Source 2] stays.") == "Plain [Source 2] stays."


@pytest.mark.parametrize("answer", [
    "Uses it [Source 2].", "Uses it 【Source 2】.", "Uses it *Source 2*.", "Uses it (Source 2).",
    "Uses it. Sources: Source 2",
])
def test_select_cited_chunks_understands_every_citation_style(answer):
    assert [c["page_number"] for c in select_cited_chunks(answer, CHUNKS)] == [2]


def test_all_sources_cited_in_mixed_styles():
    answer = "A [Source 1], B 【Source 4】, C *Source 5*"
    assert [c["page_number"] for c in select_cited_chunks(answer, CHUNKS)] == [1, 4, 5]


@pytest.mark.parametrize("text,expected", [
    ("The guide does not provide a salary figure for a data scientist in Germany.", True),
    ("The document does not mention anything about visas.", True),
    # a corrective answer to a false premise is a real answer, not a refusal:
    ("The guide does not recommend paying a fee for internships; it warns against it.", False),
    ("The document does not contain information about salaries.", True),
    ("I couldn't find sufficiently relevant information in the uploaded documents.", True),
    ("Red flags include asking for fees. The guide cannot find every scam, though.", False),
    ("", False),
])
def test_is_refusal(text, expected):
    assert is_refusal(text) is expected


def _engine(answer, chunks=CHUNKS):
    engine = object.__new__(RAGEngine)
    engine.graph = SimpleNamespace(invoke=lambda **k: {"answer": answer, "chunks": chunks, "is_grounded": True})
    return engine


def test_engine_fancy_citations_filter_sources_and_normalise_answer():
    result = _engine("Websites: Google Docs 【Source 1】, Canva 【Source 1】.").query("which websites?")
    assert result.answer == "Websites: Google Docs [Source 1], Canva [Source 1]."
    assert [s.page_number for s in result.sources] == [1]
    assert result.chunks_retrieved == 1


def test_engine_refusal_has_no_sources_and_low_confidence():
    result = _engine("The document does not contain a salary figure.").query("salary in Germany?")
    assert result.sources == [] and result.chunks_retrieved == 0 and result.confidence == "LOW"


def _sse_final(response):
    final = None
    for line in response.text.splitlines():
        if line.startswith("data: ") and '"sources"' in line:
            final = json.loads(line[6:])
    return final


@pytest.fixture()
def streaming_setup(monkeypatch):
    class Store:
        def list_documents(self, owner_id=None):
            return [{"doc_id": "d1", "filename": "g.pdf", "total_pages": 5, "total_chunks": 5, "file_size_kb": 1.0}]
        def search(self, *a, **k):
            return CHUNKS
    monkeypatch.setattr(routes, "get_vector_store", lambda: Store())
    main._hits.clear()

    def install(answer):
        llm = SimpleNamespace(stream=lambda messages: [SimpleNamespace(content=answer)])
        graph = SimpleNamespace(llm=llm, check_evidence=lambda state: {"is_grounded": True})
        monkeypatch.setattr(routes, "get_rag_engine",
            lambda: SimpleNamespace(graph=graph, _assess_confidence=RAGEngine._assess_confidence))
    yield install
    main._hits.clear()


def test_stream_final_event_normalises_and_filters_sources(streaming_setup):
    streaming_setup("Google Docs 【Source 1】, Canva 【Source 1】")
    final = _sse_final(client.post("/api/query/stream", json={"question": "which websites?"}))
    assert final["answer"] == "Google Docs [Source 1], Canva [Source 1]"
    assert [s["page_number"] for s in final["sources"]] == [1] and final["chunks_retrieved"] == 1


def test_stream_refusal_has_no_sources(streaming_setup):
    streaming_setup("The guide does not contain a salary figure for Germany.")
    final = _sse_final(client.post("/api/query/stream", json={"question": "salary?"}))
    assert final["sources"] == [] and final["confidence"] == "LOW"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_markdown_renderer_formats_and_escapes():
    script = """
    const {renderMarkdown: r} = require('./frontend/static/js/markdown.js');
    const out = r('Bold **9 sections** [Source 5]\\n\\n- a\\n- b\\n\\n| H1 | H2 |\\n|---|---|\\n| x | y |\\n\\n<script>alert(1)</script>');
    process.stdout.write(out);
    """
    root = Path(__file__).resolve().parents[1]
    html = subprocess.run(["node", "-e", script], cwd=root, capture_output=True, text=True, check=True).stdout
    assert "<strong>9 sections</strong>" in html and '<span class="cite">[Source 5]</span>' in html
    assert "<ul><li>a</li><li>b</li></ul>" in html and "<th>H1</th>" in html and "<td>y</td>" in html
    assert "<script>" not in html and "&lt;script&gt;" in html
