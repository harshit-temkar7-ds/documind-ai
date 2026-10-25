
from unittest.mock import Mock, patch

from backend.services.rag_engine import RAGEngine
from backend.services.rag_graph import DocuMindRAGGraph


# Confidence scoring tests

def test_confidence_is_low_when_no_chunks():
    assert RAGEngine._assess_confidence([]) == "LOW"


def test_confidence_is_high_for_strong_similarity():
    chunks = [
        {"similarity": 0.90},
        {"similarity": 0.85},
    ]
    assert RAGEngine._assess_confidence(chunks) == "HIGH"


def test_confidence_is_medium_for_moderate_similarity():
    chunks = [
        {"similarity": 0.55},
        {"similarity": 0.50},
    ]
    assert RAGEngine._assess_confidence(chunks) == "MEDIUM"


def test_confidence_is_low_for_weak_similarity():
    chunks = [
        {"similarity": 0.20},
        {"similarity": 0.15},
    ]
    assert RAGEngine._assess_confidence(chunks) == "LOW"


# RAG engine adapter tests

def test_query_adapts_graph_result():
    fake_graph = Mock()
    fake_graph.invoke.return_value = {
        "answer": "Python and SQL are listed.",
        "chunks": [
            {
                "doc_name": "Resume.pdf",
                "page_number": 1,
                "text": "Technical skills: Python and SQL.",
                "similarity": 0.85,
            }
        ],
        "is_grounded": True,
        "error": None,
    }

    with patch(
        "backend.services.rag_engine.get_rag_graph",
        return_value=fake_graph,
    ):
        engine = RAGEngine()
        result = engine.query("Which skills are listed?")

    assert result.answer == "Python and SQL are listed."
    assert result.query == "Which skills are listed?"
    assert result.chunks_retrieved == 1
    assert result.sources[0].doc_name == "Resume.pdf"
    assert result.sources[0].page_number == 1
    assert result.is_grounded is True
    assert result.confidence == "HIGH"
    fake_graph.invoke.assert_called_once()


def test_query_handles_empty_graph_result():
    fake_graph = Mock()
    fake_graph.invoke.return_value = {
        "answer": "",
        "chunks": [],
        "is_grounded": False,
        "error": None,
    }

    with patch(
        "backend.services.rag_engine.get_rag_graph",
        return_value=fake_graph,
    ):
        engine = RAGEngine()
        result = engine.query("What is in this document?")

    assert result.chunks_retrieved == 0
    assert result.is_grounded is False
    assert result.confidence == "LOW"
    assert "couldn't find relevant information" in result.answer


# LangGraph retrieval-routing tests

def test_retrieval_gate_finishes_when_no_chunks():
    assert DocuMindRAGGraph.route_after_retrieval(
        {"chunks": []}
    ) == "finish"


def test_retrieval_gate_finishes_for_weak_evidence():
    state = {
        "chunks": [
            {"similarity": 0.10},
            {"similarity": 0.15},
        ]
    }

    assert DocuMindRAGGraph.route_after_retrieval(state) == "finish"


def test_retrieval_gate_generates_for_sufficient_similarity():
    state = {
        "chunks": [
            {"similarity": 0.35},
            {"similarity": 0.25},
        ]
    }

    assert DocuMindRAGGraph.route_after_retrieval(state) == "generate"
