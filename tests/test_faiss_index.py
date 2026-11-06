"""Unit tests for the FAISS wrapper (no embedding model, no Chroma)."""
import numpy as np
import pytest

from backend.services.faiss_index import FaissIndex


def unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v)


def test_returns_exact_cosine_similarity_in_rank_order():
    index = FaissIndex()
    index.add(["a", "b", "c"], [[1, 0], [1, 1], [0, 1]], ["d1", "d1", "d1"])
    hits = index.search([1, 0.1], top_k=3)
    assert [chunk for chunk, _ in hits] == ["a", "b", "c"]
    expected = float(unit([1, 1]) @ unit([1, 0.1]))
    assert dict(hits)["b"] == pytest.approx(expected, abs=1e-6)


def test_vectors_are_normalised_so_scale_does_not_matter():
    index = FaissIndex()
    index.add(["big", "small"], [[100, 0], [0, 0.001]], ["d", "d"])
    hits = dict(index.search([5, 0], top_k=2))
    assert hits["big"] == pytest.approx(1.0, abs=1e-6) and hits["small"] == pytest.approx(0.0, abs=1e-6)


def test_matches_brute_force_on_random_data():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(500, 32)).astype("float32")
    query = rng.normal(size=32).astype("float32")
    index = FaissIndex()
    index.add([f"c{i}" for i in range(500)], data, ["d"] * 500)
    normed = data / np.linalg.norm(data, axis=1, keepdims=True)
    expected = np.argsort(-(normed @ unit(query)))[:10]
    assert [int(c[1:]) for c, _ in index.search(query, 10)] == expected.tolist()


def test_document_and_owner_filters():
    index = FaissIndex()
    index.add(["a1", "a2"], [[1, 0], [0.9, 0.1]], ["docA", "docA"], ["alice", "alice"])
    index.add(["b1"], [[0.8, 0.2]], ["docB"], ["bob"])
    assert {c for c, _ in index.search([1, 0], 5, doc_ids={"docB"})} == {"b1"}
    assert {c for c, _ in index.search([1, 0], 5, owner_id="alice")} == {"a1", "a2"}
    assert index.search([1, 0], 5, doc_ids={"docB"}, owner_id="alice") == []


def test_strict_filter_searches_deeper_than_the_first_batch():
    # 300 vectors of doc A are all closer to the query than the 5 vectors of doc B.
    rng = np.random.default_rng(1)
    near = np.array([1.0, 0.0, 0.0, 0.0]) + rng.normal(scale=0.01, size=(300, 4))
    far = np.array([0.0, 1.0, 0.0, 0.0]) + rng.normal(scale=0.01, size=(5, 4))
    index = FaissIndex()
    index.add([f"a{i}" for i in range(300)], near, ["A"] * 300)
    index.add([f"b{i}" for i in range(5)], far, ["B"] * 5)
    hits = index.search([1, 0, 0, 0], top_k=5, doc_ids={"B"})
    assert len(hits) == 5 and all(c.startswith("b") for c, _ in hits)


def test_remove_doc_only_removes_that_document():
    index = FaissIndex()
    index.add(["a1", "a2"], [[1, 0], [0, 1]], ["A", "A"])
    index.add(["b1"], [[1, 1]], ["B"])
    assert index.remove_doc("A") == 2 and index.ntotal == 1
    assert [c for c, _ in index.search([1, 0], 5)] == ["b1"]
    assert index.remove_doc("A") == 0  # already gone


def test_adding_an_existing_chunk_replaces_it():
    index = FaissIndex()
    index.add(["c1"], [[1, 0]], ["d"])
    index.add(["c1"], [[0, 1]], ["d"])
    assert index.ntotal == 1
    assert index.search([0, 1], 1)[0][1] == pytest.approx(1.0, abs=1e-6)


def test_empty_index_and_edge_cases():
    index = FaissIndex()
    assert index.search([1, 0], 5) == []
    index.add(["a"], [[1, 0]], ["d"])
    assert index.search([1, 0], 0) == []
    with pytest.raises(ValueError):
        index.search([1, 0, 0], 1)      # wrong query dimension
    with pytest.raises(ValueError):
        index.add(["z"], [[1, 0, 0]], ["d"])  # wrong vector dimension
    index.clear()
    assert index.ntotal == 0 and index.search([1, 0], 1) == []
