"""
Compare similarity-search speed: FAISS (exact, flat) vs ChromaDB (HNSW, approximate).

    python scripts/benchmark_search.py                      # 2,000 and 20,000 random vectors
    python scripts/benchmark_search.py --sizes 5000 50000 --queries 300

Measures search only (no embedding model needed), using random 384-d unit vectors like MiniLM's.
Also reports recall: how many of Chroma's top-k results match the exact FAISS answer.
Numbers depend on your CPU - run it on your own machine before quoting results.
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.services.faiss_index import FaissIndex  # noqa: E402


def run(n: int, dim: int, queries: int, top_k: int, seed: int = 0) -> dict:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, dim)).astype("float32")
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    qs = rng.normal(size=(queries, dim)).astype("float32")
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    ids = [f"c{i}" for i in range(n)]

    t = time.perf_counter()
    faiss_index = FaissIndex()
    faiss_index.add(ids, data, ["doc"] * n)
    faiss_build = time.perf_counter() - t

    with tempfile.TemporaryDirectory() as tmp:
        client = chromadb.PersistentClient(path=tmp, settings=ChromaSettings(anonymized_telemetry=False))
        collection = client.get_or_create_collection("bench", metadata={"hnsw:space": "cosine"},
                                                     embedding_function=None)
        t = time.perf_counter()
        for start in range(0, n, 5000):
            end = min(start + 5000, n)
            collection.add(ids=ids[start:end], embeddings=data[start:end].tolist())
        chroma_build = time.perf_counter() - t

        for q in qs[:10]:  # warm-up
            faiss_index.search(q, top_k)
            collection.query(query_embeddings=[q.tolist()], n_results=top_k)

        t = time.perf_counter()
        exact = [{c for c, _ in faiss_index.search(q, top_k)} for q in qs]
        faiss_ms = (time.perf_counter() - t) * 1000 / queries

        t = time.perf_counter()
        approx = [set(collection.query(query_embeddings=[q.tolist()], n_results=top_k)["ids"][0]) for q in qs]
        chroma_ms = (time.perf_counter() - t) * 1000 / queries

    recall = float(np.mean([len(a & e) / top_k for a, e in zip(approx, exact)]))
    return {"n": n, "faiss_ms": faiss_ms, "chroma_ms": chroma_ms, "recall": recall,
            "faiss_build": faiss_build, "chroma_build": chroma_build}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2000, 20000])
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print(f"{'vectors':>9} | {'FAISS ms/query':>15} | {'Chroma ms/query':>16} | {'speed-up':>8} | "
          f"{'Chroma recall@k':>15} | {'build FAISS/Chroma (s)':>23}")
    print("-" * 100)
    for n in args.sizes:
        r = run(n, args.dim, args.queries, args.top_k)
        print(f"{r['n']:>9,} | {r['faiss_ms']:>15.3f} | {r['chroma_ms']:>16.3f} | "
              f"{r['chroma_ms'] / r['faiss_ms']:>7.1f}x | {r['recall']:>14.1%} | "
              f"{r['faiss_build']:>10.2f} / {r['chroma_build']:<10.2f}")


if __name__ == "__main__":
    main()
