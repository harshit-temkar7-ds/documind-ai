"""Report retrieval hit-rate@k and mean reciprocal rank for a JSON evaluation set."""
import argparse
import json
from pathlib import Path
from backend.services.vector_store import get_vector_store

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    store = get_vector_store()
    hits = reciprocal = 0
    for row in rows:
        results = store.search(row["question"], top_k=args.k)
        rank = next((i for i, item in enumerate(results, 1)
                     if item["page_number"] == row["expected_page"]), None)
        hits += rank is not None
        reciprocal += 1 / rank if rank else 0
    total = max(1, len(rows))
    print(f"hit-rate@{args.k}: {hits/total:.3f}\nMRR@{args.k}: {reciprocal/total:.3f}")

if __name__ == "__main__":
    main()
