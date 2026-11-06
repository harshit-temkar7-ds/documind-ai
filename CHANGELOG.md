# Changes in this review

## Update 3 - FAISS
- New `backend/services/faiss_index.py`: exact cosine search (`IndexFlatIP` + `IndexIDMap`) with
  per-document delete and doc/owner filtering (over-fetches until enough results pass the filter).
- `VectorStoreService` uses FAISS for similarity search and ChromaDB for text/metadata. Chroma stays
  the source of truth; FAISS is rebuilt from it at startup and whenever the counts differ.
- `VECTOR_BACKEND=faiss|chroma` (default `faiss`); automatic fallback to Chroma if `faiss-cpu` is missing.
- `/api/health` reports `vector_backend`; `scripts/benchmark_search.py` compares speed and recall.
- Tests: 72 -> 89 (FAISS unit tests incl. brute-force check, real FAISS+Chroma integration,
  FAISS-vs-Chroma ranking equivalence, restart/self-heal/fallback).

## Update 2 (found while testing with a real PDF and `openai/gpt-oss-20b`)
- **Citations:** gpt-oss writes `【Source 1】` (special brackets) and sometimes `*Source 1*`; only
  `[Source 1]` was recognised, so the "N sources" count was wrong (5 shown for an answer citing 1).
  All styles are accepted now and normalised to `[Source N]`.
- **Refusals** ("the document does not provide / mention / contain ...") now show **0 sources**
  and LOW confidence instead of five unrelated chunks. A corrective answer ("does not recommend")
  is not treated as a refusal.
- **Markdown** (bold, lists, tables, `code`) is rendered in answers by a small HTML-escaping
  renderer (`frontend/static/js/markdown.js`); "1 source" is singular.
- Tests: 53 -> 72.


**Crash / install blockers**
- `vector_store.py`: `Dict` was not imported -> `NameError` on startup (all tests failed).
- Chroma's `document_registry` upsert had no embeddings, so Chroma silently downloaded a second
  ONNX embedding model at runtime. Now `embedding_function=None` + dummy vector.
- Requirements: removed unused `langchain`/`langchain-community`, pinned `posthog==3.11.0`,
  split dev tools into `requirements-dev.txt`, documented Python 3.10-3.12 support.

**Bugs**
- Default chunk size 500 tokens exceeded the embedding model's 256-token limit -> now 200/30.
- OCR fallback crashed (500) when Tesseract was missing -> falls back to a clear 422.
- "Document already indexed" message never appeared.
- Streaming endpoint retrieved with the un-condensed follow-up question.
- Placeholder `GROQ_API_KEY=replace_with_your_key` counted as configured.
- `total_pages` ignored blank trailing pages; tiktoken failure when offline; `list_documents`
  hid storage errors; refusal detection looked anywhere in the answer; invalid `[Source N]`
  citation produced zero sources; confidence ignored non-cited chunks.

**Security / robustness**
- Chat `history` validated (role, length, max 10 turns); request ids sanitised; rate-limit
  buckets and finished jobs are pruned; frontend uses delegated events instead of inline `onclick`;
  validation errors are shown readably.

**Ops**
- Dockerfile: Python 3.12, tiktoken vocabulary baked in; CI uses CPU torch + Python 3.12;
  `run.ps1` / `run.sh` one-command setup.

**Tests**: 28 -> 53 (regression tests for every item above + real-ChromaDB integration test).
