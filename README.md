# DocuMind AI

Ask questions about your PDFs and get answers with page-level citations.
FastAPI + PyMuPDF + sentence-transformers + **FAISS** + ChromaDB + LangGraph + Groq.
**Live demo:** https://huggingface.co/spaces/HarryDataScience/documind-ai

## Quick start

**You need Python 3.10, 3.11 or 3.12** (3.13/3.14 do not work: `torch 2.5.1` has no wheels for them).
Get a free API key at <https://console.groq.com>.

### Windows (PowerShell)

```powershell
.\run.ps1
```

The script creates `.venv`, installs everything, creates `.env` (and opens it so you can paste
`GROQ_API_KEY`), then starts the app. Or do it manually:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
pip install -r requirements-dev.txt
Copy-Item .env.example .env           # then edit .env and set GROQ_API_KEY
python main.py
```

### Linux / macOS

```bash
./run.sh        # first run creates .env; add GROQ_API_KEY, then run it again
```

### Docker

```bash
cp .env.example .env     # Windows: Copy-Item .env.example .env   (then set GROQ_API_KEY)
docker-compose up --build
```

Open <http://localhost:8000> (API docs at `/docs`). The first upload/question loads the
embedding model (~90 MB download when running outside Docker).

## How it works

```
Upload:  PDF -> PyMuPDF text (per page, reading order, OCR fallback) -> clean -> token-based
         chunks (200 tokens, 30 overlap) -> MiniLM embeddings (384-d, normalised) -> ChromaDB (text + metadata) and FAISS (vectors)
Query:   question -> (condense follow-up using chat history) -> embed -> top-K cosine search
         -> weak evidence? -> "no evidence" answer
         -> otherwise Groq LLM answers with [Source N] citations -> grounding check
```

* LangGraph nodes: `condense_question` (only with history) -> `retrieve` -> gate -> `generate` -> `check_evidence`.
* **Vector search:** FAISS (`IndexFlatIP` on L2-normalised vectors = exact cosine similarity) finds the
  nearest chunk ids; ChromaDB stores the chunk text, metadata and the document list (one collection
  `documents`, filters on `doc_id` / `owner_id`). Chroma is the source of truth: the FAISS index lives
  in memory and is rebuilt from Chroma at startup (about 0.5 s per 10,000 chunks) and automatically
  whenever the two disagree. `VECTOR_BACKEND=chroma` switches to Chroma's own HNSW search; if
  `faiss-cpu` is missing the app falls back to it with a warning.
* Why FAISS here: exact search (100% recall) and faster than Chroma's approximate index at this size
  (`python scripts/benchmark_search.py`). A flat index scans every vector, so past roughly a million
  chunks switch to an approximate FAISS index (IVF/HNSW).
* `confidence` (HIGH/MEDIUM/LOW) comes from retrieval similarity; `is_grounded` is a lexical-overlap
  heuristic. **Neither proves the answer is correct** - always check the cited pages.
* Only the top-K chunks are shown to the LLM, so questions about the **whole** document ("how many
  sections are there?") can be wrong while still showing `is_grounded=true`.
* Reasoning models such as `openai/gpt-oss-*` need a large answer budget (`LLM_MAX_TOKENS`, default 4096).

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | - | Required for answers. Placeholder values are treated as "not set". |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Any chat model available to your Groq key (see Troubleshooting). |
| `LLM_MAX_TOKENS` | `4096` | Answer budget. Reasoning models (gpt-oss) need a large value. |
| `API_KEY` | empty | If set, clients must send `X-API-Key`; data is isolated per key. Empty = open (dev). |
| `ALLOWED_ORIGINS` | `http://localhost:8000` | Comma-separated CORS origins. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `CHUNK_UNIT` | `200` / `30` / `tokens` | Keep chunks <= 256 tokens: the embedding model ignores the rest. |
| `TOP_K_RESULTS` | `5` | Chunks sent to the LLM. |
| `RETRIEVAL_THRESHOLD` | `0.20` | Below this top similarity the LLM is not called. |
| `GROUNDING_OVERLAP_THRESHOLD` | `0.30` | Word-overlap needed for `is_grounded=true`. |
| `VECTOR_BACKEND` | `faiss` | `faiss` or `chroma` - which engine runs the similarity search. |
| `HYBRID_SEARCH` | `false` | Re-rank the top vector candidates with BM25 (RRF fusion). |
| `ENABLE_RERANK` | `false` | Cross-encoder re-ranking (downloads another model). |
| `ENABLE_FAITHFULNESS_CHECK` | `false` | Extra LLM call that judges whether the answer is supported. |
| `TORCH_THREADS` | `1` | CPU threads for embeddings. |
| `DEBUG` | `false` | `true` enables auto-reload. |

## API

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | Public. |
| POST | `/api/upload` | PDF, max 50 MB. >= 10 MB is indexed in the background and returns a `job_id`. |
| GET | `/api/jobs/{id}` | Background upload status (`queued/processing/done/failed`). |
| GET | `/api/documents` | List indexed documents. |
| DELETE | `/api/documents/{doc_id}` | Remove vectors and the stored PDF. |
| POST | `/api/query` | `{"question": "...", "top_k": 5, "doc_ids": [], "history": [{"role":"user","content":"..."}]}` (history: max 10 turns, 4000 chars each). |
| POST | `/api/query/stream` | Same input, Server-Sent Events: `token` events, then one `final` event. |

Query response:

```json
{
  "answer": "The report identifies three findings [Source 1].",
  "sources": [{"doc_name": "report.pdf", "page_number": 4, "text": "Relevant excerpt", "similarity": 0.82}],
  "query": "What are the findings?", "model_used": "openai/gpt-oss-20b",
  "latency_ms": 1200.0, "chunks_retrieved": 1, "is_grounded": true, "confidence": "HIGH"
}
```

Errors: `400` bad upload, `401` bad API key, `404` nothing indexed / unknown id, `409` content owned by
another key, `422` unreadable or scanned PDF without OCR / invalid body, `429` rate limit
(10 uploads/min, 30 queries/min per IP), `500` generic message with a `request_id`.

## Project structure

```text
main.py                         app, CORS, rate limiting, request ids
backend/api/{routes,schemas}.py HTTP layer
backend/core/{config,logger}.py settings + logging
backend/services/
    pdf_processor.py            extraction, OCR fallback, chunking
    embeddings.py               sentence-transformers wrapper (lazy singleton)
    vector_store.py             ChromaDB storage + indexing/search/delete (+ optional BM25/rerank)
    faiss_index.py              FAISS exact-cosine index with delete and doc/owner filters
    rag_graph.py                LangGraph workflow, prompts, grounding check
    rag_engine.py               adapts graph output to the API response
frontend/                       single-page UI (vanilla JS)
tests/  scripts/                pytest suite, retrieval evaluation script
```

## Development

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
ruff check .
python scripts/eval_retrieval.py scripts/eval_example.json   # hit-rate@k / MRR on your own PDFs
python scripts/benchmark_search.py                           # FAISS vs Chroma search speed + recall
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `No matching distribution found for torch==2.5.1` | You are on Python 3.13/3.14. Use Python 3.12 (`py -3.12 -m venv .venv`). |
| `running scripts is disabled` (PowerShell) | `Set-ExecutionPolicy -Scope Process RemoteSigned` |
| Answers say "Answer generation failed" / `model_not_found` | Your key cannot use that model. List the ones it can: `$k=(Select-String -Path .env -Pattern '^GROQ_API_KEY=(.*)$').Matches.Groups[1].Value; (Invoke-RestMethod https://api.groq.com/openai/v1/models -Headers @{Authorization="Bearer $k"}).data.id` and set `LLM_MODEL` to a chat model from that list. |
| Scanned PDF returns 422 | Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and put it on `PATH` (Docker image includes it). |
| `VECTOR_BACKEND=faiss but faiss-cpu is not installed` | `pip install -r requirements.txt` again (the app keeps working on Chroma meanwhile). |
| Port 8000 busy | Set `PORT=8001` in `.env`. |

## Known limitations

* `confidence` / `is_grounded` are heuristics, not guarantees of correctness.
* Rate limits and background jobs live in process memory: run **one** worker (Chroma's local
  client is also single-process). Use a real queue / server database to scale out.
* "Hybrid" search re-ranks the top vector candidates with BM25; it is not a full-corpus keyword search.
* OCR needs Tesseract. Tables and multi-column layouts are extracted as plain text.
* English-focused embedding model (`all-MiniLM-L6-v2`).
* The FAISS index is in-memory (rebuilt from Chroma at startup); it is exact, not approximate, so very
  large corpora (millions of chunks) need an IVF/HNSW index instead.
* Data written by older versions (one Chroma collection per document) is not migrated: delete
  `data/vectorstore` and re-upload your PDFs.
* Default mode has no authentication - set `API_KEY` before exposing the service.
