# Contributing

Use Python 3.10-3.12 and a virtual environment: `pip install -r requirements-dev.txt`, then run
`ruff check .` and `pytest tests/ -v` before opening a PR. Keep API routes and response fields
backwards compatible unless a change is explicitly discussed. Add a test for every bug fix.
Never commit `.env`, uploaded PDFs, or generated vector-store files.
