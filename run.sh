#!/usr/bin/env bash
# One-command setup + run for Linux/macOS. Requires Python 3.10-3.12.
set -euo pipefail
cd "$(dirname "$0")"
PY=""
for v in 3.12 3.11 3.10; do command -v "python$v" >/dev/null 2>&1 && PY="python$v" && break; done
[ -n "$PY" ] || { echo "Python 3.10-3.12 not found (3.13+ is unsupported by torch 2.5.1)"; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
[ -f .env ] || { cp .env.example .env; echo "Created .env - add your GROQ_API_KEY, then re-run."; exit 0; }
mkdir -p data/uploads
python main.py
