# ---------- build stage: install Python deps (CPU-only torch) ----------
FROM python:3.12-slim AS build
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# ---------- runtime stage ----------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model-cache \
    TIKTOKEN_CACHE_DIR=/app/model-cache/tiktoken
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser
COPY --from=build /install /usr/local
COPY . .
# Bake the embedding model and tiktoken vocabulary into the image so the first
# upload/query does not need internet access.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2')" \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && mkdir -p /app/data/uploads /app/data/vectorstore \
    && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
# Chroma's local PersistentClient is not safe with several worker processes: keep ONE worker.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
