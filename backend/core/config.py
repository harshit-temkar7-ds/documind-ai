"""
DocuMind AI — Centralized Configuration
Loads all settings from environment variables via .env file.
"""
from dotenv import load_dotenv
load_dotenv()
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """All application settings — loaded from .env file."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    # ── LLM ───────────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    llm_model: str = "openai/gpt-oss-20b"
    # Reasoning models spend part of this budget on hidden thinking, so keep it generous.
    llm_max_tokens: int = 4096

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Storage paths ─────────────────────────────────────────────────────────
    vectorstore_path: str = "./data/vectorstore"
    upload_folder: str = "./data/uploads"

    # ── Chunking ──────────────────────────────────────────────────────────────
    # all-MiniLM-L6-v2 reads at most 256 tokens, so keep chunks safely below that.
    chunk_size: int = 200
    chunk_overlap: int = 30
    chunk_unit: str = "tokens"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_results: int = 5
    retrieval_threshold: float = 0.20
    grounding_overlap_threshold: float = 0.30
    # "faiss": FAISS does the similarity search (Chroma stores text + metadata).
    # "chroma": Chroma's own HNSW index does the search. Falls back to chroma if faiss-cpu is missing.
    vector_backend: str = "faiss"
    hybrid_search: bool = False
    enable_rerank: bool = False
    enable_faithfulness_check: bool = False
    api_key: str = ""
    allowed_origins: str = "http://localhost:8000"
    torch_threads: int = 1

    # ── API ───────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    @field_validator("GROQ_API_KEY", "api_key", mode="before")
    @classmethod
    def _ignore_placeholder_keys(cls, value):
        """Treat template values such as 'replace_with_your_key' as 'not configured'."""
        if isinstance(value, str) and value.strip().lower().startswith("replace_with"):
            return ""
        return value.strip() if isinstance(value, str) else value

    @field_validator("vector_backend", mode="before")
    @classmethod
    def _check_vector_backend(cls, value):
        value = str(value).strip().lower()
        if value not in {"faiss", "chroma"}:
            raise ValueError("VECTOR_BACKEND must be 'faiss' or 'chroma'")
        return value

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    # ── App Meta ──────────────────────────────────────────────────────────────
    app_name: str = "DocuMind AI"
    app_version: str = "1.0.0"
    app_description: str = "RAG-powered document intelligence system"

    def ensure_directories(self):
        """Create required directories if they don't exist."""
        Path(self.vectorstore_path).mkdir(parents=True, exist_ok=True)
        Path(self.upload_folder).mkdir(parents=True, exist_ok=True)


# Singleton instance — import this everywhere
settings = Settings()
settings.ensure_directories()
