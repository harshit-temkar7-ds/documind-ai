"""
DocuMind AI — Centralized Configuration
Loads all settings from environment variables via .env file.
"""
from dotenv import load_dotenv
load_dotenv()
from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    """All application settings — loaded from .env file."""

    # ── LLM ───────────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    llm_model: str = "llama3-8b-8192"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Storage paths ─────────────────────────────────────────────────────────
    vectorstore_path: str = "./data/vectorstore"
    upload_folder: str = "./data/uploads"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_results: int = 5

    # ── API ───────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # ── App Meta ──────────────────────────────────────────────────────────────
    app_name: str = "DocuMind AI"
    app_version: str = "1.0.0"
    app_description: str = "RAG-powered document intelligence system"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def ensure_directories(self):
        """Create required directories if they don't exist."""
        Path(self.vectorstore_path).mkdir(parents=True, exist_ok=True)
        Path(self.upload_folder).mkdir(parents=True, exist_ok=True)


# Singleton instance — import this everywhere
settings = Settings()
settings.ensure_directories()
