from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    app_name: str = "ResumeIQ"
    app_env: str = os.getenv("APP_ENV", "development")
    api_provider: str = os.getenv("API_PROVIDER", "auto")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "auto")
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    openai_summary_model: str = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_KEY", "")
    huggingface_embedding_model: str = os.getenv(
        "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    sentence_transformer_model: str = os.getenv(
        "SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    resume_parser: str = os.getenv("RESUME_PARSER", "auto")
    vector_backend: str = os.getenv("VECTOR_BACKEND", "faiss")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "")
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "8"))

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


settings = Settings()
