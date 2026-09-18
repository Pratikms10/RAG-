"""Small, explicit configuration surface for the service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings; environment variables are read once at startup."""

    data_dir: Path = Path("data")
    embedding_backend: str = "auto"  # auto | local | openai
    answer_backend: str = "auto"  # auto | extractive | openai
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4.1-mini"
    chunk_size_words: int = 180
    chunk_overlap_words: int = 40
    min_relevance_score: float = 0.18
    dense_weight: float = 0.70
    lexical_weight: float = 0.30
    max_upload_bytes: int = 12 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.getenv("RAG_DATA_DIR", "data")),
            embedding_backend=os.getenv("RAG_EMBEDDING_BACKEND", "auto").lower(),
            answer_backend=os.getenv("RAG_ANSWER_BACKEND", "auto").lower(),
            openai_embedding_model=os.getenv(
                "RAG_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            openai_chat_model=os.getenv("RAG_OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        )

    def validate(self) -> None:
        if self.embedding_backend not in {"auto", "local", "openai"}:
            raise ValueError("RAG_EMBEDDING_BACKEND must be auto, local, or openai")
        if self.answer_backend not in {"auto", "extractive", "openai"}:
            raise ValueError("RAG_ANSWER_BACKEND must be auto, extractive, or openai")
        if not 80 <= self.chunk_size_words <= 600:
            raise ValueError("chunk_size_words must be between 80 and 600")
        if not 0 <= self.chunk_overlap_words < self.chunk_size_words:
            raise ValueError("chunk_overlap_words must be non-negative and smaller than chunk_size_words")
        if self.dense_weight < 0 or self.lexical_weight < 0:
            raise ValueError("retrieval weights must be non-negative")
        if abs((self.dense_weight + self.lexical_weight) - 1.0) > 0.001:
            raise ValueError("dense_weight and lexical_weight must add up to 1.0")
