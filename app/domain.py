"""Domain objects kept independent of FastAPI and storage details."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    filename: str
    page_number: int | None
    ordinal: int
    text: str
    word_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    filename: str
    format: str
    sha256: str
    page_count: int
    chunk_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(**data)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    grounded: bool
    insufficient_evidence: bool
    answer_backend: str
    warnings: tuple[str, ...]
    sources: tuple[RetrievedChunk, ...]
