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
    chunking_strategy: str = "sentence-window-v1"
    chunk_size_words: int = 180
    chunk_overlap_words: int = 40

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(**data)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0
    matched_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """A single sentence that supplies directly inspectable support for a query."""

    chunk_id: str
    text: str
    matched_terms: tuple[str, ...]
    coverage: float


@dataclass(frozen=True, slots=True)
class RetrievalAssessment:
    query_terms: tuple[str, ...]
    matched_terms: tuple[str, ...]
    term_coverage: float
    required_term_matches: int
    top_score: float | None
    minimum_score: float
    requires_single_span: bool
    best_span_coverage: float
    evidence_spans: tuple[EvidenceSpan, ...]
    sufficient: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    hits: tuple[RetrievedChunk, ...]
    assessment: RetrievalAssessment


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    grounded: bool
    insufficient_evidence: bool
    answer_backend: str
    warnings: tuple[str, ...]
    sources: tuple[RetrievedChunk, ...]
    retrieval: RetrievalTrace
