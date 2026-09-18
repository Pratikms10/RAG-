"""Pydantic request and response contracts exposed by FastAPI."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2_000, examples=["When is the safety drill?"])
    top_k: int = Field(default=4, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def no_blank_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None
    score: float
    dense_score: float
    lexical_score: float
    matched_terms: list[str]
    excerpt: str


class EvidenceSpanResponse(BaseModel):
    chunk_id: str
    text: str
    matched_terms: list[str]
    coverage: float


class RetrievalTraceResponse(BaseModel):
    query_terms: list[str]
    matched_terms: list[str]
    term_coverage: float
    required_term_matches: int
    top_score: float | None
    minimum_score: float
    requires_single_span: bool
    best_span_coverage: float
    evidence_spans: list[EvidenceSpanResponse]
    sufficient: bool
    reasons: list[str]
    candidates: list[SourceChunk]


class AnswerResponse(BaseModel):
    answer: str
    grounded: bool
    insufficient_evidence: bool
    answer_backend: str
    warnings: list[str]
    source_chunks: list[SourceChunk]
    retrieval_trace: RetrievalTraceResponse


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    format: str
    pages_with_text: int
    chunks_added: int
    chunking_strategy: str
    chunk_size_words: int
    chunk_overlap_words: int
    embedding_backend: str
    warning: str | None = None


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    format: str
    pages_with_text: int
    chunk_count: int
    chunking_strategy: str
    chunk_size_words: int
    chunk_overlap_words: int


class ChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None
    ordinal: int
    word_count: int
    text: str


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    embedding_backend: str | None


class ClearResponse(BaseModel):
    status: str
