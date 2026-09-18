"""FastAPI entry points for document ingestion, retrieval, and provenance-rich answers."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings
from app.domain import Chunk, RetrievalTrace, RetrievedChunk
from app.errors import (
    DocumentParseError,
    DuplicateDocumentError,
    EmbeddingProviderError,
    EmptyDocumentError,
    IndexCompatibilityError,
    UnsupportedDocumentError,
)
from app.schemas import (
    AnswerResponse,
    ChunkResponse,
    ClearResponse,
    DocumentResponse,
    EvidenceSpanResponse,
    HealthResponse,
    IngestResponse,
    QuestionRequest,
    RetrievalTraceResponse,
    SourceChunk,
)
from app.service import RAGService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    service = RAGService(settings)
    app = FastAPI(
        title="Grounded Local RAG API",
        version="0.2.0",
        description="PDF/TXT retrieval with hybrid ranking, citations, and an inspectable evidence gate.",
    )
    app.state.rag_service = service
    static_directory = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_directory), name="static")

    @app.get("/", include_in_schema=False)
    def evidence_lab() -> FileResponse:
        return FileResponse(static_directory / "index.html")

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        snapshot = _service(request).store.snapshot()
        return HealthResponse(
            status="ok",
            documents=len(snapshot.documents),
            chunks=len(snapshot.chunks),
            embedding_backend=snapshot.metadata.get("embedding_backend"),
        )

    @app.post("/documents", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
    async def upload_document(
        request: Request,
        file: Annotated[UploadFile, File(description="A UTF-8 TXT or text-based PDF")],
        chunk_size_words: Annotated[int, Form(ge=80, le=600)] = 180,
        chunk_overlap_words: Annotated[int, Form(ge=0, le=250)] = 40,
    ) -> IngestResponse:
        if chunk_overlap_words >= chunk_size_words:
            raise HTTPException(status_code=422, detail="chunk_overlap_words must be smaller than chunk_size_words")
        payload = await file.read()
        if len(payload) > _service(request).settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Upload exceeds the 12 MB limit.")
        try:
            result = _service(request).ingest(
                file.filename or "upload.txt", payload, chunk_size_words, chunk_overlap_words
            )
        except (UnsupportedDocumentError, DocumentParseError, EmptyDocumentError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except DuplicateDocumentError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except IndexCompatibilityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except EmbeddingProviderError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        document = result.document
        return IngestResponse(
            document_id=document.id,
            filename=document.filename,
            format=document.format,
            pages_with_text=document.page_count,
            chunks_added=document.chunk_count,
            chunking_strategy=document.chunking_strategy,
            chunk_size_words=document.chunk_size_words,
            chunk_overlap_words=document.chunk_overlap_words,
            embedding_backend=result.embedding_backend,
            warning=result.warning,
        )

    @app.get("/documents", response_model=list[DocumentResponse])
    def list_documents(request: Request) -> list[DocumentResponse]:
        return [
            DocumentResponse(
                document_id=document.id,
                filename=document.filename,
                format=document.format,
                pages_with_text=document.page_count,
                chunk_count=document.chunk_count,
                chunking_strategy=document.chunking_strategy,
                chunk_size_words=document.chunk_size_words,
                chunk_overlap_words=document.chunk_overlap_words,
            )
            for document in _service(request).documents()
        ]

    @app.get("/documents/{document_id}/chunks", response_model=list[ChunkResponse])
    def list_document_chunks(request: Request, document_id: str) -> list[ChunkResponse]:
        chunks = _service(request).chunks_for_document(document_id)
        if not chunks:
            raise HTTPException(status_code=404, detail="Document was not found or has no chunks.")
        return [_chunk_response(chunk) for chunk in chunks]

    @app.get("/documents/{document_id}/chunks/{chunk_id}", response_model=ChunkResponse)
    def get_document_chunk(request: Request, document_id: str, chunk_id: str) -> ChunkResponse:
        chunk = _service(request).chunk(document_id, chunk_id)
        if chunk is None:
            raise HTTPException(status_code=404, detail="Chunk was not found in this document.")
        return _chunk_response(chunk)

    @app.delete("/documents", response_model=ClearResponse)
    def clear_documents(request: Request) -> ClearResponse:
        _service(request).clear()
        return ClearResponse(status="cleared")

    @app.post("/questions", response_model=AnswerResponse)
    def ask_question(request: Request, body: QuestionRequest) -> AnswerResponse:
        try:
            result = _service(request).answer(body.question, body.top_k)
        except (EmbeddingProviderError, IndexCompatibilityError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return AnswerResponse(
            answer=result.answer,
            grounded=result.grounded,
            insufficient_evidence=result.insufficient_evidence,
            answer_backend=result.answer_backend,
            warnings=list(result.warnings),
            source_chunks=[_source_chunk(hit) for hit in result.sources],
            retrieval_trace=_trace_response(result.retrieval),
        )

    @app.post("/retrievals", response_model=RetrievalTraceResponse)
    def preview_retrieval(request: Request, body: QuestionRequest) -> RetrievalTraceResponse:
        """Inspect rankings and the evidence gate without invoking answer generation."""
        try:
            trace = _service(request).retrieve(body.question, body.top_k)
        except (EmbeddingProviderError, IndexCompatibilityError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _trace_response(trace)

    return app


def _service(request: Request) -> RAGService:
    return request.app.state.rag_service


def _source_chunk(hit: RetrievedChunk) -> SourceChunk:
    excerpt = hit.chunk.text if len(hit.chunk.text) <= 320 else hit.chunk.text[:317].rstrip() + "..."
    return SourceChunk(
        chunk_id=hit.chunk.id,
        document_id=hit.chunk.document_id,
        filename=hit.chunk.filename,
        page_number=hit.chunk.page_number,
        score=round(hit.score, 4),
        dense_score=round(hit.dense_score, 4),
        lexical_score=round(hit.lexical_score, 4),
        matched_terms=list(hit.matched_terms),
        excerpt=excerpt,
    )


def _trace_response(trace: RetrievalTrace) -> RetrievalTraceResponse:
    assessment = trace.assessment
    return RetrievalTraceResponse(
        query_terms=list(assessment.query_terms),
        matched_terms=list(assessment.matched_terms),
        term_coverage=round(assessment.term_coverage, 4),
        required_term_matches=assessment.required_term_matches,
        top_score=round(assessment.top_score, 4) if assessment.top_score is not None else None,
        minimum_score=assessment.minimum_score,
        requires_single_span=assessment.requires_single_span,
        best_span_coverage=round(assessment.best_span_coverage, 4),
        evidence_spans=[
            EvidenceSpanResponse(
                chunk_id=span.chunk_id,
                text=span.text,
                matched_terms=list(span.matched_terms),
                coverage=round(span.coverage, 4),
            )
            for span in assessment.evidence_spans
        ],
        sufficient=assessment.sufficient,
        reasons=list(assessment.reasons),
        candidates=[_source_chunk(hit) for hit in trace.hits],
    )


def _chunk_response(chunk: Chunk) -> ChunkResponse:
    return ChunkResponse(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        filename=chunk.filename,
        page_number=chunk.page_number,
        ordinal=chunk.ordinal,
        word_count=chunk.word_count,
        text=chunk.text,
    )
