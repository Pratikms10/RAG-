"""FastAPI entry points for document ingestion, retrieval, and provenance-rich answers."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status

from app.config import Settings
from app.domain import RetrievedChunk
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
    ClearResponse,
    DocumentResponse,
    HealthResponse,
    IngestResponse,
    QuestionRequest,
    SourceChunk,
)
from app.service import RAGService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    service = RAGService(settings)
    app = FastAPI(
        title="Grounded Local RAG API",
        version="0.1.0",
        description="PDF/TXT retrieval with local vectors, citations, and an evidence gate.",
    )
    app.state.rag_service = service

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
            )
            for document in _service(request).documents()
        ]

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
        )

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
        excerpt=excerpt,
    )
