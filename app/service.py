"""Application service that coordinates parsing, embeddings, storage, and grounding gates."""

from __future__ import annotations

from dataclasses import dataclass

from app.answering import assess_evidence, generate_answer
from app.config import Settings
from app.domain import AnswerResult, Chunk, Document, RetrievalTrace
from app.embeddings import EmbeddingOutcome, LocalHashEmbeddingProvider, provider_for
from app.errors import EmbeddingProviderError
from app.ingestion import build_document_and_chunks
from app.retrieval import rank_chunks
from app.storage import LocalVectorStore


@dataclass(frozen=True, slots=True)
class IngestResult:
    document: Document
    embedding_backend: str
    warning: str | None = None


class RAGService:
    def __init__(self, settings: Settings) -> None:
        settings.validate()
        self.settings = settings
        self.store = LocalVectorStore(settings.data_dir)

    def ingest(
        self,
        filename: str,
        payload: bytes,
        chunk_size_words: int | None = None,
        chunk_overlap_words: int | None = None,
    ) -> IngestResult:
        size = chunk_size_words or self.settings.chunk_size_words
        overlap = chunk_overlap_words if chunk_overlap_words is not None else self.settings.chunk_overlap_words
        document, chunks = build_document_and_chunks(filename, payload, size, overlap)
        outcome = self._embed_for_ingestion([chunk.text for chunk in chunks])
        self.store.add(document, chunks, outcome.vectors, outcome.backend_name)
        return IngestResult(document, outcome.backend_name, outcome.warning)

    def answer(self, question: str, top_k: int) -> AnswerResult:
        retrieval = self.retrieve(question, top_k)
        if not retrieval.hits:
            return AnswerResult(
                answer="I do not have any indexed documents yet. Upload a PDF or TXT file first.",
                grounded=False,
                insufficient_evidence=True,
                answer_backend="none",
                warnings=(),
                sources=(),
                retrieval=retrieval,
            )
        if not retrieval.assessment.sufficient:
            return AnswerResult(
                answer="I could not find enough evidence in the uploaded documents to answer that question.",
                grounded=False,
                insufficient_evidence=True,
                answer_backend="none",
                warnings=(),
                sources=retrieval.hits,
                retrieval=retrieval,
            )
        generated = generate_answer(
            question,
            retrieval.hits,
            self.settings,
            retrieval.assessment.evidence_spans,
        )
        answer_sources = tuple(
            hit
            for hit in retrieval.hits
            if hit.chunk.id in generated.source_chunk_ids
        )
        if generated.answer is None:
            return AnswerResult(
                answer="I could not find enough evidence in the uploaded documents to answer that question.",
                grounded=False,
                insufficient_evidence=True,
                answer_backend=generated.backend_name,
                warnings=tuple(filter(None, [generated.warning])),
                sources=answer_sources,
                retrieval=retrieval,
            )
        return AnswerResult(
            answer=generated.answer,
            grounded=True,
            insufficient_evidence=False,
            answer_backend=generated.backend_name,
            warnings=tuple(filter(None, [generated.warning])),
            sources=answer_sources,
            retrieval=retrieval,
        )

    def retrieve(self, question: str, top_k: int) -> RetrievalTrace:
        """Return the ranking and decision trace without generating an answer."""
        snapshot = self.store.snapshot()
        if not snapshot.chunks:
            return RetrievalTrace((), assess_evidence(question, (), self.settings.min_relevance_score))
        provider = provider_for(self.settings, str(snapshot.metadata["embedding_backend"]))
        query_vector = provider.embed_query(question)
        hits = tuple(
            rank_chunks(
                snapshot.chunks,
                snapshot.vectors,
                query_vector,
                question,
                top_k,
                self.settings.dense_weight,
                self.settings.lexical_weight,
            )
        )
        return RetrievalTrace(hits, assess_evidence(question, hits, self.settings.min_relevance_score))

    def documents(self) -> tuple[Document, ...]:
        return self.store.snapshot().documents

    def chunks_for_document(self, document_id: str) -> tuple[Chunk, ...]:
        return tuple(chunk for chunk in self.store.snapshot().chunks if chunk.document_id == document_id)

    def chunk(self, document_id: str, chunk_id: str) -> Chunk | None:
        return next(
            (
                chunk
                for chunk in self.store.snapshot().chunks
                if chunk.document_id == document_id and chunk.id == chunk_id
            ),
            None,
        )

    def clear(self) -> None:
        self.store.clear()

    def _embed_for_ingestion(self, texts: list[str]) -> EmbeddingOutcome:
        provider = provider_for(self.settings)
        try:
            return EmbeddingOutcome(provider.embed_documents(texts), provider.backend_name)
        except EmbeddingProviderError:
            # In auto mode an unavailable hosted provider falls back *before* writing any vectors.
            # Explicit openai mode surfaces the failure so an operator does not unknowingly downgrade.
            if self.settings.embedding_backend != "auto" or provider.backend_name != "openai":
                raise
            fallback = LocalHashEmbeddingProvider()
            return EmbeddingOutcome(
                fallback.embed_documents(texts),
                fallback.backend_name,
                "OpenAI embeddings failed; indexed with the local lexical fallback instead.",
            )
