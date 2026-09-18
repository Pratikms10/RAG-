"""A tiny persisted vector store: vectors.npz plus inspectable JSON provenance."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.domain import Chunk, Document, RetrievedChunk
from app.errors import DuplicateDocumentError, IndexCompatibilityError


@dataclass(frozen=True, slots=True)
class IndexSnapshot:
    chunks: tuple[Chunk, ...]
    documents: tuple[Document, ...]
    vectors: np.ndarray
    metadata: dict[str, Any]


class LocalVectorStore:
    """Append-only-in-practice local vector storage for a single-user demo service."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._vectors_path = directory / "vectors.npy"
        self._chunks_path = directory / "chunks.json"
        self._documents_path = directory / "documents.json"
        self._metadata_path = directory / "metadata.json"
        self._lock = threading.RLock()

    def snapshot(self) -> IndexSnapshot:
        with self._lock:
            return self._load_locked()

    def add(self, document: Document, chunks: list[Chunk], vectors: np.ndarray, backend_name: str) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("Each chunk must have exactly one 2D embedding vector.")
        with self._lock:
            current = self._load_locked()
            if any(item.id == document.id for item in current.documents):
                raise DuplicateDocumentError(
                    f"This exact file is already indexed as document {document.id}."
                )
            if current.vectors.size:
                old_backend = current.metadata["embedding_backend"]
                old_dimensions = int(current.metadata["dimensions"])
                if backend_name != old_backend or vectors.shape[1] != old_dimensions:
                    raise IndexCompatibilityError(
                        "The current index uses a different embedding backend. "
                        "Clear it with DELETE /documents before changing backends."
                    )
                merged_vectors = np.vstack([current.vectors, vectors]).astype(np.float32)
            else:
                merged_vectors = vectors.astype(np.float32)
            merged_chunks = [*current.chunks, *chunks]
            merged_documents = [*current.documents, document]
            metadata = {
                "index_schema_version": 2,
                "embedding_backend": backend_name,
                "dimensions": int(merged_vectors.shape[1]),
                "vector_count": int(merged_vectors.shape[0]),
                "retrieval_strategy": "hybrid-dense-bm25-v1",
            }
            self._write_json(self._chunks_path, [item.to_dict() for item in merged_chunks])
            self._write_json(self._documents_path, [item.to_dict() for item in merged_documents])
            self._write_json(self._metadata_path, metadata)
            self._write_vectors(merged_vectors)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        snapshot = self.snapshot()
        if not snapshot.chunks:
            return []
        if query_vector.ndim != 1 or query_vector.shape[0] != snapshot.vectors.shape[1]:
            raise IndexCompatibilityError("Query embedding dimensions do not match the persisted vector index.")
        scores = snapshot.vectors @ query_vector
        limit = min(top_k, len(snapshot.chunks))
        indices = np.argsort(-scores)[:limit]
        return [
            RetrievedChunk(chunk=snapshot.chunks[int(index)], score=float(scores[int(index)]))
            for index in indices
        ]

    def clear(self) -> None:
        with self._lock:
            for path in (
                self._vectors_path,
                self._chunks_path,
                self._documents_path,
                self._metadata_path,
            ):
                path.unlink(missing_ok=True)

    def _load_locked(self) -> IndexSnapshot:
        if not self._vectors_path.exists():
            return IndexSnapshot((), (), np.empty((0, 0), dtype=np.float32), {})
        vectors = np.load(self._vectors_path, allow_pickle=False).astype(np.float32)
        chunks_data = self._read_json(self._chunks_path)
        documents_data = self._read_json(self._documents_path)
        metadata = self._read_json(self._metadata_path)
        chunks = tuple(Chunk.from_dict(item) for item in chunks_data)
        documents = tuple(Document.from_dict(item) for item in documents_data)
        if vectors.shape[0] != len(chunks) or metadata.get("vector_count") != len(chunks):
            raise IndexCompatibilityError("The local index files are inconsistent. Clear and re-ingest documents.")
        return IndexSnapshot(chunks, documents, vectors, metadata)

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexCompatibilityError(f"Could not read local index file {path.name}.") from exc

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def _write_vectors(self, vectors: np.ndarray) -> None:
        temporary = self._vectors_path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, vectors.astype(np.float32), allow_pickle=False)
        temporary.replace(self._vectors_path)
