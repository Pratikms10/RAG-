"""Embedding providers with an offline-safe default and explicit API error handling."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from app.config import Settings
from app.errors import EmbeddingProviderError

_TOKEN = re.compile(r"[a-z0-9][a-z0-9'-]*", re.IGNORECASE)


class EmbeddingProvider(Protocol):
    backend_name: str
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class LocalHashEmbeddingProvider:
    """Deterministic hashed word + bigram vectors with no network dependency.

    It is a deliberately modest fallback: it favors transparent lexical retrieval
    over claiming that offline embeddings have semantic understanding.
    """

    backend_name = "local-hash-v1"
    dimensions = 512

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._embed(text) for text in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text).astype(np.float32)

    def _embed(self, text: str) -> np.ndarray:
        tokens = _TOKEN.findall(text.lower())
        vector = np.zeros(self.dimensions, dtype=np.float32)
        features: list[tuple[str, float]] = [(token, 1.0) for token in tokens]
        features.extend((f"{left}::{right}", 0.7) for left, right in zip(tokens, tokens[1:]))
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            vector[index] += weight
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


class OpenAIEmbeddingProvider:
    """Small adapter around a real embedding API call, isolated for easy testing."""

    backend_name = "openai"

    def __init__(self, model: str) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EmbeddingProviderError("OPENAI_API_KEY is required for the OpenAI embedding backend.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingProviderError("The openai package is not installed.") from exc
        self._client = OpenAI(api_key=api_key, timeout=20.0, max_retries=1)
        self._model = model
        self.dimensions = 0  # filled after the first successful response

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._request(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return self._request([text])[0]

    def _request(self, inputs: list[str]) -> np.ndarray:
        try:
            response = self._client.embeddings.create(model=self._model, input=inputs)
            vectors = np.asarray([item.embedding for item in response.data], dtype=np.float32)
        except Exception as exc:  # SDK/network/provider failures must not become silent hallucinations.
            raise EmbeddingProviderError(
                f"OpenAI embedding call failed ({type(exc).__name__}). Check the key, network, and model."
            ) from exc
        if len(vectors) != len(inputs) or vectors.ndim != 2:
            raise EmbeddingProviderError("OpenAI returned an unexpected embedding response shape.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
        self.dimensions = int(vectors.shape[1])
        return vectors


@dataclass(frozen=True, slots=True)
class EmbeddingOutcome:
    vectors: np.ndarray
    backend_name: str
    warning: str | None = None


def provider_for(settings: Settings, backend_name: str | None = None) -> EmbeddingProvider:
    """Resolve a provider, optionally pinning it to a persisted index backend."""
    requested = backend_name or settings.embedding_backend
    if requested in {"local", "local-hash-v1"}:
        return LocalHashEmbeddingProvider()
    if requested in {"openai"}:
        return OpenAIEmbeddingProvider(settings.openai_embedding_model)
    if requested == "auto":
        return OpenAIEmbeddingProvider(settings.openai_embedding_model) if os.getenv("OPENAI_API_KEY") else LocalHashEmbeddingProvider()
    raise EmbeddingProviderError(f"Unknown embedding backend: {requested}")
