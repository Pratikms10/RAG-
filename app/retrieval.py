"""Transparent hybrid retrieval: embedding similarity plus BM25-style lexical evidence."""

from __future__ import annotations

from collections import Counter
from math import log
from typing import Sequence

import numpy as np

from app.domain import Chunk, RetrievedChunk
from app.errors import IndexCompatibilityError
from app.text import content_terms, ordered_content_terms, tokens

BM25_K1 = 1.2
BM25_B = 0.75


def rank_chunks(
    chunks: Sequence[Chunk],
    vectors: np.ndarray,
    query_vector: np.ndarray,
    question: str,
    top_k: int,
    dense_weight: float,
    lexical_weight: float,
) -> list[RetrievedChunk]:
    """Rank chunks with a readable hybrid score.

    Dense similarity helps paraphrases when a hosted embedding model is configured;
    lexical BM25 rewards exact factual terms. Returning both components makes a
    selection debuggable instead of asking a reviewer to trust a single number.
    """
    if not chunks:
        return []
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise IndexCompatibilityError("The local vector index is inconsistent with its chunks.")
    if query_vector.ndim != 1 or query_vector.shape[0] != vectors.shape[1]:
        raise IndexCompatibilityError("Query embedding dimensions do not match the persisted vector index.")

    query_terms = ordered_content_terms(question)
    dense_scores = np.clip(vectors @ query_vector, 0.0, None)
    lexical_scores = _bm25_scores(chunks, query_terms)
    maximum_lexical = float(lexical_scores.max()) if lexical_scores.size else 0.0
    normalized_lexical = lexical_scores / maximum_lexical if maximum_lexical > 0 else lexical_scores
    hybrid_scores = dense_weight * dense_scores + lexical_weight * normalized_lexical

    limit = min(top_k, len(chunks))
    indices = np.argsort(-hybrid_scores, kind="stable")[:limit]
    return [
        RetrievedChunk(
            chunk=chunks[int(index)],
            score=float(hybrid_scores[int(index)]),
            dense_score=float(dense_scores[int(index)]),
            lexical_score=float(normalized_lexical[int(index)]),
            matched_terms=tuple(
                term for term in query_terms if term in content_terms(chunks[int(index)].text)
            ),
        )
        for index in indices
    ]


def _bm25_scores(chunks: Sequence[Chunk], query_terms: tuple[str, ...]) -> np.ndarray:
    """Compute a compact BM25-style lexical score without another service or package."""
    if not query_terms:
        return np.zeros(len(chunks), dtype=np.float32)
    document_tokens = [tokens(chunk.text) for chunk in chunks]
    frequencies = [Counter(token for token in document if token in query_terms) for document in document_tokens]
    lengths = np.asarray([max(1, len(document)) for document in document_tokens], dtype=np.float32)
    average_length = float(lengths.mean()) if len(lengths) else 1.0
    corpus_size = len(chunks)
    scores = np.zeros(corpus_size, dtype=np.float32)
    for term in query_terms:
        document_frequency = sum(1 for counts in frequencies if term in counts)
        if document_frequency == 0:
            continue
        inverse_frequency = log(1 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5))
        for index, counts in enumerate(frequencies):
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + BM25_K1 * (1 - BM25_B + BM25_B * lengths[index] / average_length)
            scores[index] += inverse_frequency * frequency * (BM25_K1 + 1) / denominator
    return scores
