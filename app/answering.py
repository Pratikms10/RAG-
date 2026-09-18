"""Grounded answer generation: API-backed when configured, extractive when offline."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Sequence

from app.config import Settings
from app.domain import RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9][a-z0-9'-]*", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "the", "to", "was", "what", "when",
    "where", "which", "who", "why", "with", "would", "you", "your",
}

GROUNDING_PROMPT = """You are a retrieval-augmented assistant. Answer only from the source chunks below.
If the chunks do not establish the answer, reply exactly: NOT_FOUND.
Do not use outside knowledge. Keep the answer to at most three sentences and place the supplied
[chunk-id] after each factual claim.

Question: {question}

Source chunks:
{context}
"""


@dataclass(frozen=True, slots=True)
class Generation:
    answer: str | None
    backend_name: str
    warning: str | None = None
    source_chunk_ids: tuple[str, ...] = ()


def content_terms(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(text) if token.lower() not in _STOPWORDS and len(token) > 1}


def evidence_is_sufficient(question: str, hits: Sequence[RetrievedChunk], minimum_score: float) -> bool:
    """Require both vector score and meaningful query-term coverage before answering."""
    if not hits or hits[0].score < minimum_score:
        return False
    question_terms = content_terms(question)
    if not question_terms:
        return False
    retrieved_terms = set().union(*(content_terms(hit.chunk.text) for hit in hits[:2]))
    covered = len(question_terms & retrieved_terms)
    required = 1 if len(question_terms) == 1 else 2
    return covered >= required and (covered / len(question_terms)) >= 0.5


def generate_extractive(question: str, hits: Sequence[RetrievedChunk]) -> Generation:
    """Choose the highest lexical-overlap sentences instead of fabricating a free-form response."""
    terms = content_terms(question)
    candidates: list[tuple[float, str, str]] = []
    for hit in hits:
        for sentence in _SENTENCE.split(hit.chunk.text):
            sentence = sentence.strip()
            overlap = len(terms & content_terms(sentence))
            if overlap:
                candidates.append((overlap + hit.score, sentence, hit.chunk.id))
    if not candidates:
        return Generation(None, "extractive")
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen: list[str] = []
    chosen_texts: set[str] = set()
    chosen_ids: list[str] = []
    covered_terms: set[str] = set()
    for _, sentence, chunk_id in candidates:
        new_terms = terms & content_terms(sentence) - covered_terms
        if sentence not in chosen_texts and new_terms:
            chosen.append(f"{sentence} [{chunk_id}]")
            chosen_texts.add(sentence)
            chosen_ids.append(chunk_id)
            covered_terms.update(new_terms)
        if len(chosen) == 2:
            break
    return Generation(
        "Based on the uploaded documents: " + " ".join(chosen),
        "extractive",
        source_chunk_ids=tuple(dict.fromkeys(chosen_ids)),
    )


def generate_openai(question: str, hits: Sequence[RetrievedChunk], settings: Settings) -> Generation:
    """Make one guarded LLM call; any failure deliberately falls back at the service layer."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return Generation(None, "openai", "OPENAI_API_KEY is not configured.")
    try:
        from openai import OpenAI

        context = "\n\n".join(f"[{hit.chunk.id}] {hit.chunk.text}" for hit in hits)
        response = OpenAI(api_key=api_key, timeout=25.0, max_retries=1).chat.completions.create(
            model=settings.openai_chat_model,
            messages=[{"role": "user", "content": GROUNDING_PROMPT.format(question=question, context=context)}],
            temperature=0,
        )
        answer = (response.choices[0].message.content or "").strip()
    except Exception as exc:  # Explicitly recover rather than returning a model-shaped guess.
        return Generation(None, "openai", f"OpenAI answer call failed ({type(exc).__name__}); used extractive fallback.")
    if not answer or answer.upper() == "NOT_FOUND":
        return Generation(None, "openai")
    # The model receives only these chunks, so they are the full provenance set
    # for the generated answer even if it omits an inline marker.
    return Generation(answer, "openai", source_chunk_ids=tuple(hit.chunk.id for hit in hits))


def generate_answer(question: str, hits: Sequence[RetrievedChunk], settings: Settings) -> Generation:
    """Use OpenAI only when explicitly available; local behavior remains runnable offline."""
    wants_openai = settings.answer_backend == "openai" or (
        settings.answer_backend == "auto" and bool(os.getenv("OPENAI_API_KEY"))
    )
    if wants_openai:
        model_result = generate_openai(question, hits, settings)
        if model_result.answer is not None:
            return model_result
        fallback = generate_extractive(question, hits)
        return Generation(
            fallback.answer,
            fallback.backend_name,
            model_result.warning,
            fallback.source_chunk_ids,
        )
    return generate_extractive(question, hits)
