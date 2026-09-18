"""Grounded answer generation: API-backed when configured, extractive when offline."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Sequence

from app.config import Settings
from app.domain import EvidenceSpan, RetrievalAssessment, RetrievedChunk
from app.text import content_terms, ordered_content_terms

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
SINGLE_SPAN_MIN_COVERAGE = 0.60

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


def assess_evidence(
    question: str, hits: Sequence[RetrievedChunk], minimum_score: float
) -> RetrievalAssessment:
    """Explain the grounding decision in a form both people and the UI can inspect."""
    query_terms = ordered_content_terms(question)
    if not hits:
        return RetrievalAssessment(
            query_terms=query_terms,
            matched_terms=(),
            term_coverage=0.0,
            required_term_matches=1 if len(query_terms) == 1 else 2,
            top_score=None,
            minimum_score=minimum_score,
            requires_single_span=_requires_single_span(question),
            best_span_coverage=0.0,
            evidence_spans=(),
            sufficient=False,
            reasons=("No indexed chunks are available yet.",),
        )
    retrieved_terms = set().union(*(content_terms(hit.chunk.text) for hit in hits[:2]))
    matched_terms = tuple(term for term in query_terms if term in retrieved_terms)
    coverage = len(matched_terms) / len(query_terms) if query_terms else 0.0
    required = 1 if len(query_terms) == 1 else 2
    top_score = hits[0].score
    spans = _rank_evidence_spans(query_terms, hits)
    best_span_coverage = spans[0].coverage if spans else 0.0
    requires_single_span = _requires_single_span(question)
    reasons: list[str] = []
    if not query_terms:
        reasons.append("The question had no meaningful terms after normalization.")
    if top_score < minimum_score:
        reasons.append(
            f"Top hybrid retrieval score {top_score:.2f} is below the {minimum_score:.2f} threshold."
        )
    if len(matched_terms) < required:
        reasons.append(f"Only {len(matched_terms)} meaningful query terms occur in the top evidence.")
    if coverage < 0.5:
        reasons.append(f"Term coverage is {coverage:.0%}, below the 50% grounding rule.")
    if requires_single_span and (
        best_span_coverage < SINGLE_SPAN_MIN_COVERAGE
        or (spans and len(spans[0].matched_terms) < required)
        or not spans
    ):
        reasons.append(
            "No single evidence sentence covers at least 60% of this one-part question; "
            "the system will not combine separate facts into a new claim."
        )
    if not reasons:
        reasons.append("Score and term-coverage checks passed; answer generation may use this evidence.")
    return RetrievalAssessment(
        query_terms=query_terms,
        matched_terms=matched_terms,
        term_coverage=coverage,
        required_term_matches=required,
        top_score=top_score,
        minimum_score=minimum_score,
        requires_single_span=requires_single_span,
        best_span_coverage=best_span_coverage,
        evidence_spans=spans,
        sufficient=not any(
            (
                not query_terms,
                top_score < minimum_score,
                len(matched_terms) < required,
                coverage < 0.5,
                requires_single_span
                and (
                    best_span_coverage < SINGLE_SPAN_MIN_COVERAGE
                    or not spans
                    or len(spans[0].matched_terms) < required
                ),
            )
        ),
        reasons=tuple(reasons),
    )


def evidence_is_sufficient(question: str, hits: Sequence[RetrievedChunk], minimum_score: float) -> bool:
    """Compatibility helper for callers that only need the boolean decision."""
    return assess_evidence(question, hits, minimum_score).sufficient


def _requires_single_span(question: str) -> bool:
    """A one-part question needs one sentence of evidence, not a stitched answer.

    Explicit conjunctions are allowed to draw on more than one sentence because
    they ask for multiple facts (for example, chemistry *and* inspection cadence).
    """
    return " and " not in f" {question.lower()} "


def _rank_evidence_spans(
    query_terms: tuple[str, ...], hits: Sequence[RetrievedChunk]
) -> tuple[EvidenceSpan, ...]:
    if not query_terms:
        return ()
    candidates: list[tuple[float, float, EvidenceSpan]] = []
    for hit in hits:
        for sentence in _SENTENCE.split(hit.chunk.text):
            sentence = sentence.strip()
            if not sentence:
                continue
            matched = tuple(term for term in query_terms if term in content_terms(sentence))
            if not matched:
                continue
            coverage = len(matched) / len(query_terms)
            span = EvidenceSpan(
                chunk_id=hit.chunk.id,
                text=sentence,
                matched_terms=matched,
                coverage=coverage,
            )
            candidates.append((coverage, hit.score, span))
    candidates.sort(key=lambda item: (item[0], item[1], len(item[2].matched_terms)), reverse=True)
    return tuple(item[2] for item in candidates[:6])


def generate_extractive(
    question: str,
    hits: Sequence[RetrievedChunk],
    evidence_spans: Sequence[EvidenceSpan] = (),
) -> Generation:
    """Choose the highest lexical-overlap sentences instead of fabricating a free-form response."""
    if evidence_spans:
        return _generate_from_evidence_spans(question, evidence_spans)
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


def _generate_from_evidence_spans(
    question: str, evidence_spans: Sequence[EvidenceSpan]
) -> Generation:
    """Turn verified support into a concise answer without adding another inference step."""
    single_span_required = _requires_single_span(question)
    ranked_spans = sorted(
        evidence_spans,
        key=lambda span: (
            span.coverage,
            _is_yes_no_question(question) and _contains_negation(span.text),
            len(span.matched_terms),
        ),
        reverse=True,
    )
    selected: list[EvidenceSpan] = []
    covered: set[str] = set()
    for span in ranked_spans:
        new_terms = set(span.matched_terms) - covered
        if not new_terms and selected:
            continue
        selected.append(span)
        covered.update(span.matched_terms)
        if single_span_required or len(selected) == 2:
            break
    if not selected:
        return Generation(None, "extractive")
    answer = " ".join(f"{span.text} [{span.chunk_id}]" for span in selected)
    return Generation(
        "Based on the uploaded documents: " + answer,
        "extractive",
        source_chunk_ids=tuple(dict.fromkeys(span.chunk_id for span in selected)),
    )


def _is_yes_no_question(question: str) -> bool:
    words = question.lower().strip().split(maxsplit=1)
    return bool(words) and words[0] in {"does", "do", "is", "are", "can", "will", "has", "have"}


def _contains_negation(text: str) -> bool:
    lowered = f" {text.lower()} "
    return any(marker in lowered for marker in (" not ", " no ", " never ", " cannot ", " can't "))


def generate_openai(
    question: str,
    hits: Sequence[RetrievedChunk],
    settings: Settings,
    evidence_spans: Sequence[EvidenceSpan] = (),
) -> Generation:
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
    # Keep the API's provenance conservative: expose the direct local evidence
    # spans that passed the gate, rather than treating every retrieved candidate
    # as if it supported the final wording.
    source_ids = tuple(dict.fromkeys(span.chunk_id for span in evidence_spans))
    return Generation(answer, "openai", source_chunk_ids=source_ids)


def generate_answer(
    question: str,
    hits: Sequence[RetrievedChunk],
    settings: Settings,
    evidence_spans: Sequence[EvidenceSpan] = (),
) -> Generation:
    """Use OpenAI only when explicitly available; local behavior remains runnable offline."""
    wants_openai = settings.answer_backend == "openai" or (
        settings.answer_backend == "auto" and bool(os.getenv("OPENAI_API_KEY"))
    )
    if wants_openai:
        model_result = generate_openai(question, hits, settings, evidence_spans)
        if model_result.answer is not None:
            return model_result
        fallback = generate_extractive(question, hits, evidence_spans)
        return Generation(
            fallback.answer,
            fallback.backend_name,
            model_result.warning,
            fallback.source_chunk_ids,
        )
    return generate_extractive(question, hits, evidence_spans)
