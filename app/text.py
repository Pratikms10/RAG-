"""Shared text normalization helpers used by retrieval and answer grounding."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9][a-z0-9'-]*", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
        "you",
        "your",
    }
)


def tokens(text: str) -> tuple[str, ...]:
    """Return lightly normalized lexical tokens while preserving source order."""
    return tuple(_normalize_token(token.lower()) for token in _TOKEN.findall(text))


def ordered_content_terms(text: str) -> tuple[str, ...]:
    """Return meaningful query terms in first-seen order, without duplicates."""
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens(text):
        if token in _STOPWORDS or len(token) <= 1 or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return tuple(result)


def content_terms(text: str) -> set[str]:
    return set(ordered_content_terms(text))


def _normalize_token(token: str) -> str:
    """Handle common inflectional variants without pretending to be a full lemmatizer."""
    if token in _STOPWORDS:
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("uses"):
        return token[:-1]
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token
