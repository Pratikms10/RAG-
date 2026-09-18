"""Document parsing and deliberately simple, inspectable chunking."""

from __future__ import annotations

import hashlib
import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from app.domain import Chunk, Document
from app.errors import DocumentParseError, EmptyDocumentError, UnsupportedDocumentError

SUPPORTED_SUFFIXES = {".pdf", ".txt"}
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
CHUNKING_STRATEGY = "sentence-window-v1"


def parse_document(filename: str, payload: bytes) -> tuple[list[tuple[int | None, str]], str]:
    """Return ordered (page_number, text) units and the detected format."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedDocumentError("Only .pdf and .txt uploads are supported.")
    if not payload:
        raise EmptyDocumentError("The uploaded file is empty.")

    if suffix == ".txt":
        text = _decode_text(payload)
        normalized = _normalize(text)
        if not normalized:
            raise EmptyDocumentError("The text file did not contain readable text.")
        return [(None, normalized)], "txt"

    try:
        reader = PdfReader(BytesIO(payload), strict=False)
        if reader.is_encrypted:
            raise DocumentParseError("Encrypted PDFs are not supported.")
        pages = [(number, _normalize(page.extract_text() or "")) for number, page in enumerate(reader.pages, 1)]
    except DocumentParseError:
        raise
    except Exception as exc:  # pypdf has several parser-specific exception types.
        raise DocumentParseError("The PDF could not be read. It may be malformed or image-only.") from exc

    nonempty_pages = [(page, text) for page, text in pages if text]
    if not nonempty_pages:
        raise EmptyDocumentError(
            "No selectable text was found in this PDF. OCR is not implemented in this version."
        )
    return nonempty_pages, "pdf"


def build_document_and_chunks(
    filename: str,
    payload: bytes,
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> tuple[Document, list[Chunk]]:
    """Create content-addressed provenance records and overlapping word windows."""
    pages, document_format = parse_document(filename, payload)
    digest = hashlib.sha256(payload).hexdigest()
    document_id = digest[:16]
    chunks: list[Chunk] = []
    ordinal = 0
    for page_number, text in pages:
        for chunk_text in chunk_text_words(text, chunk_size_words, chunk_overlap_words):
            ordinal += 1
            chunks.append(
                Chunk(
                    id=f"{document_id}-p{page_number or 0:03d}-c{ordinal:03d}",
                    document_id=document_id,
                    filename=filename,
                    page_number=page_number,
                    ordinal=ordinal,
                    text=chunk_text,
                    word_count=len(chunk_text.split()),
                )
            )
    if not chunks:
        raise EmptyDocumentError("No chunks could be made from this document.")
    return (
        Document(
            id=document_id,
            filename=filename,
            format=document_format,
            sha256=digest,
            page_count=len(pages),
            chunk_count=len(chunks),
            chunking_strategy=CHUNKING_STRATEGY,
            chunk_size_words=chunk_size_words,
            chunk_overlap_words=chunk_overlap_words,
        ),
        chunks,
    )


def chunk_text_words(text: str, chunk_size_words: int, overlap_words: int) -> list[str]:
    """Chunk near a target size without cutting ordinary sentences.

    The 40-word overlap is retained at natural sentence boundaries whenever
    possible. A punctuation-free or exceptionally long sentence falls back to
    deterministic word windows so that ingestion cannot silently drop text.
    """
    sentences = [sentence.strip() for sentence in _SENTENCE_BREAK.split(text) if sentence.strip()]
    if not sentences or any(len(sentence.split()) > chunk_size_words for sentence in sentences):
        return _fixed_word_windows(text, chunk_size_words, overlap_words)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > chunk_size_words:
            chunks.append(" ".join(current))
            current = _overlap_sentences(current, overlap_words)
            current_words = sum(len(item.split()) for item in current)
            if current and current_words + sentence_words > chunk_size_words:
                # Preserve the target size when one long sentence would make
                # sentence-level overlap counterproductive.
                current = []
                current_words = 0
        current.append(sentence)
        current_words += sentence_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def _fixed_word_windows(text: str, chunk_size_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = chunk_size_words - overlap_words
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def _overlap_sentences(sentences: list[str], overlap_words: int) -> list[str]:
    if overlap_words == 0:
        return []
    overlap: list[str] = []
    words = 0
    for sentence in reversed(sentences):
        overlap.insert(0, sentence)
        words += len(sentence.split())
        if words >= overlap_words:
            break
    return overlap


def _decode_text(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentParseError("The text file could not be decoded.")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()
