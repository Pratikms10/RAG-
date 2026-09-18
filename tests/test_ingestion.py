from __future__ import annotations

from app.ingestion import build_document_and_chunks, chunk_text_words


def test_overlapping_word_chunking_keeps_boundary_context():
    text = " ".join(f"word{i}" for i in range(10))
    chunks = chunk_text_words(text, chunk_size_words=6, overlap_words=2)
    assert chunks == [
        "word0 word1 word2 word3 word4 word5",
        "word4 word5 word6 word7 word8 word9",
    ]


def test_document_id_is_content_addressed_and_chunks_have_provenance():
    document, chunks = build_document_and_chunks("memo.txt", b"A useful fact about radar calibration.", 80, 10)
    assert document.id == document.sha256[:16]
    assert chunks[0].document_id == document.id
    assert chunks[0].page_number is None
    assert chunks[0].id.startswith(document.id)
