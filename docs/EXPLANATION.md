# Design Notes - Grounded Local RAG

This file is the text source for the single-page submission PDF at `outputs/RAG_Explanation.pdf`.

**Design parameter.** I chose 180-word chunks with a 40-word overlap. Reference documents in this exercise are likely short but factual; 180 words usually keeps a complete procedure or policy together, while 40 words protects an answer that crosses a boundary. I deliberately kept the chunker as an inspectable word-window function instead of hiding it behind a framework. The exact chunks and their IDs are saved in `data/chunks.json`.

**Observed failure.** While verifying the build, two overlapping `pip install` processes tried to write the same Pydantic metadata file and produced an `OSError: [Errno 13] Permission denied` on `pydantic-2.13.5.dist-info/INSTALLER`. The cause was concurrent environment setup, not the API. I let the first process finish, then reran installation; it is a reminder that setup must be reproducible and serialized. Separately, the API has a tested malformed-PDF path: pypdf parsing errors become a clear 422 response rather than an uncaught exception.

**Metric.** I measured local query latency using `python scripts/benchmark.py`: 20 identical questions against the sample handbook, after ingestion. The final run recorded 0.81 ms median and 2.40 ms p95 in-process (reported in the README). It confirms that the local fallback is fast enough for a demo; it does *not* establish retrieval accuracy. The response also exposes each cosine score so a reviewer can inspect why a chunk was selected.

**Not finished / next.** The current offline fallback is hashed lexical embeddings, not a semantic local model, and I did not build a labeled retrieval benchmark, OCR for scanned PDFs, metadata filters, or multi-user concurrency. Next I would create a 30-50 question gold set, compare OpenAI embeddings against a local sentence-transformer with recall@k and answer-faithfulness checks, tune the threshold from evidence, and add OCR only if the target documents require it.
