# Grounded Local RAG

A compact, inspectable RAG service for PDF and TXT reference documents. It indexes documents locally, retrieves the most relevant chunks, answers only when there is enough evidence, and returns the exact source chunks behind each answer.

This is intentionally not built on a black-box RAG framework. The essential decisions - parsing, word-window chunking, embedding selection, local persistence, cosine retrieval, grounding gate, and answer fallback - are visible in small Python modules.

## What it demonstrates

| Requirement | Implementation |
| --- | --- |
| PDF and TXT input | `POST /documents`; `pypdf` extracts text-based PDFs and TXT is decoded locally. |
| Chunk and embed | Overlapping 180-word windows by default; OpenAI embeddings when configured, deterministic local vectors otherwise. |
| Vector store | `data/vectors.npy` holds normalized vectors; `chunks.json`, `documents.json`, and `metadata.json` hold inspectable provenance and index metadata. |
| Retrieval | Cosine similarity over normalized vectors; the top `k` chunks, scores, excerpts, and IDs are returned. |
| Grounded answer | A score-plus-term-coverage evidence gate runs before answer synthesis. The local fallback extracts supporting sentences; optional OpenAI generation receives only retrieved chunks. |
| No-answer behavior | The API responds `grounded: false` and `insufficient_evidence: true` rather than inventing an answer. |
| Pydantic validation | `QuestionRequest` validates nonblank questions and `top_k`; FastAPI validates upload parameters. |
| Error handling | Document parser errors map to 422; duplicate/index conflicts map to 409; OpenAI embedding errors map to 503 or, in `auto` mode during a new ingest, a disclosed local fallback. |

## Quick start

The default mode is fully local: it does **not** require an API key, network call, database, or model download.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs> for the interactive OpenAPI UI.

In another PowerShell window, upload the sample document:

```powershell
curl.exe -X POST http://127.0.0.1:8000/documents -F "file=@examples/aurora_station_handbook.txt"
```

Ask an answerable question:

```powershell
curl.exe -X POST http://127.0.0.1:8000/questions -H "Content-Type: application/json" -d '{"question":"When does the Aurora Station safety drill begin?","top_k":3}'
```

Ask an unanswerable question:

```powershell
curl.exe -X POST http://127.0.0.1:8000/questions -H "Content-Type: application/json" -d '{"question":"What is the annual leave policy?"}'
```

Expected shape (abbreviated):

```json
{
  "answer": "Based on the uploaded documents: The Aurora Station safety drill begins at 09:30 every Tuesday... [chunk-id]",
  "grounded": true,
  "insufficient_evidence": false,
  "answer_backend": "extractive",
  "source_chunks": [
    {"chunk_id": "...", "filename": "aurora_station_handbook.txt", "page_number": null, "score": 0.63, "excerpt": "..."}
  ]
}
```

`DELETE /documents` clears the local index when you want a clean demo or change embedding backends.

## PDF input

Text-based PDFs work through the same upload route:

```powershell
curl.exe -X POST http://127.0.0.1:8000/documents -F "file=@C:\path\to\reference.pdf"
```

Scanned/image-only PDFs return 422 rather than silently producing an empty index. OCR is deliberately listed as unfinished work below.

## Optional OpenAI mode

Copy `.env.example` values into your shell (do not commit the secret):

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:RAG_EMBEDDING_BACKEND = "openai"
$env:RAG_ANSWER_BACKEND = "openai"
python -m uvicorn app.main:app --reload
```

In OpenAI mode, `app/embeddings.py` calls `client.embeddings.create(...)` and `app/answering.py` calls `chat.completions.create(...)`. Both calls have explicit error handling. In `auto` mode, a new ingest falls back to the local embedder if the hosted embedding call fails before data is persisted; an explicit `openai` configuration fails loudly with a 503 instead of silently changing quality.

The prompt is in `app/answering.py` as `GROUNDING_PROMPT`. It tells the model to use only retrieved source chunks, return `NOT_FOUND` for insufficient context, and attach chunk IDs after factual claims. The API still returns the provenance records independently of the model text.

## Design and quality notes

### Chunking and retrieval

The default 180-word chunk with 40-word overlap was chosen to preserve a self-contained factual paragraph while retaining boundary context. It is intentionally configurable on upload (`chunk_size_words`, `chunk_overlap_words`) so the decision can be evaluated. Vectors are L2-normalized; therefore the dot product in `LocalVectorStore.search` is cosine similarity.

Before generating an answer, `evidence_is_sufficient` requires:

1. The top vector score to meet the configured 0.18 threshold.
2. At least two meaningful query terms (one for a one-term question) to occur in the first two retrieved chunks.
3. At least 50% coverage of meaningful query terms.

That conservative second check protects against a broad topic match becoming a fabricated answer. It can lead to false negatives, which is preferable to unsupported answers in this task.

### Why a custom local store?

For a single-user take-home, a NumPy vector matrix plus JSON is durable, inspectable, and has no service dependency. Each stored chunk includes its file name, document hash ID, page number (when PDF), ordinal, full text, and word count. A production system would use a database/vector engine for concurrent writes, filtering, access controls, and scale; adding one here would not improve the core retrieval reasoning.

### What works vs. what does not

Works:

- TXT uploads and text-based PDF uploads.
- Persistent local index across server restarts.
- Source chunk IDs, filenames, page numbers, scores, and excerpts in every answer response.
- Explicit no-answer behavior for unsupported questions and an empty index.
- Offline local operation and optional guarded OpenAI calls.
- Tests for TXT retrieval, actual PDF retrieval, malformed PDFs, invalid request data, overlap behavior, storage reset, and simulated hosted-model failures.

Not finished:

- OCR for scanned PDFs, tables/layout-aware PDF parsing, metadata filtering, document deletion by ID, authentication, multi-user locking, and a frontend.
- A labeled retrieval/faithfulness evaluation set. The latency benchmark is not an accuracy claim.
- The local fallback is lexical feature hashing, not a semantic transformer. Use OpenAI embeddings or add a benchmarked local semantic model for broader paraphrase retrieval.

## Verification

Run the suite:

```powershell
python -m pytest
```

Current result: **10 passed** (TXT, real generated PDF, unanswerable question, malformed input, Pydantic validation, chunk overlap, provenance, storage reset, and guarded hosted-model failures).

Run the repeatable local latency check:

```powershell
python scripts/benchmark.py
```

On the build machine on 18 September 2026, the final run reported **0.81 ms median** and **2.40 ms p95** for 20 in-process local retrieval questions after the sample document was indexed. It excludes upload, server, and network overhead and should be remeasured on another machine.

## Submission artifacts

- One-page explanation PDF: [`outputs/RAG_Explanation.pdf`](outputs/RAG_Explanation.pdf)
- Its Markdown source: [`docs/EXPLANATION.md`](docs/EXPLANATION.md)
- 3-5 minute recording plan: [`walkthrough/VIDEO_SCRIPT.md`](walkthrough/VIDEO_SCRIPT.md)

For the video, use `examples/walkthrough_unseen_input.txt` (or a fresh equivalent memo), show the answerable and unanswerable calls, then show `app/answering.py` and `GROUNDING_PROMPT`. The account owner must record, upload to Google Drive, change sharing to **Anyone with the link**, and test the link in an incognito window.

## Repository publishing checklist

1. Run `python -m pytest` and `python scripts/benchmark.py` from a fresh virtual environment.
2. Commit the existing multi-commit history and push to a **public** GitHub repository.
3. Open the repository link in a private/incognito browser window.
4. Upload the walkthrough video, set it to **Anyone with the link**, and test it in an incognito window.
5. Submit the repository URL, [`outputs/RAG_Explanation.pdf`](outputs/RAG_Explanation.pdf), the public Drive URL, and actual hours spent.
