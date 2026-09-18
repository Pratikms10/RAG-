# Evidence Lab — Grounded Local RAG

Evidence Lab is an inspectable RAG system for PDF and TXT reference documents. Upload sources, ask a question, and see the answer alongside the ranked chunks, dense/lexical retrieval signals, and the sentence-level evidence that allowed—or blocked—the result.

It is designed around the take-home requirement that an answer come from the uploaded documents rather than model memory. The default path runs offline; hosted OpenAI embeddings and generation are optional, guarded integrations.

## What it demonstrates

| Requirement | Implementation |
| --- | --- |
| PDF + TXT input | `POST /documents` parses UTF-8 TXT and text-based PDFs with page provenance. Image-only PDFs fail explicitly instead of silently indexing nothing. |
| Chunk and embed | Sentence-aware windows target 180 words with 40-word overlap. The configuration is persisted per document. Local deterministic embeddings work offline; OpenAI embeddings are optional. |
| Local vector store | A NumPy vector matrix plus JSON metadata persists vectors, chunks, document hashes, page numbers, and chunking settings. |
| Retrieval | Hybrid ranking combines normalized dense similarity (70%) and BM25-style lexical evidence (30%). Every candidate exposes both signals and matched terms. |
| Grounded answer | A gate checks score, term coverage, and direct sentence support before answer generation. The offline answer is extractive; optional OpenAI generation is source-constrained. |
| Provenance | Responses return source chunk IDs, filenames, PDF page numbers, excerpts, ranking signals, and the sentence spans that passed the gate. |
| No-answer behavior | Unsupported questions return `grounded: false` / `insufficient_evidence: true`; the retrieval trace explains why. |
| Validation + errors | Pydantic validates questions and FastAPI validates uploads. Parser, duplicate, index, and hosted-provider failures have explicit HTTP behavior. |

## Evidence Lab interface

Start the service and open **http://127.0.0.1:8000**. The built-in local UI supports drag-and-drop upload, chunk configuration, answers, source inspection, and a readable retrieval trace. Open **http://127.0.0.1:8000/docs** for the FastAPI contract.

The interface makes the quality-critical details visible:

1. Candidate chunks ranked by hybrid score, with dense and lexical components.
2. Meaningful query terms, term coverage, and the threshold used.
3. Direct source sentences and their coverage of the question.
4. A visible `GROUNDED` or `INSUFFICIENT EVIDENCE` verdict before prose is trusted.

## Quick start

The default setup needs no API key, network call, database, or model download.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m uvicorn app.main:app --reload
```

In another PowerShell window, add a TXT source and ask a grounded question:

```powershell
curl.exe -X POST http://127.0.0.1:8000/documents -F "file=@examples/aurora_station_handbook.txt"

curl.exe -X POST http://127.0.0.1:8000/questions `
  -H "Content-Type: application/json" `
  -d '{"question":"When and where does the Aurora Station safety drill begin?","top_k":3}'
```

Then try a question not established by the source:

```powershell
curl.exe -X POST http://127.0.0.1:8000/questions `
  -H "Content-Type: application/json" `
  -d '{"question":"What is the annual leave policy?"}'
```

The second response intentionally refuses to answer. Its `retrieval_trace` distinguishes a retrieval miss from an evidence-gate refusal.

## API surface

| Route | Purpose |
| --- | --- |
| `POST /documents` | Ingest a PDF/TXT with optional `chunk_size_words` and `chunk_overlap_words`. |
| `GET /documents` | List indexed documents and their persisted chunking settings. |
| `GET /documents/{document_id}/chunks` | Inspect every chunk in a document. |
| `GET /documents/{document_id}/chunks/{chunk_id}` | Inspect one cited chunk exactly. |
| `POST /retrievals` | Preview ranking and the evidence gate without generating prose. |
| `POST /questions` | Return an answer/refusal, sources, and full retrieval trace. |
| `DELETE /documents` | Reset the local index for a clean demo. |

Abbreviated answer response:

```json
{
  "answer": "Based on the uploaded documents: ... [chunk-id]",
  "grounded": true,
  "insufficient_evidence": false,
  "answer_backend": "extractive",
  "source_chunks": [
    {
      "chunk_id": "...",
      "filename": "aurora_station_handbook.txt",
      "page_number": null,
      "dense_score": 0.42,
      "lexical_score": 1.0,
      "score": 0.59
    }
  ],
  "retrieval_trace": {
    "term_coverage": 1.0,
    "best_span_coverage": 1.0,
    "evidence_spans": [{"chunk_id": "...", "text": "..."}]
  }
}
```

## Design decisions worth reviewing

### Sentence-aware 180 / 40 chunking

The default is a **180-word target with 40 words of overlap**. It is large enough to keep a short procedure or paragraph coherent, while overlap protects facts at a boundary. Unlike a raw fixed window, the preferred chunker respects sentence boundaries because the grounding layer evaluates support at sentence level. The original fixed-word routine remains as a fallback when an unusually long sentence must be split.

### Transparent hybrid retrieval

`app/retrieval.py` calculates a 70/30 combination of dense dot-product similarity and normalized BM25-style lexical score. Exact entities, numbers, and terms matter disproportionately in document QA, while dense similarity can improve paraphrase recall when a semantic embedding backend is configured. Showing both components keeps the ranking debuggable rather than magical.

### A real observed failure, and the fix

An early gate joined meaningful terms across separate retrieved sentences. It incorrectly grounded: *“Does the safety officer inspect battery cabinets?”* One sentence named the officer; another described cabinet inspection; neither established that relationship. The current gate requires a single source sentence to cover at least **60%** of a one-part question. Multi-fact questions may use more than one span, but the selected spans remain visible.

### Why no black-box RAG framework?

Parsing, chunking, embedding selection, local persistence, hybrid ranking, evidence assessment, and answer fallback live in small modules under `app/`. A NumPy matrix and JSON files fit a single-user take-home: durable, inspectable, and trivial to reset. A production system would add a vector database, filters, auth, locking, and observability only when those requirements exist.

## Evaluation and verification

Run all automated tests:

```powershell
python -m pytest
```

Run the reproducible retrieval evaluation:

```powershell
python scripts/evaluate.py
```

The evaluation is a **small, manually-authored 15-case regression corpus**, not a production accuracy benchmark. It includes multi-chunk TXT, a two-page PDF (including a page-2 provenance check), direct/multi-fact questions, an explicit negative fact, and three unsupported/adversarial questions. It separately reports Retrieval Recall@k, MRR, grounding, fact coverage, citation coverage, abstention, and latency so one headline percentage cannot hide a weak component.

The committed [evaluation report](outputs/evaluation/REPORT.md) records the exact configuration and every case. On the verified local-hash + hybrid run used to build this repository: Recall@1/Recall@3/MRR@3 were 1.00; answer fact pass, citation pass, and unsupported abstention were 1.00; median end-to-end in-process latency was 7.60 ms and p95 was 11.09 ms. Those values apply only to this small fixed suite and should be re-run on another machine.

## Optional OpenAI mode

Set credentials only in your shell—never commit them:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:RAG_EMBEDDING_BACKEND = "openai"
$env:RAG_ANSWER_BACKEND = "openai"
python -m uvicorn app.main:app --reload
```

`app/embeddings.py` calls `client.embeddings.create(...)`; `app/answering.py` calls `chat.completions.create(...)`. Both calls have explicit failure handling. In `auto` mode, a failed embedding call falls back to the local provider **before** persistence and returns a warning. Explicit `openai` mode returns a 503 rather than silently changing retrieval quality. `GROUNDING_PROMPT` instructs the model to answer only from supplied chunks or return `NOT_FOUND`; the system independently returns only source chunks with direct evidence spans.

## What works and what is deliberately unfinished

Works now:

- PDF and TXT ingestion, including PDF page provenance.
- Local persistent vectors and inspectable JSON metadata.
- Sentence-aware chunking, hybrid retrieval, retrieval-only tracing, and exact chunk inspection.
- Offline extractive answers, optional guarded OpenAI calls, and explicit no-answer behavior.
- A local Evidence Lab UI plus FastAPI OpenAPI docs.
- Tests for parsing, PDF provenance, validation, fallback/error behavior, chunking, cross-sentence false grounding, trace visibility, and evaluation fixtures.

Not finished:

- OCR for scanned PDFs, tables/layout-aware parsing, metadata filters, per-document delete, auth, multi-user writes, and production telemetry.
- A semantic local model: the offline fallback is deterministic feature hashing, not a sentence transformer.
- Independently collected, held-out evaluation data. The 15 fixed cases are a regression suite and do not establish general accuracy.
- Structured claim-by-claim verification of optional LLM wording. The default extractive path is intentionally the most conservative demo path.

## Submission artifacts

- One-page explanation PDF: [`outputs/RAG_Explanation.pdf`](outputs/RAG_Explanation.pdf)
- Explanation source: [`docs/EXPLANATION.md`](docs/EXPLANATION.md)
- Measured regression report: [`outputs/evaluation/REPORT.md`](outputs/evaluation/REPORT.md)
- 3–5 minute recording plan: [`walkthrough/VIDEO_SCRIPT.md`](walkthrough/VIDEO_SCRIPT.md)

For the walkthrough, upload a memo created immediately before recording (not the checked-in sample), show one grounded result and one refusal, open the Evidence Lab trace, then explain `app/answering.py` and `GROUNDING_PROMPT`. The account owner must record the video, upload it to Google Drive, set sharing to **Anyone with the link**, and verify the link from an incognito/private window before submitting.

## Publishing checklist

1. Run `python -m pytest` and `python scripts/evaluate.py` from the repository root.
2. Commit and push the multi-commit history to the public GitHub repository.
3. Open the GitHub link in an incognito/private browser window.
4. Record the walkthrough, share the Drive link with **Anyone with the link**, and test it in incognito.
5. Submit the repository URL, the one-page PDF, video link, and actual hours spent.
