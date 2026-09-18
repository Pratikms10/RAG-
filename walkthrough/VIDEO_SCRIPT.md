# 3-5 minute walkthrough plan

Use this plan after installing the project. Record a fresh terminal session and upload the resulting video to Google Drive yourself; sharing and public-link verification require the account owner.

## 0:00-0:35 - what the system does

Show the repository root and say: “This is a FastAPI RAG service that accepts PDF and TXT files, chunks them, writes vectors locally, retrieves the relevant chunks, and returns an answer plus exact chunk provenance. It refuses an answer when the evidence gate is not met.”

## 0:35-1:20 - start the API and upload a novel input

Run:

```powershell
python -m uvicorn app.main:app --reload
```

In a second terminal, upload `examples/walkthrough_unseen_input.txt` (or a newly written memo you have not used in the README):

```powershell
curl.exe -X POST http://127.0.0.1:8000/documents -F "file=@examples/walkthrough_unseen_input.txt"
```

Point out `chunks_added` and `embedding_backend`. If using the default local mode, say it is an offline lexical fallback; with `OPENAI_API_KEY`, the same code makes real OpenAI embedding and answer calls.

## 1:20-2:10 - show a grounded answer and sources

```powershell
curl.exe -X POST http://127.0.0.1:8000/questions -H "Content-Type: application/json" -d '{"question":"Who is the calibration lead and which platforms are affected?","top_k":3}'
```

Read the answer, then point to `source_chunks[].chunk_id`, `filename`, `score`, and `excerpt`. Explain that these are persisted provenance records, not fabricated citations.

## 2:10-2:40 - show the required failure case

```powershell
curl.exe -X POST http://127.0.0.1:8000/questions -H "Content-Type: application/json" -d '{"question":"What is the annual leave policy?"}'
```

Show `grounded: false` and `insufficient_evidence: true`. Say the service checks retrieval score *and* query-term coverage before it calls an answer generator.

## 2:40-3:35 - walk through one code section

Open `app/answering.py` and explain `evidence_is_sufficient`. It needs a minimum cosine score and at least two meaningful query terms (or one for a one-term query) in the top retrieved context. That is the guardrail against answering from vague similarity alone.

Then open `app/ingestion.py` and briefly point out the 180-word window and 40-word overlap. The overlap protects evidence that would otherwise straddle a boundary.

## 3:35-4:10 - show one prompt

Open `GROUNDING_PROMPT` in `app/answering.py`. State that when `OPENAI_API_KEY` is configured, it tells the model to answer only from source chunks, emit `NOT_FOUND` if the chunks are insufficient, and attach chunk IDs. Mention the `try/except` around the API call and the extractive fallback.

## 4:10-4:35 - close honestly

Show `outputs/RAG_Explanation.pdf` and `README.md`. Say the current version has a local persistent store but not OCR, metadata filtering, or a labeled retrieval evaluation suite; those are the next steps.

After recording: upload the video to Google Drive, set **General access** to **Anyone with the link**, and verify the link in an incognito window before submitting.
