# 3–5 minute Evidence Lab walkthrough

Record a fresh local session, then upload the finished video to Google Drive yourself. Before submitting, change **General access** to **Anyone with the link** and open the link in an incognito/private window. The screen recording should use a small memo created immediately before recording—not a checked-in example—so the reviewer sees genuinely new input.

## Prep: create an unseen input

In a blank text editor, create and save a short `launch_brief.txt` with three or four facts such as:

```text
The Atlas launch rehearsal begins at 08:45 on Wednesday in Bay C.
Priya Nair is the rehearsal lead.
Visitors must wear a purple pass and remain with their host.
The brief does not specify any annual-leave policy.
```

Do not add this file to the repository. It is only for the recording.

## 0:00–0:30 — problem and approach

Show the repository root and say:

> “Evidence Lab is a FastAPI RAG system for PDF and TXT. It has a local vector store, hybrid retrieval, and an evidence gate. The goal is not just to produce an answer—it must show why the answer is supported or refuse it.”

Briefly point to `README.md` and the one-page explanation PDF.

## 0:30–1:15 — run the system and upload the fresh input

Start the API:

```powershell
python -m uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000**. Drag the freshly created `launch_brief.txt` into the Evidence Lab interface. Point out:

- The configured **180-word target / 40-word overlap**.
- The document card showing local indexing and chunking strategy.
- The method panel: dense similarity is 70%, lexical evidence is 30%.

State honestly: “This recording uses the offline local-hash fallback so it runs without credentials. With an `OPENAI_API_KEY`, the same app can make guarded embedding and answer calls.”

## 1:15–2:05 — grounded answer and inspectable evidence

Ask: **“When and where does the Atlas launch rehearsal begin?”**

Show the `GROUNDED` verdict, answer, and source sentence. Expand the candidate chunk to point out:

- filename and chunk ID;
- dense, lexical, and hybrid scores;
- meaningful query terms and sentence coverage.

Say: “The answer came from this sentence in the uploaded memo. These are persisted source records, not made-up citations.”

## 2:05–2:45 — required no-answer behavior

Ask: **“What annual leave policy applies to Atlas staff?”**

Show `INSUFFICIENT EVIDENCE` and the visible gate reason. Say: “The source says that the brief does not specify an annual-leave policy, so the system refuses to invent one. This is the required no-answer path.”

## 2:45–3:35 — show the observed failure and fix

Open `app/answering.py`, first at `assess_evidence`, then at `_rank_evidence_spans`. Explain:

> “I observed a failure where a safety-officer sentence and a battery-cabinet sentence were combined into a relationship the document never stated. Now a one-part question needs a single source sentence to cover at least 60% of meaningful terms. That stops cross-sentence fact stitching.”

Show the `negative_stitched` row in `outputs/evaluation/REPORT.md` or run:

```powershell
python scripts/evaluate.py
```

Call it a small fixed regression suite, not a general accuracy benchmark.

## 3:35–4:15 — retrieval and one prompt

Open `app/retrieval.py` and briefly show `rank_chunks`: 70% dense similarity plus 30% BM25-style lexical score. Then open `GROUNDING_PROMPT` in `app/answering.py` and say:

> “When optional OpenAI mode is configured, this prompt says to answer only from supplied chunks or return `NOT_FOUND`. The code catches provider failures and falls back to the local extractive path.”

## 4:15–4:40 — close honestly

Show `outputs/RAG_Explanation.pdf`, the 15-case evaluation report, and `README.md`. Close with:

> “What I would do next is use held-out human-authored evaluation data, compare a semantic local model with OpenAI embeddings, and add OCR and layout-aware PDF parsing. I deliberately did not add deployment or authentication because they would not improve the retrieval-quality reasoning in this scope.”

After recording, upload the video to Google Drive, set it to **Anyone with the link**, test it while signed out/incognito, and submit that link with the public repository URL and explanation PDF.
