# Design Notes - Evidence-First Local RAG

This file is the source for the single-page submission PDF at `outputs/RAG_Explanation.pdf`.

**Design parameter.** I chose a 180-word target with a 40-word overlap, but changed the implementation from fixed word windows to sentence-aware windows. The target keeps a short factual procedure together; overlap protects a fact near a boundary. Preserving sentence boundaries matters because the grounding rule now verifies support at sentence level. The exact strategy and parameters are persisted with every document, and chunks remain inspectable through the API.

**Observed failure.** An early evidence gate unioned terms across the top retrieved chunk. On the Aurora handbook, “Does the safety officer inspect battery cabinets?” incorrectly returned `grounded: true`: one sentence mentioned the safety officer while another mentioned cabinet inspection, but no sentence established that relationship. The cause was aggregate term coverage masquerading as support. I fixed it by requiring one sentence to cover at least 60% of a one-part question; the new trace exposes the failed span check instead of inventing a connection.

**Metric.** I added a fixed, manually-authored 15-case regression set spanning TXT, a two-page PDF, direct facts, multi-fact questions, an explicit negative fact, and three unsupported/adversarial questions. In the final local-hash + hybrid dense/BM25 run: Recall@1=1.00, Recall@3=1.00, citation/fact pass=1.00, unsupported abstention=1.00, median end-to-end in-process latency=7.60 ms and p95=11.09 ms. This is a small regression result, not a production accuracy claim.

**Not finished / next.** The offline fallback is still hashed lexical embeddings, not a semantic local model; the compact suite was authored for regression rather than independently collected at scale. I did not add OCR, tables/layout-aware parsing, metadata filtering, authentication, or multi-user writes. Next I would collect a held-out human-authored corpus, compare OpenAI embeddings with a local sentence-transformer using Recall@k and citation faithfulness, then tune the 60% evidence threshold only on a separate development split.
