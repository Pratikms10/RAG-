"""Run a small, fixed, manually-authored retrieval regression suite.

This is intentionally a transparency tool, not a claim of production-scale
accuracy. It records every case and separates retrieval, grounding, facts, and
citations so a single percentage never conceals a weak component.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.service import RAGService


CASES_PATH = ROOT / "evals" / "cases.jsonl"
EXAMPLE_DIR = ROOT / "examples"
FIXTURE_DIR = ROOT / "evals" / "fixtures"


def load_cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line]


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    temporary_index = ROOT / "work" / "evaluation-index"
    shutil.rmtree(temporary_index, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(data_dir=temporary_index, embedding_backend="local", answer_backend="extractive")
    service = RAGService(settings)
    for path in sorted(
        [*EXAMPLE_DIR.glob("*.txt"), *FIXTURE_DIR.glob("*.txt"), *FIXTURE_DIR.glob("*.pdf")]
    ):
        service.ingest(path.name, path.read_bytes())

    cases = load_cases()
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for case in cases:
        started = time.perf_counter()
        retrieval = service.retrieve(case["question"], top_k=3)
        answer = service.answer(case["question"], top_k=3)
        latency_ms = (time.perf_counter() - started) * 1_000
        latencies.append(latency_ms)
        expected_answer = case["expected"] == "answer"
        gold = case.get("gold")
        first_gold_rank = None
        if gold:
            needle = normalize(gold["contains"])
            for index, hit in enumerate(retrieval.hits, 1):
                page_matches = gold.get("page_number") is None or hit.chunk.page_number == gold.get("page_number")
                if (
                    hit.chunk.filename == gold["filename"]
                    and page_matches
                    and needle in normalize(hit.chunk.text)
                ):
                    first_gold_rank = index
                    break
        facts = case.get("required_answer_facts", [])
        answer_text = normalize(answer.answer)
        fact_pass = all(normalize(fact) in answer_text for fact in facts) if facts else None
        citation_pass = None
        if gold and answer.sources:
            needle = normalize(gold["contains"])
            citation_pass = any(
                hit.chunk.filename == gold["filename"]
                and (gold.get("page_number") is None or hit.chunk.page_number == gold.get("page_number"))
                and needle in normalize(hit.chunk.text)
                for hit in answer.sources
            )
        outcome = _failure_type(expected_answer, answer.grounded, first_gold_rank, fact_pass, citation_pass)
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "expected": case["expected"],
                "grounded": answer.grounded,
                "first_gold_rank": first_gold_rank,
                "fact_pass": fact_pass,
                "citation_pass": citation_pass,
                "latency_ms": round(latency_ms, 3),
                "failure_type": outcome,
                "answer": answer.answer,
                "gate_reasons": list(answer.retrieval.assessment.reasons),
                "retrieved_chunk_ids": [hit.chunk.id for hit in retrieval.hits],
            }
        )

    answerable = [row for row in rows if row["expected"] == "answer"]
    negatives = [row for row in rows if row["expected"] == "abstain"]
    metrics = {
        "suite": "small manually-authored regression set; not a production benchmark",
        "backend": "local-hash-v1 + hybrid dense/BM25 retrieval",
        "chunking": "sentence-window-v1, 180 words target, 40 words overlap",
        "cases": len(rows),
        "answerable_cases": len(answerable),
        "unsupported_cases": len(negatives),
        "retrieval_recall_at_1": _rate([row["first_gold_rank"] == 1 for row in answerable]),
        "retrieval_recall_at_3": _rate([row["first_gold_rank"] is not None for row in answerable]),
        "mrr_at_3": round(
            sum(1 / row["first_gold_rank"] for row in answerable if row["first_gold_rank"]) / len(answerable), 4
        ),
        "answerable_grounded_rate": _rate([row["grounded"] for row in answerable]),
        "answer_fact_pass_rate": _rate([bool(row["fact_pass"]) for row in answerable]),
        "citation_pass_rate": _rate([bool(row["citation_pass"]) for row in answerable]),
        "unsupported_abstention_rate": _rate([not row["grounded"] for row in negatives]),
        "false_grounded_rate_on_unsupported": _rate([row["grounded"] for row in negatives]),
        "latency_median_ms": round(statistics.median(latencies), 3),
        "latency_p95_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 3),
    }
    (output_dir / "results.json").write_text(
        json.dumps({"metrics": metrics, "cases": rows}, indent=2), encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(render_report(metrics, rows), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def _rate(values: list[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _failure_type(
    expected_answer: bool,
    grounded: bool,
    first_gold_rank: int | None,
    fact_pass: bool | None,
    citation_pass: bool | None,
) -> str:
    if not expected_answer and grounded:
        return "unsafe_grounded_answer"
    if not expected_answer:
        return "correct_abstention"
    if first_gold_rank is None:
        return "gold_evidence_not_retrieved"
    if not grounded:
        return "evidence_gate_false_negative"
    if not fact_pass:
        return "answer_fact_miss"
    if not citation_pass:
        return "citation_provenance_miss"
    return "pass"


def render_report(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    metric_lines = "\n".join(f"- **{key.replace('_', ' ')}:** {value}" for key, value in metrics.items())
    table = "\n".join(
        "| {id} | {category} | {expected} | {rank} | {grounded} | {facts} | {citation} | {failure} |".format(
            id=row["id"],
            category=row["category"],
            expected=row["expected"],
            rank=row["first_gold_rank"] or "-",
            grounded="yes" if row["grounded"] else "no",
            facts="-" if row["fact_pass"] is None else ("yes" if row["fact_pass"] else "no"),
            citation="-" if row["citation_pass"] is None else ("yes" if row["citation_pass"] else "no"),
            failure=row["failure_type"],
        )
        for row in rows
    )
    return f"""# Retrieval Evaluation Report

{metrics['suite']}. This result applies only to this fixed corpus and the recorded local backend/configuration.

## Configuration and metrics

{metric_lines}

## Per-case evidence

| Case | Category | Expected | First gold rank | Grounded | Facts | Citation | Outcome |
| --- | --- | --- | ---: | --- | --- | --- | --- |
{table}

## Failure taxonomy

- `gold_evidence_not_retrieved`: ranking failure.
- `evidence_gate_false_negative`: gold evidence ranked but the grounding rule abstained.
- `answer_fact_miss`: answer was grounded but missed a required fact.
- `citation_provenance_miss`: answer did not return the expected source chunk.
- `unsafe_grounded_answer`: unsupported question received a grounded answer.
"""


if __name__ == "__main__":
    main()
