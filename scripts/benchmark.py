"""Measure a small repeatable local retrieval sample without needing a running server."""

from __future__ import annotations

import shutil
import statistics
import sys
import time
from pathlib import Path

# Make `python scripts/benchmark.py` work as well as `python -m scripts.benchmark`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.service import RAGService


def main() -> None:
    benchmark_dir = Path("work/benchmark-index")
    shutil.rmtree(benchmark_dir, ignore_errors=True)
    service = RAGService(Settings(data_dir=benchmark_dir, embedding_backend="local", answer_backend="extractive"))
    payload = Path("examples/aurora_station_handbook.txt").read_bytes()
    start = time.perf_counter()
    ingest = service.ingest("aurora_station_handbook.txt", payload)
    ingest_ms = (time.perf_counter() - start) * 1000
    measurements: list[float] = []
    for _ in range(20):
        start = time.perf_counter()
        result = service.answer("When does the Aurora Station safety drill begin?", top_k=3)
        measurements.append((time.perf_counter() - start) * 1000)
        assert result.grounded
    print(f"chunks={ingest.document.chunk_count} embedding_backend={ingest.embedding_backend}")
    print(f"ingest_ms={ingest_ms:.2f}")
    print(f"query_median_ms={statistics.median(measurements):.2f}")
    print(f"query_p95_ms={sorted(measurements)[int(len(measurements) * 0.95) - 1]:.2f}")


if __name__ == "__main__":
    main()
