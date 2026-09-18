from __future__ import annotations

from app.answering import Generation
from app.config import Settings
from app.errors import EmbeddingProviderError
from app.service import RAGService


def test_auto_mode_discloses_local_fallback_after_hosted_embedding_error(monkeypatch, tmp_path):
    class FailingHostedProvider:
        backend_name = "openai"

        def embed_documents(self, texts):
            raise EmbeddingProviderError("simulated provider outage")

    monkeypatch.setattr("app.service.provider_for", lambda settings: FailingHostedProvider())
    service = RAGService(Settings(data_dir=tmp_path / "index", embedding_backend="auto"))

    result = service.ingest("facts.txt", b"The green beacon is inspected every Monday.")

    assert result.embedding_backend == "local-hash-v1"
    assert result.warning == "OpenAI embeddings failed; indexed with the local lexical fallback instead."


def test_answer_model_failure_uses_grounded_extractive_fallback(monkeypatch, tmp_path):
    service = RAGService(
        Settings(data_dir=tmp_path / "index", embedding_backend="local", answer_backend="openai")
    )
    service.ingest("facts.txt", b"The green beacon is inspected every Monday.")
    monkeypatch.setattr(
        "app.answering.generate_openai",
        lambda question, hits, settings, evidence_spans=(): Generation(
            None, "openai", "simulated answer outage"
        ),
    )

    answer = service.answer("When is the green beacon inspected?", top_k=3)

    assert answer.grounded is True
    assert answer.answer_backend == "extractive"
    assert "Monday" in answer.answer
    assert answer.warnings == ("simulated answer outage",)
