from __future__ import annotations

from app.config import Settings
from app.service import RAGService


def test_clear_removes_local_index(tmp_path):
    service = RAGService(Settings(data_dir=tmp_path / "index", embedding_backend="local", answer_backend="extractive"))
    service.ingest("facts.txt", b"The blue team meets on Thursday at noon.")
    assert service.documents()
    service.clear()
    assert not service.documents()
