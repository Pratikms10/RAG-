from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings


def client_for(tmp_path):
    return TestClient(
        create_app(Settings(data_dir=tmp_path / "index", embedding_backend="local", answer_backend="extractive"))
    )


def upload_aurora(client: TestClient):
    payload = (
        "Aurora Station Field Handbook. The Aurora Station safety drill begins at 09:30 every Tuesday "
        "in the east assembly bay. The safety officer checks attendance before the group returns to work. "
        "The battery laboratory uses cobalt-free sodium-ion cells for short duration backup power. "
        "Each cabinet must be inspected every 30 days by a trained technician."
    )
    return client.post("/documents", files={"file": ("aurora.txt", payload.encode(), "text/plain")})


def test_sentence_level_guard_rejects_stitched_relationship(tmp_path):
    client = client_for(tmp_path)
    upload_aurora(client)

    response = client.post("/questions", json={"question": "Does the safety officer inspect battery cabinets?"})

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["retrieval_trace"]["requires_single_span"] is True
    assert body["retrieval_trace"]["best_span_coverage"] < 0.5
    assert any("will not combine separate facts" in item for item in body["retrieval_trace"]["reasons"])


def test_explicit_two_part_question_can_use_two_evidence_sentences(tmp_path):
    client = client_for(tmp_path)
    upload_aurora(client)

    response = client.post(
        "/questions",
        json={"question": "What cell chemistry does the battery laboratory use and how often are cabinets inspected?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert "sodium-ion" in body["answer"]
    assert "30 days" in body["answer"]
    assert body["retrieval_trace"]["requires_single_span"] is False
    assert len(body["retrieval_trace"]["evidence_spans"]) >= 2


def test_trace_exposes_hybrid_signals_and_exact_chunk_lookup(tmp_path):
    client = client_for(tmp_path)
    ingested = upload_aurora(client).json()

    trace = client.post("/retrievals", json={"question": "When does the safety drill begin?", "top_k": 3})

    assert trace.status_code == 200
    body = trace.json()
    assert body["sufficient"] is True
    candidate = body["candidates"][0]
    assert candidate["dense_score"] >= 0
    assert candidate["lexical_score"] >= 0
    assert candidate["matched_terms"]

    chunk = client.get(f"/documents/{ingested['document_id']}/chunks/{candidate['chunk_id']}")
    assert chunk.status_code == 200
    assert "09:30" in chunk.json()["text"]


def test_evidence_lab_is_served(tmp_path):
    client = client_for(tmp_path)
    response = client.get("/")
    assert response.status_code == 200
    assert "Evidence Lab" in response.text


def test_two_page_pdf_keeps_page_two_provenance(tmp_path):
    client = client_for(tmp_path)
    pdf_path = "evals/fixtures/orbital_appendix.pdf"
    with open(pdf_path, "rb") as stream:
        upload = client.post(
            "/documents", files={"file": ("orbital_appendix.pdf", stream.read(), "application/pdf")}
        )
    assert upload.status_code == 201

    response = client.post(
        "/questions", json={"question": "Who leads optical calibration and when?", "top_k": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert "navigation engineer" in body["answer"].lower()
    assert "14:20" in body["answer"]
    assert any(source["page_number"] == 2 for source in body["source_chunks"])
