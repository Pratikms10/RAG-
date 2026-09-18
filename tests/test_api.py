from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from reportlab.pdfgen.canvas import Canvas

from app.api import create_app
from app.config import Settings


def client_for(tmp_path):
    app = create_app(Settings(data_dir=tmp_path / "index", embedding_backend="local", answer_backend="extractive"))
    return TestClient(app)


def upload_handbook(client: TestClient):
    text = (
        "Aurora Base Manual. The annual safety drill begins at 09:30 every Tuesday. "
        "Visitors need a yellow badge and an escort. Sodium-ion cabinets are inspected every 30 days."
    )
    return client.post("/documents", files={"file": ("manual.txt", text.encode(), "text/plain")})


def test_txt_upload_question_and_chunk_provenance(tmp_path):
    client = client_for(tmp_path)
    response = upload_handbook(client)
    assert response.status_code == 201
    assert response.json()["chunks_added"] == 1
    assert response.json()["embedding_backend"] == "local-hash-v1"

    answer = client.post("/questions", json={"question": "When does the annual safety drill begin?"})
    body = answer.json()
    assert answer.status_code == 200
    assert body["grounded"] is True
    assert "09:30" in body["answer"]
    assert body["source_chunks"]
    assert body["source_chunks"][0]["filename"] == "manual.txt"
    assert body["source_chunks"][0]["chunk_id"].startswith(body["source_chunks"][0]["document_id"])


def test_unanswerable_question_does_not_invent(tmp_path):
    client = client_for(tmp_path)
    upload_handbook(client)
    answer = client.post("/questions", json={"question": "What is the annual leave policy?"})
    body = answer.json()
    assert answer.status_code == 200
    assert body["grounded"] is False
    assert body["insufficient_evidence"] is True
    assert "could not find enough evidence" in body["answer"].lower()


def test_text_pdf_upload_is_searchable(tmp_path):
    client = client_for(tmp_path)
    pdf = BytesIO()
    canvas = Canvas(pdf)
    canvas.drawString(72, 720, "The observatory opens its dome at 19:15 on clear nights.")
    canvas.save()
    response = client.post(
        "/documents", files={"file": ("observatory.pdf", pdf.getvalue(), "application/pdf")}
    )
    assert response.status_code == 201
    assert response.json()["format"] == "pdf"
    answer = client.post("/questions", json={"question": "When does the observatory open its dome?"})
    assert answer.status_code == 200
    assert answer.json()["grounded"] is True
    assert "19:15" in answer.json()["answer"]


def test_rejects_unsupported_and_malformed_pdf(tmp_path):
    client = client_for(tmp_path)
    bad_type = client.post("/documents", files={"file": ("notes.docx", b"not a docx", "application/octet-stream")})
    assert bad_type.status_code == 422

    malformed_pdf = client.post(
        "/documents", files={"file": ("broken.pdf", b"%PDF-1.7 truncated", "application/pdf")}
    )
    assert malformed_pdf.status_code == 422
    assert "could not be read" in malformed_pdf.json()["detail"]


def test_request_validation_and_empty_index(tmp_path):
    client = client_for(tmp_path)
    invalid = client.post("/questions", json={"question": "  ", "top_k": 0})
    assert invalid.status_code == 422

    empty = client.post("/questions", json={"question": "When is the drill?"})
    assert empty.status_code == 200
    assert empty.json()["insufficient_evidence"] is True
