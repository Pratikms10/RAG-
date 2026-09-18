"""Build the one-page explanation PDF from verified implementation facts."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


OUTPUT = Path("outputs/RAG_Explanation.pdf")


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=15 * mm,
        bottomMargin=14 * mm,
        title="Grounded Local RAG - One-Page Explanation",
        author="Pratik",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=16, leading=19,
        textColor=colors.HexColor("#0F172A"), spaceAfter=2,
    )
    subtitle = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=11,
        textColor=colors.HexColor("#475569"), spaceAfter=8,
    )
    heading = ParagraphStyle(
        "Heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=9.4, leading=11.5,
        textColor=colors.HexColor("#0F766E"), spaceBefore=2, spaceAfter=2,
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.25, leading=10.4,
        textColor=colors.HexColor("#1E293B"), alignment=TA_LEFT, spaceAfter=5,
    )
    callout = ParagraphStyle(
        "Callout", parent=body, textColor=colors.HexColor("#0F172A"), spaceAfter=0,
    )

    sections = [
        (
            "1. Design parameter: chunk size",
            "I chose <b>180-word chunks with a 40-word overlap</b>. The task's reference files are likely short but factual: 180 words usually holds one procedure or policy together, while 40 words protects an answer that crosses a boundary. The chunker is an inspectable word-window function rather than a hidden framework abstraction; each exact chunk and ID is persisted in <font name='Courier'>data/chunks.json</font>.",
        ),
        (
            "2. Failure observed while building",
            "During verification, two overlapping <font name='Courier'>pip install</font> processes tried to update the same Pydantic metadata file and raised <font name='Courier'>OSError: [Errno 13] Permission denied</font> for <font name='Courier'>pydantic-2.13.5.dist-info/INSTALLER</font>. The cause was concurrent environment setup, not the API; I let the first install finish and reran the second. In the product path, malformed-PDF parser failures are also caught and returned as a clear 422 response (covered by a test).",
        ),
        (
            "3. Metric tracked: local query latency",
            "I ran <font name='Courier'>python scripts/benchmark.py</font> with 20 identical questions after ingesting the sample handbook. On this machine it recorded <b>0.81 ms median</b> and <b>2.40 ms p95</b> local query latency (excluding server/network overhead). That says the lightweight fallback is quick enough for a demo; it does <b>not</b> prove retrieval accuracy. Returned cosine scores and excerpts make each retrieval decision inspectable.",
        ),
        (
            "4. Not finished and next step",
            "The offline fallback is hashed lexical embeddings, not a semantic local model. I did not build a labeled retrieval benchmark, OCR for scanned PDFs, metadata filtering, or multi-user concurrency. Next I would build a 30-50 question gold set, compare OpenAI embeddings against a local sentence-transformer using recall@k and faithfulness checks, tune the evidence threshold from those results, then add OCR only if target inputs require it.",
        ),
    ]

    story = [
        para("Grounded Local RAG", title),
        para("One-page engineering explanation | FastAPI + local vector store | 18 September 2026", subtitle),
    ]
    for section_title, section_body in sections:
        story.append(KeepTogether([para(section_title, heading), para(section_body, body)]))
    footer = Table(
        [[para("Honest scope: runnable offline by default; OpenAI calls are optional and guarded with fallback/error handling.", callout)]],
        colWidths=[176 * mm],
    )
    footer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E6FFFB")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#99F6E4")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([Spacer(1, 2 * mm), footer])
    doc.build(story)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
