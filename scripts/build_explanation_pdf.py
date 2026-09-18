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
            "I chose a <b>180-word target with a 40-word overlap</b>, but changed fixed word windows to sentence-aware windows. The target keeps a short factual procedure together; overlap protects a boundary fact. Preserving sentence boundaries matters because support is now verified at sentence level. The strategy and parameters are persisted with every document, and chunks remain inspectable through the API.",
        ),
        (
            "2. Failure observed while building",
            "An early gate unioned terms across a retrieved chunk. “Does the safety officer inspect battery cabinets?” incorrectly returned <font name='Courier'>grounded: true</font>: one sentence mentioned the officer and another cabinet inspection, but no sentence established that relationship. Aggregate term coverage masqueraded as support. I fixed it by requiring one sentence to cover at least <b>60%</b> of a one-part question; the trace now exposes the failed span check rather than inventing a connection.",
        ),
        (
            "3. Metric tracked: local query latency",
            "I added a fixed, manually-authored <b>15-case</b> regression set across TXT, a two-page PDF, direct and multi-fact questions, an explicit negative fact, and three unsupported/adversarial questions. Final local-hash + hybrid dense/BM25 run: <b>Recall@1=1.00</b>, Recall@3=1.00, fact/citation pass=1.00, unsupported abstention=1.00, median latency=<b>7.60 ms</b>, p95=11.09 ms. This is a small regression result, <b>not</b> a production accuracy claim.",
        ),
        (
            "4. Not finished and next step",
            "The offline fallback is still hashed lexical embeddings, not a semantic local model; the compact suite is for regression rather than independently collected at scale. I did not add OCR, tables/layout-aware parsing, metadata filtering, authentication, or multi-user writes. Next I would collect a held-out human-authored corpus, compare OpenAI embeddings with a local sentence-transformer using Recall@k and citation faithfulness, then tune the 60% evidence threshold only on a separate development split.",
        ),
    ]

    story = [
        para("Grounded Local RAG", title),
        para("One-page engineering explanation | FastAPI + local vector store | 18 September 2026", subtitle),
    ]
    for section_title, section_body in sections:
        story.append(KeepTogether([para(section_title, heading), para(section_body, body)]))
    footer = Table(
        [[para("Honest scope: runnable offline by default; hybrid ranking and sentence-level evidence are inspectable; OpenAI calls remain optional and guarded.", callout)]],
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
