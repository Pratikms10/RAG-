"""Create a two-page, text-based PDF fixture for provenance regression testing."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas


OUTPUT = Path("evals/fixtures/orbital_appendix.pdf")


def draw_wrapped(canvas: Canvas, text: str, x: float, y: float, width: float) -> float:
    words = text.split()
    line: list[str] = []
    while words:
        candidate = " ".join([*line, words[0]])
        if line and stringWidth(candidate, "Helvetica", 11) > width:
            canvas.drawString(x, y, " ".join(line))
            y -= 17
            line = []
        else:
            line.append(words.pop(0))
    if line:
        canvas.drawString(x, y, " ".join(line))
        y -= 17
    return y


def draw_page(canvas: Canvas, title: str, section: str, body: str, page: int) -> None:
    canvas.setFillColor("#102240")
    canvas.rect(0, 720, 612, 72, fill=1, stroke=0)
    canvas.setFillColor("#EAF4FF")
    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(54, 756, title)
    canvas.setFillColor("#6BC7B7")
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(54, 734, f"OPERATIONAL APPENDIX  |  PAGE {page}")
    canvas.setFillColor("#102240")
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(54, 678, section)
    canvas.setFont("Helvetica", 11)
    draw_wrapped(canvas, body, 54, 646, 504)
    canvas.setStrokeColor("#C8D5E4")
    canvas.line(54, 55, 558, 55)
    canvas.setFillColor("#63758D")
    canvas.setFont("Helvetica", 8)
    canvas.drawString(54, 38, "Synthetic test document - used only to verify PDF text extraction and page provenance.")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(OUTPUT), pagesize=letter, pageCompression=1)
    draw_page(
        canvas,
        "Orbital Operations Appendix",
        "Page 1 - Navigation timing",
        "The sunrise maintenance window begins at 05:40. Navigation timing changes require a signed update from the mission planner. This page intentionally contains no optical calibration assignment.",
        1,
    )
    canvas.showPage()
    draw_page(
        canvas,
        "Orbital Operations Appendix",
        "Page 2 - Optical calibration",
        "Optical calibration is led by the navigation engineer at 14:20 each Thursday. The flight director approves only the completed calibration record after the navigation engineer records the result.",
        2,
    )
    canvas.save()
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
