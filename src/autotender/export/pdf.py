"""Xuất PDF theo đúng thể thức văn bản hành chính Việt Nam (Mục 8).

Một template dùng chung cho preview HTML + in trực tiếp + xuất PDF (WeasyPrint).
Nếu WeasyPrint lỗi (rủi ro đã biết trên Windows do thiếu GTK, Mục 14), tự động
fallback sang ReportLab với font DejaVu Sans nhúng sẵn (hỗ trợ đầy đủ dấu tiếng Việt).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from autotender.hitl.store import HitlStore
from autotender.schemas import HSMTDocument
from autotender.utils.logging import get_logger

logger = get_logger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
FONTS_DIR = Path(__file__).parent / "fonts"

_CHAPTER_TITLES = {
    "chuong_III": "Chương III — Tiêu chuẩn đánh giá E-HSDT",
    "chuong_V": "Chương V — Yêu cầu về kỹ thuật",
}


def _group_by_chapter(doc: HSMTDocument) -> dict[str, list]:
    chapters: dict[str, list] = {}
    for s in doc.sections:
        chapter = s.section_id.split(".")[0]
        chapters.setdefault(chapter, []).append(s)
    return chapters


def render_html(doc: HSMTDocument, approval_log: list[dict], show_watermark: bool = True) -> str:
    """Render HTML dùng CHUNG cho preview trên màn hình, in trực tiếp và xuất PDF."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(["html"])
    )
    template = env.get_template("hsmt.html.j2")
    css = (TEMPLATES_DIR / "hsmt.css").read_text(encoding="utf-8")
    approved_count, total_count = doc.approval_progress

    return template.render(
        doc=doc,
        css=css,
        chapters=_group_by_chapter(doc),
        chapter_titles=_CHAPTER_TITLES,
        approval_log=approval_log,
        approved_count=approved_count,
        total_count=total_count,
        generated_date=date.today().isoformat(),
        show_watermark=show_watermark,
    )


def export_pdf(doc: HSMTDocument, store: HitlStore, output_path: str | Path, show_watermark: bool = True) -> Path:
    """Xuất PDF. Thử WeasyPrint trước, tự động fallback ReportLab nếu lỗi."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    approval_log = store.get_approval_log(doc.doc_id)
    html = render_html(doc, approval_log, show_watermark)

    try:
        from weasyprint import HTML

        HTML(string=html, base_url=str(TEMPLATES_DIR)).write_pdf(str(output_path))
        logger.info("Xuất PDF bằng WeasyPrint: %s", output_path)
        return output_path
    except Exception as e:  # noqa: BLE001 — lỗi phổ biến trên Windows do thiếu GTK (Mục 14)
        logger.warning("WeasyPrint thất bại (%s), fallback sang ReportLab.", e)

    _export_pdf_reportlab(doc, approval_log, output_path, show_watermark)
    logger.info("Xuất PDF bằng ReportLab (fallback): %s", output_path)
    return output_path


def _export_pdf_reportlab(doc: HSMTDocument, approval_log: list[dict], output_path: Path, show_watermark: bool) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        NextPageTemplate,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    pdfmetrics.registerFont(TTFont("VN", str(FONTS_DIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("VN-Bold", str(FONTS_DIR / "DejaVuSans-Bold.ttf")))

    styles = {
        "title": ParagraphStyle("title", fontName="VN-Bold", fontSize=18, alignment=1, spaceAfter=12),
        "h2": ParagraphStyle("h2", fontName="VN-Bold", fontSize=14, alignment=1, spaceBefore=12, spaceAfter=8),
        "h3": ParagraphStyle("h3", fontName="VN-Bold", fontSize=13, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("body", fontName="VN", fontSize=13, leading=19, alignment=4),
        "warning": ParagraphStyle("warning", fontName="VN-Bold", fontSize=11, textColor="red"),
    }

    story = []
    story.append(Paragraph("HỒ SƠ MỜI THẦU", styles["title"]))
    story.append(Paragraph(doc.package.package_name, styles["h2"]))
    story.append(Paragraph(f"Chủ đầu tư: {doc.package.investor}", styles["body"]))
    story.append(Paragraph(f"Số hiệu tài liệu: {doc.doc_id}", styles["body"]))
    story.append(Spacer(1, 10 * mm))
    story.append(
        Paragraph(
            "Dự thảo do hệ thống hỗ trợ tạo lập — bắt buộc thẩm định và phê duyệt theo quy định "
            "pháp luật trước khi phát hành.",
            styles["warning"],
        )
    )
    story.append(PageBreak())

    approved_count, total_count = doc.approval_progress
    if approved_count < total_count:
        story.append(Paragraph("⚠️ CẢNH BÁO: TÀI LIỆU CÒN MỤC CHƯA PHÊ DUYỆT", styles["warning"]))
        story.append(Paragraph(f"{approved_count}/{total_count} mục đã phê duyệt.", styles["body"]))
        for s in doc.sections:
            if s.status != "approved":
                story.append(Paragraph(f"• {s.section_id} — {s.title} ({s.status})", styles["body"]))
        story.append(PageBreak())

    story.append(Paragraph("MỤC LỤC", styles["h2"]))
    for s in doc.sections:
        story.append(Paragraph(f"{s.section_id} — {s.title}", styles["body"]))
    story.append(PageBreak())

    chapters = _group_by_chapter(doc)
    for chapter, sections in chapters.items():
        story.append(Paragraph(_CHAPTER_TITLES.get(chapter, chapter), styles["h2"]))
        for s in sections:
            story.append(Paragraph(f"{s.section_id}. {s.title}", styles["h3"]))
            story.append(Paragraph(s.current_text.replace("\n", "<br/>"), styles["body"]))
            story.append(Spacer(1, 4 * mm))
        story.append(PageBreak())

    story.append(Paragraph("PHỤ LỤC — NHẬT KÝ PHÊ DUYỆT", styles["h2"]))
    table_data = [["Mục", "Tiêu đề", "Trạng thái", "Người duyệt", "Thời điểm"]]
    for row in approval_log:
        table_data.append(
            [row["section_id"], row["title"], row["status"], row["approved_by"] or "-", row["approved_at"] or "-"]
        )
    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "VN"),
                ("FONTNAME", (0, 0), (-1, 0), "VN-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, "black"),
            ]
        )
    )
    story.append(table)

    document = SimpleDocTemplate(str(output_path), pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm, leftMargin=30 * mm, rightMargin=20 * mm)
    document.build(story)
