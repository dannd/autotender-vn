"""Xuất DOCX — HSMT thực tế thường được chỉnh sửa tiếp trên Word (Mục 8)."""

from __future__ import annotations

from pathlib import Path

from autotender.hitl.store import HitlStore
from autotender.models.generator import CHAPTER_TITLES
from autotender.schemas import HSMTDocument, HSMTSection

_CHAPTER_RANK = {chapter: i for i, chapter in enumerate(CHAPTER_TITLES)}


def _ordered_sections(sections: list[HSMTSection]) -> list[HSMTSection]:
    """`sections` sắp theo thứ tự chương CHÍNH THỨC (I→VIII, xem CHAPTER_TITLES), không
    theo thứ tự sinh/nạp gốc — người dùng có thể sinh Chương V trước Chương I, nhưng tài
    liệu xuất ra vẫn phải đúng thứ tự chương chuẩn. `sorted` ổn định (stable) nên thứ tự
    các mục TRONG cùng 1 chương được giữ nguyên."""
    return sorted(sections, key=lambda s: _CHAPTER_RANK.get(s.section_id.split(".")[0], len(_CHAPTER_RANK)))


def export_docx(doc: HSMTDocument, store: HitlStore, output_path: str | Path) -> Path:
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Mm, Pt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = docx.Document()
    section = document.sections[0]
    section.page_height = Mm(297)
    section.page_width = Mm(210)
    section.top_margin = Mm(20)
    section.bottom_margin = Mm(20)
    section.left_margin = Mm(30)
    section.right_margin = Mm(20)

    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(13)

    title = document.add_heading("HỒ SƠ MỜI THẦU", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = document.add_paragraph(doc.package.package_name)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].bold = True

    document.add_paragraph(f"Chủ đầu tư: {doc.package.investor}")
    document.add_paragraph(f"Số hiệu tài liệu: {doc.doc_id}")

    warning = document.add_paragraph()
    warning_run = warning.add_run(
        "Dự thảo do hệ thống hỗ trợ tạo lập — bắt buộc thẩm định và phê duyệt theo quy định "
        "pháp luật trước khi phát hành."
    )
    warning_run.bold = True
    document.add_page_break()

    approved_count, total_count = doc.approval_progress
    if approved_count < total_count:
        document.add_heading("⚠️ CẢNH BÁO: TÀI LIỆU CÒN MỤC CHƯA PHÊ DUYỆT", level=1)
        document.add_paragraph(f"{approved_count}/{total_count} mục đã phê duyệt.")
        for s in doc.sections:
            if s.status != "approved":
                document.add_paragraph(f"{s.section_id} — {s.title} ({s.status})", style="List Bullet")
        document.add_page_break()

    ordered_sections = _ordered_sections(doc.sections)

    document.add_heading("Mục lục", level=1)
    for s in ordered_sections:
        document.add_paragraph(f"{s.section_id} — {s.title}")
    document.add_page_break()

    chapters: dict[str, list] = {}
    for s in ordered_sections:
        chapter = s.section_id.split(".")[0]
        chapters.setdefault(chapter, []).append(s)

    for chapter, sections in chapters.items():
        document.add_heading(CHAPTER_TITLES.get(chapter, chapter), level=1)
        for s in sections:
            document.add_heading(f"{s.section_id}. {s.title}", level=2)
            document.add_paragraph(s.current_text)
        document.add_page_break()

    document.add_heading("Phụ lục — Nhật ký phê duyệt", level=1)
    table = document.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, name in enumerate(["Mục", "Tiêu đề", "Trạng thái", "Người duyệt", "Thời điểm"]):
        hdr[i].text = name
    for row in store.get_approval_log(doc.doc_id):
        cells = table.add_row().cells
        cells[0].text = row["section_id"]
        cells[1].text = row["title"]
        cells[2].text = row["status"]
        cells[3].text = row["approved_by"] or "-"
        cells[4].text = row["approved_at"] or "-"

    document.save(str(output_path))
    return output_path
