from pathlib import Path

from autotender.ingest.docx_reader import extract_docx_text
from autotender.ingest.pdf_reader import extract_pdf_text


def _make_sample_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    # Font mặc định "helv" của PyMuPDF không có glyph cho dấu tiếng Việt tổ hợp sẵn,
    # nên dùng Arial (có sẵn trên Windows) để test phản ánh đúng nội dung tiếng Việt thật.
    page.insert_text(
        (72, 72),
        "Nguyễn Thị Hường — gói thầu số 05: Mua sắm thiết bị",
        fontfile="C:/Windows/Fonts/arial.ttf",
        fontname="F0",
    )
    doc.save(str(path))
    doc.close()


def _make_sample_docx(path: Path) -> None:
    import docx

    document = docx.Document()
    document.add_paragraph("Nguyễn Thị Hường — gói thầu số 05: Mua sắm thiết bị")
    document.save(str(path))


def test_extract_pdf_text_reads_vietnamese_diacritics(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _make_sample_pdf(pdf_path)

    result = extract_pdf_text(pdf_path)

    assert len(result.pages) == 1
    assert "Nguyễn Thị Hường" in result.full_text
    assert result.any_ocr_used is False


def test_extract_pdf_text_flags_scan_page_without_ocr(tmp_path):
    import fitz

    pdf_path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    result = extract_pdf_text(pdf_path)
    assert result.any_ocr_used is True
    assert "CẦN NGƯỜI DÙNG BỔ SUNG" in result.full_text


def test_extract_docx_text_reads_vietnamese_diacritics(tmp_path):
    docx_path = tmp_path / "sample.docx"
    _make_sample_docx(docx_path)

    text = extract_docx_text(docx_path)
    assert "Nguyễn Thị Hường" in text
