from pathlib import Path

from autotender.ingest.docx_reader import extract_docx_text
from autotender.ingest.pdf_reader import extract_pdf_text


def _get_test_font() -> str | None:
    bundled = Path(__file__).resolve().parent.parent / "src" / "autotender" / "export" / "fonts" / "DejaVuSans.ttf"
    if bundled.exists():
        return str(bundled)
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def _make_sample_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    fontfile = _get_test_font()
    if fontfile:
        page.insert_text(
            (72, 72),
            "Nguyễn Thị Hường — gói thầu số 05: Mua sắm thiết bị",
            fontfile=fontfile,
            fontname="F0",
        )
    else:
        page.insert_text(
            (72, 72),
            "Nguyễn Thị Hường — gói thầu số 05: Mua sắm thiết bị",
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
