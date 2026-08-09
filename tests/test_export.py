import json
from datetime import datetime

from autotender.export.docx import export_docx
from autotender.export.pdf import export_pdf, render_html
from autotender.export.print import build_print_html
from autotender.hitl.store import HitlStore
from autotender.schemas import HSMTDocument, HSMTSection, TenderNotice


def _make_doc(fully_approved: bool = False) -> HSMTDocument:
    now = datetime.now()
    package = TenderNotice(
        tbmt_id="IB1", package_name="Nguyễn Thị Hường — gói thầu số 05: Mua sắm thiết bị",
        investor="Sở Y tế", source_url="https://x",
    )
    status = "approved" if fully_approved else "draft"
    sections = [
        HSMTSection(
            section_id="chuong_III.muc_1", title="Tiêu chuẩn năng lực", generated_text="Nội dung mục 1.",
            status=status, model_tier=3, generated_at=now,
            approved_by="reviewer1" if fully_approved else None, approved_at=now if fully_approved else None,
        ),
        HSMTSection(
            section_id="chuong_V.muc_1", title="Phạm vi cung cấp", generated_text="Nội dung mục 2.",
            status=status, model_tier=3, generated_at=now,
            approved_by="reviewer1" if fully_approved else None, approved_at=now if fully_approved else None,
        ),
    ]
    return HSMTDocument(doc_id="doc_export_test", package=package, sections=sections, created_at=now, updated_at=now)


def _make_store(tmp_path, doc: HSMTDocument) -> HitlStore:
    store = HitlStore(tmp_path / "test.db")
    store.save_document(doc)
    return store


def test_render_html_contains_vietnamese_and_warning_when_not_approved(tmp_path):
    doc = _make_doc(fully_approved=False)
    store = _make_store(tmp_path, doc)
    html = render_html(doc, store.get_approval_log(doc.doc_id))

    assert "Nguyễn Thị Hường" in html
    assert "Cảnh báo" in html
    assert "Dự thảo do hệ thống hỗ trợ tạo lập" in html
    store.close()


def test_render_html_no_warning_when_fully_approved(tmp_path):
    doc = _make_doc(fully_approved=True)
    store = _make_store(tmp_path, doc)
    html = render_html(doc, store.get_approval_log(doc.doc_id))

    assert "Cảnh báo" not in html
    store.close()


def test_export_pdf_creates_valid_file_with_fallback(tmp_path):
    doc = _make_doc(fully_approved=True)
    store = _make_store(tmp_path, doc)
    out_path = tmp_path / "out.pdf"

    result = export_pdf(doc, store, out_path)

    assert result.exists()
    assert result.stat().st_size > 0
    with open(result, "rb") as f:
        assert f.read(5) == b"%PDF-"
    store.close()


def test_export_docx_creates_valid_file(tmp_path):
    doc = _make_doc(fully_approved=True)
    store = _make_store(tmp_path, doc)
    out_path = tmp_path / "out.docx"

    result = export_docx(doc, store, out_path)

    assert result.exists()
    assert result.stat().st_size > 0
    with open(result, "rb") as f:
        assert f.read(2) == b"PK"  # docx là file zip
    store.close()


def test_build_print_html_embeds_document_and_calls_print():
    html_snippet = build_print_html("<html><body>Test</body></html>")
    assert "print-frame" in html_snippet
    assert ".print()" in html_snippet


def test_build_print_html_escapes_backtick_and_template_expression():
    """Hồi quy XSS: nội dung tài liệu (do người dùng nhập/sửa) từng bị nhét thẳng vào JS
    template literal (backtick) — một backtick hoặc `${...}` trong nội dung có thể thoát
    khỏi chuỗi và chèn mã JS tuỳ ý. Nội dung giờ phải nằm trọn trong 1 chuỗi JSON round-trip
    được (không phá vỡ cấu trúc script bao ngoài)."""
    malicious = "Tên gói thầu`; alert(document.cookie); const x = `${1+1}"
    html_snippet = build_print_html(malicious)

    # `json.dumps` là cách mã hoá thật sự dùng trong code — nội dung độc hại phải xuất hiện
    # ĐÚNG NGUYÊN VĂN dưới dạng chuỗi JSON đã escape đó (chứng minh nó bị "giam" trong 1
    # chuỗi hợp lệ, không phá vỡ cấu trúc script bao ngoài), thay vì đứng trần dạng mã JS.
    assert json.dumps(malicious) in html_snippet
    assert "JSON.parse(" in html_snippet


def test_build_print_html_escapes_closing_script_tag():
    malicious = "</script><script>alert(1)</script>"
    html_snippet = build_print_html(malicious)
    assert "</script><script>alert(1)</script>" not in html_snippet


def test_export_pdf_renders_vietnamese_diacritics_correctly(tmp_path):
    """Test bắt buộc theo Mục 8 SPEC: chuỗi tiếng Việt phải hiển thị đúng 100% dấu."""
    import fitz

    now = datetime.now()
    test_string = "Nguyễn Thị Hường — gói thầu số 05: Mua sắm thiết bị"
    package = TenderNotice(tbmt_id="IB05", package_name=test_string, investor="Sở Y tế", source_url="https://x")
    sections = [
        HSMTSection(
            section_id="chuong_III.muc_1", title="Test", generated_text=test_string,
            status="approved", model_tier=3, generated_at=now, approved_by="x", approved_at=now,
        )
    ]
    doc = HSMTDocument(doc_id="doc_font_test", package=package, sections=sections, created_at=now, updated_at=now)
    store = _make_store(tmp_path, doc)
    out_path = tmp_path / "font_test.pdf"

    export_pdf(doc, store, out_path)

    pdf = fitz.open(str(out_path))
    extracted_text = pdf[0].get_text()
    assert test_string in extracted_text
    store.close()
