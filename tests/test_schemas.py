from datetime import datetime

from autotender.schemas import HSMTDocument, HSMTSection, TenderNotice


def test_tender_notice_minimal():
    tn = TenderNotice(
        tbmt_id="IB1",
        package_name="Test",
        investor="Inv",
        source_url="https://example.com",
    )
    assert tn.currency == "VND"
    assert tn.attachments == []


def test_hsmt_section_current_text_prefers_edited():
    sec = HSMTSection(
        section_id="chuong_III.1",
        title="Test",
        generated_text="ban goc",
        edited_text="ban da sua",
        model_tier=3,
        generated_at=datetime.now(),
    )
    assert sec.current_text == "ban da sua"


def test_hsmt_section_current_text_falls_back_to_generated():
    sec = HSMTSection(
        section_id="chuong_III.1",
        title="Test",
        generated_text="ban goc",
        model_tier=3,
        generated_at=datetime.now(),
    )
    assert sec.current_text == "ban goc"


def test_hsmt_document_approval_progress():
    tn = TenderNotice(tbmt_id="IB1", package_name="T", investor="I", source_url="https://x")
    now = datetime.now()
    sections = [
        HSMTSection(section_id="a", title="A", generated_text="x", status="approved", model_tier=3, generated_at=now),
        HSMTSection(section_id="b", title="B", generated_text="y", status="draft", model_tier=3, generated_at=now),
    ]
    doc = HSMTDocument(doc_id="d1", package=tn, sections=sections, created_at=now, updated_at=now)
    assert doc.approval_progress == (1, 2)
    assert doc.is_fully_approved is False
