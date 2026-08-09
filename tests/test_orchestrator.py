from autotender.models.generator import SECTION_DEFINITIONS
from autotender.pipeline.orchestrator import Orchestrator
from autotender.schemas import TenderNotice


def test_orchestrator_full_flow_from_text_to_generated_sections(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    orch = Orchestrator()

    raw_text = (
        "Tên gói thầu: Mua sắm máy chủ và thiết bị lưu trữ\n"
        "Chủ đầu tư: Sở Thông tin và Truyền thông tỉnh Y\n"
        "Giá gói thầu: 8.500.000.000 đồng\n"
        "Nguồn vốn: Ngân sách nhà nước năm 2026\n"
        "Hình thức lựa chọn nhà thầu: đấu thầu rộng rãi trong nước\n"
        "Loại hợp đồng: Trọn gói\n"
        "Thời gian thực hiện hợp đồng: 90 ngày\n"
    )
    text = orch.ingest_text(raw_text)
    fields = orch.extract_fields(text)
    assert any(f.name == "VALUE" for f in fields)

    package = TenderNotice(
        tbmt_id="IB1", package_name="Mua sắm máy chủ", investor="Sở TTTT", source_url="https://x"
    )
    doc = orch.create_document("doc1", package, fields)
    assert doc.sections == []

    section = orch.generate_section("chuong_III.muc_1", fields)
    assert section.status == "draft"
    assert section.model_tier == 3
    assert len(section.citations) > 0

    all_sections = orch.generate_all_sections(fields)
    assert len(all_sections) == len(SECTION_DEFINITIONS)

    chuong_iii_sections = orch.generate_chuong_iii(fields)
    assert len(chuong_iii_sections) == 4
    assert all(s.section_id.startswith("chuong_III.") for s in chuong_iii_sections)
    assert all(s.status == "draft" for s in chuong_iii_sections)
    assert all(len(s.citations) > 0 for s in chuong_iii_sections)

    chuong_v_sections = orch.generate_chuong_v(fields)
    assert len(chuong_v_sections) == 4
    assert all(s.section_id.startswith("chuong_V.") for s in chuong_v_sections)

    # Chỉ soạn Chương III (thiếu mọi chương khác) -> phải bị gắn cờ R5 cho từng mục còn thiếu
    incomplete_flags = orch.check_completeness(chuong_iii_sections)
    assert len(incomplete_flags) == len(SECTION_DEFINITIONS) - len(chuong_iii_sections)
    assert all(f.rule_code == "R5" for f in incomplete_flags)

    # Soạn đủ cả 8 chương (đã sinh ở all_sections) -> không còn cờ R5 nào
    complete_flags = orch.check_completeness(all_sections)
    assert complete_flags == []


def test_create_document_syncs_confirmed_package_name_into_fields(monkeypatch):
    """Hồi quy: KHLCNT ghi "Tên dự án:" (không phải "Tên gói thầu:" mà regex M2 nhận diện)
    nên NER không trích được PACKAGE_NAME/INVESTOR — người dùng phải tự gõ ở form xác nhận
    (Trang 2). Trước fix, giá trị đó chỉ lưu vào `TenderNotice`, không vào `fields`, nên M5
    vẫn sinh ra placeholder dù người dùng đã sửa đúng."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    orch = Orchestrator()

    raw_text = "Tên dự án: Nâng cấp hệ thống mạng\nGiá gói thầu: 2.000.000.000 đồng\n"
    fields = orch.extract_fields(orch.ingest_text(raw_text))
    assert not any(f.name == "PACKAGE_NAME" for f in fields)
    assert not any(f.name == "INVESTOR" for f in fields)

    package = TenderNotice(
        tbmt_id="IB2", package_name="Nâng cấp hệ thống mạng", investor="Sở Y tế tỉnh Z", source_url="https://x"
    )
    doc = orch.create_document("doc2", package, fields)

    doc_package_name = next(f.value for f in doc.fields if f.name == "PACKAGE_NAME")
    doc_investor = next(f.value for f in doc.fields if f.name == "INVESTOR")
    assert doc_package_name == "Nâng cấp hệ thống mạng"
    assert doc_investor == "Sở Y tế tỉnh Z"

    section = orch.generate_section("chuong_III.muc_1", doc.fields)
    assert "[CẦN NGƯỜI DÙNG BỔ SUNG: tên gói thầu]" not in section.generated_text
    assert "[CẦN NGƯỜI DÙNG BỔ SUNG: chủ đầu tư]" not in section.generated_text
    assert "Nâng cấp hệ thống mạng" in section.generated_text
    assert "Sở Y tế tỉnh Z" in section.generated_text


def test_create_document_does_not_mutate_caller_fields_list(monkeypatch):
    """`create_document` phải trả về DANH SÁCH MỚI trong `doc.fields`, không sửa ngay trên
    `fields` mà caller truyền vào — tránh side-effect bất ngờ nếu caller còn dùng lại biến
    `fields` gốc sau khi gọi (vd để sinh mục ở nơi khác, xem test phía trên)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    orch = Orchestrator()

    fields = orch.extract_fields(orch.ingest_text("Tên dự án: X\n"))
    original_len = len(fields)
    package = TenderNotice(tbmt_id="IB3", package_name="X", investor="Y", source_url="https://x")

    orch.create_document("doc3", package, fields)

    assert len(fields) == original_len
    assert not any(f.name == "PACKAGE_NAME" for f in fields)
