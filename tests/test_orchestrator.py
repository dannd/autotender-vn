from autotender.pipeline.orchestrator import Orchestrator
from autotender.schemas import TenderNotice


def test_orchestrator_full_flow_from_text_to_generated_sections():
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

    classification = orch.classify_package(text)
    assert classification.label in {"hang_hoa", "xay_lap", "tu_van", "phi_tu_van", "hon_hop"}

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
    assert len(all_sections) == 8

    chuong_iii_sections = orch.generate_chuong_iii(fields)
    assert len(chuong_iii_sections) == 4
    assert all(s.section_id.startswith("chuong_III.") for s in chuong_iii_sections)
    assert all(s.status == "draft" for s in chuong_iii_sections)
    assert all(len(s.citations) > 0 for s in chuong_iii_sections)

    chuong_v_sections = orch.generate_chuong_v(fields)
    assert len(chuong_v_sections) == 4
    assert all(s.section_id.startswith("chuong_V.") for s in chuong_v_sections)

    # Chỉ soạn Chương III (thiếu Chương V) -> phải bị gắn cờ R5 cho 4 mục Chương V còn thiếu
    incomplete_flags = orch.check_completeness(chuong_iii_sections)
    assert len(incomplete_flags) == 4
    assert all(f.rule_code == "R5" for f in incomplete_flags)

    # Soạn đủ cả Chương III + Chương V -> không còn cờ R5 nào
    complete_flags = orch.check_completeness(chuong_iii_sections + chuong_v_sections)
    assert complete_flags == []
