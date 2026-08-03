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
