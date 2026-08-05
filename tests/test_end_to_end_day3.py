"""Test end-to-end (theo tiêu chí Ngày 3): KHLCNT mẫu -> NER -> sinh Chương III bằng template."""

from autotender.ingest.synth_document import build_synthetic_khlcnt_text
from autotender.models.generator import GeneratorModule
from autotender.models.ner import NERModule
from autotender.schemas import TenderNotice


def test_end_to_end_generate_chuong_iii_from_sample_khlcnt(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    notice = TenderNotice(
        tbmt_id="IB2601099",
        package_name="Gói thầu số 99: Mua sắm máy chủ và thiết bị lưu trữ",
        investor="Sở Thông tin và Truyền thông tỉnh Y",
        package_value=8_500_000_000,
        funding_source="Ngân sách nhà nước năm 2026",
        selection_method="đấu thầu rộng rãi trong nước",
        contract_type="Trọn gói",
        package_type="hàng hóa",
        execution_time="90 ngày",
        source_url="https://example.com",
    )

    khlcnt_text = build_synthetic_khlcnt_text(notice)

    ner = NERModule()
    fields = ner.extract(khlcnt_text)
    assert ner.active_tier == 3
    assert any(f.name == "VALUE" for f in fields)

    generator = GeneratorModule()
    section = generator.generate_section("chuong_III.muc_1", fields)

    assert generator.active_tier == 3
    assert "Mua sắm máy chủ" in section.text
    assert len(section.citations) > 0
    assert section.flags == []  # số liệu đều khớp field trích xuất, không có cảnh báo R4
