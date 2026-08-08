from datetime import datetime, timezone

from autotender.models.compliance import REQUIRED_HSMT_COMPONENTS, check_document_completeness, ComplianceModule
from autotender.schemas import HSMTSection


def _all_required_section_ids() -> list[str]:
    return [f"{chapter}.{muc}" for chapter, mucs in REQUIRED_HSMT_COMPONENTS.items() for muc in mucs]


def _section(section_id: str, text: str = "Nội dung đầy đủ, không phải placeholder.") -> HSMTSection:
    return HSMTSection(
        section_id=section_id, title=section_id, generated_text=text, status="draft",
        model_tier=3, generated_at=datetime.now(timezone.utc),
    )


def test_compliance_flags_brand_name():
    module = ComplianceModule()
    flags = module.check_text("Yêu cầu cung cấp máy chủ hãng Dell hoặc tương đương.")
    assert any(f.rule_code == "R1" for f in flags)


def test_compliance_flags_excessive_revenue_requirement():
    module = ComplianceModule()
    flags = module.check_text("Nhà thầu phải có doanh thu bình quân 5 lần giá gói thầu trong 3 năm gần nhất.")
    assert any(f.rule_code == "R2" for f in flags)


def test_compliance_flags_tailored_spec():
    module = ComplianceModule()
    flags = module.check_text("Thiết bị phải đáp ứng thông số kỹ thuật độc quyền, chỉ có sản phẩm này mới đạt được.")
    assert any(f.rule_code == "R3" for f in flags)


def test_compliance_no_flag_for_clean_text():
    module = ComplianceModule()
    flags = module.check_text("Nhà thầu cung cấp hàng hóa đáp ứng thông số kỹ thuật tối thiểu nêu tại E-HSMT.")
    assert flags == []


def test_compliance_does_not_flag_negated_rule_description():
    """Câu mô tả NGUYÊN TẮC cấm ('không được... duy nhất trên thị trường') không phải vi phạm thật."""
    module = ComplianceModule()
    flags = module.check_text(
        "Không được đưa ra thông số kỹ thuật của một sản phẩm cụ thể duy nhất trên thị trường."
    )
    assert flags == []


def test_completeness_flags_r5_for_every_missing_required_section():
    flags = check_document_completeness(sections=[])

    assert len(flags) == len(_all_required_section_ids())
    assert all(f.rule_code == "R5" for f in flags)


def test_completeness_no_flag_when_all_required_sections_present_with_real_content():
    sections = [_section(sid) for sid in _all_required_section_ids()]

    flags = check_document_completeness(sections)

    assert flags == []


def test_completeness_flags_r5_for_placeholder_only_section():
    sections = [_section(sid) for sid in _all_required_section_ids()]
    sections[0] = _section(sections[0].section_id, text="[CẦN NGƯỜI DÙNG BỔ SUNG: tên gói thầu]")

    flags = check_document_completeness(sections)

    assert len(flags) == 1
    assert flags[0].rule_code == "R5"
    assert sections[0].section_id in flags[0].explanation
