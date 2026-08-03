from autotender.models.compliance import ComplianceModule


def test_compliance_tier3_flags_brand_name():
    module = ComplianceModule()
    flags = module.check_text("Yêu cầu cung cấp máy chủ hãng Dell hoặc tương đương.")
    assert module.active_tier == 3
    assert any(f.rule_code == "R1" for f in flags)


def test_compliance_tier3_flags_excessive_revenue_requirement():
    module = ComplianceModule()
    flags = module.check_text("Nhà thầu phải có doanh thu bình quân 5 lần giá gói thầu trong 3 năm gần nhất.")
    assert any(f.rule_code == "R2" for f in flags)


def test_compliance_tier3_flags_tailored_spec():
    module = ComplianceModule()
    flags = module.check_text("Thiết bị phải đáp ứng thông số kỹ thuật độc quyền, chỉ có sản phẩm này mới đạt được.")
    assert any(f.rule_code == "R3" for f in flags)


def test_compliance_tier3_no_flag_for_clean_text():
    module = ComplianceModule()
    flags = module.check_text("Nhà thầu cung cấp hàng hóa đáp ứng thông số kỹ thuật tối thiểu nêu tại E-HSMT.")
    assert flags == []


def test_compliance_tier3_does_not_flag_negated_rule_description():
    """Câu mô tả NGUYÊN TẮC cấm ('không được... duy nhất trên thị trường') không phải vi phạm thật."""
    module = ComplianceModule()
    flags = module.check_text(
        "Không được đưa ra thông số kỹ thuật của một sản phẩm cụ thể duy nhất trên thị trường."
    )
    assert flags == []

