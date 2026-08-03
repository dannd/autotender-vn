from autotender.models.generator import GeneratorModule, verify_numeric_consistency
from autotender.schemas import ExtractedField


def _fields() -> list[ExtractedField]:
    return [
        ExtractedField(name="PACKAGE_NAME", value="Mua sắm thiết bị CNTT", confidence=0.9, source="regex"),
        ExtractedField(name="INVESTOR", value="Sở Y tế tỉnh X", confidence=0.9, source="regex"),
        ExtractedField(name="VALUE", value="5.200.000.000", confidence=0.9, source="regex"),
        ExtractedField(name="FUNDING", value="Ngân sách nhà nước năm 2026", confidence=0.9, source="regex"),
        ExtractedField(name="DURATION", value="90 ngày", confidence=0.9, source="regex"),
    ]


def test_generator_tier3_fills_slots_without_any_checkpoint():
    module = GeneratorModule()
    result = module.generate_section("chuong_III.muc_4", _fields())

    assert module.active_tier == 3
    assert "Mua sắm thiết bị CNTT" in result.text
    assert "5.200.000.000" in result.text
    assert len(result.citations) > 0


def test_generator_rejects_out_of_scope_section():
    module = GeneratorModule()
    try:
        module.generate_section("chuong_I.muc_1", _fields())
        assert False, "phải raise ValueError cho section ngoài phạm vi"
    except ValueError:
        pass


def test_verify_numeric_consistency_flags_unknown_number():
    fields = _fields()
    text = "Giá gói thầu là 5.200.000.000 đồng nhưng doanh thu yêu cầu tối thiểu 99.999.999.999 đồng."
    flags = verify_numeric_consistency(text, fields)
    assert len(flags) == 1
    assert flags[0].rule_code == "R4"
    assert "99.999.999.999" in flags[0].sentence


def test_verify_numeric_consistency_no_flag_when_numbers_match():
    fields = _fields()
    text = "Giá gói thầu là 5.200.000.000 đồng, thời gian thực hiện 90 ngày."
    flags = verify_numeric_consistency(text, fields)
    assert flags == []
