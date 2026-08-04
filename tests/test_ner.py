from autotender.models.ner import NERModule

SAMPLE_TEXT = """
Tên gói thầu: Mua sắm thiết bị công nghệ thông tin phục vụ chuyển đổi số
Chủ đầu tư: Sở Giáo dục và Đào tạo tỉnh Nghệ An
Giá gói thầu: 5.200.000.000 đồng
Nguồn vốn: Ngân sách nhà nước năm 2026
Hình thức lựa chọn nhà thầu: đấu thầu rộng rãi trong nước
Loại hợp đồng: Trọn gói
Thời gian thực hiện hợp đồng: 90 ngày
"""


def test_ner_tier3_extracts_expected_fields_without_any_checkpoint():
    module = NERModule()
    fields = module.extract(SAMPLE_TEXT)

    assert module.active_tier == 3
    names = {f.name for f in fields}
    assert "VALUE" in names
    assert "FUNDING" in names
    assert "INVESTOR" in names

    value_field = next(f for f in fields if f.name == "VALUE")
    assert "5.200.000.000" in value_field.value
    assert value_field.source == "regex"
    assert 0 < value_field.confidence <= 1


def test_ner_tier3_does_not_extract_missing_info_placeholder_as_a_value():
    """Regression: regex DURATION từng khớp nhầm chính placeholder báo thiếu thông tin
    (phát hiện khi đánh giá trên 232 bản ghi thật — xem reports/metrics.json)."""
    text = (
        "Tên gói thầu: Mua sắm thiết bị\n"
        "Chủ đầu tư: Sở Y tế\n"
        "Thời gian thực hiện hợp đồng: [CẦN NGƯỜI DÙNG BỔ SUNG: thời gian thực hiện]\n"
    )
    module = NERModule()
    fields = module.extract(text)
    assert all("[CẦN NGƯỜI DÙNG BỔ SUNG" not in f.value for f in fields)
    assert not any(f.name == "DURATION" for f in fields)
