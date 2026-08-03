from autotender.models.classifier import ClassifierModule


def test_classifier_tier3_detects_xay_lap():
    module = ClassifierModule()
    result = module.classify("Thi công xây dựng công trình đường giao thông nông thôn")
    assert module.active_tier == 3
    assert result.label == "xay_lap"
    assert result.label_display == "xây lắp"


def test_classifier_tier3_detects_hang_hoa():
    module = ClassifierModule()
    result = module.classify("Mua sắm thiết bị công nghệ thông tin phục vụ chuyển đổi số")
    assert result.label == "hang_hoa"


def test_classifier_tier3_default_when_no_keyword_match():
    module = ClassifierModule()
    result = module.classify("noi dung khong ro rang khong khop tu khoa nao")
    assert result.label == "hang_hoa"
    assert result.confidence == 0.2
