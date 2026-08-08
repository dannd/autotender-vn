from autotender.generation import law_classifier as law_classifier_module
from autotender.generation.claude_client import ClaudeUnavailableError
from autotender.generation.law_classifier import classify_relevant_law_ids


def test_classify_returns_valid_law_ids_from_claude_response(monkeypatch):
    monkeypatch.setattr(
        law_classifier_module, "call_claude", lambda **kwargs: "nd_45_2026_ndcp, luat_22_2023_qh15"
    )
    result = classify_relevant_law_ids("Nghị định 45/2026 quy định gì về nghiệm thu phần mềm?")
    assert result == {"nd_45_2026_ndcp", "luat_22_2023_qh15"}


def test_classify_ignores_unknown_labels_and_keeps_valid_ones(monkeypatch):
    monkeypatch.setattr(law_classifier_module, "call_claude", lambda **kwargs: "nd_214_2025_ndcp, khong_ton_tai")
    result = classify_relevant_law_ids("câu hỏi bất kỳ")
    assert result == {"nd_214_2025_ndcp"}


def test_classify_returns_none_when_no_valid_labels(monkeypatch):
    monkeypatch.setattr(law_classifier_module, "call_claude", lambda **kwargs: "không rõ")
    result = classify_relevant_law_ids("câu hỏi bất kỳ")
    assert result is None


def test_classify_returns_none_when_claude_unavailable(monkeypatch):
    def _raise(**kwargs):
        raise ClaudeUnavailableError("no api key")

    monkeypatch.setattr(law_classifier_module, "call_claude", _raise)
    result = classify_relevant_law_ids("câu hỏi bất kỳ")
    assert result is None
