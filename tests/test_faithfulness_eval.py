import json

import pytest

from autotender.eval import faithfulness_eval as fe_module
from autotender.eval.faithfulness_eval import judge_faithfulness
from autotender.generation.claude_client import ClaudeUnavailableError
from autotender.schemas import RetrievedChunk


def _citation(text: str = "Hồ sơ mời thầu bao gồm chỉ dẫn nhà thầu.") -> RetrievedChunk:
    return RetrievedChunk(chunk_id="c1", text=text, source_doc="Điều 44", score=1.0, law_id="x", dieu_so=44)


def test_judge_faithfulness_parses_valid_json_response(monkeypatch):
    fake_response = json.dumps(
        {"faithfulness": 0.9, "completeness": 0.8, "unsupported_claims": [], "reasoning": "Khớp căn cứ."}
    )
    monkeypatch.setattr(fe_module, "call_claude", lambda **kwargs: fake_response)

    result = judge_faithfulness("Hồ sơ mời thầu gồm gì?", [_citation()], "Hồ sơ mời thầu gồm chỉ dẫn nhà thầu.")

    assert result.faithfulness == 0.9
    assert result.completeness == 0.8
    assert result.unsupported_claims == []


def test_judge_faithfulness_flags_unsupported_claims(monkeypatch):
    fake_response = json.dumps(
        {
            "faithfulness": 0.2,
            "completeness": 0.5,
            "unsupported_claims": ["Giá gói thầu 999 tỷ đồng (không có trong context)"],
            "reasoning": "Văn bản bịa số liệu.",
        }
    )
    monkeypatch.setattr(fe_module, "call_claude", lambda **kwargs: fake_response)

    result = judge_faithfulness("q", [_citation()], "text bịa đặt")

    assert result.faithfulness == 0.2
    assert len(result.unsupported_claims) == 1


def test_judge_faithfulness_raises_on_invalid_json(monkeypatch):
    monkeypatch.setattr(fe_module, "call_claude", lambda **kwargs: "không phải JSON")

    with pytest.raises(ClaudeUnavailableError):
        judge_faithfulness("q", [_citation()], "text")


def test_judge_faithfulness_raises_on_missing_fields(monkeypatch):
    monkeypatch.setattr(fe_module, "call_claude", lambda **kwargs: json.dumps({"faithfulness": 0.5}))

    with pytest.raises(ClaudeUnavailableError):
        judge_faithfulness("q", [_citation()], "text")


def test_judge_faithfulness_propagates_claude_unavailable(monkeypatch):
    def _raise(**kwargs):
        raise ClaudeUnavailableError("no key")

    monkeypatch.setattr(fe_module, "call_claude", _raise)

    with pytest.raises(ClaudeUnavailableError):
        judge_faithfulness("q", [_citation()], "text")
