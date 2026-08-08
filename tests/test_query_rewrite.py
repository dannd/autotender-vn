from autotender.generation import query_rewrite as query_rewrite_module
from autotender.generation.claude_client import ClaudeUnavailableError
from autotender.generation.query_rewrite import rewrite_query


def test_rewrite_query_returns_claude_output(monkeypatch):
    monkeypatch.setattr(
        query_rewrite_module, "call_claude", lambda **kwargs: "Thủ tục lập hồ sơ mời thầu."
    )
    result = rewrite_query("thầu qua mạng thế nào")
    assert result == "Thủ tục lập hồ sơ mời thầu."


def test_rewrite_query_falls_back_to_original_on_claude_unavailable(monkeypatch):
    def _raise(**kwargs):
        raise ClaudeUnavailableError("no api key")

    monkeypatch.setattr(query_rewrite_module, "call_claude", _raise)
    result = rewrite_query("thầu qua mạng thế nào")
    assert result == "thầu qua mạng thế nào"


def test_rewrite_query_falls_back_to_original_on_empty_response(monkeypatch):
    monkeypatch.setattr(query_rewrite_module, "call_claude", lambda **kwargs: "   ")
    result = rewrite_query("thầu qua mạng thế nào")
    assert result == "thầu qua mạng thế nào"
