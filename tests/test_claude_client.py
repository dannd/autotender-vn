import pytest

from autotender.generation.claude_client import ClaudeUnavailableError, call_claude


def test_call_claude_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ClaudeUnavailableError, match="ANTHROPIC_API_KEY"):
        call_claude(system="s", user_prompt="u", model="claude-sonnet-5")


def test_call_claude_returns_text_from_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    anthropic = pytest.importorskip("anthropic")

    class _Block:
        type = "text"
        text = "câu trả lời giả lập"

    class _Response:
        content = [_Block()]

    class _FakeMessages:
        def create(self, **kwargs):
            assert kwargs["model"] == "claude-sonnet-5"
            assert kwargs["system"] == "s"
            return _Response()

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.messages = _FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)

    result = call_claude(system="s", user_prompt="u", model="claude-sonnet-5")
    assert result == "câu trả lời giả lập"


def test_call_claude_wraps_api_errors(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    anthropic = pytest.importorskip("anthropic")

    class _FailingClient:
        def __init__(self, *a, **kw):
            raise RuntimeError("kết nối thất bại")

    monkeypatch.setattr(anthropic, "Anthropic", _FailingClient)

    with pytest.raises(ClaudeUnavailableError):
        call_claude(system="s", user_prompt="u", model="claude-sonnet-5")
