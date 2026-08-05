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


def test_call_claude_omits_temperature_by_default(monkeypatch):
    """Model mới (claude-sonnet-5) từ chối request có `temperature` tường minh (lỗi 400
    thật, xác nhận khi chạy live) — mặc định KHÔNG được gửi tham số này."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    anthropic = pytest.importorskip("anthropic")

    captured = {}

    class _Block:
        type = "text"
        text = "ok"

    class _Response:
        content = [_Block()]

    class _FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Response()

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.messages = _FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)

    call_claude(system="s", user_prompt="u", model="claude-sonnet-5")
    assert "temperature" not in captured


def test_call_claude_disables_thinking_by_default(monkeypatch):
    """Phát hiện thực tế khi chạy live: claude-sonnet-5 bật extended thinking mặc định,
    có thể tiêu hết `max_tokens` bằng thinking trước khi sinh text (response chỉ có block
    `thinking`) — mặc định phải tắt để kết quả ổn định."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    anthropic = pytest.importorskip("anthropic")

    captured = {}

    class _Block:
        type = "text"
        text = "ok"

    class _Response:
        content = [_Block()]

    class _FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Response()

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.messages = _FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)

    call_claude(system="s", user_prompt="u", model="claude-sonnet-5")
    assert captured["thinking"] == {"type": "disabled"}

    captured.clear()
    call_claude(system="s", user_prompt="u", model="claude-sonnet-5", disable_thinking=False)
    assert "thinking" not in captured


def test_call_claude_raises_informative_error_when_only_thinking_block(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    anthropic = pytest.importorskip("anthropic")

    class _ThinkingBlock:
        type = "thinking"

    class _Response:
        content = [_ThinkingBlock()]
        stop_reason = "max_tokens"

    class _FakeMessages:
        def create(self, **kwargs):
            return _Response()

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.messages = _FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)

    with pytest.raises(ClaudeUnavailableError, match="max_tokens"):
        call_claude(system="s", user_prompt="u", model="claude-sonnet-5")


def test_call_claude_wraps_api_errors(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    anthropic = pytest.importorskip("anthropic")

    class _FailingClient:
        def __init__(self, *a, **kw):
            raise RuntimeError("kết nối thất bại")

    monkeypatch.setattr(anthropic, "Anthropic", _FailingClient)

    with pytest.raises(ClaudeUnavailableError):
        call_claude(system="s", user_prompt="u", model="claude-sonnet-5")
