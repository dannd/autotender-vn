import pytest

import autotender.generation.claude_client as claude_client
from autotender.config import AppSettings, ClaudeBudgetConfig, ClaudeModelPricing
from autotender.generation.claude_client import BudgetExceededError, ClaudeUnavailableError, call_claude, get_session_cost_usd


@pytest.fixture(autouse=True)
def _reset_session_cost():
    """`_session_cost_usd` là biến toàn cục cấp module (cố ý — cộng dồn theo process, xem
    docstring `ClaudeBudgetConfig`) nên phải reset giữa các test để tránh rò rỉ trạng thái."""
    claude_client._session_cost_usd = 0.0
    yield
    claude_client._session_cost_usd = 0.0


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


def _fake_settings(usd_cap_per_process: float = 5.0) -> AppSettings:
    settings = AppSettings()
    settings.claude_budget = ClaudeBudgetConfig(
        usd_cap_per_process=usd_cap_per_process,
        pricing_usd_per_mtok={
            "claude-sonnet-5": ClaudeModelPricing(input_usd_per_mtok=3.0, output_usd_per_mtok=15.0),
        },
    )
    return settings


def test_call_claude_raises_budget_exceeded_before_calling_api(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(claude_client, "get_app_settings", lambda: _fake_settings(usd_cap_per_process=1.0))
    claude_client._session_cost_usd = 1.0

    with pytest.raises(BudgetExceededError):
        call_claude(system="s", user_prompt="u", model="claude-sonnet-5")


def test_budget_exceeded_is_a_claude_unavailable_error(monkeypatch):
    """Callers đã bắt `ClaudeUnavailableError` để rơi xuống tier dự phòng — hết ngân sách
    phải đi qua đúng đường đó, không cần sửa thêm ở nơi gọi."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(claude_client, "get_app_settings", lambda: _fake_settings(usd_cap_per_process=1.0))
    claude_client._session_cost_usd = 1.0

    with pytest.raises(ClaudeUnavailableError):
        call_claude(system="s", user_prompt="u", model="claude-sonnet-5")


def test_call_claude_records_cost_from_usage(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(claude_client, "get_app_settings", lambda: _fake_settings())
    anthropic = pytest.importorskip("anthropic")

    class _Block:
        type = "text"
        text = "ok"

    class _Usage:
        input_tokens = 1_000_000
        output_tokens = 1_000_000

    class _Response:
        content = [_Block()]
        usage = _Usage()

    class _FakeMessages:
        def create(self, **kwargs):
            return _Response()

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.messages = _FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)

    assert get_session_cost_usd() == 0.0
    call_claude(system="s", user_prompt="u", model="claude-sonnet-5")
    # 1M input @ $3/Mtok + 1M output @ $15/Mtok = $18
    assert get_session_cost_usd() == pytest.approx(18.0)


def test_call_claude_skips_cost_for_response_without_usage(monkeypatch):
    """Response giả lập (test double) không có `usage` — không được đoán mò chi phí,
    chỉ đơn giản bỏ qua (xem `_record_cost_and_check_budget`)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(claude_client, "get_app_settings", lambda: _fake_settings())
    anthropic = pytest.importorskip("anthropic")

    class _Block:
        type = "text"
        text = "ok"

    class _Response:
        content = [_Block()]

    class _FakeMessages:
        def create(self, **kwargs):
            return _Response()

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.messages = _FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)

    call_claude(system="s", user_prompt="u", model="claude-sonnet-5")
    assert get_session_cost_usd() == 0.0


def test_call_claude_skips_cost_for_unpriced_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(claude_client, "get_app_settings", lambda: _fake_settings())
    anthropic = pytest.importorskip("anthropic")

    class _Block:
        type = "text"
        text = "ok"

    class _Usage:
        input_tokens = 1_000_000
        output_tokens = 1_000_000

    class _Response:
        content = [_Block()]
        usage = _Usage()

    class _FakeMessages:
        def create(self, **kwargs):
            return _Response()

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.messages = _FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)

    call_claude(system="s", user_prompt="u", model="model-khong-co-trong-bang-gia")
    assert get_session_cost_usd() == 0.0
