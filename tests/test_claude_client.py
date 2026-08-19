import pytest

import autotender.generation.llm_client as llm_client
from autotender.config import AppSettings, ClaudeBudgetConfig, ClaudeModelPricing, LLMGatewayConfig
from autotender.generation.llm_client import (
    BudgetExceededError,
    ClaudeUnavailableError,
    LLMUnavailableError,
    call_claude,
    call_llm,
    get_session_cost_usd,
    reset_session_cost_usd,
)


@pytest.fixture(autouse=True)
def _reset_session_cost():
    reset_session_cost_usd()
    yield
    reset_session_cost_usd()


def test_call_llm_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(LLMUnavailableError, match="API Key"):
        call_llm(system="s", user_prompt="u", model="claude-3-5-sonnet-20241022")


def test_call_llm_returns_text_from_response(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "fake-key-for-test")
    openai = pytest.importorskip("openai")

    class _Message:
        content = "câu trả lời giả lập"

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]
        usage = None

    class _FakeChatCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == "claude-3-5-sonnet-20241022"
            assert kwargs["messages"][0]["content"] == "s"
            assert kwargs["messages"][1]["content"] == "u"
            return _Response()

    class _FakeChat:
        completions = _FakeChatCompletions()

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.chat = _FakeChat()

    monkeypatch.setattr(llm_client, "_create_client", lambda *a, **kw: _FakeClient())

    result = call_llm(system="s", user_prompt="u", model="claude-3-5-sonnet-20241022")
    assert result == "câu trả lời giả lập"


def test_call_claude_alias_works(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "fake-key-for-test")

    class _Message:
        content = "ok alias"

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]
        usage = None

    class _FakeChatCompletions:
        def create(self, **kwargs):
            return _Response()

    class _FakeChat:
        completions = _FakeChatCompletions()

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.chat = _FakeChat()

    monkeypatch.setattr(llm_client, "_create_client", lambda *a, **kw: _FakeClient())

    result = call_claude(system="s", user_prompt="u")
    assert result == "ok alias"


def _fake_settings(usd_cap_per_process: float = 5.0) -> AppSettings:
    settings = AppSettings()
    settings.llm_gateway = LLMGatewayConfig(
        usd_cap_per_process=usd_cap_per_process,
        pricing_usd_per_mtok={
            "claude-3-5-sonnet-20241022": ClaudeModelPricing(input_usd_per_mtok=3.0, output_usd_per_mtok=15.0),
        },
    )
    settings.claude_budget = ClaudeBudgetConfig(
        usd_cap_per_process=usd_cap_per_process,
        pricing_usd_per_mtok={
            "claude-3-5-sonnet-20241022": ClaudeModelPricing(input_usd_per_mtok=3.0, output_usd_per_mtok=15.0),
        },
    )
    return settings


def test_call_llm_raises_budget_exceeded_before_calling_api(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(llm_client, "get_app_settings", lambda: _fake_settings(usd_cap_per_process=1.0))
    llm_client._session_cost_usd = 1.0

    with pytest.raises(BudgetExceededError):
        call_llm(system="s", user_prompt="u", model="claude-3-5-sonnet-20241022")


def test_budget_exceeded_is_a_claude_unavailable_error(monkeypatch):
    """Callers đã bắt `ClaudeUnavailableError` để rơi xuống tier dự phòng — hết ngân sách
    phải đi qua đúng đường đó."""
    monkeypatch.setenv("LLM_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(llm_client, "get_app_settings", lambda: _fake_settings(usd_cap_per_process=1.0))
    llm_client._session_cost_usd = 1.0

    with pytest.raises(ClaudeUnavailableError):
        call_claude(system="s", user_prompt="u", model="claude-3-5-sonnet-20241022")


def test_call_llm_records_cost_from_usage(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(llm_client, "get_app_settings", lambda: _fake_settings())

    class _Message:
        content = "ok"

    class _Choice:
        message = _Message()

    class _Usage:
        prompt_tokens = 1_000_000
        completion_tokens = 1_000_000

    class _Response:
        choices = [_Choice()]
        usage = _Usage()

    class _FakeChatCompletions:
        def create(self, **kwargs):
            return _Response()

    class _FakeChat:
        completions = _FakeChatCompletions()

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.chat = _FakeChat()

    monkeypatch.setattr(llm_client, "_create_client", lambda *a, **kw: _FakeClient())

    assert get_session_cost_usd() == 0.0
    call_llm(system="s", user_prompt="u", model="claude-3-5-sonnet-20241022")
    # 1M prompt @ $3/Mtok + 1M completion @ $15/Mtok = $18
    assert get_session_cost_usd() == pytest.approx(18.0)
