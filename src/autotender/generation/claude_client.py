"""Wrapper tương thích ngược quanh `autotender.generation.llm_client`.

Mọi lệnh gọi từ `claude_client` được tự động chuyển tiếp tới `llm_client`
(Universal OpenAI-compatible Gateway, hỗ trợ WokuShop, Anthropic, OpenAI, DeepSeek).
"""

from __future__ import annotations

from autotender.generation.llm_client import (
    BudgetExceededError,
    ClaudeUnavailableError,
    LLMUnavailableError,
    call_claude,
    call_llm,
    get_api_key,
    get_base_url,
    get_session_cost_usd,
    is_configured,
    reset_session_cost_usd,
)

__all__ = [
    "BudgetExceededError",
    "ClaudeUnavailableError",
    "LLMUnavailableError",
    "call_claude",
    "call_llm",
    "get_api_key",
    "get_base_url",
    "get_session_cost_usd",
    "is_configured",
    "reset_session_cost_usd",
]
