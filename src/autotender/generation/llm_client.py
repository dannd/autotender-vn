"""Universal OpenAI-compatible LLM Gateway Client (WokuShop / OpenAI / DeepSeek / vLLM).

Hỗ trợ kết nối qua endpoint chuẩn OpenAI (`https://llm.wokushop.com/v1`, OpenAI, vLLM hoặc on-premise).
Tự động nạp API key từ các biến môi trường:
- `LLM_API_KEY` (ưu tiên 1)
- `OPENAI_API_KEY` (ưu tiên 2)
- `ANTHROPIC_API_KEY` (ưu tiên 3, hỗ trợ tương thích ngược)

Hỗ trợ:
- Cấu hình linh hoạt URL qua `LLM_BASE_URL` (mặc định: `https://llm.wokushop.com/v1`)
- Cơ chế Exponential Backoff Retry với thư viện `tenacity`
- Quản lý và kiểm soát trần chi phí `_session_cost_usd` chống chi tiêu ngoài tầm kiểm soát
- Graceful Degradation: trả về ngoại lệ chuẩn hóa `LLMUnavailableError` để tầng nghiệp vụ fallback
"""

from __future__ import annotations

import os
import threading
from typing import Any

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from autotender.config import PROJECT_ROOT, get_app_settings
from autotender.utils.logging import get_logger

load_dotenv(PROJECT_ROOT / ".env")

logger = get_logger(__name__)


class LLMUnavailableError(Exception):
    """Không gọi được LLM API (thiếu key, lỗi mạng, lỗi API...) — tầng gọi cần bắt exception này để fallback."""


class ClaudeUnavailableError(LLMUnavailableError):
    """Alias tương thích ngược với mã nguồn cũ."""


class BudgetExceededError(ClaudeUnavailableError):
    """Đã chạm trần ngân sách LLM cho phiên chạy process."""


_cost_lock = threading.Lock()
_session_cost_usd = 0.0


def get_session_cost_usd() -> float:
    """Tổng chi phí LLM ước tính đã sử dụng từ lúc process khởi động."""
    with _cost_lock:
        return _session_cost_usd


def reset_session_cost_usd() -> None:
    """Reset bộ đếm chi phí (dùng trong test)."""
    global _session_cost_usd
    with _cost_lock:
        _session_cost_usd = 0.0


def get_api_key() -> str | None:
    """Lấy API Key từ các biến môi trường hỗ trợ."""
    return (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )


def get_base_url() -> str:
    """Lấy Base URL của Gateway (mặc định WokuShop)."""
    settings = get_app_settings()
    return os.environ.get("LLM_BASE_URL") or settings.llm_gateway.base_url


def is_configured() -> bool:
    """Kiểm tra nhanh xem đã có API Key cấu hình hay chưa."""
    return bool(get_api_key())


def _record_cost_and_check_budget(model: str, usage: Any) -> None:
    """Cộng dồn chi phí request vào _session_cost_usd."""
    global _session_cost_usd
    if usage is None:
        return

    settings = get_app_settings()
    pricing = settings.llm_gateway.pricing_usd_per_mtok.get(model) or settings.claude_budget.pricing_usd_per_mtok.get(model)
    if pricing is None:
        logger.debug("Chưa cấu hình đơn giá cho model '%s' — bỏ qua tính chi phí.", model)
        return

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
    cost = (prompt_tokens / 1_000_000) * pricing.input_usd_per_mtok + (completion_tokens / 1_000_000) * pricing.output_usd_per_mtok

    with _cost_lock:
        _session_cost_usd += cost


def _create_client(api_key: str, base_url: str, timeout: int = 60):
    """Tạo client OpenAI tương thích."""
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def call_llm(
    system: str,
    user_prompt: str,
    model: str | None = None,
    max_tokens: int = 1536,
    temperature: float = 0.0,
    disable_thinking: bool = True,
) -> str:
    """Gọi LLM qua OpenAI-compatible API Gateway với cơ chế bảo vệ ngân sách và retry.

    Parameters
    ----------
    system : str
        System prompt định hình vai trò và nguyên tắc trả lời.
    user_prompt : str
        Nội dung câu hỏi hoặc yêu cầu của người dùng.
    model : str | None
        Tên mô hình (mặc định lấy từ cấu hình: claude-3-5-sonnet-20241022 / deepseek-chat / gpt-4o).
    max_tokens : int
        Số token tối đa cho kết quả trả về.
    temperature : float
        Độ sáng tạo của mô hình (mặc định 0.0 cho tính tất định trong văn bản pháp lý).
    disable_thinking : bool
        Tắt suy luận mở rộng (extended thinking) đối với các model hỗ trợ để tiết kiệm token.
    """
    api_key = get_api_key()
    if not api_key:
        raise LLMUnavailableError("Chưa cấu hình API Key (cần thiết lập LLM_API_KEY, OPENAI_API_KEY hoặc ANTHROPIC_API_KEY).")

    settings = get_app_settings()
    model = model or settings.llm_gateway.default_model
    base_url = get_base_url()
    budget_cap = settings.llm_gateway.usd_cap_per_process

    current_cost = get_session_cost_usd()
    if current_cost >= budget_cap:
        raise BudgetExceededError(
            f"Đã đạt trần ngân sách LLM (~${current_cost:.2f} / ${budget_cap:.2f} mỗi process) — "
            "tự động dừng gọi API để tránh phát sinh chi phí ngoài tầm kiểm soát."
        )

    try:
        from openai import APIConnectionError, APIError, InternalServerError, OpenAI, RateLimitError
    except ImportError as e:
        raise LLMUnavailableError("Thư viện `openai` chưa được cài đặt.") from e

    client = _create_client(api_key, base_url, timeout=settings.llm_gateway.timeout_seconds)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

    # Retry decorator cục bộ cho request này
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, InternalServerError)),
        reraise=True,
    )
    def _execute_request():
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return client.chat.completions.create(**kwargs)

    try:
        response = _execute_request()
    except Exception as e:  # noqa: BLE001
        logger.warning("Gọi LLM Gateway thất bại (%s): %s", base_url, e)
        raise LLMUnavailableError(f"Gọi LLM Gateway thất bại: {e}") from e

    _record_cost_and_check_budget(model, getattr(response, "usage", None))

    if not response.choices or not response.choices[0].message:
        raise LLMUnavailableError("LLM Gateway trả về phản hồi rỗng.")

    content = response.choices[0].message.content
    return (content or "").strip()


# Alias tương thích ngược hoàn toàn với `claude_client.call_claude`
call_claude = call_llm
