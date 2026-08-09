"""Wrapper mỏng quanh Anthropic SDK — dùng LLM có sẵn (Claude), KHÔNG tự train (đúng
hướng đề cương RAG+LLM). Đọc API key từ biến môi trường chuẩn `ANTHROPIC_API_KEY` (SDK tự
đọc, không cần biến riêng của dự án) — nếu thiếu, raise `ClaudeUnavailableError` để tầng
gọi (`BaseModule._try_tier1`) bắt và rơi xuống tier dự phòng, KHÔNG làm sập ứng dụng.

Tự nạp `.env` ở gốc dự án (nếu có) khi import module này — cho phép cấu hình
`ANTHROPIC_API_KEY` một lần trong file cục bộ (đã gitignore, KHÔNG commit) thay vì phải
`export`/`set` biến môi trường thủ công mỗi phiên terminal hay mỗi lần chạy Streamlit.
`load_dotenv` không ghi đè biến môi trường đã có sẵn trong shell (`override=False` mặc
định) — biến môi trường thật (vd CI/CD) luôn được ưu tiên hơn `.env`.
"""

from __future__ import annotations

import os
import threading

from dotenv import load_dotenv

from autotender.config import PROJECT_ROOT, get_app_settings
from autotender.utils.logging import get_logger

load_dotenv(PROJECT_ROOT / ".env")

logger = get_logger(__name__)


class ClaudeUnavailableError(Exception):
    """Không gọi được Claude API (thiếu key, lỗi mạng, lỗi API...) — không phải lỗi nghiêm
    trọng, tầng gọi cần bắt exception này và dùng phương án dự phòng."""


class BudgetExceededError(ClaudeUnavailableError):
    """Đã tiêu hết ngân sách Claude API cho phiên chạy (xem `ClaudeBudgetConfig` trong
    `autotender.config`) — kế thừa `ClaudeUnavailableError` NÊN mọi nơi gọi `call_claude`
    (đã bắt `ClaudeUnavailableError` để rơi xuống tier dự phòng) tự động xử lý đúng, không
    cần sửa thêm: hết ngân sách → ứng dụng vẫn chạy được (chỉ mất phần sinh bằng LLM), không
    crash và không âm thầm tiêu tiếp."""


_cost_lock = threading.Lock()
_session_cost_usd = 0.0


def get_session_cost_usd() -> float:
    """Tổng chi phí Claude API ước tính đã dùng từ lúc process này khởi động — dùng để
    hiển thị trên Trang 6 (Bảng điều khiển) và trong `call_claude` để kiểm tra trần."""
    with _cost_lock:
        return _session_cost_usd


def _record_cost_and_check_budget(model: str, response) -> None:
    """Cộng dồn chi phí request vừa xong vào `_session_cost_usd`. Không raise ở đây (request
    đã thực hiện xong, không thể huỷ) — việc chặn request TIẾP THEO nằm ở đầu `call_claude`."""
    global _session_cost_usd
    usage = getattr(response, "usage", None)
    if usage is None:
        return  # test double / API cũ không trả usage — bỏ qua, không đoán mò chi phí
    settings = get_app_settings()
    pricing = settings.claude_budget.pricing_usd_per_mtok.get(model)
    if pricing is None:
        logger.warning("Chưa cấu hình giá cho model '%s' trong claude_budget.pricing_usd_per_mtok — bỏ qua tính chi phí request này.", model)
        return
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cost = (input_tokens / 1_000_000) * pricing.input_usd_per_mtok + (output_tokens / 1_000_000) * pricing.output_usd_per_mtok
    with _cost_lock:
        _session_cost_usd += cost


def is_configured() -> bool:
    """Kiểm tra nhanh (không gọi mạng) xem có nên thử Tier 1 hay không — dùng để tránh làm
    công việc truy xuất/rerank tốn thời gian rồi mới phát hiện thiếu API key."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def call_claude(
    system: str, user_prompt: str, model: str, max_tokens: int = 1024,
    temperature: float | None = None, disable_thinking: bool = True,
) -> str:
    """Gọi Claude API 1 lượt (không giữ lịch sử hội thoại — mỗi câu hỏi/mục độc lập theo
    đúng thiết kế RAG: ngữ cảnh luôn đến từ retrieval, không phải từ hội thoại trước đó).

    `temperature=None` (mặc định) — KHÔNG gửi tham số này trong request. Một số model mới
    (vd `claude-sonnet-5`) trả lỗi 400 "temperature is deprecated for this model" nếu tham
    số này được truyền tường minh — xác nhận thực tế khi chạy live, không phải giả định.

    `disable_thinking=True` (mặc định) — `claude-sonnet-5` bật extended thinking mặc định;
    phát hiện thực tế khi chạy live: nếu model "suy nghĩ" lâu, thinking token tính vào
    `max_tokens`, có thể tiêu hết ngân sách trước khi sinh ra bất kỳ text nào (response
    chỉ có block `thinking`, không có `text` — lỗi khó hiểu nếu không biết nguyên nhân).
    Tác vụ RAG-grounded ở đây (trả lời/soạn mục dựa trích dẫn có sẵn) không cần suy luận
    nhiều bước lộ ra ngoài, nên tắt thinking để kết quả ổn định, dự đoán được chi phí.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ClaudeUnavailableError("Biến môi trường ANTHROPIC_API_KEY chưa được cấu hình.")

    budget = get_app_settings().claude_budget
    current_cost = get_session_cost_usd()
    if current_cost >= budget.usd_cap_per_process:
        raise BudgetExceededError(
            f"Đã đạt trần ngân sách Claude API (~${current_cost:.2f} / ${budget.usd_cap_per_process:.2f} mỗi "
            "process) — dừng gọi API để tránh phát sinh chi phí ngoài kiểm soát. Tăng "
            "`claude_budget.usd_cap_per_process` trong configs/app.yaml nếu cần dùng tiếp, hoặc khởi động lại "
            "process (bộ đếm không bền qua restart)."
        )

    try:
        import anthropic
    except ImportError as e:
        raise ClaudeUnavailableError("Thư viện `anthropic` chưa được cài đặt.") from e

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if disable_thinking:
        kwargs["thinking"] = {"type": "disabled"}

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(**kwargs)
    except Exception as e:  # noqa: BLE001 — mọi lỗi API (rate limit, mạng, key sai...) đều là "không dùng được"
        raise ClaudeUnavailableError(f"Gọi Claude API thất bại: {e}") from e

    _record_cost_and_check_budget(model, response)

    text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    if not text_parts:
        block_types = [getattr(b, "type", "?") for b in response.content]
        raise ClaudeUnavailableError(
            f"Claude API trả về phản hồi không có nội dung text (stop_reason={response.stop_reason}, "
            f"block types={block_types}) — thường do max_tokens quá thấp, tăng tham số này nếu lặp lại."
        )
    return "\n".join(text_parts).strip()
