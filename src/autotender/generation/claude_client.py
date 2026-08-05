"""Wrapper mỏng quanh Anthropic SDK — dùng LLM có sẵn (Claude), KHÔNG tự train (đúng
hướng đề cương RAG+LLM). Đọc API key từ biến môi trường chuẩn `ANTHROPIC_API_KEY` (SDK tự
đọc, không cần biến riêng của dự án) — nếu thiếu, raise `ClaudeUnavailableError` để tầng
gọi (`BaseModule._try_tier1`) bắt và rơi xuống tier dự phòng, KHÔNG làm sập ứng dụng.
"""

from __future__ import annotations

import os


class ClaudeUnavailableError(Exception):
    """Không gọi được Claude API (thiếu key, lỗi mạng, lỗi API...) — không phải lỗi nghiêm
    trọng, tầng gọi cần bắt exception này và dùng phương án dự phòng."""


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

    text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    if not text_parts:
        block_types = [getattr(b, "type", "?") for b in response.content]
        raise ClaudeUnavailableError(
            f"Claude API trả về phản hồi không có nội dung text (stop_reason={response.stop_reason}, "
            f"block types={block_types}) — thường do max_tokens quá thấp, tăng tham số này nếu lặp lại."
        )
    return "\n".join(text_parts).strip()
