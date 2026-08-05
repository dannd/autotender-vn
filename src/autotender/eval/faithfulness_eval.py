"""Đánh giá faithfulness (không bịa đặt) + completeness của văn bản sinh ra, dùng LLM-as-judge
qua Claude API (Giai đoạn 3, đề cương RAG+LLM).

GIỚI HẠN QUAN TRỌNG cần nêu trong báo cáo (đúng như đề cương Mục 7 gợi ý xác nhận lại với
giảng viên): giám khảo (judge) và mô hình sinh CÙNG là Claude — có thể có thiên lệch tự ưu
ái nhẹ (self-preference bias), một hiện tượng đã được ghi nhận trong literature LLM-as-judge.
Không có ngân sách để dùng judge khác họ (vd GPT-4, Gemini) trong phạm vi đồ án 15 ngày —
đây là giới hạn phương pháp luận, không phải lỗi triển khai.

Rubric 2 chiều, mỗi câu hỏi/mục chấm độc lập:
- faithfulness (0.0-1.0): mọi khẳng định trong văn bản sinh ra có được trích đoạn căn cứ hỗ
  trợ hay không (0 = bịa đặt hoàn toàn, 1 = mọi khẳng định đều có căn cứ).
- completeness (0.0-1.0): văn bản có trả lời/đáp ứng đầy đủ câu hỏi/mục yêu cầu dựa trên
  thông tin SẴN CÓ trong trích đoạn hay không (0 = không liên quan, 1 = đầy đủ).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from autotender.generation.claude_client import ClaudeUnavailableError, call_claude
from autotender.schemas import RetrievedChunk

_JUDGE_SYSTEM_PROMPT = (
    "Bạn là giám khảo chấm điểm khách quan cho hệ thống RAG pháp lý về đấu thầu Việt Nam. "
    "Bạn sẽ nhận: (1) câu hỏi hoặc yêu cầu soạn thảo, (2) các trích đoạn văn bản pháp luật "
    "làm căn cứ (context), (3) văn bản do hệ thống sinh ra dựa trên context đó. "
    "Chấm điểm THEO ĐÚNG 2 tiêu chí sau, đừng chấm dựa trên văn phong hay độ dài:\n\n"
    "1. faithfulness (0.0-1.0): mọi khẳng định trong văn bản sinh ra có thực sự được hỗ trợ "
    "bởi context hay không. Nếu văn bản nêu số liệu/điều khoản KHÔNG có trong context (bịa "
    "đặt), điểm phải thấp (gần 0) bất kể nội dung nghe có hợp lý đến đâu.\n"
    "2. completeness (0.0-1.0): văn bản có tận dụng đầy đủ thông tin liên quan có sẵn trong "
    "context để trả lời/đáp ứng câu hỏi hay không (không tính thiếu sót do context vốn không "
    "có thông tin đó).\n\n"
    "Trả lời DUY NHẤT bằng JSON hợp lệ, không thêm text khác, đúng schema:\n"
    '{"faithfulness": <số 0.0-1.0>, "completeness": <số 0.0-1.0>, '
    '"unsupported_claims": [<liệt kê câu/ý không có căn cứ trong context, rỗng nếu không có>], '
    '"reasoning": "<giải thích ngắn gọn>"}'
)


@dataclass
class FaithfulnessJudgment:
    faithfulness: float
    completeness: float
    unsupported_claims: list[str] = field(default_factory=list)
    reasoning: str = ""


def _build_judge_prompt(question: str, citations: list[RetrievedChunk], generated_text: str) -> str:
    context_str = "\n\n".join(f"[{c.source_doc}]\n{c.text}" for c in citations)
    return (
        f"Câu hỏi/yêu cầu: {question}\n\n"
        f"Context (trích đoạn văn bản pháp luật):\n\n{context_str}\n\n"
        f"Văn bản do hệ thống sinh ra:\n\n{generated_text}\n\n"
        f"Hãy chấm điểm theo đúng schema JSON đã nêu."
    )


def judge_faithfulness(
    question: str, citations: list[RetrievedChunk], generated_text: str, model: str = "claude-sonnet-5"
) -> FaithfulnessJudgment:
    """Gọi Claude API làm giám khảo. Raise `ClaudeUnavailableError` nếu không gọi được (thiếu
    API key...) — tầng gọi (script eval) tự quyết định báo "N/A" thay vì giả lập điểm số."""
    prompt = _build_judge_prompt(question, citations, generated_text)
    # Không truyền temperature — model mới (claude-sonnet-5) từ chối tham số này (xem
    # docstring claude_client.call_claude). Giám khảo vẫn đủ ổn định nhờ rubric + yêu cầu
    # JSON schema cố định trong system prompt, không cần ép temperature=0.
    # max_tokens=1024 (không phải 512): phát hiện thực tế khi chạy live — 512 đôi khi cắt
    # cụt JSON giữa danh sách `unsupported_claims` dài, khiến response không parse được.
    raw = call_claude(system=_JUDGE_SYSTEM_PROMPT, user_prompt=prompt, model=model, max_tokens=1024)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ClaudeUnavailableError(f"Judge trả về JSON không hợp lệ: {raw[:200]}") from e

    try:
        return FaithfulnessJudgment(
            faithfulness=float(parsed["faithfulness"]),
            completeness=float(parsed["completeness"]),
            unsupported_claims=list(parsed.get("unsupported_claims", [])),
            reasoning=str(parsed.get("reasoning", "")),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ClaudeUnavailableError(f"Judge trả về JSON thiếu trường bắt buộc: {parsed}") from e
