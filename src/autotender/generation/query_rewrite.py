"""Query rewriting (HyDE-lite) trước khi truy hồi — chỉ dùng cho Mức 1 Hỏi-đáp, nơi câu hỏi
đến trực tiếp từ người dùng (khác Mức 2, nơi câu truy vấn là `SECTION_DEFINITIONS[...]["query"]`
đã được viết sẵn bằng thuật ngữ pháp lý chuẩn, không cần viết lại).

Câu hỏi người dùng thường ngắn/khẩu ngữ ("thầu qua mạng thế nào") thiếu thuật ngữ pháp lý
chuẩn hoá khiến truy hồi (đặc biệt BM25, vốn khớp từ chứ không hiểu ngữ nghĩa) dễ bỏ sót. Ý
tưởng HyDE (Hypothetical Document Embeddings — Gao et al., 2022): sinh một đoạn văn bản GIẢ
ĐỊNH trả lời câu hỏi bằng thuật ngữ đúng miền, rồi dùng chính đoạn đó (thay vì câu hỏi gốc) để
truy hồi — đoạn giả định không cần đúng nội dung, chỉ cần "nghe giống" văn bản luật thật để kéo
vector/từ khoá gần các chunk liên quan hơn so với câu hỏi khẩu ngữ gốc.

Bản "lite" ở đây: gọi Claude Haiku (rẻ, nhanh hơn Sonnet) viết lại NGẮN GỌN, không sinh hẳn
một "văn bản giả định" dài như HyDE gốc — đủ để chuẩn hoá thuật ngữ mà không tốn nhiều token.
Best-effort: lỗi/thiếu API key → rơi về câu hỏi gốc, KHÔNG raise ra ngoài (đúng nguyên tắc
"luôn có phương án chạy được" áp dụng xuyên suốt dự án).
"""

from __future__ import annotations

from autotender.generation.claude_client import ClaudeUnavailableError, call_claude
from autotender.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_REWRITE_MODEL = "claude-haiku-4-5-20251001"

_REWRITE_SYSTEM_PROMPT = (
    "Bạn hỗ trợ một hệ thống truy hồi văn bản pháp luật đấu thầu Việt Nam. Nhiệm vụ: viết lại "
    "câu hỏi của người dùng thành 1 đoạn ngắn (2-3 câu) dùng ĐÚNG thuật ngữ pháp lý đấu thầu "
    "chuẩn (vd \"hồ sơ mời thầu\", \"bên mời thầu\", \"hình thức lựa chọn nhà thầu\", \"tiêu "
    "chuẩn đánh giá\", \"hồ sơ dự thầu\"...) để tăng độ khớp khi tìm kiếm ngữ nghĩa/từ khoá. "
    "TUYỆT ĐỐI KHÔNG trả lời câu hỏi, KHÔNG bịa số điều/khoản/nghị định cụ thể — chỉ diễn đạt "
    "lại ý câu hỏi bằng thuật ngữ chuẩn hơn. Chỉ trả về đoạn văn bản đó, không thêm giải thích "
    "hay lời dẫn."
)


def rewrite_query(question: str, model: str = DEFAULT_REWRITE_MODEL) -> str:
    """Trả về câu hỏi đã "chuẩn hoá thuật ngữ" để dùng làm câu truy vấn cho hybrid search —
    KHÔNG dùng để thay câu hỏi hiển thị/trả lời cho người dùng (`QAAnswer.question` vẫn giữ
    nguyên câu hỏi gốc). Rơi về `question` gốc nếu Claude không dùng được hoặc trả về rỗng.
    """
    try:
        rewritten = call_claude(
            system=_REWRITE_SYSTEM_PROMPT, user_prompt=question, model=model, max_tokens=200,
        )
    except ClaudeUnavailableError as e:
        logger.info("Query rewrite bỏ qua (Claude không dùng được), dùng câu hỏi gốc: %s", e)
        return question
    return rewritten.strip() or question
