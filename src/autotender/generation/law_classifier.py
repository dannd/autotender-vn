"""Phân loại loại văn bản liên quan (metadata filtering theo `law_id`) TRƯỚC KHI truy hồi —
chỉ dùng cho Mức 1 Hỏi-đáp (câu hỏi tự do). Khác `query_rewrite.py` (viết lại NỘI DUNG câu
hỏi): module này chỉ chọn TẬP CON văn bản để giới hạn phạm vi tìm kiếm, không đổi câu hỏi.

Đo "trần" bằng oracle filter (biết trước đáp án đúng, xem `docs/DATA_CARD.md` Mục 12.3) cho
thấy lợi ích tiềm năng LỚN (hybrid+rerank nDCG@5 0.627→0.734, MRR 0.587→0.707 trên 46 câu eval)
— khác hẳn kết quả TIÊU CỰC của query rewrite (Mục 12.2). Module này hiện thực bộ phân loại
THẬT (không biết trước đáp án) để đo xem có đạt được một phần lợi ích đó hay không
(`scripts/run_retrieval_eval.py --classify-filter`).
"""

from __future__ import annotations

from autotender.generation.claude_client import ClaudeUnavailableError, call_claude
from autotender.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"

# Mô tả ngắn mỗi văn bản — dùng làm lựa chọn cho bộ phân loại. Khớp `law_id` thật trong
# `data/samples/legal_corpus/*.jsonl` (xác nhận qua `article.law_id`, không đoán).
LAW_DESCRIPTIONS: dict[str, str] = {
    "luat_22_2023_qh15": "Luật Đấu thầu 22/2023/QH15 — nguyên tắc chung, hình thức/phương thức lựa chọn nhà thầu, hành vi cấm, xử lý vi phạm, hợp đồng, ưu đãi",
    "nd_214_2025_ndcp": "Nghị định 214/2025/NĐ-CP — quy trình chi tiết lựa chọn nhà thầu, nội dung HSMT, tiêu chuẩn đánh giá, thời gian đấu thầu",
    "nd_45_2026_ndcp": "Nghị định 45/2026/NĐ-CP — CHUYÊN NGÀNH CNTT: đầu tư ứng dụng CNTT, phần mềm nội bộ, thử nghiệm, nghiệm thu, bảo hành phần mềm",
    "tt_01_2024_bkhdt": "Thông tư 01/2024/TT-BKHĐT — mẫu hồ sơ, đăng tải thông tin đấu thầu qua mạng",
    "tt_22_2024_bkhdt": "Thông tư 22/2024/TT-BKHĐT — mẫu hồ sơ, đăng tải thông tin đấu thầu qua mạng (thay Thông tư 06/2024)",
}

_SYSTEM_PROMPT = (
    "Bạn phân loại câu hỏi về đấu thầu Việt Nam thuộc văn bản pháp luật nào trong danh sách sau, "
    "để giới hạn phạm vi tìm kiếm:\n"
    + "\n".join(f"- {law_id}: {desc}" for law_id, desc in LAW_DESCRIPTIONS.items())
    + "\n\nChỉ trả về danh sách law_id (đúng chính tả như trên) PHÙ HỢP nhất, cách nhau bằng dấu "
    "phẩy, KHÔNG giải thích. Nếu câu hỏi có thể liên quan nhiều văn bản hoặc bạn không chắc, liệt "
    "kê TẤT CẢ law_id có khả năng liên quan — thà thừa còn hơn thiếu (bỏ sót văn bản đúng làm mất "
    "hoàn toàn kết quả tìm kiếm, trong khi thừa chỉ làm phạm vi tìm kiếm rộng hơn một chút)."
)


def classify_relevant_law_ids(question: str, model: str = DEFAULT_CLASSIFIER_MODEL) -> set[str] | None:
    """Trả về tập `law_id` liên quan đến `question`, hoặc `None` nếu Claude không dùng được
    hoặc phân loại ra rỗng/toàn nhãn lạ (an toàn: không lọc gì — coi như tất cả văn bản đều
    có thể liên quan — thay vì lọc sai làm mất kết quả đúng)."""
    try:
        raw = call_claude(system=_SYSTEM_PROMPT, user_prompt=question, model=model, max_tokens=100)
    except ClaudeUnavailableError as e:
        logger.info("Law classifier bỏ qua (Claude không dùng được), không lọc: %s", e)
        return None

    candidates = {tok.strip() for tok in raw.split(",")}
    valid = candidates & set(LAW_DESCRIPTIONS)
    if not valid:
        logger.info("Law classifier không nhận diện được law_id hợp lệ từ %r, không lọc.", raw)
        return None
    return valid
