"""M6 — Compliance Guard (Mục 6/M6), module trọng tâm.

Rule-based thuần: từ điển nhãn hiệu phổ biến + regex ngưỡng bất hợp lý. Khung `BaseModule`
3-tier (cross-encoder XLM-R fine-tuned / zero-shot) từng có ở đây đã bị bỏ: cả 2 tầng đó
chưa từng chạy thật trong đồ án (không có checkpoint đã train), nên được đơn giản hoá
thành logic trực tiếp thay vì giữ code chết không ai gọi tới. Xem lịch sử số liệu Tier 3
(F1 thật đã đo) tại docs/MODEL_CARD.md.

6 lớp: R1 (nhãn hiệu/xuất xứ cụ thể), R2 (năng lực/doanh thu bất hợp lý),
R3 (thông số "may đo"), R4 (số liệu sai lệch KHLCNT — đã xử lý riêng ở M5 verifier),
R5 (thiếu thành phần bắt buộc — `check_document_completeness`, xem bên dưới), OK (hợp
lệ). Recall được ưu tiên hơn precision (Mục 6/M6: bỏ sót vi phạm nguy hiểm hơn báo
động giả).
"""

from __future__ import annotations

import re

from autotender.config import get_models_settings
from autotender.schemas import ComplianceFlag, HSMTSection, RetrievedChunk
from autotender.utils.vn_text import split_sentences


class ComplianceModule:
    def __init__(self) -> None:
        self._cfg = get_models_settings().compliance

    def check_text(self, text: str, evidence: list[RetrievedChunk] | None = None) -> list[ComplianceFlag]:
        evidence = evidence or []
        flags: list[ComplianceFlag] = []
        brands = self._cfg.get("brand_dictionary", _DEFAULT_BRANDS)
        revenue_ratio = self._cfg.get("revenue_ratio_threshold", 3.0)

        for sentence in split_sentences(text):
            lowered = sentence.lower()

            # R1 — nhãn hiệu/xuất xứ cụ thể
            for brand in brands:
                if brand.lower() in lowered:
                    flags.append(
                        ComplianceFlag(
                            rule_code="R1",
                            severity="cao",
                            sentence=sentence,
                            explanation=f"Câu có nêu nhãn hiệu/thương hiệu cụ thể ('{brand}') — vi phạm nguyên tắc không hạn chế cạnh tranh.",
                            evidence=evidence,
                            confidence=0.75,
                        )
                    )
                    break

            # R2 — yêu cầu doanh thu bất hợp lý (doanh thu > revenue_ratio lần giá gói thầu)
            m = re.search(r"doanh thu[^\d]{0,40}(\d[\d.,]*)\s*(?:lần|x)\s*giá gói thầu", lowered)
            if m:
                ratio = float(m.group(1).replace(",", "."))
                if ratio > revenue_ratio:
                    flags.append(
                        ComplianceFlag(
                            rule_code="R2",
                            severity="cao",
                            sentence=sentence,
                            explanation=f"Yêu cầu doanh thu gấp {ratio} lần giá gói thầu, vượt ngưỡng hợp lý ({revenue_ratio} lần).",
                            evidence=evidence,
                            confidence=0.7,
                        )
                    )

            # R3 — thông số "may đo" (dấu hiệu từ khoá "duy nhất", "chỉ có", "độc quyền").
            # Bỏ qua nếu câu là PHỦ ĐỊNH ("không được", "không đưa ra"...) — đây là câu mô tả
            # NGUYÊN TẮC cấm hành vi đó, không phải câu đang vi phạm nó.
            r3_keywords = ["duy nhất trên thị trường", "chỉ có sản phẩm", "độc quyền"]
            matched_kw = next((kw for kw in r3_keywords if kw in lowered), None)
            is_negated = matched_kw and re.search(
                r"không\s+(?:được|đưa ra|nêu|thiết kế)", lowered[: lowered.find(matched_kw)]
            )
            if matched_kw and not is_negated:
                flags.append(
                    ComplianceFlag(
                        rule_code="R3",
                        severity="cao",
                        sentence=sentence,
                        explanation="Câu có dấu hiệu mô tả thông số kỹ thuật chỉ một sản phẩm cụ thể mới đáp ứng được.",
                        evidence=evidence,
                        confidence=0.6,
                    )
                )

        return flags


_DEFAULT_BRANDS = [
    "Cisco", "Dell", "HP", "Hewlett Packard", "Samsung", "IBM", "Lenovo", "Intel",
    "Microsoft", "Apple", "Huawei", "Sony", "Canon", "Epson", "Fujitsu", "Oracle", "SAP",
]

# Phải khớp với `models/generator.py::_PLACEHOLDER` — trùng lặp thay vì import tên "private"
# giữa 2 module, tránh phụ thuộc chéo không cần thiết.
_PLACEHOLDER_MARKER = "[CẦN NGƯỜI DÙNG BỔ SUNG"

# Điều 44 Luật Đấu thầu 22/2023/QH15 (bản hợp nhất) đã được đơn giản hoá chỉ còn 4 khoản
# tổng quát, giao Chính phủ quy định chi tiết (Khoản 4) — nội dung HSMT chi tiết THẬT SỰ
# nằm ở Điều 26 Khoản 2 Nghị định 214/2025/NĐ-CP (gói hàng hóa), liệt kê đủ 7 thành phần
# a-g. Đây là căn cứ ĐÚNG với luật hiện hành — không dùng lại mô tả "7 thành phần Điều 44"
# cũ trong đề cương gốc, vốn dựa trên NĐ 24/2024/NĐ-CP đã hết hiệu lực (xem docs/DATA_CARD.md
# Mục 10). Chỉ liệt kê phần trong PHẠM VI hệ thống hỗ trợ soạn (Chương III + Chương V —
# xem Mục "Việc KHÔNG làm" trong kế hoạch: không sinh trọn bộ HSMT gồm cả Chương I/II/IV/VI).
REQUIRED_HSMT_COMPONENTS: dict[str, dict[str, str]] = {
    "chuong_III": {
        "muc_1": "Tiêu chuẩn đánh giá về năng lực, kinh nghiệm — Điều 26 Khoản 2 điểm c, Nghị định 214/2025/NĐ-CP",
        "muc_2": "Tiêu chuẩn đánh giá về kỹ thuật — Điều 26 Khoản 2 điểm c, Nghị định 214/2025/NĐ-CP",
        "muc_3": "Tiêu chuẩn đánh giá về tài chính/giá — Điều 26 Khoản 2 điểm c, Nghị định 214/2025/NĐ-CP",
        "muc_4": "Quy định về nhãn hiệu, xuất xứ hàng hóa — Điều 26 Khoản 9, Nghị định 214/2025/NĐ-CP",
    },
    "chuong_V": {
        "muc_1": "Phạm vi cung cấp — Điều 26 Khoản 2 điểm đ, Nghị định 214/2025/NĐ-CP",
        "muc_2": "Yêu cầu về thông số kỹ thuật — Điều 26 Khoản 2 điểm đ, Nghị định 214/2025/NĐ-CP",
        "muc_3": "Yêu cầu về bảo hành, bảo trì — Điều 26 Khoản 2 điểm đ, Nghị định 214/2025/NĐ-CP",
        "muc_4": "Yêu cầu về tiến độ thực hiện — Điều 26 Khoản 2 điểm đ, Nghị định 214/2025/NĐ-CP",
    },
}


def check_document_completeness(sections: list[HSMTSection]) -> list[ComplianceFlag]:
    """Kiểm tra đủ thành phần bắt buộc (trong phạm vi Chương III + Chương V mà hệ thống hỗ
    trợ soạn) theo Điều 26 Khoản 2 Nghị định 214/2025/NĐ-CP — bản chi tiết hoá hiện hành của
    Điều 44 Luật Đấu thầu. Đây là kiểm tra CẤP TÀI LIỆU (mục nào đã/chưa soạn), khác với
    `check_text` (kiểm tra CẤP CÂU, R1-R3) — luôn chạy được, không qua model, không phải
    Tier 1/2/3 vì bản chất là đối chiếu danh sách, không phải suy luận.
    """
    present = {s.section_id: s for s in sections}
    flags: list[ComplianceFlag] = []
    for chapter, mucs in REQUIRED_HSMT_COMPONENTS.items():
        for muc_key, legal_basis in mucs.items():
            section_id = f"{chapter}.{muc_key}"
            section = present.get(section_id)
            if section is None:
                flags.append(
                    ComplianceFlag(
                        rule_code="R5",
                        severity="cao",
                        sentence=f"[Thiếu mục {section_id}]",
                        explanation=f"Chưa soạn mục bắt buộc theo {legal_basis}.",
                        evidence=[],
                        confidence=1.0,
                    )
                )
                continue
            if not section.current_text.strip() or _PLACEHOLDER_MARKER in section.current_text:
                flags.append(
                    ComplianceFlag(
                        rule_code="R5",
                        severity="trung_binh",
                        sentence=section.current_text[:200],
                        explanation=f"Mục {section_id} còn thiếu thông tin (placeholder chưa điền), theo {legal_basis}.",
                        evidence=section.citations,
                        confidence=0.9,
                    )
                )
    return flags
