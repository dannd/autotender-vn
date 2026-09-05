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
# Mục 10). Khớp với `models/generator.py::SECTION_DEFINITIONS` (8 chương I-VIII) — điểm g
# "hồ sơ, bản vẽ khác (nếu có)" không map 1-1 với 1 chương chuẩn nên không có mục riêng.
REQUIRED_HSMT_COMPONENTS: dict[str, dict[str, str]] = {
    "chuong_I": {
        "muc_1": "Quy định chung — Điều 26 Khoản 2 điểm a, Nghị định 214/2025/NĐ-CP",
        "muc_2": "Chuẩn bị hồ sơ dự thầu — Điều 26 Khoản 2 điểm a, Nghị định 214/2025/NĐ-CP",
        "muc_3": "Nộp, mở và đánh giá hồ sơ dự thầu — Điều 26 Khoản 2 điểm a, Nghị định 214/2025/NĐ-CP",
        "muc_4": "Thương thảo, hoàn thiện và ký kết hợp đồng — Điều 26 Khoản 2 điểm a, Nghị định 214/2025/NĐ-CP",
    },
    "chuong_II": {
        "muc_1": "Thông tin chung về gói thầu — Điều 26 Khoản 2 điểm b, Nghị định 214/2025/NĐ-CP",
        "muc_2": "Bảo đảm dự thầu, bảo đảm thực hiện hợp đồng và tiến độ đấu thầu — Điều 26 Khoản 2 điểm b, Nghị định 214/2025/NĐ-CP",
    },
    "chuong_III": {
        "muc_1": "Tiêu chuẩn đánh giá về năng lực, kinh nghiệm — Điều 26 Khoản 2 điểm c, Nghị định 214/2025/NĐ-CP",
        "muc_2": "Tiêu chuẩn đánh giá về kỹ thuật — Điều 26 Khoản 2 điểm c, Nghị định 214/2025/NĐ-CP",
        "muc_3": "Tiêu chuẩn đánh giá về tài chính/giá — Điều 26 Khoản 2 điểm c, Nghị định 214/2025/NĐ-CP",
        "muc_4": "Quy định về nhãn hiệu, xuất xứ hàng hóa — Điều 26 Khoản 9, Nghị định 214/2025/NĐ-CP",
    },
    "chuong_IV": {
        "muc_1": "Mẫu đơn dự thầu và giấy uỷ quyền — Điều 26 Khoản 2 điểm d, Nghị định 214/2025/NĐ-CP",
        "muc_2": "Mẫu bảo lãnh dự thầu và cam kết của nhà thầu — Điều 26 Khoản 2 điểm d, Nghị định 214/2025/NĐ-CP",
    },
    "chuong_V": {
        "muc_1": "Phạm vi cung cấp — Điều 26 Khoản 2 điểm đ, Nghị định 214/2025/NĐ-CP",
        "muc_2": "Yêu cầu về thông số kỹ thuật — Điều 26 Khoản 2 điểm đ, Nghị định 214/2025/NĐ-CP",
        "muc_3": "Yêu cầu về bảo hành, bảo trì — Điều 26 Khoản 2 điểm đ, Nghị định 214/2025/NĐ-CP",
        "muc_4": "Yêu cầu về tiến độ thực hiện — Điều 26 Khoản 2 điểm đ, Nghị định 214/2025/NĐ-CP",
    },
    "chuong_VI": {
        "muc_1": "Định nghĩa, phạm vi và loại hợp đồng — Điều 26 Khoản 2 điểm e, Nghị định 214/2025/NĐ-CP",
        "muc_2": "Quyền và nghĩa vụ của các bên — Điều 26 Khoản 2 điểm e, Nghị định 214/2025/NĐ-CP",
        "muc_3": "Sửa đổi, thanh lý hợp đồng và xử lý vi phạm — Điều 26 Khoản 2 điểm e, Nghị định 214/2025/NĐ-CP",
    },
    "chuong_VII": {
        "muc_1": "Điều khoản cụ thể về giá, thanh toán và tiến độ — Điều 26 Khoản 2 điểm e, Nghị định 214/2025/NĐ-CP",
        "muc_2": "Điều khoản cụ thể về bảo hành, bảo trì và nghiệm thu — Điều 26 Khoản 2 điểm e, Nghị định 214/2025/NĐ-CP",
    },
    "chuong_VIII": {
        "muc_1": "Mẫu hợp đồng và phụ lục hợp đồng — Điều 26 Khoản 2 điểm e, Nghị định 214/2025/NĐ-CP",
    },
}


def check_document_completeness(sections: list[HSMTSection]) -> list[ComplianceFlag]:
    """Kiểm tra đủ thành phần bắt buộc (8 chương I-VIII hệ thống hỗ trợ soạn) theo Điều 26
    Khoản 2 Nghị định 214/2025/NĐ-CP — bản chi tiết hoá hiện hành của Điều 44 Luật Đấu thầu.
    Đây là kiểm tra CẤP TÀI LIỆU (mục nào đã/chưa soạn), khác với
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


def check_it_specific_compliance(sections: list[HSMTSection]) -> list[ComplianceFlag]:
    """Rà soát chuyên sâu các điều kiện bắt buộc đối với gói thầu Phần mềm và CNTT.

    1. An toàn thông tin mạng theo cấp độ (Nghị định 85/2016/NĐ-CP & Nghị định 73/2019/NĐ-CP):
       Chương V (Yêu cầu kỹ thuật) phải có yêu cầu về ATTT/bảo mật hệ thống.
    2. Bàn giao toàn bộ mã nguồn & Sở hữu trí tuệ (Điều 55 Nghị định 73/2019/NĐ-CP):
       Chương VI hoặc VII (Điều kiện hợp đồng) phải có điều khoản quy định Chủ đầu tư sở hữu
       mã nguồn (source code) và cơ sở dữ liệu.
    """
    present = {s.section_id: s for s in sections}
    flags: list[ComplianceFlag] = []

    # 1. Rà soát yêu cầu An toàn thông tin trong Chương V
    tech_sec = present.get("chuong_V.muc_2") or present.get("chuong_V.muc_1")
    if tech_sec and tech_sec.current_text.strip() and _PLACEHOLDER_MARKER not in tech_sec.current_text:
        text_lower = tech_sec.current_text.lower()
        has_security = any(kw in text_lower for kw in ["an toàn thông tin", "cấp độ", "bảo mật", "attt", "owasp", "mã hóa", "tls"])
        if not has_security:
            flags.append(
                ComplianceFlag(
                    rule_code="R3",
                    severity="trung_binh",
                    sentence=tech_sec.current_text[:200],
                    explanation=(
                        "Gói thầu phần mềm/CNTT thiếu yêu cầu bắt buộc về bảo đảm An toàn thông tin theo cấp độ "
                        "(theo Nghị định số 85/2016/NĐ-CP và Điều 5 Nghị định số 73/2019/NĐ-CP)."
                    ),
                    evidence=tech_sec.citations,
                    confidence=0.8,
                )
            )

    # 2. Rà soát điều khoản Bàn giao mã nguồn trong Chương VI hoặc VII
    contract_sec = present.get("chuong_VI.muc_2") or present.get("chuong_VII.muc_2")
    if contract_sec and contract_sec.current_text.strip() and _PLACEHOLDER_MARKER not in contract_sec.current_text:
        text_lower = contract_sec.current_text.lower()
        has_source_code = any(kw in text_lower for kw in ["mã nguồn", "source code", "sở hữu trí tuệ", "quyền tác giả", "cơ sở dữ liệu"])
        if not has_source_code:
            flags.append(
                ComplianceFlag(
                    rule_code="R3",
                    severity="cao",
                    sentence=contract_sec.current_text[:200],
                    explanation=(
                        "Thiếu điều khoản bắt buộc về quyền sở hữu trí tuệ và bàn giao toàn bộ mã nguồn (source code), "
                        "cơ sở dữ liệu cho Chủ đầu tư (theo Điều 55 Nghị định số 73/2019/NĐ-CP)."
                    ),
                    evidence=contract_sec.citations,
                    confidence=0.85,
                )
            )

    return flags

