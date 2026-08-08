"""M2 — NER trích xuất trường thông tin từ văn bản KHLCNT (Mục 6/M2).

Rule-based thuần (regex + từ điển từ khoá) — module này không dùng mô hình học máy.
Khung `BaseModule` 3-tier (fine-tuned checkpoint / zero-shot pretrained) từng có ở đây đã
bị bỏ: cả 2 tầng đó chưa từng chạy thật trong đồ án (không có checkpoint đã train, và
Tier 2 dùng sai tên task pipeline của `transformers`), nên được đơn giản hoá thành logic
trực tiếp thay vì giữ code chết không ai gọi tới. Xem lịch sử số liệu Tier 3 (F1 thật đã
đo) tại docs/MODEL_CARD.md.
"""

from __future__ import annotations

import re

from autotender.schemas import ExtractedField

ENTITY_LABELS = [
    "PACKAGE_NAME", "VALUE", "FUNDING", "METHOD", "CONTRACT_TYPE", "DURATION", "INVESTOR", "LOCATION",
]

# Mỗi entry: (label, regex, group_index_giá_trị).
_PATTERNS: list[tuple[str, re.Pattern, int]] = [
    (
        "VALUE",
        re.compile(r"(?:giá gói thầu|giá trị gói thầu|giá trị)[:\s]*([\d.,]+)\s*(?:đồng|VNĐ|VND)", re.IGNORECASE),
        1,
    ),
    (
        "FUNDING",
        re.compile(r"(?:nguồn vốn)[:\s]*([^\n.;]+)", re.IGNORECASE),
        1,
    ),
    (
        "METHOD",
        re.compile(
            r"(đấu thầu rộng rãi[^\n.;]*|chào hàng cạnh tranh[^\n.;]*|chỉ định thầu[^\n.;]*|"
            r"tự thực hiện[^\n.;]*|mua sắm trực tiếp[^\n.;]*)",
            re.IGNORECASE,
        ),
        1,
    ),
    (
        "CONTRACT_TYPE",
        re.compile(r"(?:loại hợp đồng)[:\s]*([^\n.;]+)", re.IGNORECASE),
        1,
    ),
    (
        "DURATION",
        re.compile(r"(?:thời gian thực hiện(?: hợp đồng)?)[:\s]*([^\n.;]+)", re.IGNORECASE),
        1,
    ),
    (
        "INVESTOR",
        re.compile(r"(?:chủ đầu tư)[:\s]*([^\n.;]+)", re.IGNORECASE),
        1,
    ),
    (
        "PACKAGE_NAME",
        re.compile(r"(?:tên gói thầu|gói thầu số\s*\d+[:\s]*)[:\s]*([^\n.;]+)", re.IGNORECASE),
        1,
    ),
]


class NERModule:
    def extract(self, text: str) -> list[ExtractedField]:
        fields: list[ExtractedField] = []
        for label, pattern, group_idx in _PATTERNS:
            match = pattern.search(text)
            if match:
                value = match.group(group_idx).strip()
                if "[CẦN NGƯỜI DÙNG BỔ SUNG" in value:
                    # Placeholder báo thiếu thông tin (Mục 2.2) — không phải giá trị thật,
                    # không được coi là đã trích xuất được (tránh khớp nhầm chính placeholder).
                    continue
                fields.append(
                    ExtractedField(
                        name=label,
                        value=value,
                        confidence=0.6,
                        char_start=match.start(group_idx),
                        char_end=match.end(group_idx),
                        source="regex",
                    )
                )
        return fields
