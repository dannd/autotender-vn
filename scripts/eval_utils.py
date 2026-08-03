"""Hàm tiện ích dùng chung cho `scripts/evaluate.py` (Mục 10)."""

from __future__ import annotations

import re

from autotender.schemas import ExtractedField

_TOKEN_RE = re.compile(r"\S+")


def tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def fields_to_bio_tags(text: str, fields: list[ExtractedField]) -> list[str]:
    """Chuyển list[ExtractedField] (có char_start/char_end) thành BIO tags theo cùng
    cách tokenize với `scripts/build_dataset.py`, để so sánh trực tiếp bằng seqeval."""
    tokens = tokenize_with_spans(text)
    tags = ["O"] * len(tokens)
    for f in fields:
        if f.char_start is None or f.char_end is None:
            continue
        first = True
        for i, (_tok, tstart, tend) in enumerate(tokens):
            if tstart >= f.char_start and tend <= f.char_end:
                tags[i] = ("B-" if first else "I-") + f.name
                first = False
    return tags


# Tập test nhỏ gán tay cho M6 Compliance (thay thế cho việc gán tay quy mô lớn hơn
# trên dữ liệu thật — xem giới hạn nghiên cứu trong DATA_CARD.md).
COMPLIANCE_TEST_SET: list[tuple[str, str]] = [
    ("Yêu cầu cung cấp máy chủ hãng Dell hoặc tương đương.", "R1"),
    ("Thiết bị phải có xuất xứ từ Cisco, không chấp nhận hãng khác.", "R1"),
    ("Nhà thầu phải có doanh thu bình quân 5 lần giá gói thầu trong 3 năm gần nhất.", "R2"),
    ("Yêu cầu doanh thu tối thiểu 4.5 lần giá gói thầu.", "R2"),
    ("Thông số kỹ thuật chỉ có sản phẩm này mới đáp ứng được trên thị trường.", "R3"),
    ("Yêu cầu độc quyền về công nghệ, không sản phẩm nào khác đáp ứng.", "R3"),
    ("Nhà thầu cung cấp hàng hóa đáp ứng thông số kỹ thuật tối thiểu nêu tại E-HSMT.", "OK"),
    ("Thời gian thực hiện hợp đồng là 90 ngày kể từ ngày ký hợp đồng.", "OK"),
    ("Không được đưa ra thông số kỹ thuật của một sản phẩm cụ thể duy nhất trên thị trường.", "OK"),
    ("Giá dự thầu được xác định sau khi sửa lỗi và hiệu chỉnh sai lệch theo quy định.", "OK"),
]
