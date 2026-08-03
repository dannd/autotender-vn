"""Sinh văn bản KHLCNT tổng hợp từ một `TenderNotice` — dùng cho distant supervision (M2)
và để demo end-to-end pipeline khi chưa có kho văn bản KHLCNT thật đã crawl (Mục 6/M2).

Đây KHÔNG phải dữ liệu "sinh tự do" đưa vào sản phẩm cuối — chỉ dùng nội bộ để:
(a) tạo nhãn NER tự động bằng cách khớp chuỗi field đã biết vào văn bản, và
(b) làm input mẫu cho toàn bộ pipeline ingest -> NER -> classify -> generate khi demo offline.
"""

from __future__ import annotations

from autotender.schemas import TenderNotice


def build_synthetic_khlcnt_text(notice: TenderNotice) -> str:
    """Ghép các trường có cấu trúc của TenderNotice thành đoạn văn bản dạng KHLCNT."""
    value_str = f"{notice.package_value:,.0f}".replace(",", ".") if notice.package_value else "chưa xác định"
    lines = [
        f"Tên gói thầu: {notice.package_name}",
        f"Chủ đầu tư: {notice.investor}",
        f"Giá gói thầu: {value_str} đồng",
        f"Nguồn vốn: {notice.funding_source or '[CẦN NGƯỜI DÙNG BỔ SUNG: nguồn vốn]'}",
        f"Hình thức lựa chọn nhà thầu: {notice.selection_method or '[CẦN NGƯỜI DÙNG BỔ SUNG: hình thức lựa chọn nhà thầu]'}",
        f"Loại hợp đồng: {notice.contract_type or '[CẦN NGƯỜI DÙNG BỔ SUNG: loại hợp đồng]'}",
        f"Thời gian thực hiện hợp đồng: {notice.execution_time or '[CẦN NGƯỜI DÙNG BỔ SUNG: thời gian thực hiện]'}",
    ]
    return "\n".join(lines)
