"""Chuẩn hoá dữ liệu thô thu thập được (JSON API hoặc HTML) về schema `TenderNotice`.

Ghi chú kỹ thuật (Mục 6/M0): endpoint JSON nội bộ thật của cổng đã được xác định qua
DevTools Network là `POST /o/egp-portal-home/services/smart/search` (module Liferay
`egp-portal-home`), trả về `page.content[]` với các trường bên dưới. Hàm
`parse_msc_api_record` implement đúng theo shape đã quan sát được để `MSCApiSource`
sẵn sàng dùng ngay khi có payload/xác thực hợp lệ (xem docstring `MSCApiSource`).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from autotender.schemas import TenderNotice

_PACKAGE_TYPE_MAP = {
    "HH": "hàng hóa",
    "XL": "xây lắp",
    "TV": "tư vấn",
    "PTV": "phi tư vấn",
    "HH_XL": "hỗn hợp",
}

_SELECTION_METHOD_MAP = {
    "1_MTHS": "đấu thầu rộng rãi 1 giai đoạn 1 túi hồ sơ",
    "1_MTHS2T": "đấu thầu rộng rãi 1 giai đoạn 2 túi hồ sơ",
    "2_MTHS": "đấu thầu rộng rãi 2 giai đoạn",
}


def _parse_dt(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def parse_msc_api_record(raw: dict[str, Any]) -> TenderNotice:
    """Parse một bản ghi trong `page.content[]` của API `smart/search` thành `TenderNotice`."""
    bid_name = raw.get("bidName") or []
    package_name = "; ".join(bid_name) if isinstance(bid_name, list) else str(bid_name)

    invest_field = raw.get("investField") or []
    package_type = None
    if invest_field:
        code = invest_field[0] if isinstance(invest_field, list) else invest_field
        package_type = _PACKAGE_TYPE_MAP.get(code, code)

    bid_price = raw.get("bidPrice") or []
    package_value = float(bid_price[0]) if bid_price else None

    locations = raw.get("locations") or []
    location_str = None
    if locations:
        loc = locations[0]
        location_str = ", ".join(filter(None, [loc.get("districtName"), loc.get("provName")]))

    return TenderNotice(
        tbmt_id=raw.get("notifyNoStand") or raw.get("notifyNo") or raw.get("id", ""),
        package_name=package_name or "[CẦN NGƯỜI DÙNG BỔ SUNG: tên gói thầu]",
        investor=raw.get("investorName", ""),
        procuring_entity=raw.get("investorName"),
        package_value=package_value,
        currency="VND",
        funding_source=None,
        selection_method=_SELECTION_METHOD_MAP.get(raw.get("bidMode"), raw.get("bidMode")),
        contract_type=raw.get("bidForm"),
        package_type=package_type,
        execution_time=location_str,
        publish_date=_parse_dt(raw.get("publicDate")),
        close_date=_parse_dt(raw.get("bidCloseDate")),
        attachments=[],
        source_url=f"https://muasamcong.mpi.gov.vn/o/{raw.get('notifyNoStand', '')}",
    )


def parse_local_sample_record(raw: dict[str, Any]) -> TenderNotice:
    """Parse một bản ghi JSON trong `data/samples/*.json` (đã đúng schema TenderNotice)."""
    return TenderNotice.model_validate(raw)


def parse_dauthau_asia_rows(html: str) -> list[TenderNotice]:
    """Parse bảng `table.bidding-table` trên trang danh sách của dauthau.asia
    (vd `https://dauthau.asia/thongbao/moithau/?page=N`) thành `TenderNotice`.

    Trang này chỉ hiển thị công khai: mã TBMT, tên gói thầu, chủ đầu tư, ngày đăng tải,
    ngày đóng thầu, link chi tiết — không có giá gói thầu/nguồn vốn ở dạng danh sách
    (các trường này để `None`, đúng nguyên tắc "không bịa đặt" — Mục 2.2 SPEC).
    """
    from selectolax.parser import HTMLParser

    tree = HTMLParser(html)
    table = tree.css_first("table.bidding-table")
    if table is None:
        return []

    notices: list[TenderNotice] = []
    for row in table.css("tr"):
        code_span = row.css_first(".bidding-code")
        if code_span is None:
            continue
        link = code_span.parent  # thẻ <a> bao ngoài span mã TBMT
        tbmt_id = code_span.text(strip=True)
        full_link_text = link.text(strip=True) if link else ""
        package_name = full_link_text.replace(tbmt_id, "", 1).strip()
        detail_href = link.attributes.get("href") if link else None

        investor_span = row.css_first(".solicitor-code")
        investor_link = investor_span.parent if investor_span else None
        investor = investor_link.text(strip=True).replace(investor_span.text(strip=True), "", 1).strip() if investor_span else ""

        date_cells = row.css("td.txt-center")
        publish_raw = date_cells[0].text(strip=True) if len(date_cells) > 0 else None
        close_raw = date_cells[1].text(strip=True) if len(date_cells) > 1 else None

        notices.append(
            TenderNotice(
                tbmt_id=tbmt_id,
                package_name=package_name or "[CẦN NGƯỜI DÙNG BỔ SUNG: tên gói thầu]",
                investor=investor or "[CẦN NGƯỜI DÙNG BỔ SUNG: chủ đầu tư]",
                publish_date=_parse_vn_datetime(publish_raw),
                close_date=_parse_vn_datetime(close_raw),
                source_url=f"https://dauthau.asia{detail_href}" if detail_href else "https://dauthau.asia/thongbao/moithau/",
            )
        )
    return notices


def _parse_vn_datetime(text: str | None) -> date | None:
    """Parse chuỗi 'HH:MM DD/MM/YYYY' (định dạng hiển thị của dauthau.asia) thành date."""
    if not text:
        return None
    try:
        return datetime.strptime(text.strip(), "%H:%M %d/%m/%Y").date()
    except ValueError:
        return None
