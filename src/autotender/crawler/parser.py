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
