from datetime import date

from autotender.crawler.parser import parse_local_sample_record, parse_msc_api_record
from autotender.schemas import TenderNotice


def test_parse_msc_api_record_maps_known_fields():
    raw = {
        "notifyNoStand": "IB2600420870-00",
        "bidName": ["Thi công xây dựng công trình"],
        "investorName": "Ban quản lý Dự án đầu tư - hạ tầng xã Hoài Đức",
        "investField": ["XL"],
        "bidPrice": [1.2982428e10],
        "locations": [{"provName": "Thành phố Hà Nội", "districtName": "Xã Hoài Đức"}],
        "publicDate": "2026-08-03T21:10:23.86",
        "bidCloseDate": "2026-08-12T14:00:00",
        "bidMode": "1_MTHS",
        "bidForm": "DTRR",
    }
    notice = parse_msc_api_record(raw)

    assert isinstance(notice, TenderNotice)
    assert notice.tbmt_id == "IB2600420870-00"
    assert notice.package_name == "Thi công xây dựng công trình"
    assert notice.package_type == "xây lắp"
    assert notice.package_value == 1.2982428e10
    assert notice.publish_date == date(2026, 8, 3)
    assert notice.close_date == date(2026, 8, 12)


def test_parse_msc_api_record_missing_bid_name_uses_placeholder():
    raw = {"notifyNoStand": "IB000", "investorName": "X"}
    notice = parse_msc_api_record(raw)
    assert notice.package_name.startswith("[CẦN NGƯỜI DÙNG BỔ SUNG")


def test_parse_local_sample_record_roundtrip():
    raw = {
        "tbmt_id": "IB1",
        "package_name": "Test",
        "investor": "Inv",
        "source_url": "https://example.com",
    }
    notice = parse_local_sample_record(raw)
    assert notice.tbmt_id == "IB1"
    assert notice.currency == "VND"
