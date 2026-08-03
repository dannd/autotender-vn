"""Kiểm tra 12 bản ghi TBMT thật (không phải tổng hợp) lấy được qua thao tác trực tiếp
trên trình duyệt — xem docs/DATA_CARD.md mục 8."""

from pathlib import Path

from autotender.schemas import TenderNotice

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "data" / "samples" / "real_pilot_sample.jsonl"


def test_real_pilot_sample_has_12_valid_records():
    assert SAMPLE_FILE.exists()
    notices = []
    with open(SAMPLE_FILE, encoding="utf-8") as f:
        for line in f:
            notices.append(TenderNotice.model_validate_json(line))
    assert len(notices) == 12
    assert all(n.tbmt_id.startswith("IB") for n in notices)
    assert all(n.package_value and n.package_value > 0 for n in notices)
    assert len({n.tbmt_id for n in notices}) == 12  # khong trung lap
