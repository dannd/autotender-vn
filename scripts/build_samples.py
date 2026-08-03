"""Sinh 20 bản ghi TenderNotice mẫu (dữ liệu tổng hợp) để commit vào data/samples/.

Dữ liệu này là TỔNG HỢP (synthetic), không phải trích xuất từ hồ sơ thật của bất kỳ
chủ đầu tư/nhà thầu cụ thể nào — dùng để đảm bảo pipeline luôn demo được (Mục 6/M0,
Mục 2.1 Degraded Mode) kể cả khi không có mạng hoặc crawler thật bị chặn.
Xem docs/DATA_CARD.md để biết chi tiết phương pháp tạo dữ liệu.
"""

from __future__ import annotations

import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from autotender.utils.console import ensure_utf8_console  # noqa: E402

ensure_utf8_console()

random.seed(42)

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"

INVESTORS = [
    "Sở Y tế tỉnh Bình Dương",
    "Sở Giáo dục và Đào tạo tỉnh Nghệ An",
    "Ban Quản lý dự án đầu tư xây dựng huyện Thanh Oai",
    "Bệnh viện Đa khoa tỉnh Hà Nam",
    "Trung tâm Kiểm soát bệnh tật tỉnh Bắc Giang",
    "UBND xã Đông Hưng",
    "Công ty TNHH MTV Cấp thoát nước Cần Thơ",
    "Sở Nông nghiệp và Môi trường tỉnh Sóc Trăng",
    "Trường Đại học Sư phạm Kỹ thuật Vinh",
    "Ban Quản lý dự án giao thông tỉnh Quảng Ngãi",
]

PACKAGE_TYPES = ["hàng hóa", "xây lắp", "tư vấn", "phi tư vấn", "hỗn hợp"]
SELECTION_METHODS = [
    "đấu thầu rộng rãi trong nước",
    "chào hàng cạnh tranh",
    "chỉ định thầu rút gọn",
    "đấu thầu rộng rãi qua mạng",
]
FUNDING_SOURCES = [
    "Ngân sách nhà nước năm 2026",
    "Vốn sự nghiệp y tế",
    "Vốn đầu tư công trung hạn 2021-2025",
    "Nguồn thu hợp pháp của đơn vị",
]

PACKAGE_TEMPLATES = {
    "hàng hóa": [
        "Mua sắm thiết bị công nghệ thông tin phục vụ chuyển đổi số",
        "Mua sắm trang thiết bị y tế phục vụ khám chữa bệnh",
        "Mua sắm bàn ghế, thiết bị dạy học năm học 2026-2027",
        "Mua sắm vật tư văn phòng phẩm định kỳ",
    ],
    "xây lắp": [
        "Thi công xây dựng công trình đường giao thông nông thôn",
        "Cải tạo, sửa chữa trụ sở làm việc",
        "Xây dựng hệ thống thoát nước khu dân cư",
    ],
    "tư vấn": [
        "Tư vấn lập báo cáo nghiên cứu khả thi dự án",
        "Tư vấn giám sát thi công xây dựng công trình",
        "Tư vấn thiết kế bản vẽ thi công",
    ],
    "phi tư vấn": [
        "Dịch vụ bảo trì hệ thống mạng và máy chủ",
        "Dịch vụ vệ sinh công nghiệp trụ sở cơ quan",
    ],
    "hỗn hợp": [
        "Cung cấp và lắp đặt hệ thống điện năng lượng mặt trời",
        "Cung cấp, lắp đặt và vận hành thử hệ thống camera giám sát",
    ],
}


def _random_date(start: date, end: date) -> date:
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))


def build_records(n: int = 20) -> list[dict]:
    records = []
    base_publish = date(2026, 1, 1)
    for i in range(1, n + 1):
        package_type = random.choice(PACKAGE_TYPES)
        title = random.choice(PACKAGE_TEMPLATES[package_type])
        publish = _random_date(base_publish, date(2026, 6, 30))
        close = publish + timedelta(days=random.randint(15, 30))
        value = round(random.uniform(2, 1) + random.uniform(0.05, 50), 3) * 1_000_000_000
        record = {
            "tbmt_id": f"IB260{1000 + i:04d}",
            "package_name": f"Gói thầu số {i:02d}: {title}",
            "investor": random.choice(INVESTORS),
            "procuring_entity": random.choice(INVESTORS),
            "package_value": round(value, -3),
            "currency": "VND",
            "funding_source": random.choice(FUNDING_SOURCES),
            "selection_method": random.choice(SELECTION_METHODS),
            "contract_type": "Trọn gói",
            "package_type": package_type,
            "execution_time": f"{random.choice([60, 90, 120, 180])} ngày",
            "publish_date": publish.isoformat(),
            "close_date": close.isoformat(),
            "attachments": [f"E-HSMT_goithau{i:02d}.pdf"],
            "source_url": f"https://muasamcong.mpi.gov.vn/o/sample/IB260{1000 + i:04d}",
        }
        records.append(record)
    return records


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records(20)
    out_file = SAMPLES_DIR / "tender_notices.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Đã ghi {len(records)} bản ghi mẫu vào {out_file}")


if __name__ == "__main__":
    main()
