"""CLI thu thập dữ liệu thông báo mời thầu (M0).

Ví dụ:
    python scripts/crawl.py --from 2025-01-01 --to 2026-06-30 --max-records 3000 --out data/raw/
    python scripts/crawl.py --source local --max-records 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.config import get_crawler_settings  # noqa: E402
from autotender.crawler.pipeline import run_crawl  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402
from autotender.utils.logging import get_logger  # noqa: E402

ensure_utf8_console()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thu thập thông báo mời thầu từ Hệ thống mạng đấu thầu quốc gia.")
    parser.add_argument("--from", dest="date_from", default="2025-01-01", help="Ngày bắt đầu (YYYY-MM-DD)")
    parser.add_argument("--to", dest="date_to", default="2026-06-30", help="Ngày kết thúc (YYYY-MM-DD)")
    parser.add_argument("--max-records", type=int, default=20, help="Số bản ghi tối đa cần thu thập")
    parser.add_argument("--out", default="data/raw", help="Thư mục ghi kết quả")
    parser.add_argument(
        "--source",
        choices=["api", "browser", "local"],
        default=None,
        help="Chỉ định 1 nguồn cụ thể (bỏ qua cơ chế fallback tự động)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_crawler_settings().crawler

    notices, source_used = run_crawl(
        cfg=settings,
        date_from=args.date_from,
        date_to=args.date_to,
        max_records=args.max_records,
        out_dir=args.out,
        only_source=args.source,
    )

    logger.info("=" * 60)
    logger.info("Hoàn tất: %d bản ghi, nguồn dùng: '%s'", len(notices), source_used)
    if notices:
        logger.info("Ví dụ bản ghi đầu tiên: %s — %s", notices[0].tbmt_id, notices[0].package_name)


if __name__ == "__main__":
    main()
