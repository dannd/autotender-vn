"""Crawl thông báo mời thầu công khai từ dauthau.asia (Mục 2.4 SPEC — thu thập có trách nhiệm).

Trang này tổng hợp lại dữ liệu công khai từ Hệ thống mạng đấu thầu quốc gia và có Điều
khoản sử dụng yêu cầu xin phép bằng văn bản để sao chép/tái sử dụng ngoài phạm vi cá
nhân/nội bộ/phi thương mại (xem docs/DATA_CARD.md mục 9). Script này CHỈ dùng cho mục
đích học thuật/nội bộ phi thương mại của đồ án — không dùng lại cho mục đích khác mà
không rà soát lại điều khoản.

Khác với M0 (muasamcong.mpi.gov.vn, cần CSRF token động qua trình duyệt thật), trang
này dùng phân trang URL tĩnh (`?page=N`), không cần trình duyệt/token — gọi thẳng bằng
httpx, tôn trọng robots.txt và rate-limit qua `MscHttpClient` dùng chung với M0.

Ví dụ: python scripts/crawl_dauthau_asia.py --max-pages 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.config import CrawlerConfig, resolve_path  # noqa: E402
from autotender.crawler.msc_client import MscHttpClient  # noqa: E402
from autotender.crawler.parser import parse_dauthau_asia_rows  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402
from autotender.utils.logging import get_logger  # noqa: E402

ensure_utf8_console()
logger = get_logger(__name__)

LISTING_PATH = "/thongbao/moithau/"


def build_config() -> CrawlerConfig:
    return CrawlerConfig(
        base_url="https://dauthau.asia",
        user_agent=(
            "AutoTenderVN-ResearchBot/0.1 (+mailto:nguyendan1987@gmail.com; "
            "muc dich: do an Deep Learning, hoc thuat, phi thuong mai)"
        ),
        min_request_interval_seconds=2.5,
        respect_robots_txt=True,
        timeout_seconds=30,
        max_retries=3,
        cache_dir="data/raw/dauthau_asia",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl công khai dauthau.asia (mục đích học thuật, phi thương mại).")
    parser.add_argument("--max-pages", type=int, default=10, help="Số trang tối đa (20 bản ghi/trang)")
    parser.add_argument("--out", default="data/samples/real_dauthau_asia_sample.jsonl", help="File output jsonl")
    args = parser.parse_args()

    cfg = build_config()
    cache_root = resolve_path(cfg.cache_dir)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_notices = []
    seen_ids: set[str] = set()

    with MscHttpClient(cfg, cache_root) as client:
        for page_num in range(1, args.max_pages + 1):
            params = None if page_num == 1 else {"page": page_num}
            logger.info("Đang tải trang %d/%d...", page_num, args.max_pages)
            try:
                html = client.request_text("GET", LISTING_PATH, params=params)
            except Exception as e:  # noqa: BLE001
                logger.warning("Trang %d thất bại, dừng lại: %s", page_num, e)
                break

            notices = parse_dauthau_asia_rows(html)
            if not notices:
                logger.info("Trang %d không còn dữ liệu, dừng lại.", page_num)
                break

            new_count = 0
            for n in notices:
                if n.tbmt_id not in seen_ids:
                    seen_ids.add(n.tbmt_id)
                    all_notices.append(n)
                    new_count += 1
            logger.info("Trang %d: %d bản ghi (%d mới, %d trùng).", page_num, len(notices), new_count, len(notices) - new_count)

    with open(out_path, "w", encoding="utf-8") as f:
        for n in all_notices:
            f.write(n.model_dump_json() + "\n")

    logger.info("=" * 60)
    logger.info("Hoàn tất: %d bản ghi thật (unique) ghi vào %s", len(all_notices), out_path)


if __name__ == "__main__":
    main()
