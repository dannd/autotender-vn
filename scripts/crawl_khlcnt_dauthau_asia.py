"""Crawl KHLCNT (Kế hoạch lựa chọn nhà thầu) công khai từ dauthau.asia, lọc theo từ khoá
(mặc định "phần mềm") — dùng để lấy thêm dữ liệu demo thật cho Trang 2 — Nạp KHLCNT.

Cùng nguồn/nguyên tắc với `crawl_dauthau_asia.py` (thu thập có trách nhiệm, chỉ mục đích
học thuật/nội bộ phi thương mại — xem docs/DATA_CARD.md mục 9): gọi thẳng bằng httpx qua
`MscHttpClient` dùng chung (rate-limit, cache, tôn trọng robots.txt — đã xác nhận
`/kehoach/luachon-nhathau/` không bị chặn), không cần trình duyệt/token vì trang tìm kiếm
này dùng query string tĩnh (`?q=<từ khoá>&type_info=2&page=N`).

Ví dụ: python scripts/crawl_khlcnt_dauthau_asia.py --keyword "phần mềm" --max-records 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.config import CrawlerConfig, resolve_path  # noqa: E402
from autotender.crawler.msc_client import MscHttpClient  # noqa: E402
from autotender.crawler.parser import parse_dauthau_asia_khlcnt_rows  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402
from autotender.utils.logging import get_logger  # noqa: E402

ensure_utf8_console()
logger = get_logger(__name__)

SEARCH_PATH = "/kehoach/luachon-nhathau/"
# type_info=2: "Kế hoạch lựa chọn nhà thầu" (khác type_info=1 "Thông báo mời thầu" mà
# crawl_dauthau_asia.py đang dùng) — xác nhận qua thao tác thật trên form tìm kiếm của trang.
_BASE_PARAMS = {"type_search": 1, "type_info": 2, "searchkind": 0}


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
        cache_dir="data/raw/dauthau_asia_khlcnt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl KHLCNT dauthau.asia lọc theo từ khoá (mục đích học thuật, phi thương mại).")
    parser.add_argument("--keyword", default="phần mềm", help="Từ khoá tìm kiếm (mặc định 'phần mềm')")
    parser.add_argument("--max-records", type=int, default=30, help="Số bản ghi tối đa cần lấy")
    parser.add_argument("--max-pages", type=int, default=30, help="Giới hạn số trang duyệt qua (phòng từ khoá quá hiếm)")
    parser.add_argument("--out", default="data/samples/khlcnt_phanmem_sample.jsonl", help="File output jsonl")
    args = parser.parse_args()

    cfg = build_config()
    cache_root = resolve_path(cfg.cache_dir)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_notices = []
    seen_ids: set[str] = set()

    with MscHttpClient(cfg, cache_root) as client:
        for page_num in range(1, args.max_pages + 1):
            if len(all_notices) >= args.max_records:
                break
            params = {**_BASE_PARAMS, "q": args.keyword, "page": page_num}
            logger.info("Đang tải trang %d (đã có %d/%d bản ghi)...", page_num, len(all_notices), args.max_records)
            try:
                html = client.request_text("GET", SEARCH_PATH, params=params)
            except Exception as e:  # noqa: BLE001
                logger.warning("Trang %d thất bại, dừng lại: %s", page_num, e)
                break

            notices = parse_dauthau_asia_khlcnt_rows(html)
            if not notices:
                logger.info("Trang %d không còn dữ liệu, dừng lại.", page_num)
                break

            new_count = 0
            for n in notices:
                if len(all_notices) >= args.max_records:
                    break
                if n.tbmt_id not in seen_ids:
                    seen_ids.add(n.tbmt_id)
                    all_notices.append(n)
                    new_count += 1
            logger.info("Trang %d: %d bản ghi (%d mới, %d trùng).", page_num, len(notices), new_count, len(notices) - new_count)

    with open(out_path, "w", encoding="utf-8") as f:
        for n in all_notices:
            f.write(n.model_dump_json() + "\n")

    logger.info("=" * 60)
    logger.info("Hoàn tất: %d bản ghi thật (unique, từ khoá '%s') ghi vào %s", len(all_notices), args.keyword, out_path)


if __name__ == "__main__":
    main()
