"""Crawl KHLCNT (Kế hoạch lựa chọn nhà thầu) công khai từ dauthau.asia, lọc theo từ khoá
(mặc định "phần mềm") — dùng để lấy thêm dữ liệu demo thật cho Trang 2 — Nạp KHLCNT.

Cùng nguồn/nguyên tắc với `crawl_dauthau_asia.py` (thu thập có trách nhiệm, chỉ mục đích
học thuật/nội bộ phi thương mại — xem docs/DATA_CARD.md mục 9): gọi thẳng bằng httpx qua
`MscHttpClient` dùng chung (rate-limit, cache, tôn trọng robots.txt — đã xác nhận
`/kehoach/luachon-nhathau/` không bị chặn), không cần trình duyệt/token vì trang tìm kiếm
này dùng query string tĩnh (`?q=<từ khoá>&type_info=2&page=N`).

`--enrich-details` vào từng trang chi tiết để lấy thêm nguồn vốn/hình thức LCNT/loại hợp
đồng/thời gian thực hiện/lĩnh vực/tóm tắt công việc (công khai, không cần đăng nhập). Giá
gói thầu vẫn bị khoá sau đăng nhập, giữ nguyên None — xem docstring
`parser.enrich_dauthau_asia_khlcnt_detail`. Khi bật cờ này, còn dựng thêm 1 văn bản dạng
"KHLCNT thô" cho mỗi bản ghi (đúng cú pháp nhãn mà `models/ner.py` nhận diện được:
"Tên gói thầu:", "Chủ đầu tư:", "Nguồn vốn:"...) để dán trực tiếp vào Trang 2 — Nạp KHLCNT
làm demo — dùng TOÀN BỘ giá trị THẬT đã crawl, KHÔNG bịa thêm trường nào (Mục 2.2 SPEC);
trường nào không lấy được (vd giá gói thầu) thì bỏ dòng đó, không để trống/giả định.

Ví dụ: python scripts/crawl_khlcnt_dauthau_asia.py --keyword "phần mềm" --max-records 30 --enrich-details
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.config import CrawlerConfig, resolve_path  # noqa: E402
from autotender.crawler.msc_client import MscHttpClient  # noqa: E402
from autotender.crawler.parser import (  # noqa: E402
    enrich_dauthau_asia_khlcnt_detail,
    extract_khlcnt_task_summary,
    parse_dauthau_asia_khlcnt_rows,
)
from autotender.schemas import TenderNotice  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402
from autotender.utils.logging import get_logger  # noqa: E402

ensure_utf8_console()
logger = get_logger(__name__)

SEARCH_PATH = "/kehoach/luachon-nhathau/"
# type_info=2: "Kế hoạch lựa chọn nhà thầu" (khác type_info=1 "Thông báo mời thầu" mà
# crawl_dauthau_asia.py đang dùng) — xác nhận qua thao tác thật trên form tìm kiếm của trang.
_BASE_PARAMS = {"type_search": 1, "type_info": 2, "searchkind": 0}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


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


def _field_line(label: str, value: str) -> str:
    # `orch.ingest_text` chạy qua `merge_broken_lines` (Mục 2.2 pipeline chuẩn hoá) TRƯỚC
    # khi đưa vào NER — dòng nào KHÔNG kết thúc bằng dấu câu sẽ bị nối vào dòng sau, khiến
    # regex tham lam (`[^\n.;]*`) của METHOD/CONTRACT_TYPE/DURATION nuốt luôn nhãn kế tiếp.
    # Luôn kết thúc mỗi dòng bằng dấu chấm (đúng cách văn bản KHLCNT thật vẫn viết) để mỗi
    # trường đứng độc lập sau khi chuẩn hoá.
    value = value.rstrip()
    if not value.endswith((".", "!", "?", ":", ";", "•", "-")):
        value += "."
    return f"{label}: {value}"


def render_khlcnt_text(notice: TenderNotice, task_summary: str | None) -> str:
    """Dựng văn bản "KHLCNT thô" pastable vào Trang 2, dùng đúng nhãn mà `models/ner.py`
    (regex) nhận diện được. Chỉ ghi dòng nào có giá trị THẬT đã crawl — không có (vd
    "Giá gói thầu" luôn bị khoá sau đăng nhập trên trang công khai) thì bỏ qua, không bịa
    (Mục 2.2 SPEC) — người dùng sẽ thấy field đó thiếu khi trích xuất, đúng thực tế."""
    lines = [_field_line("Tên gói thầu", notice.package_name)]
    lines.append(_field_line("Chủ đầu tư", notice.investor))
    if notice.procuring_entity and notice.procuring_entity != notice.investor:
        lines.append(_field_line("Bên mời thầu", notice.procuring_entity))
    if notice.funding_source:
        lines.append(_field_line("Nguồn vốn", notice.funding_source))
    if notice.selection_method:
        lines.append(_field_line("Hình thức lựa chọn nhà thầu", notice.selection_method))
    if notice.contract_type:
        lines.append(_field_line("Loại hợp đồng", notice.contract_type))
    if notice.execution_time:
        lines.append(_field_line("Thời gian thực hiện hợp đồng", notice.execution_time))
    if notice.package_type:
        lines.append(_field_line("Lĩnh vực", notice.package_type))
    if notice.publish_date:
        lines.append(_field_line("Ngày đăng tải", notice.publish_date.strftime("%d/%m/%Y")))
    if task_summary:
        lines.append("")
        lines.append(f"Tóm tắt công việc chính: {task_summary}")
    lines.append("")
    lines.append(f"Nguồn: {notice.source_url} (dauthau.asia, mã {notice.tbmt_id})")
    return "\n".join(lines)


def _slugify(tbmt_id: str) -> str:
    return _SLUG_RE.sub("-", tbmt_id.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl KHLCNT dauthau.asia lọc theo từ khoá (mục đích học thuật, phi thương mại).")
    parser.add_argument("--keyword", default="phần mềm", help="Từ khoá tìm kiếm (mặc định 'phần mềm')")
    parser.add_argument("--max-records", type=int, default=30, help="Số bản ghi tối đa cần lấy")
    parser.add_argument("--max-pages", type=int, default=30, help="Giới hạn số trang duyệt qua (phòng từ khoá quá hiếm)")
    parser.add_argument("--out", default="data/samples/khlcnt_phanmem_sample.jsonl", help="File output jsonl")
    parser.add_argument(
        "--enrich-details", action="store_true",
        help="Vào từng trang chi tiết lấy thêm nguồn vốn/hình thức LCNT/loại hợp đồng/thời gian thực "
        "hiện + dựng văn bản demo pastable vào Trang 2 (xem thư mục --demo-texts-dir).",
    )
    parser.add_argument("--max-details", type=int, default=None, help="Giới hạn số trang chi tiết ghé thăm (mặc định: tất cả)")
    parser.add_argument("--demo-texts-dir", default="data/samples/khlcnt_demo_texts", help="Thư mục ghi văn bản demo (.txt) khi bật --enrich-details")
    args = parser.parse_args()

    cfg = build_config()
    cache_root = resolve_path(cfg.cache_dir)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_notices: list[TenderNotice] = []
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

        if args.enrich_details:
            n_targets = min(args.max_details, len(all_notices)) if args.max_details else len(all_notices)
            demo_dir = resolve_path(args.demo_texts_dir)
            demo_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Bắt đầu ghé %d trang chi tiết (rate-limit %.1fs/trang)...", n_targets, cfg.min_request_interval_seconds)
            for i in range(n_targets):
                notice = all_notices[i]
                detail_path = notice.source_url.replace("https://dauthau.asia", "")
                try:
                    detail_html = client.request_text("GET", detail_path)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Bỏ qua trang chi tiết lỗi (%s): %s", notice.tbmt_id, e)
                    continue
                all_notices[i] = enrich_dauthau_asia_khlcnt_detail(notice, detail_html)
                summary = extract_khlcnt_task_summary(detail_html)
                text = render_khlcnt_text(all_notices[i], summary)
                (demo_dir / f"{_slugify(notice.tbmt_id)}.txt").write_text(text, encoding="utf-8")
                if (i + 1) % 10 == 0:
                    logger.info("  Đã xong %d/%d trang chi tiết...", i + 1, n_targets)
            logger.info("Đã ghi %d văn bản demo vào %s", n_targets, demo_dir)

    with open(out_path, "w", encoding="utf-8") as f:
        for n in all_notices:
            f.write(n.model_dump_json() + "\n")

    logger.info("=" * 60)
    logger.info("Hoàn tất: %d bản ghi thật (unique, từ khoá '%s') ghi vào %s", len(all_notices), args.keyword, out_path)


if __name__ == "__main__":
    main()
