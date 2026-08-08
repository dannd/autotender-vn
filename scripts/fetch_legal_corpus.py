"""Fetch văn bản pháp luật thật (nguyên văn) cho kho tri thức RAG.

Nguồn: xaydungchinhsach.chinhphu.vn (Báo điện tử Chính phủ) — đã kiểm tra robots.txt
cho phép, không có bot-specific disallow (khác vbpl.vn/thuvienphapluat.vn — xem
docstring `autotender.knowledge.legal_fetch`).

Ví dụ: python scripts/fetch_legal_corpus.py --law luat_22_2023_qh15
       python scripts/fetch_legal_corpus.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.config import resolve_path  # noqa: E402
from autotender.knowledge.legal_fetch import (  # noqa: E402
    LegalDocSource,
    fetch_and_parse,
    fetch_and_parse_gxd,
    fetch_and_parse_luatvietnam,
)
from autotender.utils.console import ensure_utf8_console  # noqa: E402
from autotender.utils.logging import get_logger  # noqa: E402

ensure_utf8_console()
logger = get_logger(__name__)

OUT_DIR = resolve_path("data/samples/legal_corpus")

# Đăng ký nguồn văn bản luật. Thêm dần qua các ngày (Nghị định, Thông tư...).
SOURCES: dict[str, LegalDocSource] = {
    "luat_22_2023_qh15": LegalDocSource(
        law_id="luat_22_2023_qh15",
        # Dùng bản HỢP NHẤT (đã tích hợp sửa đổi từ Luật 57/2024/QH15 và Luật 90/2025/QH15)
        # thay vì bản gốc 2023 đơn thuần — đúng hiện trạng pháp luật áp dụng năm 2026.
        law_name="Luật Đấu thầu số 22/2023/QH15 (hợp nhất với Luật 57/2024/QH15, Luật 90/2025/QH15)",
        url="https://dauthau.gxd.vn/van-ban/luat/luat-dau-thau-22-57-90-2025.html",
        start_marker="Chương I QUY ĐỊNH CHUNG",
        start_occurrence=2,  # trang có Mục lục lặp lại heading trước phần nội dung thật
        end_marker=(
            "Đoạn đầu khoản 1 này được sửa đổi, bổ sung theo quy định tại điểm a khoản 1 "
            "Điều 1 Luật số 90/2025/QH15."
        ),
    ),
    "nd_214_2025_ndcp": LegalDocSource(
        law_id="nd_214_2025_ndcp",
        # Nghị định 214/2025/NĐ-CP (hiệu lực 4/8/2025) THAY THẾ Nghị định 24/2024/NĐ-CP —
        # phát hiện qua đối chiếu Điều 44 Luật (bản hợp nhất) đã đơn giản hoá đáng kể so với
        # nội dung chi tiết cũ, tra cứu lại thì NĐ 24/2024 đã hết hiệu lực. KHÔNG dùng NĐ
        # 24/2024 làm căn cứ pháp luật hiện hành.
        law_name="Nghị định 214/2025/NĐ-CP (thay thế Nghị định 24/2024/NĐ-CP, hiệu lực từ 04/8/2025)",
        url="https://dauthau.gxd.vn/van-ban/dau-thau/nghi-dinh-214-2025.html",
        start_marker="Chương I NHỮNG QUY ĐỊNH CHUNG",
        start_occurrence=2,  # trang có Mục lục lặp lại heading trước phần nội dung thật
        end_marker="Last Updated:",
    ),
}

# Nguồn cần parse theo cấu trúc THẺ HTML thật (không phải quét text phẳng) — trang có
# heading Điều lặp/thiếu do lỗi biên soạn nguồn, xem docstring
# `knowledge.legal_fetch.parse_gxd_theme_articles`. Đăng ký riêng vì dùng
# `fetch_and_parse_gxd(url, law_id, law_name, ...)` thay vì `LegalDocSource`.
GXD_HTML_SOURCES: dict[str, dict] = {
    "tt_01_2024_bkhdt": {
        "law_name": "Thông tư 01/2024/TT-BKHĐT (hướng dẫn cung cấp/đăng tải thông tin đấu thầu, mẫu hồ sơ)",
        "url": "https://dauthau.gxd.vn/van-ban/dau-thau/thong-tu-01-2024-TT-BKHDT.html",
        "initial_chuong_so": "I",
        "initial_chuong_title": "QUY ĐỊNH CHUNG",
    },
    "tt_22_2024_bkhdt": {
        "law_name": "Thông tư 22/2024/TT-BKHĐT (thay Thông tư 06/2024/TT-BKHĐT, hướng dẫn cung cấp/đăng tải thông tin đấu thầu, mẫu hồ sơ)",
        "url": "https://dauthau.gxd.vn/van-ban/dau-thau/thong-tu-22-2024-TT-BKHDT.html",
        "initial_chuong_so": "I",
        "initial_chuong_title": "QUY ĐỊNH CHUNG",
    },
}

# Nghị định 45/2026/NĐ-CP — văn bản chuyên ngành CNTT (Mục 1.5 đề cương). Bản chính thức
# duy nhất tại datafiles.chinhphu.vn là PDF SCAN ẢNH (0 ký tự text, xem DATA_CARD.md), nên
# dùng bản transcript dạng text tại luatvietnam.vn thay thế — cùng nội dung pháp luật thật,
# chỉ khác nguồn hiển thị (văn bản QPPL không thuộc đối tượng bảo hộ quyền tác giả, Điều 15
# Luật SHTT — xem docstring `LegalArticle`). Dùng `fetch_and_parse_luatvietnam` (httpx thuần,
# không cần Playwright) vì trang server-render sẵn nội dung.
LUATVIETNAM_SOURCES: dict[str, dict] = {
    "nd_45_2026_ndcp": {
        "law_name": "Nghị định 45/2026/NĐ-CP (quản lý đầu tư ứng dụng CNTT dùng vốn ngân sách nhà nước)",
        "url": "https://luatvietnam.vn/dau-tu/nghi-dinh-45-2026-nd-cp-quan-ly-dau-tu-cong-nghe-thong-tin-tu-ngan-sach-nha-nuoc-425010-d1.html",
    },
}


def save_articles(law_id: str, articles: list) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{law_id}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(a.model_dump_json() + "\n")
    return out_path


def main() -> None:
    all_ids = list(SOURCES) + list(GXD_HTML_SOURCES) + list(LUATVIETNAM_SOURCES)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--law", choices=all_ids, help="Chỉ fetch 1 văn bản")
    parser.add_argument("--all", action="store_true", help="Fetch tất cả văn bản đã đăng ký")
    args = parser.parse_args()

    if not args.law and not args.all:
        parser.error("Cần chỉ định --law <id> hoặc --all")

    law_ids = all_ids if args.all else [args.law]

    for law_id in law_ids:
        try:
            if law_id in SOURCES:
                articles = fetch_and_parse(SOURCES[law_id])
            elif law_id in GXD_HTML_SOURCES:
                cfg = GXD_HTML_SOURCES[law_id]
                articles = fetch_and_parse_gxd(
                    cfg["url"], law_id, cfg["law_name"],
                    initial_chuong_so=cfg.get("initial_chuong_so"), initial_chuong_title=cfg.get("initial_chuong_title"),
                )
            else:
                cfg = LUATVIETNAM_SOURCES[law_id]
                articles = fetch_and_parse_luatvietnam(cfg["url"], law_id, cfg["law_name"])
        except Exception as e:  # noqa: BLE001
            logger.error("Fetch/parse thất bại cho %s: %s", law_id, e)
            continue
        out_path = save_articles(law_id, articles)
        logger.info("Đã ghi %d Điều vào %s", len(articles), out_path)


if __name__ == "__main__":
    main()
