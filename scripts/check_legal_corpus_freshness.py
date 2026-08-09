"""Kiểm tra độ mới của kho tri thức luật thật (data/samples/legal_corpus/) — 2 chế độ.

Mặc định (KHÔNG cần mạng): báo số ngày kể từ lần fetch gần nhất (`fetched_at` lưu trong
mỗi Điều, xem knowledge/legal_fetch.py::LegalArticle) cho từng văn bản, cảnh báo văn bản
nào đã quá `--max-age-days` ngày (mặc định 90). Luật đấu thầu VN sửa đổi khá thường xuyên
— chính kho tri thức này đã phải thay Nghị định 24/2024 bằng 214/2025 giữa dự án (xem
scripts/fetch_legal_corpus.py) — nên corpus "im lặng" quá lâu là rủi ro thật cho một hệ RAG
pháp lý (soạn HSMT dựa trên luật đã hết hiệu lực), không phải lo xa.

`--check-live` (CẦN mạng + `playwright install chromium`): fetch lại từng nguồn bằng đúng
logic trong scripts/fetch_legal_corpus.py, so sánh SỐ ĐIỀU và hash nội dung với bản cục bộ
hiện có — phát hiện nguồn đã đổi (luật mới ban hành/thay thế, trang chỉnh sửa) mà local
corpus chưa cập nhật theo. Không tự động ghi đè corpus cũ — chỉ báo cáo, người dùng tự
quyết định chạy lại `fetch_legal_corpus.py` sau khi xem xét thay đổi là gì.

Ví dụ: python scripts/check_legal_corpus_freshness.py
       python scripts/check_legal_corpus_freshness.py --max-age-days 30
       python scripts/check_legal_corpus_freshness.py --check-live
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from autotender.config import resolve_path  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402
from autotender.utils.logging import get_logger  # noqa: E402

ensure_utf8_console()
logger = get_logger(__name__)

CORPUS_DIR = resolve_path("data/samples/legal_corpus")


def _load_local_articles(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _content_hash(articles: list[dict]) -> str:
    combined = "\n".join(a["text"] for a in articles)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _fetch_live_articles(law_id: str) -> list:
    """Fetch lại 1 nguồn bằng đúng dispatch logic của fetch_legal_corpus.py (import module
    đó thay vì chép lại — 1 nơi duy nhất biết cách fetch từng loại nguồn)."""
    import fetch_legal_corpus as flc

    if law_id in flc.SOURCES:
        return flc.fetch_and_parse(flc.SOURCES[law_id])
    if law_id in flc.GXD_HTML_SOURCES:
        cfg = flc.GXD_HTML_SOURCES[law_id]
        return flc.fetch_and_parse_gxd(
            cfg["url"], law_id, cfg["law_name"],
            initial_chuong_so=cfg.get("initial_chuong_so"), initial_chuong_title=cfg.get("initial_chuong_title"),
        )
    cfg = flc.LUATVIETNAM_SOURCES[law_id]
    return flc.fetch_and_parse_luatvietnam(cfg["url"], law_id, cfg["law_name"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-age-days", type=int, default=90, help="Cảnh báo nếu fetch cách đây quá N ngày (mặc định 90)")
    parser.add_argument("--check-live", action="store_true", help="Fetch lại nguồn thật để so sánh (cần mạng + playwright)")
    args = parser.parse_args()

    corpus_files = sorted(CORPUS_DIR.glob("*.jsonl"))
    if not corpus_files:
        parser.error(f"Không tìm thấy file .jsonl nào trong {CORPUS_DIR}")

    now = datetime.now(timezone.utc)
    any_issue = False

    for path in corpus_files:
        law_id = path.stem
        articles = _load_local_articles(path)
        if not articles:
            logger.warning("%s: file rỗng, bỏ qua.", law_id)
            continue

        fetched_at = datetime.fromisoformat(articles[0]["fetched_at"])
        age_days = (now - fetched_at).days
        stale = age_days > args.max_age_days
        any_issue = any_issue or stale
        status = "⚠️  QUÁ HẠN" if stale else "✅ còn mới"
        print(f"{status} | {law_id:<24} | {len(articles):>3} Điều | fetch cách đây {age_days} ngày ({fetched_at.date()})")

        if args.check_live:
            try:
                live_articles = _fetch_live_articles(law_id)
            except Exception as e:  # noqa: BLE001 — lỗi fetch (mạng, thay đổi cấu trúc trang...) không nên chặn các law_id khác
                logger.error("  → fetch live thất bại cho %s: %s", law_id, e)
                any_issue = True
                continue
            live_dicts = [a.model_dump(mode="json") for a in live_articles]
            local_count, live_count = len(articles), len(live_dicts)
            local_hash, live_hash = _content_hash(articles), _content_hash(live_dicts)
            if local_hash != live_hash:
                any_issue = True
                print(
                    f"  → ⚠️  NỘI DUNG NGUỒN ĐÃ THAY ĐỔI kể từ lần fetch gần nhất "
                    f"(local {local_count} Điều vs live {live_count} Điều) — rà soát rồi chạy lại "
                    f"`python scripts/fetch_legal_corpus.py --law {law_id}`."
                )
            else:
                print("  → nội dung khớp với nguồn live.")

    if any_issue:
        print("\nCó văn bản cần rà soát lại — xem cảnh báo ⚠️  ở trên.")
        sys.exit(1)
    print("\nToàn bộ kho tri thức còn trong hạn.")


if __name__ == "__main__":
    main()
