"""Orchestrator M0: chọn nguồn thu thập theo thứ tự ưu tiên, resume, ghi kết quả.

Thứ tự fallback (Degraded Mode, Mục 2.1): api -> browser -> local.
Nếu người dùng chỉ định `--source` cụ thể thì chỉ dùng đúng nguồn đó (không fallback),
để tiện debug từng tầng riêng lẻ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from autotender.config import CrawlerConfig, resolve_path
from autotender.crawler.sources import (
    LocalSampleSource,
    MSCApiSource,
    MSCBrowserSource,
    TenderSource,
    TenderSourceError,
)
from autotender.schemas import TenderNotice
from autotender.utils.logging import get_logger

logger = get_logger(__name__)

_FALLBACK_ORDER = ["api", "browser", "local"]


@dataclass
class CrawlState:
    """Trạng thái crawl, lưu vào `_state.json` để resume được sau khi ngắt."""

    last_source_used: str
    records_fetched: int
    completed: bool
    updated_at: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CrawlState | None":
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(**data)
        except (json.JSONDecodeError, TypeError):
            return None


def _build_sources(cfg: CrawlerConfig, only: str | None) -> list[TenderSource]:
    cache_root = resolve_path(cfg.cache_dir)
    samples_dir = resolve_path(cfg.samples_dir)
    registry: dict[str, TenderSource] = {
        "api": MSCApiSource(cfg, cache_root),
        "browser": MSCBrowserSource(cfg, cache_root),
        "local": LocalSampleSource(samples_dir),
    }
    if only:
        if only not in registry:
            raise ValueError(f"Nguồn không hợp lệ: {only}. Chọn trong {list(registry)}.")
        return [registry[only]]
    return [registry[name] for name in _FALLBACK_ORDER]


def run_crawl(
    cfg: CrawlerConfig,
    date_from: str,
    date_to: str,
    max_records: int,
    out_dir: str,
    only_source: str | None = None,
) -> tuple[list[TenderNotice], str]:
    """Chạy pipeline thu thập với fallback tự động. Trả về (notices, source_used)."""
    out_path = resolve_path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    state_path = resolve_path(cfg.state_file)

    sources = _build_sources(cfg, only_source)
    last_error: Exception | None = None

    for source in sources:
        logger.info("Thử thu thập bằng nguồn: %s", source.name)
        try:
            notices = list(source.fetch(date_from, date_to, max_records))
            if not notices:
                logger.warning("Nguồn %s không trả về bản ghi nào, thử nguồn kế tiếp.", source.name)
                continue
            state = CrawlState(
                last_source_used=source.name,
                records_fetched=len(notices),
                completed=True,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            state.save(state_path)
            _write_notices(notices, out_path / "notices.jsonl")
            logger.info("Thu thập thành công %d bản ghi từ nguồn '%s'.", len(notices), source.name)
            return notices, source.name
        except TenderSourceError as e:
            logger.warning("Nguồn %s thất bại: %s", source.name, e)
            last_error = e
            continue

    # Không nguồn nào thành công — chỉ xảy ra nếu cả LocalSampleSource cũng lỗi
    # (ví dụ thiếu file mẫu), đây là lỗi cấu hình dự án, không phải lỗi runtime bình thường.
    raise RuntimeError(
        f"Tất cả nguồn thu thập đều thất bại (kể cả LocalSampleSource). Lỗi cuối: {last_error}"
    )


def _write_notices(notices: list[TenderNotice], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for n in notices:
            f.write(n.model_dump_json() + "\n")
