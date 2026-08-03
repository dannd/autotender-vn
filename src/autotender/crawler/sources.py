"""Interface `TenderSource` và 3 cách triển khai thu thập dữ liệu (Mục 6/M0).

Thứ tự ưu tiên khi chạy pipeline: MSCApiSource -> MSCBrowserSource -> LocalSampleSource.
`LocalSampleSource` KHÔNG BAO GIỜ được phép lỗi — đây là tầng đảm bảo demo luôn chạy được.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from autotender.config import CrawlerConfig
from autotender.crawler.msc_client import MscHttpClient
from autotender.crawler.parser import parse_local_sample_record, parse_msc_api_record
from autotender.schemas import TenderNotice
from autotender.utils.logging import get_logger

logger = get_logger(__name__)


class TenderSourceError(Exception):
    """Nguồn dữ liệu không thể thu thập được (mạng lỗi, API đổi hợp đồng dữ liệu, bị chặn...)."""


class TenderSource(ABC):
    """Giao diện chung: mọi nguồn thu thập đều trả về iterator các `TenderNotice`."""

    name: str

    @abstractmethod
    def fetch(self, date_from: str, date_to: str, max_records: int) -> Iterator[TenderNotice]:
        """Sinh ra tối đa `max_records` bản ghi TenderNotice trong khoảng [date_from, date_to]."""
        raise NotImplementedError


class MSCApiSource(TenderSource):
    """Gọi API JSON nội bộ thật của muasamcong.mpi.gov.vn.

    Endpoint đã xác định qua DevTools Network: `POST /o/egp-portal-home/services/smart/search`
    (xem `autotender.crawler.parser` để biết shape response đã quan sát được).

    GIỚI HẠN ĐÃ BIẾT (ghi rõ trong báo cáo — Mục 14 của SPEC):
    cổng portal (Liferay) trả về HTTP 400 cho các payload JSON được thử một cách hợp lý
    (page/size, pageable, criteria, filter...) kể cả khi gọi cùng-origin có cookie phiên
    hợp lệ từ trình duyệt thật — hàm ý endpoint yêu cầu một hợp đồng payload nội bộ chưa
    xác định được trong thời gian cho phép của đồ án 7 ngày. Class này vẫn implement đầy đủ
    cơ chế gọi + parse để sẵn sàng dùng ngay khi payload đúng được xác định (ví dụ: người
    dùng tự bắt gói tin qua DevTools và điền vào `payload_template`), nhưng KHÔNG được coi
    là nguồn dữ liệu tin cậy cho pipeline — mọi lỗi sẽ raise `TenderSourceError` để
    orchestrator tự động rơi xuống `MSCBrowserSource` rồi `LocalSampleSource`.
    """

    name = "api"
    SEARCH_PATH = "/o/egp-portal-home/services/smart/search"

    def __init__(self, cfg: CrawlerConfig, cache_root: Path, payload_template: dict | None = None):
        self._cfg = cfg
        self._cache_root = cache_root
        # Payload mặc định — CHƯA xác nhận đúng hợp đồng thật của server (xem docstring).
        self._payload_template = payload_template or {"page": 0, "size": 20}

    def fetch(self, date_from: str, date_to: str, max_records: int) -> Iterator[TenderNotice]:
        fetched = 0
        page = 0
        page_size = min(max_records, 20) or 20
        with MscHttpClient(self._cfg, self._cache_root) as client:
            while fetched < max_records:
                payload = {**self._payload_template, "page": page, "size": page_size}
                try:
                    data = client.request_json("POST", self.SEARCH_PATH, json_body=payload)
                except Exception as e:  # noqa: BLE001 — muốn bắt mọi lỗi mạng/HTTP/parse
                    raise TenderSourceError(f"MSCApiSource thất bại (page={page}): {e}") from e

                content = (data.get("page") or {}).get("content", [])
                if not content:
                    break
                for raw in content:
                    if fetched >= max_records:
                        return
                    try:
                        yield parse_msc_api_record(raw)
                        fetched += 1
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Bỏ qua bản ghi lỗi parse: %s", e)
                if (data.get("page") or {}).get("last", True):
                    break
                page += 1


class MSCBrowserSource(TenderSource):
    """Fallback dùng Playwright headless để render SPA khi API trực tiếp không gọi được.

    Chiến lược: mở trang tìm kiếm thật, để JS nội bộ của trang tự gọi API `smart/search`
    bằng session cookie hợp lệ của chính nó, sau đó bắt lại response qua
    `page.on("response")` — tận dụng đúng cơ chế xác thực của trang thay vì đoán payload.
    """

    name = "browser"

    def __init__(self, cfg: CrawlerConfig, cache_root: Path):
        self._cfg = cfg
        self._cache_root = cache_root

    def fetch(self, date_from: str, date_to: str, max_records: int) -> Iterator[TenderNotice]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise TenderSourceError(
                "Playwright chưa được cài đặt (`pip install playwright && playwright install chromium`)."
            ) from e

        captured: list[dict] = []
        fetched = 0

        def _on_response(response) -> None:
            if "smart/search" in response.url and response.request.method == "POST":
                try:
                    captured.append(response.json())
                except Exception:  # noqa: BLE001 — response không phải JSON hợp lệ, bỏ qua
                    pass

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(user_agent=self._cfg.user_agent)
                page = context.new_page()
                page.on("response", _on_response)
                page.goto(self._cfg.base_url, wait_until="networkidle", timeout=self._cfg.timeout_seconds * 1000)
                browser.close()
        except Exception as e:  # noqa: BLE001
            raise TenderSourceError(f"MSCBrowserSource thất bại: {e}") from e

        for payload in captured:
            for raw in (payload.get("page") or {}).get("content", []):
                if fetched >= max_records:
                    return
                try:
                    yield parse_msc_api_record(raw)
                    fetched += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning("Bỏ qua bản ghi lỗi parse: %s", e)

        if fetched == 0:
            raise TenderSourceError("MSCBrowserSource không bắt được response smart/search nào.")


class LocalSampleSource(TenderSource):
    """Đọc từ `data/samples/*.json` — LUÔN HOẠT ĐỘNG, không phụ thuộc mạng.

    Đây là tầng đảm bảo phần mềm luôn demo được (Mục 2.1, Degraded Mode) và cũng là
    nguồn dữ liệu chính thức cho tiêu chí nghiệm thu "20 bản ghi mẫu" (Mục 6/M0).
    """

    name = "local"

    def __init__(self, samples_dir: Path):
        self._samples_dir = samples_dir

    def fetch(self, date_from: str, date_to: str, max_records: int) -> Iterator[TenderNotice]:
        notices_file = self._samples_dir / "tender_notices.jsonl"
        if not notices_file.exists():
            raise TenderSourceError(
                f"Không tìm thấy {notices_file} — chạy scripts/build_dataset.py hoặc kiểm tra data/samples/."
            )
        count = 0
        with open(notices_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if count >= max_records:
                    break
                raw = json.loads(line)
                yield parse_local_sample_record(raw)
                count += 1
