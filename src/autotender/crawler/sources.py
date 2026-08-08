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

    CẬP NHẬT ĐIỀU TRA (xem docs/DATA_CARD.md mục 8 để biết chi tiết đầy đủ): có ít nhất
    2 endpoint `smart/search` khác nhau tuỳ portlet:
    - `POST /o/egp-portal-home/services/smart/search` (widget tìm kiếm nhanh ở trang chủ)
    - `POST /o/egp-portal-contractor-selection-v2/services/smart/search?token=<...>` (trang
      tra cứu đầy đủ tại `/web/guest/contractor-selection?render=search`, có phân trang thật)

    Cả hai đều yêu cầu một tham số `token` trên query string — đây là CSRF token động do
    Liferay sinh ra và nhúng vào từng lần render trang, KHÔNG PHẢI cố định, và KHÔNG THỂ suy
    ra được chỉ từ việc đọc mã nguồn client hay đoán payload. Đây là lý do mọi lần gọi POST
    trực tiếp (không qua trình duyệt thật) đều nhận HTTP 400 dù có cookie phiên hợp lệ.

    GIỚI HẠN ĐÃ BIẾT: vì token phải lấy từ một trang đã render (cần chạy JS thật), class này
    KHÔNG THỂ tự lấy token nếu chỉ dùng httpx thuần — bắt buộc phải qua `MSCBrowserSource`
    (Playwright) để có phiên trình duyệt thật sinh ra token hợp lệ. `MSCApiSource` vẫn giữ lại
    làm nơi implement parser/cache dùng chung, nhưng luôn raise `TenderSourceError` trong thực
    tế nếu không được cấp `payload_template` kèm token hợp lệ (lấy thủ công qua DevTools).
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
    """Fallback dùng Playwright để render trang tra cứu thật khi gọi API trực tiếp không được.

    Chiến lược: mở trang tra cứu đầy đủ (`/web/guest/contractor-selection?render=search`),
    bấm nút "Tìm kiếm" để trang tự gọi API `smart/search` kèm CSRF `token` động do chính nó
    sinh ra (xem docstring `MSCApiSource`), bắt lại response qua `page.on("response")`, rồi
    bấm nút phân trang (`.btn-next`) lặp lại để lấy thêm trang, có nghỉ giữa các lần bấm theo
    đúng rate-limit (Mục 2.4 SPEC).

    GIỚI HẠN ĐÃ XÁC NHẬN (quan trọng — đọc trước khi debug lỗi timeout):
    Trang này có WAF/anti-bot chặn hoặc treo vô thời hạn các phiên Playwright headless khởi
    chạy từ môi trường máy chủ/cloud/CI (đã kiểm chứng: cùng đoạn code này chạy OK khi thao
    tác qua trình duyệt tương tác thật, nhưng `chromium.launch(headless=True)` gọi trực tiếp
    từ một quy trình Python trong môi trường sandbox/cloud bị treo ở bước `page.goto` dù dùng
    `channel="msedge"` hay `headless=False`). Nếu chạy source này mà liên tục timeout ở
    `page.goto`, nhiều khả năng đây là do IP/mạng của máy đang chạy bị WAF liệt vào danh sách
    chặn tự động hoá — thử lại từ mạng dân dụng (nhà/văn phòng) thay vì máy chủ/cloud.
    Ngoài ra, WAF có vẻ cũng theo dõi tốc độ thao tác trong 1 phiên: bấm phân trang quá
    nhanh/nhiều trong thời gian ngắn có thể khiến toàn bộ phiên (kể cả điều hướng thường)
    bị từ chối — nên `_MIN_PAGINATION_INTERVAL_SECONDS` KHÔNG được hạ thấp hơn giá trị mặc định.
    """

    name = "browser"
    SEARCH_URL = "https://muasamcong.mpi.gov.vn/web/guest/contractor-selection?render=search"
    _MIN_PAGINATION_INTERVAL_SECONDS = 2.5
    _MAX_PAGES = 40

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

        import time

        captured: list[dict] = []

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
                page.goto(self.SEARCH_URL, wait_until="networkidle", timeout=self._cfg.timeout_seconds * 1000)
                page.click(".button__search", timeout=self._cfg.timeout_seconds * 1000)
                page.wait_for_timeout(int(self._MIN_PAGINATION_INTERVAL_SECONDS * 1000))

                def _total_captured() -> int:
                    return sum(len((pl.get("page") or {}).get("content", [])) for pl in captured)

                clicks = 0
                while _total_captured() < max_records and clicks < self._MAX_PAGES:
                    next_btn = page.locator(".btn-next").first
                    if next_btn.count() == 0 or next_btn.is_disabled():
                        break
                    next_btn.click()
                    clicks += 1
                    time.sleep(self._MIN_PAGINATION_INTERVAL_SECONDS)
                browser.close()
        except Exception as e:  # noqa: BLE001
            raise TenderSourceError(f"MSCBrowserSource thất bại: {e}") from e

        fetched = 0
        seen_ids: set[str] = set()
        for payload in captured:
            for raw in (payload.get("page") or {}).get("content", []):
                if fetched >= max_records:
                    return
                notify_no = raw.get("notifyNo")
                if notify_no and notify_no in seen_ids:
                    continue
                if notify_no:
                    seen_ids.add(notify_no)
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
                f"Không tìm thấy {notices_file} — chạy scripts/build_samples.py hoặc kiểm tra data/samples/."
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
