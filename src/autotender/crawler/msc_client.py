"""HTTP client dùng chung cho crawler — thu thập dữ liệu có trách nhiệm (Mục 2.4).

Nguyên tắc bắt buộc:
- Tôn trọng robots.txt.
- Rate limit tối thiểu 1 request / min_request_interval_seconds giây.
- User-Agent khai báo rõ mục đích nghiên cứu.
- Cache toàn bộ response xuống đĩa, không crawl lại dữ liệu đã có.
"""

from __future__ import annotations

import hashlib
import json
import ssl
import time
from pathlib import Path
from typing import Any
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import httpx

from autotender.config import CrawlerConfig
from autotender.utils.logging import get_logger

logger = get_logger(__name__)


def _lenient_ssl_context() -> ssl.SSLContext:
    """Một số cổng thông tin nhà nước dùng tham số DH key yếu, OpenSSL 3 mặc định từ chối.

    Hạ SECLEVEL để vẫn thiết lập được kết nối TLS (dữ liệu vẫn mã hoá, chỉ nới lỏng
    yêu cầu độ dài khoá DH tối thiểu). Chỉ áp dụng cho domain thu thập công khai này.
    """
    ctx = ssl.create_default_context()
    ctx.set_ciphers("DEFAULT@SECLEVEL=1")
    return ctx


class RobotsDisallowedError(Exception):
    """robots.txt của site không cho phép truy cập đường dẫn này."""


class MscHttpClient:
    """Wrapper quanh httpx.Client với rate-limit, cache-to-disk và kiểm tra robots.txt."""

    def __init__(self, cfg: CrawlerConfig, cache_root: Path):
        self._cfg = cfg
        self._cache_root = cache_root
        self._cache_root.mkdir(parents=True, exist_ok=True)
        self._last_request_ts = 0.0
        self._robots: robotparser.RobotFileParser | None = None
        self._client = httpx.Client(
            verify=_lenient_ssl_context(),
            timeout=cfg.timeout_seconds,
            headers={"User-Agent": cfg.user_agent},
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MscHttpClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- robots.txt -----------------------------------------------------
    def _load_robots(self) -> robotparser.RobotFileParser:
        if self._robots is not None:
            return self._robots
        rp = robotparser.RobotFileParser()
        robots_url = urljoin(self._cfg.base_url, "/robots.txt")
        try:
            resp = self._client.get(robots_url)
            rp.parse(resp.text.splitlines())
        except httpx.HTTPError as e:
            logger.warning("Không tải được robots.txt (%s), mặc định coi là Disallow toàn bộ.", e)
            rp.parse(["User-agent: *", "Disallow: /"])
        self._robots = rp
        return rp

    def _check_allowed(self, url: str) -> None:
        if not self._cfg.respect_robots_txt:
            return
        rp = self._load_robots()
        if not rp.can_fetch(self._cfg.user_agent, url):
            raise RobotsDisallowedError(f"robots.txt từ chối truy cập: {url}")

    # -- rate limit -------------------------------------------------------
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        wait = self._cfg.min_request_interval_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    # -- cache --------------------------------------------------------------
    def _cache_path(self, method: str, url: str, payload: dict[str, Any] | None) -> Path:
        key_src = f"{method}:{url}:{json.dumps(payload or {}, sort_keys=True)}"
        key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
        host = urlparse(url).netloc.replace(":", "_")
        return self._cache_root / host / f"{key}.json"

    def request_json(
        self,
        method: str,
        path_or_url: str,
        json_body: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Gọi API JSON nội bộ, có cache và rate-limit. Raise nếu không thành công."""
        url = urljoin(self._cfg.base_url, path_or_url)
        cache_file = self._cache_path(method, url, json_body)
        if use_cache and cache_file.exists():
            logger.debug("Cache hit: %s", url)
            return json.loads(cache_file.read_text(encoding="utf-8"))

        self._check_allowed(url)
        self._throttle()

        last_error: Exception | None = None
        for attempt in range(1, self._cfg.max_retries + 1):
            try:
                resp = self._client.request(method, url, json=json_body)
                resp.raise_for_status()
                data = resp.json()
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                return data
            except (httpx.HTTPError, ValueError) as e:
                last_error = e
                logger.warning("Lần thử %d/%d thất bại cho %s: %s", attempt, self._cfg.max_retries, url, e)
                time.sleep(min(2**attempt, 10))
        assert last_error is not None
        raise last_error

    def request_text(self, method: str, path_or_url: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> str:
        """Gọi trang HTML thường (GET), có cache và rate-limit. Dùng cho nguồn không có API JSON."""
        url = urljoin(self._cfg.base_url, path_or_url)
        cache_key = {"params": params or {}}
        cache_file = self._cache_path(method, url, cache_key)
        if use_cache and cache_file.exists():
            logger.debug("Cache hit: %s", url)
            return cache_file.read_text(encoding="utf-8")

        self._check_allowed(url)
        self._throttle()

        last_error: Exception | None = None
        for attempt in range(1, self._cfg.max_retries + 1):
            try:
                resp = self._client.request(method, url, params=params)
                resp.raise_for_status()
                text = resp.text
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(text, encoding="utf-8")
                return text
            except httpx.HTTPError as e:
                last_error = e
                logger.warning("Lần thử %d/%d thất bại cho %s: %s", attempt, self._cfg.max_retries, url, e)
                time.sleep(min(2**attempt, 10))
        assert last_error is not None
        raise last_error
