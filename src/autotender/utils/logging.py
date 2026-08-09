"""Cấu hình logging thống nhất cho toàn bộ hệ thống.

Mặc định in dạng text dễ đọc khi chạy cục bộ/dev (`streamlit run`, `python scripts/...`).
Đặt biến môi trường `AUTOTENDER_LOG_FORMAT=json` để chuyển sang JSON 1-dòng-1-record —
cần khi triển khai thật đằng sau hệ thống thu log tập trung (ELK/Loki/CloudWatch...), vốn
parse JSON đáng tin cậy hơn nhiều so với cố gắng regex text tự do.
"""

from __future__ import annotations

import json
import logging
import os
import sys

from autotender.utils.console import ensure_utf8_console

_CONFIGURED = False

_RESERVED_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # `extra={...}` truyền vào logger.info(..., extra={...}) được gắn thẳng vào
        # LogRecord như thuộc tính bậc-1 — lọc bỏ các thuộc tính chuẩn của LogRecord để
        # chỉ giữ lại phần custom, gộp vào JSON thay vì lồng vào 1 khoá phụ.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    ensure_utf8_console()
    handler = logging.StreamHandler(stream=sys.stdout)
    if os.environ.get("AUTOTENDER_LOG_FORMAT", "text").strip().lower() == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
