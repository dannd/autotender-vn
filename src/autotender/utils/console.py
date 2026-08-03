"""Đảm bảo stdout/stderr in được tiếng Việt trên console Windows (mặc định cp1252)."""

from __future__ import annotations

import sys


def ensure_utf8_console() -> None:
    if sys.platform == "win32":
        for stream_name in ("stdout", "stderr"):
            stream = getattr(sys, stream_name)
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
