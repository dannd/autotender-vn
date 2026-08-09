"""Nhật ký kiểm toán bất biến (append-only) gắn với người dùng đã đăng nhập
(`app/auth_ui.py`) — ai đã đăng nhập/đăng xuất, sửa/duyệt/từ chối mục nào, xuất file nào,
lúc nào. Đây là điều kiện bắt buộc cho một công cụ có giá trị pháp lý trong đấu thầu: khi
có tranh chấp, phải trả lời được chính xác "ai đã làm gì, lúc nào" — không thể suy luận
ngược từ trạng thái hiện tại.

Bất biến được ép ở 2 lớp: (1) class này chỉ có `record()` (INSERT) và `list_events()`
(SELECT) — không có `update`/`delete`; (2) trigger SQLite chặn UPDATE/DELETE ở tầng DB,
phòng trường hợp có code khác lỡ thao tác trực tiếp trên bảng.
"""

from __future__ import annotations

import functools
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    username TEXT NOT NULL,
    action TEXT NOT NULL,
    doc_id TEXT,
    section_id TEXT,
    detail TEXT
);

CREATE TRIGGER IF NOT EXISTS audit_log_no_update
BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log là bảng chỉ-ghi-thêm (append-only), không được sửa');
END;

CREATE TRIGGER IF NOT EXISTS audit_log_no_delete
BEFORE DELETE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log là bảng chỉ-ghi-thêm (append-only), không được xoá');
END;
"""


def _locked(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class AuditLog:
    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @_locked
    def record(
        self,
        username: str,
        action: str,
        doc_id: str | None = None,
        section_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO audit_log (ts, username, action, doc_id, section_id, detail) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), username, action, doc_id, section_id, detail),
        )
        self._conn.commit()

    @_locked
    def list_events(self, limit: int = 200) -> list[dict]:
        rows = self._conn.execute(
            "SELECT ts, username, action, doc_id, section_id, detail FROM audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
