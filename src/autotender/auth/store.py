"""Xác thực người dùng — SQLite, mật khẩu băm PBKDF2-HMAC-SHA256 (thư viện chuẩn, không
thêm dependency mới cho một tính năng nhỏ). Mục tiêu: đủ an toàn cho bản beta dùng nội bộ
vài người (cán bộ đấu thầu), KHÔNG nhắm thay thế SSO/OIDC doanh nghiệp thật — nếu triển
khai cho tổ chức lớn, nên thay bằng `st.login` (OIDC, có sẵn từ Streamlit 1.42+) trỏ vào
nhà cung cấp danh tính thật thay vì tự quản lý mật khẩu.

Vì sao cần: bản demo/đồ án ban đầu không có xác thực — bất kỳ ai vào được địa chỉ mạng đều
sửa/duyệt/xuất tài liệu được, không có cách nào biết CHÍNH XÁC ai đã phê duyệt mục nào (yêu
cầu bắt buộc cho một công cụ có giá trị pháp lý trong đấu thầu — xem `hitl/store.py`,
`approved_by` giờ phải là định danh người dùng thật đã đăng nhập, không phải chuỗi tự gõ).
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_PBKDF2_ITERATIONS = 600_000  # khuyến nghị OWASP 2023 cho PBKDF2-HMAC-SHA256

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'editor',
    created_at TEXT NOT NULL
);
"""


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS).hex()


class UserAlreadyExistsError(Exception):
    pass


class AuthStore:
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

    def create_user(self, username: str, password: str, display_name: str, role: str = "editor") -> None:
        with self._lock:
            existing = self._conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
            if existing is not None:
                raise UserAlreadyExistsError(f"Tài khoản '{username}' đã tồn tại.")
            salt = secrets.token_bytes(16)
            self._conn.execute(
                "INSERT INTO users (username, display_name, password_hash, salt, role, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (username, display_name, _hash_password(password, salt), salt.hex(), role, datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()

    def verify_password(self, username: str, password: str) -> dict | None:
        """Trả về thông tin người dùng (không kèm hash) nếu đúng mật khẩu, `None` nếu sai
        hoặc tài khoản không tồn tại — KHÔNG phân biệt 2 trường hợp này trong thông báo lỗi
        ở tầng gọi (tránh lộ thông tin tài khoản nào tồn tại — user enumeration)."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if row is None:
            # Vẫn chạy hash 1 lần dù không có user, để thời gian phản hồi gần giống nhau
            # giữa "sai username" và "sai password" — giảm rủi ro dò username qua timing.
            _hash_password(password, secrets.token_bytes(16))
            return None
        expected = row["password_hash"]
        actual = _hash_password(password, bytes.fromhex(row["salt"]))
        if not secrets.compare_digest(expected, actual):
            return None
        return {"username": row["username"], "display_name": row["display_name"], "role": row["role"]}

    def list_users(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT username, display_name, role, created_at FROM users ORDER BY username"
            ).fetchall()
        return [dict(r) for r in rows]
