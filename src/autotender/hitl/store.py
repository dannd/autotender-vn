"""HITL store — SQLite, lưu trạng thái tài liệu/mục, nhật ký sửa và feedback (Mục 9).

Human-in-the-loop là mặc định (Mục 2.3): không mục nào được coi là "hoàn thành" nếu
chưa qua `approve_section`. Mọi lần sửa `edited_text` đều được ghi vào `edit_log`.
"""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from autotender.schemas import ComplianceFlag, HSMTDocument, HSMTSection, RetrievedChunk, TenderNotice


def _locked(method):
    """Serialize mọi thao tác đọc/ghi qua `self._lock` — một `sqlite3.Connection` dùng
    `check_same_thread=False` (bắt buộc vì Streamlit chạy mỗi phiên người dùng trên 1
    thread riêng, xem `app/common.py::get_store` — `@st.cache_resource` dùng CHUNG 1
    instance `HitlStore` cho MỌI người dùng) không tự động an toàn khi nhiều thread gọi
    đồng thời — đặc biệt các thao tác đọc-rồi-ghi (`edit_section_text`, `approve_section`,
    `reject_section`: đọc section hiện tại rồi ghi lại toàn bộ) có thể mất dữ liệu nếu 2
    người dùng thao tác cùng lúc trên cùng 1 mục (lost update). Khoá ở mức Python đơn giản
    và đủ an toàn cho quy mô vài người dùng đồng thời của bản beta — không nhắm tới
    throughput cao (xem docs/DATA_CARD.md/README.md phần "chưa cấp thiết ở quy mô hiện tại"
    về việc chưa cần chuyển sang Postgres)."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, package_json TEXT, created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS sections (
    section_id TEXT, doc_id TEXT, title TEXT, generated_text TEXT,
    edited_text TEXT, status TEXT, model_tier INT,
    citations_json TEXT, flags_json TEXT,
    approved_by TEXT, approved_at TEXT, generated_at TEXT,
    PRIMARY KEY(doc_id, section_id)
);
CREATE TABLE IF NOT EXISTS edit_log (
    id INTEGER PRIMARY KEY, doc_id TEXT, section_id TEXT,
    before_text TEXT, after_text TEXT, edited_at TEXT
);
CREATE TABLE IF NOT EXISTS flag_feedback (
    id INTEGER PRIMARY KEY, doc_id TEXT, section_id TEXT,
    rule_code TEXT, user_verdict TEXT, note TEXT, created_at TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class HitlStore:
    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL: cho phép đọc song song trong lúc đang ghi (mặc định SQLite khoá cả file khi
        # ghi) — giảm khả năng "database is locked" khi nhiều người dùng thao tác gần nhau.
        # busy_timeout: nếu vẫn đụng khoá, CHỜ tới 5s rồi mới lỗi, thay vì lỗi ngay lập tức.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "HitlStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- documents ------------------------------------------------------------
    @_locked
    def save_document(self, doc: HSMTDocument) -> None:
        self._conn.execute(
            """INSERT INTO documents (doc_id, package_json, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(doc_id) DO UPDATE SET package_json=excluded.package_json, updated_at=excluded.updated_at""",
            (doc.doc_id, doc.package.model_dump_json(), doc.created_at.isoformat(), _now()),
        )
        for section in doc.sections:
            self.upsert_section(doc.doc_id, section, log_edit=False)
        self._conn.commit()

    @_locked
    def get_document(self, doc_id: str) -> HSMTDocument | None:
        row = self._conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        package = TenderNotice.model_validate_json(row["package_json"])
        sections = self.list_sections(doc_id)
        return HSMTDocument(
            doc_id=doc_id,
            package=package,
            sections=sections,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @_locked
    def list_documents(self) -> list[dict]:
        rows = self._conn.execute("SELECT doc_id, updated_at FROM documents ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]

    @_locked
    def delete_document(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM sections WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM edit_log WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM flag_feedback WHERE doc_id = ?", (doc_id,))
        self._conn.commit()

    # -- sections ---------------------------------------------------------------
    @_locked
    def upsert_section(self, doc_id: str, section: HSMTSection, log_edit: bool = True) -> None:
        existing = self._conn.execute(
            "SELECT edited_text FROM sections WHERE doc_id = ? AND section_id = ?", (doc_id, section.section_id)
        ).fetchone()

        if log_edit and existing is not None:
            before = existing["edited_text"] or ""
            after = section.edited_text or ""
            if before != after:
                self._conn.execute(
                    "INSERT INTO edit_log (doc_id, section_id, before_text, after_text, edited_at) VALUES (?, ?, ?, ?, ?)",
                    (doc_id, section.section_id, before, after, _now()),
                )

        self._conn.execute(
            """INSERT INTO sections
               (section_id, doc_id, title, generated_text, edited_text, status, model_tier,
                citations_json, flags_json, approved_by, approved_at, generated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(doc_id, section_id) DO UPDATE SET
                 title=excluded.title, generated_text=excluded.generated_text,
                 edited_text=excluded.edited_text, status=excluded.status, model_tier=excluded.model_tier,
                 citations_json=excluded.citations_json, flags_json=excluded.flags_json,
                 approved_by=excluded.approved_by, approved_at=excluded.approved_at""",
            (
                section.section_id,
                doc_id,
                section.title,
                section.generated_text,
                section.edited_text,
                section.status,
                section.model_tier,
                json.dumps([c.model_dump() for c in section.citations], default=str),
                json.dumps([f.model_dump() for f in section.flags], default=str),
                section.approved_by,
                section.approved_at.isoformat() if section.approved_at else None,
                section.generated_at.isoformat(),
            ),
        )
        self._conn.execute("UPDATE documents SET updated_at = ? WHERE doc_id = ?", (_now(), doc_id))
        self._conn.commit()

    @_locked
    def get_section(self, doc_id: str, section_id: str) -> HSMTSection | None:
        row = self._conn.execute(
            "SELECT * FROM sections WHERE doc_id = ? AND section_id = ?", (doc_id, section_id)
        ).fetchone()
        return self._row_to_section(row) if row else None

    @_locked
    def list_sections(self, doc_id: str) -> list[HSMTSection]:
        rows = self._conn.execute(
            "SELECT * FROM sections WHERE doc_id = ? ORDER BY section_id", (doc_id,)
        ).fetchall()
        return [self._row_to_section(r) for r in rows]

    @staticmethod
    def _row_to_section(row: sqlite3.Row) -> HSMTSection:
        citations = [RetrievedChunk.model_validate(c) for c in json.loads(row["citations_json"] or "[]")]
        flags = [ComplianceFlag.model_validate(f) for f in json.loads(row["flags_json"] or "[]")]
        return HSMTSection(
            section_id=row["section_id"],
            title=row["title"],
            generated_text=row["generated_text"],
            edited_text=row["edited_text"],
            status=row["status"],
            citations=citations,
            flags=flags,
            model_tier=row["model_tier"],
            generated_at=datetime.fromisoformat(row["generated_at"]),
            approved_by=row["approved_by"],
            approved_at=datetime.fromisoformat(row["approved_at"]) if row["approved_at"] else None,
        )

    # -- human actions (Mục 2.3) --------------------------------------------
    # `@_locked` ở đây QUAN TRỌNG hơn các chỗ khác: mỗi hàm là đọc-rồi-ghi (get_section rồi
    # upsert_section lại toàn bộ) — nếu 2 người dùng thao tác cùng lúc trên cùng 1 mục mà
    # không có khoá, người ghi sau có thể ghi đè mất thay đổi của người ghi trước dựa trên
    # dữ liệu đọc đã cũ (lost update). Khoá đảm bảo trọn cặp đọc-ghi là 1 khối nguyên tử.
    @_locked
    def edit_section_text(self, doc_id: str, section_id: str, new_text: str) -> None:
        section = self.get_section(doc_id, section_id)
        if section is None:
            raise ValueError(f"Không tìm thấy mục {section_id} trong tài liệu {doc_id}")
        section.edited_text = new_text
        section.status = "edited"
        self.upsert_section(doc_id, section, log_edit=True)

    @_locked
    def approve_section(self, doc_id: str, section_id: str, approved_by: str) -> None:
        section = self.get_section(doc_id, section_id)
        if section is None:
            raise ValueError(f"Không tìm thấy mục {section_id} trong tài liệu {doc_id}")
        section.status = "approved"
        section.approved_by = approved_by
        section.approved_at = datetime.now(timezone.utc)
        self.upsert_section(doc_id, section, log_edit=False)

    @_locked
    def reject_section(self, doc_id: str, section_id: str) -> None:
        section = self.get_section(doc_id, section_id)
        if section is None:
            raise ValueError(f"Không tìm thấy mục {section_id} trong tài liệu {doc_id}")
        section.status = "rejected"
        section.approved_by = None
        section.approved_at = None
        self.upsert_section(doc_id, section, log_edit=False)

    @_locked
    def get_edit_log(self, doc_id: str, section_id: str | None = None) -> list[dict]:
        if section_id:
            rows = self._conn.execute(
                "SELECT * FROM edit_log WHERE doc_id = ? AND section_id = ? ORDER BY edited_at",
                (doc_id, section_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM edit_log WHERE doc_id = ? ORDER BY edited_at", (doc_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    @_locked
    def get_approval_log(self, doc_id: str) -> list[dict]:
        """Nhật ký phê duyệt cho phụ lục PDF (Mục 8): ai duyệt mục nào, lúc nào."""
        rows = self._conn.execute(
            "SELECT section_id, title, status, approved_by, approved_at FROM sections "
            "WHERE doc_id = ? ORDER BY section_id",
            (doc_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- flag feedback (Trang 4, dùng để huấn luyện lại M6) ----------------------
    @_locked
    def record_flag_feedback(self, doc_id: str, section_id: str, rule_code: str, verdict: str, note: str = "") -> None:
        self._conn.execute(
            "INSERT INTO flag_feedback (doc_id, section_id, rule_code, user_verdict, note, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, section_id, rule_code, verdict, note, _now()),
        )
        self._conn.commit()

    @_locked
    def list_flag_feedback(self, doc_id: str | None = None) -> list[dict]:
        if doc_id:
            rows = self._conn.execute("SELECT * FROM flag_feedback WHERE doc_id = ?", (doc_id,)).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM flag_feedback").fetchall()
        return [dict(r) for r in rows]
