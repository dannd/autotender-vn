import threading
from datetime import datetime

from autotender.hitl.feedback import export_feedback
from autotender.hitl.store import HitlStore
from autotender.schemas import ExtractedField, HSMTDocument, HSMTSection, TenderNotice


def _make_doc() -> HSMTDocument:
    now = datetime.now()
    package = TenderNotice(tbmt_id="IB1", package_name="Test", investor="Inv", source_url="https://x")
    sections = [
        HSMTSection(section_id="chuong_III.muc_1", title="A", generated_text="ban goc", model_tier=3, generated_at=now),
        HSMTSection(section_id="chuong_III.muc_2", title="B", generated_text="ban goc 2", model_tier=3, generated_at=now),
    ]
    return HSMTDocument(doc_id="doc1", package=package, sections=sections, created_at=now, updated_at=now)


def test_save_and_get_document_roundtrip(tmp_path):
    store = HitlStore(tmp_path / "test.db")
    doc = _make_doc()
    store.save_document(doc)

    loaded = store.get_document("doc1")
    assert loaded is not None
    assert loaded.package.package_name == "Test"
    assert len(loaded.sections) == 2
    assert loaded.sections[0].status == "draft"
    store.close()


def test_save_and_get_document_roundtrip_preserves_fields(tmp_path):
    """Hồi quy: `documents` từng KHÔNG có cột lưu `fields` — `save_document`/`get_document`
    âm thầm làm rỗng `doc.fields` sau mỗi lần lưu/tải lại, nên mọi giá trị NER trích được
    (kể cả PACKAGE_NAME/INVESTOR đã đồng bộ từ form xác nhận, xem
    `orchestrator._sync_package_into_fields`) đều biến mất khi Trang 3 tải lại tài liệu qua
    `store.get_document`, dù `create_document` trả về đúng trong bộ nhớ."""
    store = HitlStore(tmp_path / "test.db")
    now = datetime.now()
    package = TenderNotice(tbmt_id="IB9", package_name="Test", investor="Inv", source_url="https://x")
    fields = [
        ExtractedField(name="PACKAGE_NAME", value="Test", confidence=1.0, source="manual"),
        ExtractedField(name="VALUE", value="1.000.000.000 đồng", confidence=0.9, source="regex"),
    ]
    doc = HSMTDocument(doc_id="doc9", package=package, fields=fields, sections=[], created_at=now, updated_at=now)
    store.save_document(doc)

    loaded = store.get_document("doc9")
    assert loaded is not None
    assert len(loaded.fields) == 2
    assert next(f.value for f in loaded.fields if f.name == "PACKAGE_NAME") == "Test"
    assert next(f.value for f in loaded.fields if f.name == "VALUE") == "1.000.000.000 đồng"
    store.close()


def test_edit_and_approve_section_updates_status(tmp_path):
    store = HitlStore(tmp_path / "test.db")
    store.save_document(_make_doc())

    store.edit_section_text("doc1", "chuong_III.muc_1", "ban da sua")
    section = store.get_section("doc1", "chuong_III.muc_1")
    assert section.status == "edited"
    assert section.edited_text == "ban da sua"
    assert section.current_text == "ban da sua"

    log = store.get_edit_log("doc1", "chuong_III.muc_1")
    assert len(log) == 1
    assert log[0]["after_text"] == "ban da sua"

    store.approve_section("doc1", "chuong_III.muc_1", approved_by="nguyendan1987")
    section = store.get_section("doc1", "chuong_III.muc_1")
    assert section.status == "approved"
    assert section.approved_by == "nguyendan1987"
    store.close()


def test_document_not_fully_approved_until_all_sections_approved(tmp_path):
    store = HitlStore(tmp_path / "test.db")
    store.save_document(_make_doc())
    store.approve_section("doc1", "chuong_III.muc_1", approved_by="x")

    doc = store.get_document("doc1")
    assert doc.is_fully_approved is False
    assert doc.approval_progress == (1, 2)

    store.approve_section("doc1", "chuong_III.muc_2", approved_by="x")
    doc = store.get_document("doc1")
    assert doc.is_fully_approved is True
    store.close()


def test_flag_feedback_and_export(tmp_path):
    store = HitlStore(tmp_path / "test.db")
    store.save_document(_make_doc())
    store.edit_section_text("doc1", "chuong_III.muc_1", "sua roi")
    store.record_flag_feedback("doc1", "chuong_III.muc_1", rule_code="R1", verdict="false_positive", note="ok")

    gen_path, comp_path = export_feedback(store, tmp_path / "out")
    assert gen_path.exists()
    assert comp_path.exists()
    assert len(gen_path.read_text(encoding="utf-8").strip().splitlines()) == 1
    assert len(comp_path.read_text(encoding="utf-8").strip().splitlines()) == 1
    store.close()


def test_approval_log_for_pdf_appendix(tmp_path):
    store = HitlStore(tmp_path / "test.db")
    store.save_document(_make_doc())
    store.approve_section("doc1", "chuong_III.muc_1", approved_by="reviewer1")

    log = store.get_approval_log("doc1")
    assert len(log) == 2
    approved_row = next(r for r in log if r["section_id"] == "chuong_III.muc_1")
    assert approved_row["approved_by"] == "reviewer1"
    store.close()


def test_opening_pre_existing_db_without_fields_column_migrates_cleanly(tmp_path):
    """Hồi quy migration: DB đã tồn tại từ TRƯỚC khi thêm cột `fields_json` (vd auth.db/hitl.db
    của các phiên chạy demo trước) phải tự nâng cấp schema khi mở lại, không lỗi
    "no such column", và tài liệu cũ (không có fields) đọc lại vẫn trả về `fields=[]` hợp lệ
    thay vì crash."""
    import sqlite3

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE documents (doc_id TEXT PRIMARY KEY, package_json TEXT, created_at TEXT, updated_at TEXT)")
    package = TenderNotice(tbmt_id="OLD1", package_name="Legacy", investor="Inv", source_url="https://x")
    now = datetime.now()
    conn.execute(
        "INSERT INTO documents (doc_id, package_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("doc_legacy", package.model_dump_json(), now.isoformat(), now.isoformat()),
    )
    conn.commit()
    conn.close()

    store = HitlStore(db_path)
    loaded = store.get_document("doc_legacy")
    assert loaded is not None
    assert loaded.fields == []
    assert loaded.package.package_name == "Legacy"
    store.close()


def test_concurrent_edits_do_not_corrupt_or_lose_writes(tmp_path):
    """Hồi quy khoá đồng thời (`@_locked`, `hitl/store.py`): `get_store()` cache DÙNG CHUNG
    1 `HitlStore` cho MỌI người dùng (`app/common.py`), mỗi phiên Streamlit chạy trên 1
    thread riêng — nhiều người dùng sửa cùng 1 mục cùng lúc trước đây có thể mất bản ghi
    (lost update, đọc-rồi-ghi không nguyên tử) hoặc lỗi "database is locked". Mô phỏng bằng
    nhiều thread liên tục sửa/duyệt cùng 1 mục — không được lỗi, và mỗi lần sửa phải sinh
    đúng 1 dòng edit_log (chứng minh đọc-ghi không bị chen ngang giữa chừng)."""
    store = HitlStore(tmp_path / "test.db")
    store.save_document(_make_doc())

    n_threads, edits_per_thread = 8, 10
    errors: list[Exception] = []

    def _worker(thread_id: int) -> None:
        try:
            for i in range(edits_per_thread):
                store.edit_section_text("doc1", "chuong_III.muc_1", f"thread-{thread_id}-edit-{i}")
                store.approve_section("doc1", "chuong_III.muc_1", approved_by=f"user-{thread_id}")
        except Exception as e:  # noqa: BLE001 — muốn bắt MỌI lỗi (kể cả "database is locked")
            errors.append(e)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    section = store.get_section("doc1", "chuong_III.muc_1")
    assert section is not None
    assert section.status == "approved"  # trạng thái cuối nhất quán, không kẹt giữa chừng
    log = store.get_edit_log("doc1", "chuong_III.muc_1")
    assert len(log) == n_threads * edits_per_thread  # không mất/nhân đôi dòng log nào
    store.close()
