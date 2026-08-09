import sqlite3

import pytest

from autotender.audit.store import AuditLog


def test_record_and_list_events_most_recent_first(tmp_path):
    log = AuditLog(tmp_path / "audit.db")
    log.record("dan", "login_success")
    log.record("dan", "approve_section", doc_id="doc_1", section_id="chuong_III.muc_1")

    events = log.list_events()

    assert len(events) == 2
    assert events[0]["action"] == "approve_section"
    assert events[0]["doc_id"] == "doc_1"
    assert events[0]["section_id"] == "chuong_III.muc_1"
    assert events[1]["action"] == "login_success"
    log.close()


def test_list_events_respects_limit(tmp_path):
    log = AuditLog(tmp_path / "audit.db")
    for i in range(5):
        log.record("dan", f"action_{i}")

    events = log.list_events(limit=2)

    assert len(events) == 2
    assert events[0]["action"] == "action_4"
    log.close()


def test_update_is_rejected_by_db_trigger(tmp_path):
    log = AuditLog(tmp_path / "audit.db")
    log.record("dan", "login_success")

    with pytest.raises(sqlite3.IntegrityError):
        log._conn.execute("UPDATE audit_log SET action = 'tampered' WHERE id = 1")
    log.close()


def test_delete_is_rejected_by_db_trigger(tmp_path):
    log = AuditLog(tmp_path / "audit.db")
    log.record("dan", "login_success")

    with pytest.raises(sqlite3.IntegrityError):
        log._conn.execute("DELETE FROM audit_log WHERE id = 1")
    log.close()
