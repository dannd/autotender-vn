"""Regression test: chuyển đổi tài liệu ở Trang 3 không được lẫn nội dung mục có cùng
section_id giữa 2 tài liệu khác nhau (session_state key phải theo cả doc_id lẫn section_id).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from autotender.hitl.store import HitlStore
from autotender.schemas import HSMTDocument, HSMTSection, TenderNotice

APP_PAGE = Path(__file__).resolve().parents[1] / "app" / "pages" / "3_Soan_thao_HSMT.py"


def _make_doc(doc_id: str, text: str) -> HSMTDocument:
    now = datetime.now()
    package = TenderNotice(tbmt_id=doc_id, package_name=f"Package {doc_id}", investor="Inv", source_url="https://x")
    sections = [
        HSMTSection(section_id="chuong_III.muc_1", title="Mục 1", generated_text=text, model_tier=3, generated_at=now)
    ]
    return HSMTDocument(doc_id=doc_id, package=package, sections=sections, created_at=now, updated_at=now)


@pytest.mark.skipif(not APP_PAGE.exists(), reason="Streamlit page not found")
def test_switching_document_does_not_leak_stale_section_text(tmp_path, monkeypatch):
    db_path = tmp_path / "test_app.db"
    monkeypatch.setenv("AUTOTENDER_DB_PATH", str(db_path))

    store = HitlStore(db_path)
    store.save_document(_make_doc("doc_A", "Nội dung riêng của doc_A"))
    store.save_document(_make_doc("doc_B", "Nội dung HOÀN TOÀN khác của doc_B"))
    store.close()

    at = AppTest.from_file(str(APP_PAGE))
    at.run(timeout=60)
    assert not at.exception

    at.selectbox[0].select("doc_A").run(timeout=60)
    text_area_a = next(w for w in at.text_area if "editor_doc_A_" in w.key)
    assert text_area_a.value == "Nội dung riêng của doc_A"

    at.selectbox[0].select("doc_B").run(timeout=60)
    text_area_b = next(w for w in at.text_area if "editor_doc_B_" in w.key)
    assert text_area_b.value == "Nội dung HOÀN TOÀN khác của doc_B"
