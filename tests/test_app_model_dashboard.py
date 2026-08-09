"""Regression test: Trang 6 (Bảng điều khiển Model) từng crash với AttributeError vì
`orch.retriever` giờ là `HybridLegalRetriever` (không phải `BaseModule` 3-tier cũ), không
có thuộc tính `active_tier` — trang phải tự chạy được (không exception) sau khi cập nhật.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PAGE = Path(__file__).resolve().parents[1] / "app" / "pages" / "6_Bang_dieu_khien_Model.py"


@pytest.mark.skipif(not APP_PAGE.exists(), reason="Streamlit page not found")
def test_model_dashboard_loads_without_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOTENDER_DB_PATH", str(tmp_path / "test_app.db"))
    monkeypatch.setenv("AUTOTENDER_AUDIT_DB_PATH", str(tmp_path / "test_audit.db"))
    monkeypatch.setenv("AUTOTENDER_SKIP_AUTH", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    at = AppTest.from_file(str(APP_PAGE))
    at.run(timeout=60)

    assert not at.exception
