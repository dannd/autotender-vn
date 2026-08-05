"""Tiện ích dùng chung cho toàn bộ 6 trang Streamlit (Mục 7)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autotender.config import get_app_settings, resolve_path  # noqa: E402
from autotender.hitl.store import HitlStore  # noqa: E402
from autotender.models.legal_qa import LegalQAModule  # noqa: E402
from autotender.pipeline.orchestrator import Orchestrator  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402

ensure_utf8_console()

TIER_BADGE = {
    1: "🟢 Tier 1 — fine-tuned",
    2: "🟡 Tier 2 — pretrained",
    3: "🔵 Tier 3 — rule-based",
}
STATUS_ICON = {"draft": "⏳", "edited": "✏️", "approved": "✅", "rejected": "❌"}
SEVERITY_COLOR = {"cao": "🔴", "trung_binh": "🟠", "thap": "🟡"}


@st.cache_resource(show_spinner="Đang khởi tạo pipeline (NER, RAG, Generator, Compliance)...")
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


@st.cache_resource(show_spinner="Đang khởi tạo module Hỏi-đáp (Mức 1)...")
def get_qa_module() -> LegalQAModule:
    return LegalQAModule()


@st.cache_resource
def get_store() -> HitlStore:
    # Cho phép ghi đè đường dẫn DB qua biến môi trường — dùng trong test (AppTest) để
    # cách ly khỏi data/processed/hitl.db thật, không ảnh hưởng hành vi khi chạy thật.
    override = os.environ.get("AUTOTENDER_DB_PATH")
    if override:
        return HitlStore(Path(override))
    settings = get_app_settings()
    return HitlStore(resolve_path(settings.app.db_path))


def tier_badge(tier: int | None) -> str:
    return TIER_BADGE.get(tier, "⚪ Không xác định")


def status_icon(status: str) -> str:
    return STATUS_ICON.get(status, "❓")


def severity_icon(severity: str) -> str:
    return SEVERITY_COLOR.get(severity, "⚪")


def init_page(title: str) -> None:
    st.set_page_config(page_title=f"AutoTender-VN — {title}", page_icon="📑", layout="wide")
    st.title(title)
