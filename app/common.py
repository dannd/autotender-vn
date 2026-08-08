"""Tiện ích dùng chung cho toàn bộ 6 trang Streamlit (Mục 7)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

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
# M5 Generator và Mức 1 Hỏi-đáp (LegalQAModule) dùng cùng khung BaseModule 3-tier
# nhưng Tier 1 của 2 module này là gọi Claude API (đề cương RAG+LLM), KHÔNG phải
# checkpoint tự fine-tune như M2 NER/M6 Compliance — nhãn riêng để không gây hiểu nhầm.
TIER_BADGE_CLAUDE = {
    1: "🟢 Tier 1 — Claude API",
    2: "🟡 Tier 2 — dự phòng (chưa dùng)",
    3: "🔵 Tier 3 — template/trích dẫn không qua LLM",
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


def tier_badge_claude(tier: int | None) -> str:
    """Badge cho module có Tier 1 = Claude API (M5 Generator, Mức 1 Hỏi-đáp) — xem
    `TIER_BADGE_CLAUDE`."""
    return TIER_BADGE_CLAUDE.get(tier, "⚪ Không xác định")


def status_icon(status: str) -> str:
    return STATUS_ICON.get(status, "❓")


def severity_icon(severity: str) -> str:
    return SEVERITY_COLOR.get(severity, "⚪")


_CUSTOM_CSS = """
<style>
/* Khai báo ngôn ngữ trang là tiếng Việt — trình đọc màn hình phát âm đúng dấu thanh,
   trình duyệt chọn đúng quy tắc ngắt dòng/gợi ý chính tả cho tiếng Việt. */
html { -webkit-text-size-adjust: 100%; }

/* Văn bản pháp lý tiếng Việt câu dài, nhiều dấu — tăng line-height giúp dễ đọc hơn
   mức mặc định vốn tối ưu cho tiếng Anh câu ngắn. */
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
  line-height: 1.65;
}

/* Số liệu (giá gói thầu, điểm truy hồi, %) canh cột đều — quan trọng cho bảng
   Đánh giá/Kiểm tra tuân thủ có nhiều dòng số liên tiếp. */
[data-testid="stMetricValue"], [data-testid="stDataFrame"], code {
  font-variant-numeric: tabular-nums;
}

/* Card viền mảnh nhất quán cho các khối trích dẫn/trường thông tin dùng
   st.container(border=True) khắp app — thay viền xám mặc định bằng màu theo theme. */
[data-testid="stExpander"] {
  border-radius: 0.5rem;
}

/* Tiêu đề trang: thêm gạch dưới mảnh theo màu accent để phân tách rõ với nội dung,
   tránh cảm giác "trang trần" của tiêu đề Streamlit mặc định. */
h1 {
  padding-bottom: 0.6rem;
  border-bottom: 2px solid var(--primary-color, #2B6E62);
}

/* Menu điều hướng (st.navigation): giãn dòng + bo góc cho từng mục, tiêu đề nhóm
   ("Soạn thảo HSMT", "Hỏi-đáp & Phân tích") in hoa nhẹ để phân biệt với mục trang. */
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNav"] li a {
  border-radius: 0.4rem;
}
[data-testid="stSidebarNav"] > div > span,
[data-testid="stSidebarNav"] p {
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 0.75rem;
  opacity: 0.7;
}
</style>
"""


def init_page(title: str) -> None:
    st.set_page_config(page_title=f"AutoTender-VN — {title}", page_icon="📑", layout="wide")
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    # `st.markdown(..., unsafe_allow_html=True)` chèn <script> qua innerHTML — trình
    # duyệt KHÔNG thực thi script chèn kiểu này (theo đặc tả DOM). `components.html`
    # dựng iframe cùng-origin có script thực thi thật, từ đó với tới document cha để
    # đặt `lang="vi"` — giúp trình đọc màn hình phát âm đúng dấu thanh tiếng Việt.
    components.html(
        "<script>window.parent.document.documentElement.lang = 'vi';</script>", height=0
    )
    # Gọi sau khi `st.navigation` đã vẽ menu — hiển thị dưới danh sách trang, không đè.
    with st.sidebar:
        st.divider()
        st.caption("📑 **AutoTender-VN** — Trợ lý soạn HSMT phần mềm/CNTT bằng RAG + Claude API")
    st.title(title)
