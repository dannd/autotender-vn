"""Trang 6 — Bảng điều khiển Model (Mục 7) — trang demo trước hội đồng.

Đã cập nhật cho bản redesign RAG+LLM: M4 (Retriever) không còn là `BaseModule` 3-tier
(là `HybridLegalRetriever`, luôn kết hợp dense+sparse, không có khái niệm "tier"); M2
(NER)/M6 (Compliance) cũng đã bỏ khung 3-tier (rule-based thuần, Tier 1/2 chưa từng chạy
thật trong đồ án — xem docs/MODEL_CARD.md); M5 (Generator) Tier 1 giờ là Claude API thay
vì checkpoint local. Chỉ M5 còn khái niệm "tier" (`active_tier`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auth_ui import current_user  # noqa: E402
from common import get_audit_log, get_orchestrator, init_page  # noqa: E402

from autotender.config import get_app_settings, get_models_settings, resolve_path  # noqa: E402
from autotender.generation.claude_client import get_session_cost_usd  # noqa: E402
from autotender.generation.claude_client import is_configured as is_claude_configured  # noqa: E402

init_page("6 — Bảng điều khiển Model")

orch = get_orchestrator()
cfg = get_models_settings()

st.subheader("M2 — NER & M6 — Compliance (rule-based thuần)")
st.dataframe(
    pd.DataFrame(
        [
            {"module": "M2 — NER", "phuong_phap": "regex + từ điển từ khoá", "so_nhan": len(cfg.ner.get("labels", []))},
            {"module": "M6 — Compliance", "phuong_phap": "từ điển nhãn hiệu + regex ngưỡng", "so_nhan": len(cfg.compliance.get("rule_codes", []))},
        ]
    ),
    use_container_width=True,
)
st.caption(
    "Cả 2 module luôn chạy trực tiếp (không có tier dự phòng) — khung 3-tier (checkpoint "
    "fine-tuned/zero-shot) trước đây chưa từng chạy thật trong đồ án nên đã được bỏ. Số liệu "
    "đánh giá thật xem bên dưới."
)

st.divider()
st.divider()
st.subheader("M4 — Retriever (Qdrant Vector DB + BM25)")
retriever = orch.retriever
c1, c2, c3 = st.columns(3)
c1.metric("Model embedding", retriever.model_key)
c2.metric("Số chunk trong kho tri thức", retriever.num_chunks)
c3.metric("Qdrant Dense Index", "✅ Đã kết nối" if retriever.has_dense_index else "⚠️ BM25-only (Qdrant offline)")
if not retriever.has_dense_index:
    st.info("Qdrant đang offline hoặc chưa ingest — hệ thống tự động chạy ở chế độ dự phòng BM25-only.")

st.divider()
st.subheader("M5 — Generator (Universal LLM Gateway / WokuShop / Claude / OpenAI)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("LLM Model", cfg.generator.get("claude_model", "-"))
c2.metric("API Key đã cấu hình", "✅" if is_claude_configured() else "❌ (dùng template dự phòng)")
c3.metric("Tier chạy gần nhất", orch.generator.active_tier or "-")
budget_cap = get_app_settings().llm_gateway.usd_cap_per_process
c4.metric("Chi phí ước tính", f"${get_session_cost_usd():.3f} / ${budget_cap:.2f}")

base_url = get_app_settings().llm_gateway.base_url
st.caption(
    f"Gateway Endpoint: `{base_url}` | Chi phí tính từ lúc process khởi động. "
    "Chạm trần sẽ tự động chuyển sang phương án dự phòng (không crash)."
)
if not is_claude_configured():
    st.info(
        "Chưa cấu hình `LLM_API_KEY` (hoặc `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) — "
        "Mức 1/Mức 2 vẫn chạy được nhưng dùng trích dẫn/template-filling dự phòng."
    )

st.divider()
st.subheader("Số liệu lịch sử M2/M6 (bản đồ án gốc 7 ngày)")
metrics_path = resolve_path("reports/metrics.json")
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.json(metrics)
    st.caption(
        "Ảnh chụp cố định từ lần chạy cuối của `scripts/evaluate.py` (đã gỡ khỏi bản redesign RAG+LLM "
        "— M3 Classifier và M4 BM25-proxy trong file này đã bị thay thế, xem docs/MODEL_CARD.md)."
    )
else:
    st.warning(f"Chưa có `{metrics_path.relative_to(resolve_path('.'))}`.")
st.caption("Số liệu retrieval/generation của bản redesign RAG+LLM xem ở **Trang 8 — Đánh giá**.")

if current_user()["role"] == "admin":
    st.divider()
    st.subheader("🔒 Nhật ký kiểm toán")
    st.caption(
        "Ai đã đăng nhập/đăng xuất, sửa/duyệt/từ chối mục nào, xuất file nào, lúc nào — "
        "bảng chỉ-ghi-thêm (append-only), không thể sửa/xoá qua ứng dụng (xem "
        "`src/autotender/audit/store.py`). Chỉ hiển thị cho tài khoản role='admin'."
    )
    events = get_audit_log().list_events(limit=200)
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
    else:
        st.caption("Chưa có sự kiện nào được ghi nhận.")
