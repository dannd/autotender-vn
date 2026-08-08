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
from common import get_orchestrator, init_page  # noqa: E402

from autotender.config import get_models_settings, resolve_path  # noqa: E402
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
st.subheader("M4 — Retriever (RAG+LLM redesign)")
retriever = orch.retriever
c1, c2, c3 = st.columns(3)
c1.metric("Model embedding", retriever.model_key)
c2.metric("Số chunk trong kho tri thức", retriever.num_chunks)
c3.metric("FAISS (dense) đã build", "✅" if retriever.has_dense_index else "❌ (chỉ BM25)")
if not retriever.has_dense_index:
    st.warning("Chưa có FAISS index — chạy `python scripts/build_legal_index.py` để có truy xuất dense/hybrid đầy đủ.")

st.divider()
st.subheader("M5 — Generator (RAG+LLM redesign)")
c1, c2, c3 = st.columns(3)
c1.metric("Model Claude", cfg.generator.get("claude_model", "-"))
c2.metric("Claude API đã cấu hình", "✅" if is_claude_configured() else "❌ (dùng template dự phòng)")
c3.metric("Tier chạy lần gần nhất", orch.generator.active_tier or "-")
if not is_claude_configured():
    st.info(
        "Chưa cấu hình `ANTHROPIC_API_KEY` — Mức 1/Mức 2 vẫn chạy được nhưng chỉ liệt kê "
        "trích dẫn/template-filling, không có câu trả lời/nội dung tổng hợp bằng Claude."
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
