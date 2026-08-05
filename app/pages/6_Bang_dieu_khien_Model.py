"""Trang 6 — Bảng điều khiển Model (Mục 7) — trang demo trước hội đồng.

Đã cập nhật cho bản redesign RAG+LLM: M4 (Retriever) không còn là `BaseModule` 3-tier
(là `HybridLegalRetriever`, luôn kết hợp dense+sparse, không có khái niệm "tier") và M5
(Generator) Tier 1 giờ là Claude API thay vì checkpoint local — bảng dưới tách riêng 2
module này khỏi nhóm 3-tier cũ (M2/M3/M6) để tránh hiển thị sai/lỗi (`active_tier` không
tồn tại trên `HybridLegalRetriever`, phát hiện qua lỗi thật khi chạy trang này).
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

st.subheader("Trạng thái module 3-tier (M2, M3, M6 — kiến trúc cũ, chưa đổi)")
module_rows = [
    {"module": "M2 — NER", "tier_hien_tai": orch.ner.active_tier or "-", "checkpoint": cfg.ner.get("tier1_checkpoint")},
    {"module": "M3 — Classifier", "tier_hien_tai": orch.classifier.active_tier or "-", "checkpoint": cfg.classifier.get("tier1_checkpoint")},
    {"module": "M6 — Compliance", "tier_hien_tai": orch.compliance.active_tier or "-", "checkpoint": cfg.compliance.get("tier1_checkpoint")},
]
for row in module_rows:
    checkpoint_path = resolve_path(row["checkpoint"]) if row["checkpoint"] else None
    row["checkpoint_ton_tai"] = "✅" if checkpoint_path and checkpoint_path.exists() else "❌ (dùng Tier 3 dự phòng)"
st.dataframe(pd.DataFrame(module_rows), use_container_width=True)
st.caption("Tier hiện tại chỉ cập nhật sau khi module đã chạy ít nhất 1 lần trong phiên này (vào trang 2/4 để kích hoạt).")

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
st.subheader("Tải checkpoint (M2/M3/M6)")
with st.form("download_checkpoint"):
    c1, c2 = st.columns(2)
    module_choice = c1.selectbox("Module", options=["ner_phobert", "classifier_phobert", "compliance_xlmr"])
    url = c2.text_input("URL / Google Drive link")
    submitted = st.form_submit_button("⬇️ Tải về models/")
if submitted:
    st.info(
        f"Tải thủ công checkpoint cho `{module_choice}` từ `{url or '(chưa nhập URL)'}` và giải nén vào "
        f"`models/{module_choice}/`. Do giới hạn môi trường demo, việc tải file lớn từ Drive cần thực hiện "
        "thủ công (gdown/rclone) ngoài ứng dụng — xem README.md mục Cài đặt."
    )

st.divider()
st.subheader("So sánh baseline vs fine-tuned (M2/M3/M6, Phase 1)")
metrics_path = resolve_path("reports/metrics.json")
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.json(metrics)
else:
    st.warning(
        f"Chưa có `{metrics_path.relative_to(resolve_path('.'))}`. Chạy `python scripts/evaluate.py` "
        "để sinh báo cáo metric + baseline + ablation (Mục 10)."
    )
st.caption("Số liệu retrieval/generation của bản redesign RAG+LLM xem ở **Trang 8 — Đánh giá**.")
