"""Trang 6 — Bảng điều khiển Model (Mục 7) — trang demo trước hội đồng."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_orchestrator, init_page  # noqa: E402

from autotender.config import get_models_settings, resolve_path  # noqa: E402

init_page("6 — Bảng điều khiển Model")

orch = get_orchestrator()
cfg = get_models_settings()

st.subheader("Trạng thái từng module")
module_rows = [
    {"module": "M2 — NER", "tier_hien_tai": orch.ner.active_tier or "-", "checkpoint": cfg.ner.get("tier1_checkpoint")},
    {"module": "M3 — Classifier", "tier_hien_tai": orch.classifier.active_tier or "-", "checkpoint": cfg.classifier.get("tier1_checkpoint")},
    {"module": "M4 — Retriever", "tier_hien_tai": orch.retriever.active_tier or "-", "checkpoint": cfg.retriever.get("tier1_checkpoint")},
    {"module": "M5 — Generator", "tier_hien_tai": orch.generator.active_tier or "-", "checkpoint": cfg.generator.get("tier1_checkpoint")},
    {"module": "M6 — Compliance", "tier_hien_tai": orch.compliance.active_tier or "-", "checkpoint": cfg.compliance.get("tier1_checkpoint")},
]
for row in module_rows:
    checkpoint_path = resolve_path(row["checkpoint"]) if row["checkpoint"] else None
    row["checkpoint_ton_tai"] = "✅" if checkpoint_path and checkpoint_path.exists() else "❌ (dùng Tier 3 dự phòng)"
st.dataframe(pd.DataFrame(module_rows), use_container_width=True)
st.caption("Tier hiện tại chỉ cập nhật sau khi module đã chạy ít nhất 1 lần trong phiên này (vào trang 2/3 để kích hoạt).")

st.divider()
st.subheader("Tải checkpoint")
with st.form("download_checkpoint"):
    c1, c2 = st.columns(2)
    module_choice = c1.selectbox("Module", options=["ner_phobert", "classifier_phobert", "retriever_bi_encoder", "generator_vit5", "compliance_xlmr"])
    url = c2.text_input("URL / Google Drive link")
    submitted = st.form_submit_button("⬇️ Tải về models/")
if submitted:
    st.info(
        f"Tải thủ công checkpoint cho `{module_choice}` từ `{url or '(chưa nhập URL)'}` và giải nén vào "
        f"`models/{module_choice}/`. Do giới hạn môi trường demo, việc tải file lớn từ Drive cần thực hiện "
        "thủ công (gdown/rclone) ngoài ứng dụng — xem README.md mục Cài đặt."
    )

st.divider()
st.subheader("So sánh baseline vs fine-tuned")
metrics_path = resolve_path("reports/metrics.json")
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.json(metrics)
else:
    st.warning(
        f"Chưa có `{metrics_path.relative_to(resolve_path('.'))}`. Chạy `python scripts/evaluate.py` "
        "để sinh báo cáo metric + baseline + ablation (Mục 10)."
    )
