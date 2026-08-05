"""Trang 8 — Đánh giá: bảng ablation retrieval/generation + phân tích embedding (Giai đoạn 3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import init_page  # noqa: E402

from autotender.config import resolve_path  # noqa: E402

init_page("8 — Đánh giá")

RETRIEVAL_METRICS_PATH = resolve_path("reports/retrieval_metrics.json")
ABLATION_PATH = resolve_path("reports/ablation_table.json")
EMBEDDING_PATH = resolve_path("reports/embedding_comparison.json")
FIGURES_DIR = resolve_path("reports/figures")

st.subheader("1. Retrieval — Recall@k / MRR / nDCG@k")
if not RETRIEVAL_METRICS_PATH.exists():
    st.info("Chưa có kết quả. Chạy `python scripts/run_retrieval_eval.py` trước.")
else:
    with open(RETRIEVAL_METRICS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    st.caption(f"{data['n_queries']} câu hỏi gán tay · model embedding: `{data['embedding_model']}`")
    rows = []
    for mode, m in data["modes"].items():
        rows.append(
            {
                "Chế độ": mode, "Recall@5": round(m["recall@5"], 3), "MRR": round(m["mrr"], 3),
                "nDCG@5": round(m["ndcg@5"], 3), "Thời gian (s)": m["elapsed_seconds"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
st.subheader("2. Generation — LLM-only vs RAG (faithfulness/completeness)")
if not ABLATION_PATH.exists():
    st.info("Chưa có kết quả. Chạy `python scripts/run_ablation_table.py` trước.")
else:
    with open(ABLATION_PATH, encoding="utf-8") as f:
        ablation = json.load(f)
    gen = ablation.get("b_generation", {})
    if gen.get("status") != "ok":
        st.warning(f"Phần B (generation ablation): {gen.get('detail', 'N/A')}")
    else:
        rows = [
            {"Điều kiện": "LLM-only (không RAG)", "Faithfulness": round(gen["no_rag"]["avg_faithfulness"], 3), "Completeness": round(gen["no_rag"]["avg_completeness"], 3), "Số câu chấm": gen["no_rag"]["n_scored"]},
            {"Điều kiện": "RAG (có trích dẫn thật)", "Faithfulness": round(gen["rag"]["avg_faithfulness"], 3), "Completeness": round(gen["rag"]["avg_completeness"], 3), "Số câu chấm": gen["rag"]["n_scored"]},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"⚠️ {gen.get('caveat', '')}")

st.divider()
st.subheader("3. Phân tích Deep Learning — so sánh không gian embedding")
if not EMBEDDING_PATH.exists():
    st.info("Chưa có kết quả. Chạy `python scripts/analyze_embeddings.py` trước.")
else:
    with open(EMBEDDING_PATH, encoding="utf-8") as f:
        emb = json.load(f)
    rows = [
        {"Model": key, "Chiều": v["dim"], "Similarity nội-Điều (TB)": round(v["intra_mean"], 4), "Similarity liên-Điều (TB)": round(v["inter_mean"], 4), "Độ tách biệt": round(v["separation"], 4)}
        for key, v in emb.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Độ tách biệt = similarity nội-Điều − liên-Điều. Càng cao, không gian embedding càng "
        "phân biệt tốt các Điều/chủ đề pháp lý khác nhau — có ích cho retrieval."
    )

    st.markdown("**Trực quan hoá t-SNE / UMAP** (tô màu theo văn bản nguồn):")
    for model_key in emb:
        cols = st.columns(2)
        for col, method in zip(cols, ["tsne", "umap"]):
            img_path = FIGURES_DIR / f"embedding_{model_key}_{method}.png"
            if img_path.exists():
                col.image(str(img_path), caption=f"{model_key} — {method.upper()}", use_container_width=True)
