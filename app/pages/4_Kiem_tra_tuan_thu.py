"""Trang 4 — Kiểm tra tuân thủ (Mục 7)."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_store, init_page, severity_icon  # noqa: E402

init_page("4 — Kiểm tra tuân thủ")

store = get_store()
docs = store.list_documents()
if not docs:
    st.info("Chưa có tài liệu nào.")
    st.stop()

doc_ids = [d["doc_id"] for d in docs]
selected_doc_id = st.selectbox("Tài liệu", options=doc_ids, index=0, key="doc_selector_kiem_tra")
doc = store.get_document(selected_doc_id)

rows = []
for section in doc.sections:
    for flag in section.flags:
        rows.append(
            {
                "section_id": section.section_id,
                "title": section.title,
                "rule_code": flag.rule_code,
                "severity": flag.severity,
                "sentence": flag.sentence,
                "explanation": flag.explanation,
                "confidence": flag.confidence,
            }
        )

if not rows:
    st.success("✅ Không có cờ tuân thủ nào trong tài liệu này.")
    st.stop()

df = pd.DataFrame(rows)
severity_order = {"cao": 0, "trung_binh": 1, "thap": 2}
df["_sort"] = df["severity"].map(severity_order)
df = df.sort_values("_sort").drop(columns="_sort")

st.subheader("Biểu đồ")
c1, c2 = st.columns(2)
with c1:
    st.caption("Số cờ theo loại quy tắc")
    st.bar_chart(Counter(df["rule_code"]))
with c2:
    total_sections = len(doc.sections)
    flagged_sections = df["section_id"].nunique()
    clean_ratio = 1 - flagged_sections / total_sections if total_sections else 0
    st.metric("Tỷ lệ mục sạch (không cờ)", f"{clean_ratio:.0%}")

st.subheader("Bộ lọc")
rule_filter = st.multiselect("Mã quy tắc", options=sorted(df["rule_code"].unique()), default=list(df["rule_code"].unique()))
filtered = df[df["rule_code"].isin(rule_filter)]

st.subheader(f"Bảng tổng hợp cờ ({len(filtered)})")
for _, row in filtered.iterrows():
    with st.expander(f"{severity_icon(row['severity'])} [{row['rule_code']}] {row['title']} — \"{row['sentence'][:60]}...\""):
        st.write(f"**Câu vi phạm:** {row['sentence']}")
        st.write(f"**Giải thích:** {row['explanation']}")
        st.caption(f"Độ tin cậy: {row['confidence']:.2f}")
        b1, b2 = st.columns(2)
        key_base = f"{row['section_id']}_{row['rule_code']}"
        if b1.button("✅ Chấp nhận cờ", key=f"accept_{key_base}"):
            store.record_flag_feedback(selected_doc_id, row["section_id"], row["rule_code"], "accepted")
            st.success("Đã ghi nhận.")
        if b2.button("🚫 Đánh dấu dương tính giả", key=f"fp_{key_base}"):
            store.record_flag_feedback(selected_doc_id, row["section_id"], row["rule_code"], "false_positive")
            st.success("Đã ghi nhận.")
