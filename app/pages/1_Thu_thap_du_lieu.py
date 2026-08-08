"""Trang 1 — Thu thập dữ liệu (Mục 7)."""

from __future__ import annotations

import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import init_page  # noqa: E402

from autotender.config import get_crawler_settings  # noqa: E402
from autotender.crawler.pipeline import run_crawl  # noqa: E402
from autotender.utils.vn_text import format_vn_number  # noqa: E402

init_page("1 — Thu thập dữ liệu")

with st.form("crawl_form"):
    c1, c2, c3, c4 = st.columns(4)
    date_from = c1.date_input("Từ ngày", value=date(2025, 1, 1))
    date_to = c2.date_input("Đến ngày", value=date(2026, 6, 30))
    max_records = c3.number_input("Số bản ghi tối đa", min_value=1, max_value=3000, value=20)
    source = c4.selectbox("Nguồn", options=["Tự động (api → browser → local)", "api", "browser", "local"], index=0)
    submitted = st.form_submit_button("🚀 Bắt đầu thu thập", use_container_width=True)

if submitted:
    only_source = None if source.startswith("Tự động") else source
    settings = get_crawler_settings().crawler
    progress = st.progress(0, text="Đang kết nối nguồn dữ liệu...")
    log_box = st.empty()

    try:
        notices, source_used = run_crawl(
            cfg=settings,
            date_from=date_from.isoformat(),
            date_to=date_to.isoformat(),
            max_records=int(max_records),
            out_dir="data/raw",
            only_source=only_source,
        )
        progress.progress(100, text=f"Hoàn tất — nguồn dùng: '{source_used}'")
        log_box.success(f"Thu thập thành công {len(notices)} bản ghi từ nguồn '{source_used}'.")
        st.session_state["last_crawl_notices"] = [n.model_dump() for n in notices]
    except Exception as e:  # noqa: BLE001 — hiển thị lỗi rõ ràng thay vì để Streamlit crash
        progress.progress(100, text="Thất bại")
        log_box.error(f"Thu thập thất bại (kể cả LocalSampleSource): {e}")

st.divider()

records = st.session_state.get("last_crawl_notices")
if not records:
    st.info("Chưa có kết quả thu thập trong phiên này. Bấm **Bắt đầu thu thập** ở trên.")
else:
    df = pd.DataFrame(records)

    st.subheader("Thẻ thống kê")
    s1, s2, s3 = st.columns(3)
    s1.metric("Tổng bản ghi", len(df))
    type_counts = Counter(df["package_type"].dropna())
    s2.metric("Số loại gói thầu", len(type_counts))
    total_value = df["package_value"].dropna().sum()
    s3.metric("Tổng giá trị (tỷ VND)", format_vn_number(total_value / 1e9, decimals=1))

    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Phân bố loại gói thầu")
        st.bar_chart(pd.Series(type_counts))
    with col_b:
        st.caption("Phân bố giá trị gói thầu (VND)")
        st.bar_chart(df.set_index("tbmt_id")["package_value"].dropna())

    st.subheader("Bảng kết quả")
    st.dataframe(df, use_container_width=True)

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "⬇️ Tải JSONL", data=df.to_json(orient="records", lines=True, force_ascii=False),
        file_name="tender_notices.jsonl", mime="application/json", use_container_width=True,
    )
    dl2.download_button(
        "⬇️ Tải CSV", data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="tender_notices.csv", mime="text/csv", use_container_width=True,
    )
