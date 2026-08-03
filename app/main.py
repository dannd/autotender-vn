"""Entrypoint Streamlit — AutoTender-VN (Mục 7)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_store, init_page, tier_badge  # noqa: E402

init_page("AutoTender-VN")

st.markdown(
    "Phần mềm hỗ trợ **tự động soạn thảo Hồ sơ mời thầu (E-HSMT)** bằng Deep Learning, "
    "có cơ chế fallback 3 tầng để luôn chạy được kể cả khi chưa có model fine-tune."
)

st.warning(
    "⚠️ Mọi nội dung do hệ thống sinh ra là **dự thảo hỗ trợ soạn thảo** — "
    "bắt buộc thẩm định và phê duyệt theo quy định pháp luật trước khi phát hành chính thức.",
    icon="⚠️",
)

st.subheader("Luồng làm việc")
cols = st.columns(6)
steps = [
    ("1️⃣ Thu thập dữ liệu", "Thu thập/tải mẫu TBMT"),
    ("2️⃣ Nạp KHLCNT", "Upload PDF/DOCX, trích trường"),
    ("3️⃣ Soạn thảo HSMT", "Sinh, sửa, phê duyệt từng mục"),
    ("4️⃣ Kiểm tra tuân thủ", "Rà soát cờ vi phạm"),
    ("5️⃣ Xuất và In", "Xuất PDF/DOCX, in trực tiếp"),
    ("6️⃣ Bảng điều khiển Model", "Xem tier/metric từng module"),
]
for col, (name, desc) in zip(cols, steps):
    with col:
        st.markdown(f"**{name}**")
        st.caption(desc)

st.divider()
st.subheader("Tài liệu đang có trong hệ thống")

store = get_store()
docs = store.list_documents()
if not docs:
    st.info("Chưa có tài liệu nào. Bắt đầu từ trang **2 — Nạp KHLCNT** ở thanh điều hướng bên trái.")
else:
    for d in docs:
        doc = store.get_document(d["doc_id"])
        if doc is None:
            continue
        approved, total = doc.approval_progress
        st.write(
            f"**{doc.doc_id}** — {doc.package.package_name} · "
            f"tiến độ phê duyệt {approved}/{total} · cập nhật {d['updated_at'][:19]}"
        )
