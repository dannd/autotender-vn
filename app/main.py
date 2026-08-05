"""Entrypoint Streamlit — AutoTender-VN (Mục 7)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_store, init_page, tier_badge  # noqa: E402

init_page("AutoTender-VN")

st.markdown(
    "Phần mềm hỗ trợ **soạn thảo Hồ sơ mời thầu (E-HSMT) gói phần mềm/CNTT** bằng RAG + "
    "Claude API, dựa trên kho tri thức luật đấu thầu thật (Luật 22/2023/QH15, Nghị định "
    "214/2025/NĐ-CP). 3 mức: **Mức 1** Hỏi-đáp có trích dẫn → **Mức 2** soạn từng mục Chương "
    "III/V → **Mức 3** ghép bản nháp. Có cơ chế dự phòng (template, không LLM) để luôn chạy "
    "được kể cả khi chưa cấu hình API key."
)

st.warning(
    "⚠️ Mọi nội dung do hệ thống sinh ra là **dự thảo hỗ trợ soạn thảo** — "
    "bắt buộc thẩm định và phê duyệt theo quy định pháp luật trước khi phát hành chính thức.",
    icon="⚠️",
)

st.subheader("Luồng làm việc")
cols = st.columns(4)
steps = [
    ("7️⃣ Hỏi-đáp (Mức 1)", "Hỏi tự do, trả lời có trích dẫn luật thật"),
    ("3️⃣ Soạn thảo HSMT (Mức 2)", "Sinh, sửa, phê duyệt từng mục Chương III/V"),
    ("4️⃣ Kiểm tra tuân thủ", "Rà soát cờ R1-R5 (vi phạm + thiếu thành phần)"),
    ("8️⃣ Đánh giá", "Bảng ablation retrieval/generation + phân tích embedding"),
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
