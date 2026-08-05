"""Trang 7 — Hỏi-đáp có trích dẫn (Mức 1, đề cương RAG+LLM)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_qa_module, init_page, tier_badge_claude  # noqa: E402

init_page("7 — Hỏi-đáp (Mức 1)")

st.markdown(
    "Đặt câu hỏi về quy định đấu thầu (Luật Đấu thầu 22/2023/QH15, Nghị định 214/2025/NĐ-CP). "
    "Câu trả lời **chỉ dựa trên trích đoạn văn bản pháp luật thật** được truy xuất — không tự "
    "suy diễn ngoài phạm vi trích dẫn."
)

_EXAMPLES = [
    "Hồ sơ mời thầu gồm những nội dung gì?",
    "Bảo đảm dự thầu được quy định như thế nào?",
    "Thời gian chuẩn bị hồ sơ dự thầu tối thiểu là bao lâu?",
]

if "qa_question" not in st.session_state:
    st.session_state["qa_question"] = ""

# QUAN TRỌNG: khối nút ví dụ phải nằm TRƯỚC khi tạo widget `text_input` cùng key —
# Streamlit cấm sửa `session_state[key]` sau khi widget với key đó đã được khởi tạo
# trong CÙNG lượt chạy script (raise StreamlitAPIException). Đặt nút trước để việc gán
# session_state xảy ra trước lúc widget đọc giá trị.
cols = st.columns(len(_EXAMPLES))
for col, example in zip(cols, _EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state["qa_question"] = example
        st.rerun()

question = st.text_input("Câu hỏi của bạn", key="qa_question", placeholder=_EXAMPLES[0])

if st.button("🔍 Hỏi", type="primary", disabled=not question):
    module = get_qa_module()
    with st.spinner("Đang truy xuất và tổng hợp câu trả lời..."):
        answer = module.ask(question)

    st.markdown(f"**{tier_badge_claude(module.active_tier)}** · model: `{answer.model_used}`")
    st.markdown("### Trả lời")
    st.write(answer.answer)

    st.markdown("### Trích dẫn")
    if not answer.citations:
        st.warning("Không tìm thấy trích đoạn nào liên quan trong kho tri thức.")
    for c in answer.citations:
        with st.expander(f"📖 {c.source_doc} (score {c.score:.3f})"):
            st.write(c.text)
