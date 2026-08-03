"""Trang 2 — Nạp KHLCNT (Mục 7)."""

from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_orchestrator, get_store, init_page, tier_badge  # noqa: E402

from autotender.schemas import TenderNotice  # noqa: E402

init_page("2 — Nạp KHLCNT")

orch = get_orchestrator()
store = get_store()

tab_upload, tab_paste = st.tabs(["📎 Upload file", "📝 Dán văn bản"])

raw_text = None
with tab_upload:
    uploaded = st.file_uploader("Chọn file KHLCNT", type=["pdf", "docx", "txt"])
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        if st.button("Trích xuất văn bản từ file", type="primary"):
            with st.spinner("Đang trích xuất..."):
                raw_text = orch.ingest_file(tmp_path)
            st.session_state["khlcnt_raw_text"] = raw_text

with tab_paste:
    pasted = st.text_area("Dán nội dung KHLCNT vào đây", height=200)
    if st.button("Dùng văn bản này"):
        st.session_state["khlcnt_raw_text"] = orch.ingest_text(pasted)

raw_text = st.session_state.get("khlcnt_raw_text")

if raw_text:
    fields = orch.extract_fields(raw_text)
    st.session_state["khlcnt_fields"] = fields
    classification = orch.classify_package(raw_text)

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Văn bản gốc")
        st.caption(f"Model tier NER: {tier_badge(orch.ner.active_tier)} · Phân loại: **{classification.label_display}** ({tier_badge(orch.classifier.active_tier)}, conf={classification.confidence:.2f})")

        highlighted = raw_text
        for f in sorted(fields, key=lambda x: -(x.char_start or 0)):
            if f.char_start is None or f.char_end is None:
                continue
            color = "#fff3b0" if f.confidence < 0.7 else "#c8f7c5"
            span = f'<mark style="background-color:{color}" title="{f.name} (conf={f.confidence:.2f})">{highlighted[f.char_start:f.char_end]}</mark>'
            highlighted = highlighted[: f.char_start] + span + highlighted[f.char_end :]
        st.markdown(f"<div style='white-space: pre-wrap; line-height:1.6'>{highlighted}</div>", unsafe_allow_html=True)

    with right:
        st.subheader("Trường thông tin trích xuất")
        for i, f in enumerate(fields):
            low_conf = f.confidence < 0.7
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"**{f.name}**{' 🟡' if low_conf else ''}")
                c2.caption(f"conf={f.confidence:.2f}")
                new_value = st.text_input(
                    "Giá trị", value=f.value, key=f"field_{hash(raw_text) & 0xFFFFFFFF}_{i}",
                    label_visibility="collapsed",
                )
                if new_value != f.value:
                    fields[i] = f.model_copy(update={"value": new_value, "source": "manual"})
                if low_conf:
                    st.caption("⚠️ Độ tin cậy thấp — vui lòng xác nhận lại giá trị này.")

    st.divider()
    st.subheader("Xác nhận thông tin gói thầu")
    with st.form("confirm_form"):
        c1, c2 = st.columns(2)
        package_name = c1.text_input("Tên gói thầu", value=next((f.value for f in fields if f.name == "PACKAGE_NAME"), ""))
        investor = c2.text_input("Chủ đầu tư", value=next((f.value for f in fields if f.name == "INVESTOR"), ""))
        confirm = st.form_submit_button("✅ Xác nhận và chuyển sang soạn thảo", type="primary")

    if confirm:
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        package = TenderNotice(
            tbmt_id=doc_id, package_name=package_name or "[CẦN NGƯỜI DÙNG BỔ SUNG: tên gói thầu]",
            investor=investor or "[CẦN NGƯỜI DÙNG BỔ SUNG: chủ đầu tư]", source_url="manual-upload",
        )
        doc = orch.create_document(doc_id, package, fields)
        store.save_document(doc)
        st.session_state["current_doc_id"] = doc_id
        st.success(f"Đã tạo tài liệu **{doc_id}**. Chuyển sang trang **3 — Soạn thảo HSMT** ở thanh điều hướng để tiếp tục.")
else:
    st.info("Upload file hoặc dán văn bản KHLCNT để bắt đầu trích xuất.")
