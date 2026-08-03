"""Trang 3 — Soạn thảo HSMT, màn hình chính (Mục 7)."""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_orchestrator, get_store, init_page, severity_icon, status_icon, tier_badge  # noqa: E402

init_page("3 — Soạn thảo HSMT")

orch = get_orchestrator()
store = get_store()

docs = store.list_documents()
if not docs:
    st.info("Chưa có tài liệu nào. Vào trang **2 — Nạp KHLCNT** để tạo tài liệu trước.")
    st.stop()

doc_ids = [d["doc_id"] for d in docs]
default_idx = doc_ids.index(st.session_state["current_doc_id"]) if st.session_state.get("current_doc_id") in doc_ids else 0
# `key` cố định là bắt buộc: nếu không có key, Streamlit tự sinh key dựa trên tham số
# `index`, mà `index` ở đây phụ thuộc session_state — khiến key đổi mỗi lần rerun và
# xoá mất lựa chọn người dùng vừa chọn (widget bị coi là "mới", quay về `index` mặc định).
selected_doc_id = st.selectbox("Tài liệu", options=doc_ids, index=default_idx, key="doc_selector_soan_thao")
st.session_state["current_doc_id"] = selected_doc_id

doc = store.get_document(selected_doc_id)

if not doc.sections:
    st.warning("Tài liệu chưa có mục nào được sinh.")
    if st.button("🪄 Sinh toàn bộ (Chương III + Chương V)", type="primary"):
        progress = st.progress(0, text="Đang sinh...")
        from autotender.models.generator import SECTION_DEFINITIONS

        total = len(SECTION_DEFINITIONS)
        for i, section_id in enumerate(SECTION_DEFINITIONS):
            section = orch.generate_section(section_id, doc.fields)
            store.upsert_section(selected_doc_id, section, log_edit=False)
            progress.progress((i + 1) / total, text=f"Đã sinh {section.title}")
        st.rerun()
    st.stop()

approved, total = doc.approval_progress
left, mid, right = st.columns([1, 2, 1])

with left:
    st.subheader("📂 Cây mục lục")
    st.caption(f"Tiến độ: {approved}/{total} mục")
    st.progress(approved / total if total else 0)

    chapters: dict[str, list] = {}
    for s in doc.sections:
        chapter = s.section_id.split(".")[0]
        chapters.setdefault(chapter, []).append(s)

    selected_section_id = st.session_state.get("selected_section_id", doc.sections[0].section_id)
    for chapter, sections in chapters.items():
        st.markdown(f"**{chapter}**")
        for s in sections:
            label = f"{status_icon(s.status)} {s.title}"
            if st.button(label, key=f"nav_{selected_doc_id}_{s.section_id}", use_container_width=True):
                selected_section_id = s.section_id
    st.session_state["selected_section_id"] = selected_section_id

    if st.button("🪄 Sinh lại toàn bộ", use_container_width=True):
        from autotender.models.generator import SECTION_DEFINITIONS

        for section_id in SECTION_DEFINITIONS:
            section = orch.generate_section(section_id, doc.fields)
            store.upsert_section(selected_doc_id, section, log_edit=False)
        st.rerun()

section = store.get_section(selected_doc_id, selected_section_id)

with mid:
    st.subheader(f"✍️ {section.title}")
    st.caption(f"{status_icon(section.status)} Trạng thái: {section.status} · {tier_badge(section.model_tier)}")

    edited = st.text_area(
        "Nội dung (có thể sửa trực tiếp)", value=section.current_text, height=350,
        key=f"editor_{selected_doc_id}_{section.section_id}",
    )

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("🪄 Sinh lại", use_container_width=True):
        new_section = orch.generate_section(section.section_id, doc.fields)
        new_section.status = "draft"
        store.upsert_section(selected_doc_id, new_section, log_edit=False)
        st.rerun()
    if b2.button("✅ Phê duyệt", use_container_width=True, type="primary"):
        if edited != section.generated_text:
            store.edit_section_text(selected_doc_id, section.section_id, edited)
        store.approve_section(selected_doc_id, section.section_id, approved_by="nguyendan1987")
        st.rerun()
    if b3.button("❌ Từ chối", use_container_width=True):
        store.reject_section(selected_doc_id, section.section_id)
        st.rerun()
    if b4.button("↩️ Khôi phục bản gốc", use_container_width=True):
        section.edited_text = None
        section.status = "draft"
        store.upsert_section(selected_doc_id, section, log_edit=False)
        st.rerun()

    if edited != section.current_text and st.button("💾 Lưu chỉnh sửa (chưa phê duyệt)"):
        store.edit_section_text(selected_doc_id, section.section_id, edited)
        st.rerun()

    if section.edited_text is not None and section.edited_text != section.generated_text:
        with st.expander("🔍 So sánh bản gốc vs bản đã sửa"):
            diff = difflib.HtmlDiff().make_table(
                section.generated_text.splitlines(), section.edited_text.splitlines(),
                fromdesc="Bản sinh ra", todesc="Bản đã sửa", context=True, numlines=2,
            )
            st.markdown(diff, unsafe_allow_html=True)

with right:
    st.subheader("📎 Căn cứ & Cờ")
    st.markdown("**Trích dẫn:**")
    for c in section.citations:
        st.caption(f"• {c.source_doc} (score {c.score:.2f})")
        with st.expander("Xem đoạn trích", expanded=False):
            st.write(c.text)

    st.markdown("**🚩 Cờ tuân thủ:**")
    if not section.flags:
        st.success("Không phát hiện vi phạm.")
    for f in section.flags:
        st.markdown(f"{severity_icon(f.severity)} **{f.rule_code}** — {f.severity}")
        st.caption(f'"{f.sentence}"')
        st.caption(f.explanation)
