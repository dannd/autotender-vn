"""Trang 5 — Xuất và In (Mục 7, Mục 8)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_store, init_page  # noqa: E402

from autotender.config import resolve_path  # noqa: E402
from autotender.export.docx import export_docx  # noqa: E402
from autotender.export.pdf import export_pdf, render_html  # noqa: E402
from autotender.export.print import build_print_html  # noqa: E402

init_page("5 — Xuất và In")

store = get_store()
docs = store.list_documents()
if not docs:
    st.info("Chưa có tài liệu nào.")
    st.stop()

doc_ids = [d["doc_id"] for d in docs]
selected_doc_id = st.selectbox("Tài liệu", options=doc_ids, index=0, key="doc_selector_xuat_in")
doc = store.get_document(selected_doc_id)

approved, total = doc.approval_progress
if approved < total:
    st.error(
        f"⚠️ Tài liệu còn {total - approved}/{total} mục CHƯA phê duyệt. "
        "Vẫn có thể xuất để xem trước, nhưng PDF/DOCX sẽ hiển thị cảnh báo rõ ràng ở đầu tài liệu.",
        icon="⚠️",
    )
    for s in doc.sections:
        if s.status != "approved":
            st.caption(f"• {s.section_id} — {s.title} ({s.status})")
else:
    st.success("✅ Tất cả các mục đã được phê duyệt.")

st.divider()
c1, c2 = st.columns(2)
show_watermark = c1.checkbox("Đóng dấu watermark 'DỰ THẢO'", value=(approved < total))
c2.caption("Đánh số trang: luôn bật (theo thể thức văn bản hành chính).")

tab_preview, tab_export = st.tabs(["👁️ Xem trước", "📤 Xuất file"])

approval_log = store.get_approval_log(selected_doc_id)
html = render_html(doc, approval_log, show_watermark=show_watermark)

with tab_preview:
    components.html(html, height=800, scrolling=True)

with tab_export:
    b1, b2, b3 = st.columns(3)

    if b1.button("📄 Xuất PDF", type="primary", use_container_width=True):
        out_path = resolve_path("data/processed") / f"{selected_doc_id}.pdf"
        with st.spinner("Đang xuất PDF (thử WeasyPrint, tự động fallback ReportLab nếu cần)..."):
            export_pdf(doc, store, out_path, show_watermark=show_watermark)
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Tải PDF", data=f.read(), file_name=f"{selected_doc_id}.pdf", mime="application/pdf")

    if b2.button("📝 Xuất DOCX", use_container_width=True):
        out_path = resolve_path("data/processed") / f"{selected_doc_id}.docx"
        with st.spinner("Đang xuất DOCX..."):
            export_docx(doc, store, out_path)
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Tải DOCX", data=f.read(), file_name=f"{selected_doc_id}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

    if b3.button("🖨️ In trực tiếp", use_container_width=True):
        components.html(build_print_html(html), height=0)
        st.caption("Hộp thoại in của trình duyệt sẽ mở trong vài giây — nếu không thấy, kiểm tra popup blocker.")
