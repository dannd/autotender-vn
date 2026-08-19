"""Trang 3 — Soạn thảo & Phê duyệt HSMT (Mức 2, Đề cương RAG+LLM).

Đã nâng cấp UI/UX toàn diện:
- Hỗ trợ Phê duyệt thông minh (Duyệt nhanh hàng loạt các mục 0 lỗi, Duyệt & Tiếp tục, Duyệt cả chương).
- Bộ lọc Focus Mode (Tất cả, Cần xử lý cờ vi phạm, Chưa duyệt).
- Không gian soạn thảo rộng mở (Workspace 75% chiều ngang, khung soạn thảo 450px).
- Nút 1-Click Quick-Fix cho cờ vi phạm R1 (Tự động thêm 'hoặc tương đương').
- Chế độ Đọc toàn văn liền mạch 8 chương (Full Document Review).
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auth_ui import current_user  # noqa: E402
from common import (  # noqa: E402
    get_audit_log,
    get_orchestrator,
    get_store,
    init_page,
    severity_icon,
    status_icon,
    tier_badge_claude,
)

init_page("3 — Soạn thảo & Phê duyệt HSMT")

orch = get_orchestrator()
store = get_store()
audit = get_audit_log()

docs = store.list_documents()
if not docs:
    st.info("Chưa có tài liệu nào. Vào trang **2 — Nạp KHLCNT** để tạo tài liệu trước.")
    st.stop()

doc_ids = [d["doc_id"] for d in docs]
default_idx = doc_ids.index(st.session_state["current_doc_id"]) if st.session_state.get("current_doc_id") in doc_ids else 0
selected_doc_id = st.selectbox("📁 Chọn bộ Hồ sơ Mời thầu (HSMT):", options=doc_ids, index=default_idx, key="doc_selector_soan_thao")
st.session_state["current_doc_id"] = selected_doc_id

doc = store.get_document(selected_doc_id)

if not doc.sections:
    st.warning("Tài liệu chưa có mục nào được sinh.")
    from autotender.models.generator import SECTION_DEFINITIONS

    st.caption(
        f"Sinh trọn bộ {len(SECTION_DEFINITIONS)} mục (8 chương I-VIII) — mỗi mục là 1 lượt "
        "gọi LLM Gateway (WokuShop/Claude/OpenAI) có trích dẫn văn bản pháp luật và rà soát tuân thủ."
    )
    if st.button("🪄 Sinh toàn bộ 8 chương HSMT", type="primary", use_container_width=True):
        progress = st.progress(0, text="Đang khởi tạo tiến trình sinh...")
        total = len(SECTION_DEFINITIONS)
        for i, section_id in enumerate(SECTION_DEFINITIONS):
            section = orch.generate_section(section_id, doc.fields)
            store.upsert_section(selected_doc_id, section, log_edit=False)
            progress.progress((i + 1) / total, text=f"Đã sinh: {section.title}")
        st.success("Đã hoàn tất sinh 8 chương HSMT!")
        st.rerun()
    st.stop()

approved_count, total_count = doc.approval_progress

# -----------------------------------------------------------------------------
# TAB NAVIGATION: SOẠN THẢO TỪNG MỤC VS ĐỌC TOÀN VĂN
# -----------------------------------------------------------------------------
tab_editor, tab_full_review = st.tabs([
    f"✍️ Soạn thảo & Duyệt từng mục ({approved_count}/{total_count})",
    "📖 Đọc toàn văn 8 chương (Review Mode)",
])

from autotender.models.generator import CHAPTER_TITLES, SECTION_DEFINITIONS

# =============================================================================
# TAB 1: SOẠN THẢO & DUYỆT TỪNG MỤC
# =============================================================================
with tab_editor:
    col_nav, col_main = st.columns([1.1, 2.9])

    # -------------------------------------------------------------------------
    # CỘT TRÁI: CÂY MỤC LỤC & BỘ LỌC & THAO TÁC HÀNG LOẠT
    # -------------------------------------------------------------------------
    with col_nav:
        st.markdown("### 📂 Cây mục lục HSMT")
        st.caption(f"Tiến độ phê duyệt: **{approved_count}/{total_count} mục** ({approved_count * 100 // total_count if total_count else 0}%)")
        st.progress(approved_count / total_count if total_count else 0)

        # Tính toán thống kê theo bộ lọc
        problem_sections = [s for s in doc.sections if len(s.flags) > 0 or s.status == "rejected"]
        unapproved_sections = [s for s in doc.sections if s.status != "approved"]
        clean_unapproved = [s for s in doc.sections if s.status != "approved" and len(s.flags) == 0]

        # Nút phê duyệt thông minh
        st.markdown("**⚡ Phê duyệt nhanh:**")
        if clean_unapproved:
            if st.button(
                f"⚡ Duyệt nhanh {len(clean_unapproved)} mục 0 cảnh báo",
                help="Phê duyệt toàn bộ các mục không có cờ vi phạm pháp lý chỉ bằng 1 cú click",
                use_container_width=True,
            ):
                username = current_user()["username"]
                for s in clean_unapproved:
                    store.approve_section(selected_doc_id, s.section_id, approved_by=username)
                    audit.record(username, "approve_section_batch", doc_id=selected_doc_id, section_id=s.section_id)
                st.toast(f"Đã duyệt thành công {len(clean_unapproved)} mục hợp lệ!")
                st.rerun()
        else:
            st.caption("✅ Không còn mục 0 cảnh báo nào chờ duyệt.")

        st.divider()

        # Bộ lọc trạng thái (Focus Mode)
        filter_mode = st.radio(
            "Lọc danh sách mục:",
            [
                f"Tất cả ({total_count})",
                f"🔴 Cần xử lý ({len(problem_sections)})",
                f"⏳ Chưa duyệt ({len(unapproved_sections)})",
            ],
            index=0,
            horizontal=False,
            key="filter_mode_selection",
        )

        chapters: dict[str, list] = {}
        for s in doc.sections:
            chapter = s.section_id.split(".")[0]
            chapters.setdefault(chapter, []).append(s)

        selected_section_id = st.session_state.get("selected_section_id", doc.sections[0].section_id)
        clicked_section_id = None

        # Hiển thị cây mục lục theo thứ tự chương chuẩn I -> VIII
        for chapter in CHAPTER_TITLES:
            sections = chapters.get(chapter)
            if not sections:
                continue

            # Lọc theo mode đã chọn
            if "Cần xử lý" in filter_mode:
                filtered_sections = [s for s in sections if len(s.flags) > 0 or s.status == "rejected"]
            elif "Chưa duyệt" in filter_mode:
                filtered_sections = [s for s in sections if s.status != "approved"]
            else:
                filtered_sections = sections

            if not filtered_sections:
                continue

            st.markdown(f"**{chapter}**")
            for s in filtered_sections:
                # Icon trạng thái + cảnh báo nếu có cờ đỏ
                flag_indicator = f" 🔴({len(s.flags)})" if s.flags else ""
                label = f"{status_icon(s.status)} {s.title}{flag_indicator}"
                btn_type = "primary" if s.section_id == selected_section_id else "secondary"
                if st.button(label, key=f"nav_{selected_doc_id}_{s.section_id}", use_container_width=True, type=btn_type):
                    clicked_section_id = s.section_id

        if clicked_section_id and clicked_section_id != selected_section_id:
            st.session_state["selected_section_id"] = clicked_section_id
            st.rerun()

        st.divider()

        with st.expander("🚩 Kiểm tra đủ 8 chương (R5)", expanded=False):
            r5_flags = orch.check_completeness(doc.sections)
            if not r5_flags:
                st.success("Đủ thành phần bắt buộc (8 chương I-VIII) theo NĐ 214/2025/NĐ-CP.")
            else:
                for f in r5_flags:
                    st.markdown(f"{severity_icon(f.severity)} {f.explanation}")

        if st.button("🪄 Sinh lại toàn bộ 8 chương", use_container_width=True):
            for section_id in SECTION_DEFINITIONS:
                section = orch.generate_section(section_id, doc.fields)
                store.upsert_section(selected_doc_id, section, log_edit=False)
            st.toast("Đã sinh lại toàn bộ nội dung!")
            st.rerun()

    # -------------------------------------------------------------------------
    # CỘT PHẢI: WORKSPACE SOẠN THẢO CHÍNH (70% CHIỀU NGANG)
    # -------------------------------------------------------------------------
    section = store.get_section(selected_doc_id, selected_section_id)
    curr_chapter = section.section_id.split(".")[0]
    chapter_sections = chapters.get(curr_chapter, [])

    with col_main:
        # Tiêu đề mục & Thông tin trạng thái
        top_c1, top_c2 = st.columns([3, 1])
        with top_c1:
            st.subheader(f"✍️ {section.title}")
            st.caption(
                f"Trạng thái: **{status_icon(section.status)} {section.status.upper()}** · "
                f"Nguồn sinh: **{tier_badge_claude(section.model_tier)}** · "
                f"Cảnh báo: **{len(section.flags)} cờ vi phạm**"
            )
        with top_c2:
            # Nút duyệt trọn gói cả chương
            if st.button("📁 Duyệt cả Chương này", help=f"Duyệt tất cả các mục thuộc {curr_chapter}", use_container_width=True):
                username = current_user()["username"]
                for s in chapter_sections:
                    store.approve_section(selected_doc_id, s.section_id, approved_by=username)
                    audit.record(username, "approve_section_chapter", doc_id=selected_doc_id, section_id=s.section_id)
                st.toast(f"Đã duyệt toàn bộ các mục thuộc {curr_chapter}!")
                st.rerun()

        # Khung soạn thảo văn bản
        edited = st.text_area(
            "Nội dung điều khoản HSMT (chỉnh sửa trực tiếp):",
            value=section.current_text,
            height=450,
            key=f"editor_{selected_doc_id}_{section.section_id}",
            help="Nội dung đã được chuẩn hóa theo mẫu luật định. Bạn có thể sửa đổi văn bản trực tiếp.",
        )

        # Thanh nút bấm hành động phê duyệt
        b1, b2, b3, b4, b5 = st.columns([1.5, 1.2, 1, 1, 1])

        # Nút 1: DUYỆT & TỰ ĐỘNG CHUYỂN MỤC TIẾP THEO
        if b1.button("✅ Phê duyệt & Tiếp tục ➔", type="primary", use_container_width=True):
            username = current_user()["username"]
            if edited != section.generated_text:
                store.edit_section_text(selected_doc_id, section.section_id, edited)
                audit.record(username, "edit_section_text", doc_id=selected_doc_id, section_id=section.section_id)
            store.approve_section(selected_doc_id, section.section_id, approved_by=username)
            audit.record(username, "approve_section", doc_id=selected_doc_id, section_id=section.section_id)

            # Tìm mục kế tiếp chưa được duyệt để focus
            remaining = [s for s in doc.sections if s.status != "approved" and s.section_id != section.section_id]
            if remaining:
                st.session_state["selected_section_id"] = remaining[0].section_id
                st.toast(f"Đã duyệt '{section.title}'! Chuyển sang '{remaining[0].title}'.")
            else:
                st.toast(f"Đã duyệt xong mục cuối cùng!")
            st.rerun()

        # Nút 2: Duyệt mục hiện tại (ở lại mục này)
        if b2.button("✅ Duyệt mục này", use_container_width=True):
            username = current_user()["username"]
            if edited != section.generated_text:
                store.edit_section_text(selected_doc_id, section.section_id, edited)
                audit.record(username, "edit_section_text", doc_id=selected_doc_id, section_id=section.section_id)
            store.approve_section(selected_doc_id, section.section_id, approved_by=username)
            audit.record(username, "approve_section", doc_id=selected_doc_id, section_id=section.section_id)
            st.toast(f"Đã duyệt mục '{section.title}'!")
            st.rerun()

        # Nút 3: Từ chối mục
        if b3.button("❌ Từ chối", use_container_width=True):
            store.reject_section(selected_doc_id, section.section_id)
            audit.record(current_user()["username"], "reject_section", doc_id=selected_doc_id, section_id=section.section_id)
            st.rerun()

        # Nút 4: Sinh lại mục hiện tại
        if b4.button("🪄 Sinh lại", use_container_width=True):
            new_section = orch.generate_section(section.section_id, doc.fields)
            new_section.status = "draft"
            store.upsert_section(selected_doc_id, new_section, log_edit=False)
            st.toast("Đã sinh lại nội dung mục!")
            st.rerun()

        # Nút 5: Khôi phục bản gốc LLM
        if b5.button("↩️ Bản gốc", use_container_width=True):
            section.edited_text = None
            section.status = "draft"
            store.upsert_section(selected_doc_id, section, log_edit=False)
            st.toast("Đã khôi phục về bản gốc sinh bởi AI.")
            st.rerun()

        if edited != section.current_text and st.button("💾 Lưu tạm chỉnh sửa (chưa duyệt)"):
            store.edit_section_text(selected_doc_id, section.section_id, edited)
            st.toast("Đã lưu nội dung chỉnh sửa.")
            st.rerun()

        # ---------------------------------------------------------------------
        # KHU VỰC CỜ TUÂN THỦ & 1-CLICK QUICK FIX
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("#### 🚩 Rà soát tuân thủ & Khuyến nghị sửa lỗi")
        if not section.flags:
            st.success("✅ Mục này hoàn toàn tuân thủ các quy định pháp luật đấu thầu hiện hành (0 cờ cảnh báo).")
        else:
            for idx, f in enumerate(section.flags):
                with st.container(border=True):
                    st.markdown(f"**{severity_icon(f.severity)} Quy tắc {f.rule_code} — Mức độ: {f.severity.upper()}**")
                    st.markdown(f"**Đoạn văn bản cảnh báo:** `{f.sentence}`")
                    st.caption(f"**Lý do vi phạm:** {f.explanation}")

                    # 1-Click Quick Fix cho quy tắc R1 (Cấm nêu nhãn hiệu cụ thể)
                    if f.rule_code == "R1":
                        brand_matches = [
                            brand for brand in [
                                "Cisco", "Dell", "HP", "Hewlett Packard", "Samsung", "IBM",
                                "Lenovo", "Intel", "Microsoft", "Apple", "Huawei", "Sony",
                                "Canon", "Epson", "Fujitsu", "Oracle", "SAP",
                            ] if re.search(r"\b" + re.escape(brand) + r"\b", f.sentence, re.IGNORECASE)
                        ]
                        if brand_matches:
                            brand_name = brand_matches[0]
                            if st.button(
                                f"💡 1-Click Quick Fix: Thêm 'hoặc tương đương' sau nhãn hiệu '{brand_name}'",
                                key=f"quick_fix_{selected_doc_id}_{section.section_id}_{idx}",
                                type="secondary",
                            ):
                                fixed_text = re.sub(
                                    rf"(\b{re.escape(brand_name)}\b)(?!\s+hoặc\s+tương\s+đương)",
                                    rf"\1 hoặc tương đương",
                                    edited,
                                    flags=re.IGNORECASE,
                                )
                                store.edit_section_text(selected_doc_id, section.section_id, fixed_text)
                                st.toast(f"Đã tự động bổ sung 'hoặc tương đương' cho nhãn hiệu {brand_name}!")
                                st.rerun()

        # ---------------------------------------------------------------------
        # KHU VỰC CĂN CỨ TRÍCH DẪN PHÁP LUẬT
        # ---------------------------------------------------------------------
        with st.expander(f"📎 Căn cứ trích dẫn pháp luật ({len(section.citations)} điều khoản)", expanded=False):
            for c in section.citations:
                st.markdown(f"**• {c.source_doc}** *(Độ khớp truy hồi: {c.score:.2f})*")
                st.info(c.text)

        # ---------------------------------------------------------------------
        # BẢNG SO SÁNH DIFF (NẾU CÓ CHỈNH SỬA)
        # ---------------------------------------------------------------------
        if section.edited_text is not None and section.edited_text != section.generated_text:
            with st.expander("🔍 So sánh bản gốc AI vs Bản người dùng sửa", expanded=False):
                diff = difflib.HtmlDiff().make_table(
                    section.generated_text.splitlines(),
                    section.edited_text.splitlines(),
                    fromdesc="Bản AI sinh",
                    todesc="Bản người dùng sửa",
                    context=True,
                    numlines=2,
                )
                st.markdown(diff, unsafe_allow_html=True)


# =============================================================================
# TAB 2: ĐỌC TOÀN VĂN 8 CHƯƠNG (FULL DOCUMENT REVIEW)
# =============================================================================
with tab_full_review:
    st.markdown("### 📖 Toàn văn Dự thảo Hồ sơ Mời thầu (E-HSMT)")
    st.caption("Chế độ xem liền mạch từ Chương I đến Chương VIII giúp cán bộ kiểm tra tổng thể bố cục và văn phong.")

    # Thanh thống kê & Hành động tổng thể
    r_col1, r_col2, r_col3 = st.columns([1.5, 1.5, 2])
    r_col1.metric("Tiến độ phê duyệt", f"{approved_count}/{total_count} mục")
    all_flags_count = sum(len(s.flags) for s in doc.sections)
    r_col2.metric("Tổng số cờ cảnh báo", f"{all_flags_count} cờ", delta=None)

    with r_col3:
        if st.button("🏆 Phê duyệt toàn bộ 8 chương", type="primary", use_container_width=True):
            username = current_user()["username"]
            for s in doc.sections:
                store.approve_section(selected_doc_id, s.section_id, approved_by=username)
                audit.record(username, "approve_all_sections", doc_id=selected_doc_id, section_id=s.section_id)
            st.toast("Đã phê duyệt toàn bộ 8 chương HSMT!")
            st.rerun()

    st.markdown("---")

    # Render liên tục 8 chương
    for chapter in CHAPTER_TITLES:
        sections = chapters.get(chapter, [])
        if not sections:
            continue

        st.markdown(f"## {chapter}")
        for s in sections:
            st.markdown(f"#### {s.title} ({status_icon(s.status)} {s.status})")
            if s.flags:
                for f in s.flags:
                    st.warning(f"{severity_icon(f.severity)} **{f.rule_code}**: {f.explanation} (Đoạn: *\"{f.sentence}\"*)")
            st.markdown(s.current_text)
            st.markdown("---")
