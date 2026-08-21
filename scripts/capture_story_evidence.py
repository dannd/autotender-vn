"""Kịch bản tự động tương tác trực tiếp với Chrome thực và chụp ảnh bằng chứng (Story Evidence).

Chạy toàn bộ luồng nghiệp vụ trên Chrome và lưu ảnh chụp màn hình chất lượng cao:
1. Đăng nhập hệ thống (admin / admin123)
2. Trang chủ AutoTender-VN
3. Trang 2: Nạp KHLCNT & Trích xuất thực thể NER
4. Trang 7: Hỏi-đáp pháp lý Mức 1 (RAG + LLM Citations)
5. Trang 3: Soạn thảo 8 chương HSMT & Phê duyệt HITL
6. Trang 4: Kiểm tra tuân thủ pháp luật (R1-R5)
7. Trang 5: Xuất & In (Preview NĐ 30/2020, DOCX & PDF)
8. Trang 8: Báo cáo đánh giá hiệu năng RAG
9. Trang 6: Quản trị Model & Nhật ký kiểm toán Audit Log
"""

import os
import re
import sys
import time
from pathlib import Path

# Ensure UTF-8 console output on Windows
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORY_DIR = PROJECT_ROOT / "story"
STORY_DIR.mkdir(parents=True, exist_ok=True)

APP_URL = "http://localhost:8501"

SAMPLE_KHLCNT_TEXT = """Quyết định phê duyệt Kế hoạch lựa chọn nhà thầu dự án: Xây dựng và triển khai Hệ thống Phần mềm Quản lý Điều hành tác nghiệp tập trung.
Chủ đầu tư: Sở Thông tin và Truyền thông.
Bên mời thầu: Ban Quản lý Dự án Công nghệ thông tin.
Tên gói thầu: Mua sắm và triển khai Hệ thống Phần mềm Quản lý Điều hành tác nghiệp.
Giá gói thầu: 4.500.000.000 VND (Bốn tỷ năm trăm triệu đồng).
Nguồn vốn: Ngân sách nhà nước cấp chi sự nghiệp CNTT năm 2026.
Hình thức lựa chọn nhà thầu: Đấu thầu rộng rãi trong nước, qua mạng.
Phương thức lựa chọn nhà thầu: Một giai đoạn một túi hồ sơ.
Loại hợp đồng: Hợp đồng trọn gói.
Thời gian thực hiện gói thầu: 180 ngày.
Địa điểm thực hiện: Thành phố Hà Nội."""


def wait_for_streamlit(page, timeout_ms=10000):
    """Chờ cho Streamlit hoàn tất re-run và render xong."""
    time.sleep(2.0)
    try:
        page.wait_for_selector(".stApp", timeout=timeout_ms)
        page.wait_for_selector("[data-testid='stStatusWidget']", state="hidden", timeout=timeout_ms)
    except Exception:
        pass
    time.sleep(1.0)


def take_evidence_screenshot(page, filename: str, caption: str):
    """Chụp ảnh màn hình độ phân giải cao và lưu vào folder story/."""
    out_path = STORY_DIR / filename
    page.screenshot(path=str(out_path), full_page=False)
    print(f"  📸 [Chụp ảnh]: {filename} — {caption}")
    return out_path


def navigate_via_sidebar(page, link_text: str):
    """Bấm vào liên kết ở thanh sidebar để chuyển trang mà không reload toàn trang."""
    try:
        page.get_by_role("link", name=link_text).click()
    except Exception:
        try:
            page.locator(f"[data-testid='stSidebarNav'] >> text='{link_text}'").click()
        except Exception:
            page.locator(f"a:has-text('{link_text}')").first.click()
    wait_for_streamlit(page)


def main():
    print("=" * 70)
    print("   BẮT ĐẦU CHẠY THỰC VỚI CHROME & CHỤP ẢNH EVIDENCE VÀO FOLDER STORY/")
    print("=" * 70)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1.5,
        )
        page = context.new_page()

        # ---------------------------------------------------------------------
        # 1. Đăng nhập hệ thống
        # ---------------------------------------------------------------------
        print("\n[Bước 1] Mở ứng dụng & Đăng nhập...")
        page.goto(APP_URL, timeout=30000)
        wait_for_streamlit(page)

        # Kiểm tra xem có form đăng nhập không
        if page.locator("input[type='password']").count() > 0:
            take_evidence_screenshot(page, "01_dang_nhap_he_thong.png", "Màn hình Đăng nhập AutoTender-VN")
            
            # Điền thông tin đăng nhập
            user_inputs = page.locator("input[type='text']").all()
            if user_inputs:
                user_inputs[0].fill("admin")
            page.locator("input[type='password']").fill("admin123")
            page.get_by_role("button", name="Đăng nhập").click()
            wait_for_streamlit(page, 5000)

        # ---------------------------------------------------------------------
        # 2. Trang chủ chính
        # ---------------------------------------------------------------------
        print("\n[Bước 2] Trang chủ giới thiệu hệ thống...")
        take_evidence_screenshot(page, "02_trang_chu_gioi_thieu.png", "Trang chủ AutoTender-VN sau khi đăng nhập")

        # ---------------------------------------------------------------------
        # 3. Trang 2: Nạp KHLCNT & Trích xuất thực thể
        # ---------------------------------------------------------------------
        print("\n[Bước 3] Điều hướng sang Trang 2 (Nạp KHLCNT)...")
        navigate_via_sidebar(page, "Nạp KHLCNT")

        # Chuyển sang Tab Dán văn bản
        try:
            page.get_by_role("tab", name="📝 Dán văn bản").click()
            time.sleep(1.0)
        except Exception as e:
            print("  Tab switch note:", e)

        # Dán nội dung KHLCNT
        text_area = page.locator("textarea")
        if text_area.count() > 0:
            text_area.first.fill(SAMPLE_KHLCNT_TEXT)
            time.sleep(0.5)
            take_evidence_screenshot(page, "03_nap_khlcnt_nhap_lieu.png", "Trang 2 — Nhập văn bản KHLCNT mẫu")
            
            # Bấm nút Dùng văn bản này
            try:
                page.get_by_role("button", name="Dùng văn bản này").click()
            except Exception:
                page.locator("button:has-text('Dùng văn bản này')").click()
            wait_for_streamlit(page)

        take_evidence_screenshot(page, "04_nap_khlcnt_trich_xuat_ner.png", "Trang 2 — Bóc tách thực thể NER & Cấu trúc trường")

        # Bấm nút Xác nhận và chuyển sang soạn thảo
        try:
            page.get_by_role("button", name="✅ Xác nhận và chuyển sang soạn thảo").click()
        except Exception:
            page.locator("button:has-text('Xác nhận và chuyển sang soạn thảo')").click()
        wait_for_streamlit(page)

        # ---------------------------------------------------------------------
        # 4. Trang 7: Hỏi - đáp Pháp lý Mức 1
        # ---------------------------------------------------------------------
        print("\n[Bước 4] Điều hướng sang Trang 7 (Hỏi-đáp Mức 1)...")
        navigate_via_sidebar(page, "Hỏi-đáp (Mức 1)")

        take_evidence_screenshot(page, "05_hoi_dap_phap_ly_cau_hoi.png", "Trang 7 — Giao diện Hỏi-đáp pháp lý đấu thầu")

        # Bấm vào câu hỏi gợi ý
        try:
            page.get_by_role("button", name="Bảo đảm dự thầu được quy định như thế nào?").click()
            wait_for_streamlit(page)
        except Exception:
            pass
        
        # Bấm nút Hỏi
        try:
            page.get_by_role("button", name="🔍 Hỏi").click()
            print("  ... Đang chờ LLM Gateway & RAG trả lời câu hỏi...")
            time.sleep(12)
            wait_for_streamlit(page, 15000)
        except Exception as e:
            print("  Ask button note:", e)

        # Mở expander trích dẫn
        expanders = page.locator("[data-testid='stExpander']").all()
        if expanders:
            expanders[0].click()
            time.sleep(1)

        take_evidence_screenshot(page, "06_hoi_dap_ket_qua_rag_citations.png", "Trang 7 — Câu trả lời có viện dẫn Điều/Khoản luật thật")

        # ---------------------------------------------------------------------
        # 5. Trang 3: Soạn thảo & Phê duyệt HSMT Mức 2
        # ---------------------------------------------------------------------
        print("\n[Bước 5] Điều hướng sang Trang 3 (Soạn thảo HSMT)...")
        navigate_via_sidebar(page, "Soạn thảo HSMT")

        # Nếu có nút Sinh toàn bộ 8 chương HSMT, bấm sinh
        btn_gen_all = page.locator("button:has-text('Sinh toàn bộ 8 chương HSMT')")
        if btn_gen_all.count() > 0:
            print("  ... Đang sinh dự thảo HSMT...")
            btn_gen_all.click()
            time.sleep(20)
            wait_for_streamlit(page, 30000)

        take_evidence_screenshot(page, "07_soan_thao_hsmt_cay_muc_luc.png", "Trang 3 — Cây mục lục 8 chương E-HSMT & Không gian soạn thảo")

        # Cuộn trang để chụp trọn vẹn nội dung mục soạn thảo
        page.evaluate("window.scrollTo(0, 350)")
        time.sleep(1)
        take_evidence_screenshot(page, "08_soan_thao_hsmt_chi_tiet_noi_dung.png", "Trang 3 — Nội dung văn bản với slot-filling và căn cứ pháp lý")

        # Bấm duyệt nhanh hàng loạt mục 0 cảnh báo nếu có
        btn_batch_approve = page.locator("button:has-text('Duyệt nhanh')")
        if btn_batch_approve.count() > 0:
            btn_batch_approve.click()
            wait_for_streamlit(page)

        btn_approve_stay = page.locator("button:has-text('Duyệt mục này')")
        if btn_approve_stay.count() > 0:
            btn_approve_stay.click()
            wait_for_streamlit(page)

        take_evidence_screenshot(page, "09_phe_duyet_hitl_thanh_cong.png", "Trang 3 — Phê duyệt thông minh HITL hoàn tất")

        # ---------------------------------------------------------------------
        # 6. Trang 4: Kiểm tra tuân thủ pháp luật
        # ---------------------------------------------------------------------
        print("\n[Bước 6] Điều hướng sang Trang 4 (Kiểm tra tuân thủ)...")
        navigate_via_sidebar(page, "Kiểm tra tuân thủ")

        take_evidence_screenshot(page, "10_kiem_tra_tuan_thu_ra_soat_co.png", "Trang 4 — Bảng tổng hợp rà soát cờ vi phạm R1-R5")

        # ---------------------------------------------------------------------
        # 7. Trang 5: Xuất & In (DOCX & PDF)
        # ---------------------------------------------------------------------
        print("\n[Bước 7] Điều hướng sang Trang 5 (Xuất và In)...")
        navigate_via_sidebar(page, "Xuất và In")

        take_evidence_screenshot(page, "11_xuat_in_preview_the_thuc_cong_vu.png", "Trang 5 — Xem trước thể thức văn bản hành chính NĐ 30/2020")

        # Chuyển sang tab Xuất file
        try:
            page.get_by_role("tab", name="📤 Xuất file").click()
            time.sleep(1.0)
        except Exception:
            pass

        take_evidence_screenshot(page, "12_xuat_in_tai_docx_pdf.png", "Trang 5 — Xuất bản và tải file DOCX / PDF")

        # ---------------------------------------------------------------------
        # 8. Trang 8: Báo cáo Đánh giá Hiệu năng RAG
        # ---------------------------------------------------------------------
        print("\n[Bước 8] Điều hướng sang Trang 8 (Đánh giá)...")
        navigate_via_sidebar(page, "Đánh giá")

        take_evidence_screenshot(page, "13_danh_gia_hieu_nang_rag.png", "Trang 8 — Biểu đồ Recall@5, MRR, nDCG & Faithfulness")

        # ---------------------------------------------------------------------
        # 9. Trang 6: Quản trị Model & Audit Log
        # ---------------------------------------------------------------------
        print("\n[Bước 9] Điều hướng sang Trang 6 (Bảng điều khiển)...")
        navigate_via_sidebar(page, "Bảng điều khiển")

        take_evidence_screenshot(page, "14_model_dashboard_audit_log.png", "Trang 6 — Quản trị Microservices & Nhật ký kiểm toán Audit Log")

        browser.close()

    print("\n" + "=" * 70)
    print(f"   ĐÃ HOÀN TẤT CHỤP TOÀN BỘ ẢNH BẰNG CHỨNG VÀO FOLDER: {STORY_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
