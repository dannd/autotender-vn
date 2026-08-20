# Hướng Dẫn Vận Hành, Kiểm Thử & Bộ Dữ Liệu Mẫu (AutoTender-VN)

> **Tài liệu dành cho:** Hội đồng Đánh giá Đồ án Deep Learning / Khóa luận Thạc sĩ, Kỹ sư Phát triển và Người thẩm định nghiệp vụ Đấu thầu.
> **Hệ thống:** AutoTender-VN — Trợ lý AI Soạn thảo & Rà soát Hồ sơ Mời thầu (E-HSMT) tại Việt Nam.

---

## MỤC LỤC
1. [Hướng dẫn Cấu hình & Khởi chạy (Run Guideline)](#1-hướng-dẫn-cấu-hình--khởi-chạy-run-guideline)
2. [Hướng dẫn Chạy Test Suite (Automated Testing)](#2-hướng-dẫn-chạy-test-suite-automated-testing)
3. [Bộ Dữ liệu Mẫu & Kịch bản Kiểm thử Thực tế (Test Scenarios)](#3-bộ-dữ-liệu-mẫu--kịch-bản-kiểm-thử-thực-tế-test-scenarios)
   - [Kịch bản 1: Pipeline Soạn thảo HSMT 8 chương hoàn chỉnh](#kịch-bản-1-pipeline-soạn-thảo-hsmt-8-chương-hoàn-chỉnh)
   - [Kịch bản 2: Rà soát Vi phạm Chỉ định Nhãn hiệu (R1 Compliance)](#kịch-bản-2-rà-soát-vi-phạm-chỉ-định-nhãn-hiệu-r1-compliance)
   - [Kịch bản 3: Rà soát Tiêu chí Doanh thu Bất hợp lý (R2 Verifier)](#kịch-bản-3-rà-soát-tiêu-chí-doanh-thu-bất-hợp-lý-r2-verifier)
   - [Kịch bản 4: Hỏi - Đáp Pháp lý Đấu thầu RAG (Mức 1 QA)](#kịch-bản-4-hỏi---đáp-pháp-lý-đấu-thầu-rag-mức-1-qa)
   - [Kịch bản 5: Xuất Báo cáo Hồ sơ HSMT Word/PDF & Biên bản Audit](#kịch-bản-5-xuất-báo-cáo-hồ-sơ-hsmt-wordpdf--biên-bản-audit)

---

## 1. Hướng Dẫn Cấu Hình & Khởi Chạy (Run Guideline)

### Bước 1: Kích hoạt môi trường ảo Python
Mở cửa sổ PowerShell tại thư mục gốc dự án:
```powershell
.venv\Scripts\Activate.ps1
```

---

### Bước 2: Cấu hình API Key (Universal LLM Gateway)
Tạo hoặc chỉnh sửa file `.env` tại thư mục gốc của dự án (`autotender-vn/.env`):

```env
# ==============================================================================
# CẤU HÌNH UNIVERSAL LLM GATEWAY (OpenAI-compatible / WokuShop / OpenAI / DeepSeek)
# ==============================================================================
LLM_API_KEY=sk-xxxx-dien-key-cua-ban-vao-day
LLM_BASE_URL=https://llm.wokushop.com/v1
LLM_MODEL=claude-3-5-sonnet-20241022

# (Tuỳ chọn: Nếu muốn thử nghiệm DeepSeek-V3/R1 trên WokuShop để tối ưu chi phí)
# LLM_MODEL=deepseek-chat

# ==============================================================================
# CẤU HÌNH VECTOR DATABASE (Qdrant)
# ==============================================================================
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=legal_chunks
```

> **Nguyên tắc Graceful Degradation / Degraded Mode:**
> Nếu bạn chưa điền API Key hoặc Qdrant đang offline, hệ thống **vẫn hoạt động 100% bình thường** ở **Tier 3 (Rule-based Regex + BM25 Sparse Search + Template-filling)** mà không bị dừng hay crash.

---

### Bước 3: Khởi chạy Ứng dụng

#### 🔹 Lựa chọn 1: Web App Streamlit (Giao diện người dùng chính)
```powershell
streamlit run app/main.py
```
👉 Mở trình duyệt truy cập: **`http://localhost:8501`**

#### 🔹 Lựa chọn 2: FastAPI REST API Server (Backend Microservice)
```powershell
uvicorn autotender.api:app --host 0.0.0.0 --port 8000 --reload
```
👉 Xem tài liệu OpenAPI / Swagger tương tác: **`http://localhost:8000/docs`**

#### 🔹 Lựa chọn 3: Khởi động Qdrant Vector DB bằng Docker (Dense Search)
```powershell
# 1. Bật Qdrant container
docker compose up -d qdrant

# 2. Xây dựng index kho tri thức pháp luật vào Qdrant
python scripts/build_legal_index.py
```

---

## 2. Hướng Dẫn Chạy Test Suite (Automated Testing)

Hệ thống sở hữu bộ test tự động hóa toàn diện gồm **160 tests**, bao phủ toàn bộ các tầng kiến trúc.

### 2.1 Chạy toàn bộ Test Suite
```powershell
pytest
```
**Kết quả chuẩn:** `160 passed in ~2 phút` (100% Pass Rate).

### 2.2 Chạy kiểm thử theo từng Module chuyên biệt

| Module cần kiểm tra | Lệnh chạy | Mô tả kiểm thử |
| :--- | :--- | :--- |
| **LLM Gateway & Budget** | `pytest tests/test_claude_client.py` | Kiểm tra kết nối Gateway, OpenAI schema, trần chi phí `$5/process` |
| **Generator & Verifier R4** | `pytest tests/test_generator.py` | Kiểm tra sinh 8 chương, cơ chế chống cờ giả số liệu kỹ thuật |
| **Legal QA (Mức 1 RAG)** | `pytest tests/test_legal_qa.py` | Kiểm tra hỏi đáp pháp lý có trích dẫn, fallback 3 tầng |
| **Hybrid Retriever** | `pytest tests/test_hybrid_retriever.py` | Kiểm tra RRF fusion, Cross-Encoder rerank, Qdrant/FAISS/BM25 fallback |
| **Compliance Rules (R1–R5)** | `pytest tests/test_compliance.py` | Kiểm tra các bộ luật cấm nhãn hiệu, trần doanh thu, thiếu HSMT |
| **End-to-End Orchestrator** | `pytest tests/test_orchestrator.py` | Kiểm tra toàn bộ luồng tích hợp từ KHLCNT đến HSMT hoàn chỉnh |

---

## 3. Bộ Dữ Liệu Mẫu & Kịch Bản Kiểm Thử Thực Tế (Test Scenarios)

---

### 📋 Kịch bản 1: Pipeline Soạn thảo HSMT 8 chương hoàn chỉnh

*Mục đích:* Kiểm tra khả năng trích xuất thực thể NER (M2), truy xuất tri thức luật RAG (M4) và sinh hồ sơ mời thầu chuẩn 8 chương (M5).

**Các bước thực hiện:**
1. Mở **Trang 1: Nhập KHLCNT**.
2. Chọn tab **"Dán văn bản KHLCNT"** và dán nội dung gói thầu mẫu sau:

```text
Tên gói thầu: Mua sắm máy chủ và thiết bị lưu trữ dữ liệu trung tâm cho hệ thống giám sát y tế
Chủ đầu tư: Sở Y tế tỉnh Nghệ An
Bên mời thầu: Ban Quản lý dự án đầu tư xây dựng ngành Y tế
Giá gói thầu: 12.500.000.000 đồng
Nguồn vốn: Ngân sách nhà nước năm 2026
Hình thức lựa chọn nhà thầu: Đấu thầu rộng rãi qua mạng
Phương thức lựa chọn nhà thầu: Một giai đoạn một túi hồ sơ
Loại hợp đồng: Trọn gói
Thời gian thực hiện hợp đồng: 90 ngày
Lĩnh vực: Hàng hóa
Thời gian bắt đầu tổ chức lựa chọn nhà thầu: Quý II/2026
Địa điểm thực hiện: Thành phố Vinh, tỉnh Nghệ An
```

3. Bấm nút **"Trích xuất thông tin (M2 NER)"** $\rightarrow$ Hệ thống bóc tách chính xác 8 trường dữ liệu.
4. Chuyển sang **Trang 3 (Soạn thảo HSMT)** $\rightarrow$ Nhấn **"Sinh toàn bộ 8 chương HSMT"**:
   - Hệ thống tự động kích hoạt LLM Gateway (WokuShop) để soạn thảo 8 chương chuẩn E-HSMT theo mẫu Thông tư 22/2024/TT-BKHĐT và Nghị định 214/2025/NĐ-CP.
   - Các trích dẫn điều khoản luật được gắn inline chính xác.
5. Chuyển sang **Trang 4 (Rà soát Tuân thủ)** $\rightarrow$ Hệ thống tự động quét 5 nhóm quy tắc R1–R5.

---

### 📋 Kịch bản 2: Rà soát Vi phạm Chỉ định Nhãn hiệu (R1 Compliance)

*Mục đích:* Kiểm tra bộ quét Compliance **R1** phát hiện hành vi cấm nêu nhãn hiệu độc quyền theo Điều 44 Luật Đấu thầu số 22/2023/QH15.

**Các bước thực hiện:**
1. Tại **Trang 3: Soạn thảo HSMT**, chọn mục **Chương III: Tiêu chuẩn đánh giá HSDT**.
2. Nhập hoặc sửa đổi nội dung với đoạn văn vi phạm sau:
```text
Yêu cầu hệ thống máy chủ phải sử dụng bộ vi xử lý Intel Xeon Gold thế hệ mới nhất, trang bị card mạng Cisco Catalyst 9300 và hệ điều hành Microsoft Windows Server 2025 bản quyền.
```
3. Chuyển sang **Trang 4: Rà soát Tuân thủ**:
   - **Kết quả hiển thị:** Gắn cờ đỏ **R1 (Nêu nhãn hiệu cụ thể)**:
     - 🔴 Nhãn hiệu vi phạm: `Intel`, `Cisco`, `Microsoft`.
     - 🔴 Căn cứ pháp lý: Khoản 3 Điều 44 Luật Đấu thầu 22/2023/QH15.
     - 💡 Khuyến nghị: Thêm cụm từ quy định bắt buộc *"hoặc tương đương"*.

---

### 📋 Kịch bản 3: Rà soát Tiêu chí Doanh thu Bất hợp lý (R2 Verifier)

*Mục đích:* Kiểm tra bộ quét Compliance **R2** phát hiện việc cài cắm yêu cầu doanh thu vượt trần $\times 3$ giá gói thầu nhằm loại trừ nhà thầu.

**Các bước thực hiện:**
1. Gói thầu có giá trị trích xuất: `12.500.000.000 VNĐ` (12.5 tỷ đồng).
2. Tại **Chương III Mục 2 (Năng lực tài chính)**, sửa đổi thành:
```text
Doanh thu bình quân hàng năm từ hoạt động sản xuất kinh doanh của 03 năm tài chính gần nhất của nhà thầu phải đạt tối thiểu từ 50.000.000.000 VNĐ (Năm mươi tỷ đồng) trở lên.
```
3. Chuyển sang **Trang 4 (Rà soát Tuân thủ)**:
   - **Kết quả hiển thị:** Gắn cờ đỏ **R2 (Doanh thu vượt tỷ lệ cho phép)** vì $50 \text{ tỷ} > 3 \times 12.5 \text{ tỷ} = 37.5 \text{ tỷ}$ (hạn chế sự tham gia của nhà thầu vừa và nhỏ theo NĐ 214/2025/NĐ-CP).

---

### 📋 Kịch bản 4: Hỏi - Đáp Pháp lý Đấu thầu RAG (Mức 1 QA)

*Mục đích:* Kiểm thử năng lực tra cứu pháp lý ngữ nghĩa, trích xuất chính xác căn cứ luật định và chống bịa đặt (Anti-hallucination).

Vào **Trang 5: Hỏi - Đáp Pháp lý**, thử nghiệm các câu hỏi sau:

| Câu hỏi kiểm thử | Trọng tâm đánh giá | Căn cứ luật định mong đợi |
| :--- | :--- | :--- |
| **"Hồ sơ mời thầu gói thầu phần mềm có được nêu tên hãng sản xuất không?"** | Cấm chỉ định nhãn hiệu trong CNTT | Điều 44 Luật Đấu thầu 22/2023 & NĐ 45/2026/NĐ-CP |
| **"Thời gian nộp E-HSDT tối thiểu đối với gói thầu dịch vụ phi tư vấn qua mạng là bao nhiêu ngày?"** | Thời hạn quy định mới | Điều 45 Luật Đấu thầu 22/2023 & NĐ 214/2025 |
| **"Các trường hợp nào được áp dụng hình thức chỉ định thầu rút gọn?"** | Điều kiện chỉ định thầu | Điều 23 Luật Đấu thầu 22/2023/QH15 |
| **"Quy định về bảo hành và nghiệm thu phần mềm nội bộ theo Nghị định 45/2026 thế nào?"** | Quy định chuyên ngành CNTT | Nghị định 45/2026/NĐ-CP |

---

### 📋 Kịch bản 5: Xuất Báo cáo Hồ sơ HSMT Word/PDF & Biên bản Audit

1. **Trang 7 (Xuất HSMT):**
   - Lựa chọn định dạng xuất: **Microsoft Word (.docx)** hoặc **Adobe PDF (.pdf)**.
   - Bấm nút **"Xuất HSMT"** và tải file về máy.
   - Kiểm tra định dạng hành chính chuẩn: Quốc hiệu, Tiêu ngữ, Bảng số liệu KHLCNT, 8 chương nội dung và mục lục chi tiết.
2. **Trang 8 (Audit Log & HITL History):**
   - Kiểm tra bảng lưu vết toàn bộ hoạt động: Thời gian chỉnh sửa, người thực hiện (admin/expert), hành động duyệt/từ chối từng chương (Human-in-the-Loop) và lượng token/chi phí USD đã sử dụng.

---

*Tài liệu được cập nhật tự động theo phiên bản phát triển mới nhất của AutoTender-VN.*
