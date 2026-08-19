# TỔNG QUAN NGHIỆP VỤ ĐẤU THẦU & HỆ THỐNG AUTOTENDER-VN
## Hướng dẫn Dễ hiểu từ Cơ bản đến Chuyên sâu dành cho Kỹ sư & Nhà phát triển

---

## 1. ĐẤU THẦU LÀ GÌ? VÌ SAO PHẢI CÓ QUY TRÌNH NÀY?

### 💡 Ví dụ Đời thường để Dễ hình dung:
> Hãy tưởng tượng bạn là Giám đốc của một Bệnh viện Công lập lớn. Bệnh viện cần xây dựng một **"Phần mềm Quản lý Khám chữa bệnh & Bệnh án Điện tử"** với ngân sách dự kiến là **10 tỷ đồng** từ ngân sách nhà nước.
> 
> Bạn **KHÔNG ĐƯỢC PHÉP** gọi ngay công ty phần mềm của người quen đến ký hợp đồng và thanh toán tiền. Làm như vậy là phạm pháp (tham nhũng, thất thoát tài sản nhà nước, giá đắt nhưng chất lượng kém).
> 
> **Quy định bắt buộc:** Bạn phải **"Đấu thầu công khai"** — đưa ra đề bài yêu cầu rõ ràng để tất cả các công ty CNTT trên toàn quốc (FPT, Viettel, VNPT, CMC,...) cùng nộp hồ sơ cạnh tranh công bằng. Công ty nào đáp ứng kỹ thuật tốt nhất với mức giá hợp lý nhất sẽ trúng thầu.

---

## 2. CÁC KHÁI NIỆM NGHIỆP VỤ CỐT LÕI

| Thuật ngữ | Tên đầy đủ | Giải thích Dễ hiểu |
|---|---|---|
| **Chủ đầu tư / Bên mời thầu** | Procuring Entity | Đơn vị cần mua sắm và chi tiền (ví dụ: Bệnh viện, Bộ ban ngành, Trường Đại học). |
| **Nhà thầu** | Bidders / Contractors | Các công ty phần mềm/CNTT tham gia nộp hồ sơ để nhận dự án. |
| **KHLCNT** | **Kế hoạch lựa chọn nhà thầu** | Văn bản "khai sinh" dự án: Xác định gói thầu tên gì, giá bao nhiêu (ví dụ 10 tỷ), làm trong bao lâu (12 tháng), tiền lấy từ đâu. |
| **E-HSMT** | **Hồ sơ mời thầu điện tử** | **"Đề thi / Đề bài"** do Chủ đầu tư phát hành lên mạng, quy định toàn bộ tiêu chuẩn kỹ thuật, điều kiện hợp đồng và cách chấm điểm. |
| **E-HSDT** | **Hồ sơ dự thầu điện tử** | **"Bài làm"** do Nhà thầu nộp lên để dự thi. |
| **VNEPS / e-GP** | Hệ thống Mạng Đấu thầu Quốc gia | Cổng thông tin điện tử của Chính phủ (`muasamcong.mpi.gov.vn`) — nơi mọi cuộc đấu thầu công đều phải diễn ra công khai. |

---

## 3. "NỖI ĐAU" THỰC TẾ TRONG VIỆC SOẠN THẢO E-HSMT

Soạn thảo một bộ E-HSMT là một công việc **cực kỳ nặng nhọc, tốn thời gian và rủi ro pháp lý cao**:

```
                              NỖI ĐAU CỦA CÁN BỘ ĐẤU THẦU
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Khối lượng tài liệu khổng lồ: 1 bộ HSMT chuẩn gồm 8 Chương, dài 100 - 200 trang.  │
│ 2. Luật thay đổi liên tục: Luật Đấu thầu 22/2023, Nghị định 214/2025 mới thay thế    │
│    Nghị định 24/2024 cũ. Cán bộ copy-paste mẫu cũ rất dễ bị viện dẫn sai luật.        │
│ 3. Rủi ro "Cài cắm - Hạn chế cạnh tranh": Nếu vô tình ghi tên nhãn hiệu (ví dụ:      │
│    "phải dùng database Oracle, chip Intel") hoặc yêu cầu doanh thu quá vô lý, gói     │
│    thầu sẽ bị Thanh tra "thổi còi", hủy thầu, thậm chí bị xử lý hình sự.             │
│ 4. Sai lệch số liệu: Tính nhầm 1 con số bảo lãnh dự thầu hay thời gian thực hiện      │
│    so với KHLCNT là hồ sơ bị coi là không hợp lệ.                                     │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. CẤU TRÚC 8 CHƯƠNG CHUẨN CỦA MỘT BỘ E-HSMT (ĐIỀU 26 NGHỊ ĐỊNH 214/2025/NĐ-CP)

Dự án `AutoTender-VN` tự động hóa việc soạn thảo trọn bộ **8 Chương** pháp định:

```
                            CẤU TRÚC 8 CHƯƠNG E-HSMT
┌───────────────────────────────────────────────────────────────────────────────────────┐
│  PHẦN 1: THỦ TỤC ĐẤU THẦU                                                             │
│  ├── Chương I: Chỉ dẫn nhà thầu (Quy định chung về nộp, mở, đánh giá hồ sơ)          │
│  ├── Chương II: Bảng dữ liệu đấu thầu (Thông tin cụ thể của gói: Giá, Địa điểm, Hạn nộp)│
│  ├── Chương III: Tiêu chuẩn đánh giá E-HSDT (Tiêu chuẩn tài chính, kỹ thuật, nhân sự) │
│  └── Chương IV: Biểu mẫu mời thầu và dự thầu (Các mẫu đơn, cam kết cho nhà thầu điền)│
├───────────────────────────────────────────────────────────────────────────────────────┤
│  PHẦN 2: YÊU CẦU VỀ KỸ THUẬT                                                          │
│  └── Chương V: Yêu cầu về kỹ thuật (Phạm vi công việc, tính năng phần mềm, bảo hành) │
├───────────────────────────────────────────────────────────────────────────────────────┤
│  PHẦN 3: ĐIỀU KIỆN HỢP ĐỒNG & BIỂU MẪU HỢP ĐỒNG                                       │
│  ├── Chương VI: Điều kiện chung của hợp đồng (ĐKC - Các điều khoản pháp lý khung)     │
│  ├── Chương VII: Điều kiện cụ thể của hợp đồng (ĐKCT - Tiến độ thanh toán, phạt vi phạm)│
│  └── Chương VIII: Biểu mẫu hợp đồng (Mẫu hợp đồng khung để ký với nhà thầu trúng thầu)│
└───────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. AUTOTENDER-VN HOẠT ĐỘNG NHƯ THẾ NÀO? (KIẾN TRÚC & NGHIỆP VỤ)

Hệ thống hoạt động như một **"Trợ lý Pháp lý & Kỹ thuật ảo siêu cấp"** cho Cán bộ Đấu thầu qua 5 bước liên hoàn:

```
                                LUỒNG HOẠT ĐỘNG CỦA HỆ THỐNG
                                
 ┌──────────────────────┐
 │ File KHLCNT          │ (Đầu vào: File PDF/Word quyết định phê duyệt KHLCNT)
 └──────────┬───────────┘
            │
            ▼ [Bước 1: Bóc tách thông tin tự động - NER]
 ┌────────────────────────────────────────────────────────────────────────┐
 │ • Tên gói thầu: "Xây dựng Hệ thống Bệnh án Điện tử EMR"               │
 │ • Giá gói thầu: 10.000.000.000 VND (10 tỷ)                             │
 │ • Chủ đầu tư: Bệnh viện Đa khoa Tỉnh X                                 │
 │ • Thời gian: 12 tháng | Nguồn vốn: Ngân sách Nhà nước                  │
 └──────────┬─────────────────────────────────────────────────────────────┘
            │
            ▼ [Bước 2: Truy hồi Luật & Nghị định chính xác - RAG Pipeline]
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Hệ thống tìm kiếm trong 684 đoạn Luật Đấu thầu 22/2023 & NĐ 214/2025:  │
 │ -> Lấy đúng quy định về bảo đảm dự thầu (1%-3% giá gói thầu)           │
 │ -> Lấy đúng quy định về hợp đồng trọn gói/theo thời gian               │
 └──────────┬─────────────────────────────────────────────────────────────┘
            │
            ▼ [Bước 3: Soạn thảo từng chương bằng AI - LLM Generation]
 ┌────────────────────────────────────────────────────────────────────────┐
 │ AI (Claude / DeepSeek qua Gateway) viết dự thảo chuẩn văn phong        │
 │ hành chính, chèn đúng số liệu gói thầu và kèm trích dẫn Điều/Khoản luật│
 └──────────┬─────────────────────────────────────────────────────────────┘
            │
            ▼ [Bước 4: Vòng kiểm soát tuân thủ độc lập - Compliance Guard]
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Hệ thống tự động rà soát 5 nhóm lỗi vi phạm nghiêm trọng (R1 -> R5):   │
 │ 🚩 R1: Có nêu nhãn hiệu cụ thể (Apple, Cisco, Oracle,...) không?       │
 │ 🚩 R2: Yêu cầu doanh thu có quá cao (> 3 lần giá gói thầu) không?      │
 │ 🚩 R3: Thông số kỹ thuật có mang tính "may đo / độc quyền" không?      │
 │ 🚩 R4: Số liệu tài chính có bị lệch so với KHLCNT ban đầu không?       │
 │ 🚩 R5: Có soạn thiếu mục nào bắt buộc theo Nghị định 214 không?        │
 └──────────┬─────────────────────────────────────────────────────────────┘
            │
            ▼ [Bước 5: Con người phê duyệt & Xuất bản - HITL & Export]
 ┌────────────────────────────────────────────────────────────────────────┐
 │ • Cán bộ đấu thầu đọc đối soát, chỉnh sửa trực tiếp trên giao diện     │
 │ • Bấm Phê duyệt (Approved) từng mục -> Ghi Audit Log bất biến          │
 │ • Xuất file Word (.DOCX) và .PDF chuẩn thể thức để phát hành           │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Ý NGHĨA NGHIỆP VỤ CỦA CÁC LUẬT TUÂN THỦ (R1 ĐẾN R5)

Tại sao hệ thống lại đặt ra các bộ lọc **R1, R2, R3, R4, R5**? Đây chính là "linh hồn" bảo vệ tính minh bạch của dự án:

### 🚫 R1 — Chống Chỉ định Nhãn hiệu (Brand Restriction)
- **Luật:** Khoản 2 Điều 44 Luật Đấu thầu 2023 nghiêm cấm nêu nhãn hiệu, xuất xứ cụ thể của hàng hóa.
- **Ví dụ vi phạm:** Ghi trong hồ sơ *"Máy chủ phải là Dell PowerEdge R750"* -> **BỊ PHẠT**.
- **Cách viết đúng:** *"Máy chủ 2 socket, CPU tối thiểu 32 nhân, RAM 64GB hoặc tương đương"*.

### 🚫 R2 — Chống Ép Năng lực Tài chính Phi lý (Excessive Turnover)
- **Luật:** Doanh thu bình quân hàng năm của nhà thầu thường chỉ được yêu cầu từ **1.5 đến 2 lần** giá gói thầu.
- **Ví dụ vi phạm:** Gói thầu 10 tỷ mà yêu cầu *"Nhà thầu phải có doanh thu 50 tỷ/năm (gấp 5 lần)"* -> Bị coi là cố tình tạo rào cản loại bỏ các doanh nghiệp vừa và nhỏ.

### 🚫 R3 — Chống Thông số Kỹ thuật "May đo" (Tailored Specs)
- **Ví dụ vi phạm:** Đưa ra các thông số độc quyền *"Chỉ có sản phẩm của hãng X mới đáp ứng được kích thước chính xác đến từng milimet"*.

### 🚫 R4 — Chống Sai lệch Số liệu Tài chính (Financial Inconsistency)
- So khớp tự động đảm bảo số tiền gói thầu, giá trị bảo đảm dự thầu (100 - 150 triệu VND) khớp từng đồng với Quyết định phê duyệt KHLCNT.

### 🚫 R5 — Chống Thiếu Thành phần Pháp định (Document Completeness)
- Đối chiếu danh mục hồ sơ với Điều 26 Nghị định 214/2025/NĐ-CP để đảm bảo không bị thiếu bất kỳ biểu mẫu hay chương bắt buộc nào khi đem đi thẩm định.

---

## 7. TỔNG KẾT: GIÁ TRỊ MANG LẠI CHO NGƯỜI DÙNG

1. **Rút ngắn thời gian:** Từ **15-20 ngày** soạn thảo thủ công xuống còn **vài giờ** (chỉ cần rà soát và hiệu chỉnh).
2. **An toàn pháp lý 100%:** Luôn cập nhật văn bản pháp luật mới nhất, loại bỏ nguy cơ bị thanh tra xuất toán do dẫn chiếu luật cũ.
3. **Phòng chống tiêu cực:** Cơ chế cảnh báo R1-R3 ngăn chặn từ sớm các biểu hiện "cài cắm tiêu chí", bảo đảm tính cạnh tranh, công bằng và minh bạch trong đấu thầu quốc gia.
