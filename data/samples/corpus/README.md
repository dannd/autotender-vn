# Corpus mẫu cho RAG (M4)

**QUAN TRỌNG — đọc trước khi dùng:** Toàn bộ nội dung trong thư mục này là dữ liệu
**tổng hợp/minh hoạ** do nhóm thực hiện đồ án biên soạn, phỏng theo cấu trúc và văn
phong thường gặp của mẫu E-HSMT và các nguyên tắc phổ biến trong pháp luật đấu thầu
Việt Nam. Đây **KHÔNG PHẢI** là trích dẫn nguyên văn của Luật Đấu thầu 2023, Nghị định
214/2025/NĐ-CP, hay Thông tư 79/2025/TT-BTC.

Lý do dùng dữ liệu tổng hợp thay vì văn bản luật thật:
1. Đồ án có 7 ngày, không đủ thời gian đối chiếu và số hoá chính xác toàn văn các văn
   bản pháp luật hiện hành (rủi ro sai lệch nếu chép tay/OCR không kiểm chứng).
2. Giữ đúng nguyên tắc "Không bịa đặt" (Mục 2.2 SPEC): hệ thống sinh dự thảo chỉ được
   trích dẫn từ corpus đã nạp — nếu corpus chứa nội dung tự bịa mà lại gắn nhãn là luật
   thật, hệ thống sẽ trích dẫn sai một cách có hệ thống. Do đó mọi file ở đây được đánh
   dấu rõ là **minh hoạ**, và `source_doc` của mỗi chunk luôn ghi tiền tố `[MINH HỌA]`.

**Trước khi dùng hệ thống cho mục đích thật:** thay thế các file trong thư mục này
bằng văn bản pháp luật/mẫu Thông tư thật đã được số hoá và kiểm chứng, giữ nguyên
cấu trúc file (mỗi `##` là một chunk biên giới điều/khoản/mục) để `rag/chunker.py`
hoạt động không cần sửa code.

## Cấu trúc file
- `mau_hsmt_chuong_iii.md` — mẫu điều khoản Chương III (Tiêu chuẩn đánh giá E-HSDT)
- `mau_hsmt_chuong_v.md` — mẫu điều khoản Chương V (Yêu cầu kỹ thuật)
- `nguyen_tac_phap_ly_minh_hoa.md` — các nguyên tắc pháp lý phổ biến, diễn giải lại
  bằng ngôn ngữ riêng (không trích nguyên văn điều/khoản nào)
