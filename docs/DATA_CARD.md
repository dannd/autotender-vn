# Data Card — AutoTender-VN

## 1. Tổng quan các nguồn dữ liệu

| Bộ dữ liệu | Vị trí | Loại | Kích thước | Cách tạo |
|---|---|---|---|---|
| Bản ghi TBMT mẫu | `data/samples/tender_notices.jsonl` | Tổng hợp (synthetic) | 20 bản ghi | `scripts/build_samples.py` |
| Bản ghi TBMT thật (pilot) | `data/samples/real_pilot_sample.jsonl` | **Thật**, thu thập thủ công | 12 bản ghi | Xem Mục 8 bên dưới |
| Corpus mẫu HSMT + nguyên tắc pháp lý | `data/samples/corpus/*.md` | Tổng hợp/minh hoạ | 3 file, 13 chunk | Biên soạn thủ công |
| Dataset NER (distant supervision) | `data/processed/ner_dataset.jsonl` | Nhãn tự động (silver) | 20 bản ghi | `scripts/build_dataset.py` |
| Dataset Classifier | `data/processed/classifier_dataset.jsonl` | Nhãn tự động | 20 bản ghi | `scripts/build_dataset.py` |
| Tập test Compliance (M6) | `scripts/eval_utils.py::COMPLIANCE_TEST_SET` | Gán tay | 10 câu | Biên soạn thủ công |

**Không có dữ liệu cá nhân nào được thu thập.** Toàn bộ dữ liệu trong repo là tổng hợp.

---

## 2. Vì sao dùng dữ liệu tổng hợp thay vì crawl dữ liệu thật

Trong quá trình triển khai M0 (crawler), nhóm đã:
1. Xác nhận `robots.txt` của `muasamcong.mpi.gov.vn` cho phép truy cập (`Disallow:` rỗng).
2. Xác định được endpoint API JSON nội bộ thật qua DevTools Network:
   `POST /o/egp-portal-home/services/smart/search` (module Liferay `egp-portal-home`),
   xác nhận trả về dữ liệu TBMT thật khớp tốt với schema `TenderNotice`.
3. Thử gọi lại endpoint này bằng nhiều payload JSON hợp lý (page/size, pageable,
   criteria, filter...), kể cả gọi cùng-origin có cookie phiên hợp lệ từ trình duyệt
   thật — tất cả đều trả về `400 Bad Request`. Hợp đồng payload thật của server không
   xác định được trong thời gian cho phép của đồ án 7 ngày.

→ Đây chính là rủi ro đã liệt kê sẵn trong SPEC (Mục 14: "Crawl bị chặn/có captcha").
Giải pháp: dùng `LocalSampleSource` với 20 bản ghi **tổng hợp** làm dữ liệu chính thức,
ghi rõ giới hạn nghiên cứu này. `MSCApiSource`/`MSCBrowserSource` vẫn được implement đầy
đủ (interface + parser đúng theo shape API thật đã quan sát) và sẵn sàng dùng ngay khi
xác định được payload đúng — xem docstring `autotender/crawler/sources.py::MSCApiSource`.

---

## 3. Bản ghi TBMT mẫu (`data/samples/tender_notices.jsonl`)

20 bản ghi được sinh bằng cách chọn ngẫu nhiên (seed cố định = 42, tái lập được) từ các
danh sách: 10 chủ đầu tư hư cấu (tên cơ quan có thật về LOẠI HÌNH nhưng không trỏ đến
đơn vị cụ thể nào), 5 loại gói thầu, 4 hình thức lựa chọn nhà thầu, các mẫu tên gói thầu
theo từng loại. Giá gói thầu, ngày tháng được sinh ngẫu nhiên trong khoảng hợp lý.

**Giới hạn:** phân bố dữ liệu (tỷ lệ loại gói thầu, khoảng giá trị...) không phản ánh
phân bố thật của dữ liệu đấu thầu quốc gia — chỉ đủ để pipeline chạy end-to-end và demo.

---

## 4. Corpus RAG (`data/samples/corpus/`)

**QUAN TRỌNG:** Nội dung 3 file `.md` trong thư mục này (`mau_hsmt_chuong_iii.md`,
`mau_hsmt_chuong_v.md`, `nguyen_tac_phap_ly_minh_hoa.md`) là **văn bản tổng hợp/minh hoạ**
do nhóm biên soạn, phỏng theo cấu trúc và các nguyên tắc phổ biến được biết đến rộng rãi
trong thực hành đấu thầu tại Việt Nam (không nêu nhãn hiệu cụ thể, yêu cầu năng lực tương
xứng quy mô gói thầu...). Đây **KHÔNG PHẢI** trích dẫn nguyên văn của Luật Đấu thầu 2023,
Nghị định 214/2025/NĐ-CP, hay Thông tư 79/2025/TT-BTC.

Lý do (liên hệ trực tiếp với nguyên tắc "Không bịa đặt" — Mục 2.2 SPEC): nếu corpus chứa
nội dung tự soạn nhưng gắn nhãn là luật thật, hệ thống RAG/generator sẽ trích dẫn sai một
cách có hệ thống và không thể phát hiện được bằng cách đọc code. Do đó mọi chunk trong
corpus này được gắn tiền tố `[MINH HỌA]` trong `source_doc`, hiển thị rõ trên UI (Trang 3,
panel "Căn cứ & Cờ") để người dùng không nhầm là căn cứ pháp lý thật.

**Trước khi dùng cho mục đích thật:** thay 3 file này bằng văn bản pháp luật/mẫu Thông tư
đã số hoá và kiểm chứng, giữ nguyên quy ước heading `## ` làm biên giới chunk.

---

## 5. Dataset NER — Distant Supervision (`scripts/build_dataset.py`)

**Phương pháp:** với mỗi `TenderNotice`, sinh văn bản KHLCNT tổng hợp
(`ingest/synth_document.py`) rồi khớp chuỗi các trường đã biết (`package_name`,
`investor`, `package_value`...) vào văn bản để tự động gán nhãn BIO — không cần gán tay.

**Giới hạn quan trọng (ảnh hưởng cách đọc kết quả trong `reports/metrics.json`):** vì nhãn
gold sinh ra bằng cách khớp CHÍNH XÁC cùng các giá trị mà module NER Tier 3 (regex) cũng
tìm cách trích xuất, kết quả entity-F1 = 1.0 phản ánh **tính nhất quán nội tại** giữa 2 cơ
chế cùng dựa trên khớp chuỗi, KHÔNG phải năng lực tổng quát hoá trên dữ liệu chưa thấy.
Đây là hạn chế cố hữu khi chưa có dữ liệu thật + tập test 200 mẫu gán tay độc lập theo
đúng yêu cầu Mục 6/M2 SPEC (việc gán tay quy mô lớn nằm ngoài phạm vi tự động hoá của
đồ án 7 ngày, cần thực hiện thủ công trên dữ liệu crawl thật khi có).

Tương tự, `m4_retrieval.bm25_proxy_recall_at_5` dùng proxy "đúng nếu top-5 chứa chunk từ
đúng file chương" thay vì tập câu hỏi–chunk gán tay độc lập.

---

## 6. Tập test Compliance (M6)

10 câu được biên soạn thủ công, phủ đều 4 lớp (R1, R2, R3, OK), dùng để tính precision/
recall/F1 trong `scripts/evaluate.py::evaluate_compliance`. Do bộ test do cùng người viết
rule biên soạn, kết quả F1 cao (gần 1.0) không đại diện cho hiệu năng trên câu văn E-HSMT
thật đa dạng hơn — cần mở rộng tập test bằng dữ liệu HSMT thật + phản hồi HITL
(`hitl/feedback.py`) trước khi dùng số liệu này để so sánh với Tier 1 thật.

---

## 7. Vòng lặp cải tiến dữ liệu (Data Flywheel)

`hitl/feedback.py` xuất `data/processed/feedback_generator_{date}.jsonl` (cặp
generated→edited) và `feedback_compliance_{date}.jsonl` (nhãn đúng/sai người dùng xác
nhận cho từng cờ). Đây là cơ chế để, khi hệ thống được dùng thật, dữ liệu chất lượng cao
hơn dữ liệu tổng hợp ở trên sẽ tích luỹ dần và dùng để huấn luyện lại M5/M6 — xem
`notebooks/03_train_retriever.ipynb` và `notebooks/04_train_generator.ipynb` để biết cách
dữ liệu phản hồi này được đưa vào vòng huấn luyện tiếp theo.

---

## 8. Nỗ lực crawl dữ liệu thật lần 2 — phát hiện mới và giới hạn

Sau lần điều tra đầu (Mục 2), nhóm thử lại với trang tra cứu đầy đủ (không phải widget tìm
kiếm nhanh ở trang chủ) và có 3 phát hiện quan trọng:

**1. Tìm ra cơ chế xác thực thật của endpoint.** Trang tra cứu đầy đủ tại
`/web/guest/contractor-selection?render=search` gọi
`POST /o/egp-portal-contractor-selection-v2/services/smart/search?token=<chuỗi rất dài>`.
Tham số `token` là **CSRF token động do Liferay sinh ra khi render trang**, thay đổi mỗi
lần — đây chính là lý do mọi lần gọi POST trực tiếp trước đó (kể cả kèm cookie phiên hợp
lệ) đều nhận HTTP 400: thiếu tham số bắt buộc này, không phải do sai cấu trúc body.

**2. Xác nhận có 624.050 bản ghi TBMT thật đang có trên hệ thống** (số liệu hiển thị trực
tiếp trên UI khi tìm kiếm không lọc). Đã lấy thành công **12 bản ghi thật đầy đủ** qua thao
tác trực tiếp trên trình duyệt tương tác (bấm nút "Tìm kiếm" → đọc response mạng), lưu tại
`data/samples/real_pilot_sample.jsonl`. Các bản ghi này là **dữ liệu công khai thật** (thông
báo mời thầu, không chứa thông tin cá nhân), khác với `tender_notices.jsonl` (tổng hợp).

**3. Rào cản cho việc tự động hoá toàn bộ (quan trọng nhất):**
- Chạy Playwright **độc lập** (headless, không qua kênh trình duyệt tương tác) từ môi
  trường sandbox/cloud của phiên làm việc này bị **treo vô thời hạn ở bước điều hướng**
  (`page.goto` timeout), kể cả khi thử `headless=False` hoặc `channel="msedge"` — trong khi
  `httpx` (không giả lập trình duyệt) vẫn gọi được `robots.txt` và endpoint bình thường
  (nhận 400 nhanh, không bị treo). Điều này gợi ý WAF của trang chặn/phát hiện dựa trên dấu
  hiệu trình duyệt tự động hoá (TLS/HTTP fingerprint), không phải chặn IP hay chặn request
  đơn thuần.
- Khi thao tác qua kênh trình duyệt tương tác (không phải script), lấy được dữ liệu thật
  bình thường — NHƯNG sau vài lần bấm phân trang liên tiếp (dùng `element.click()` qua
  JavaScript, tốc độ nhanh hơn thao tác người dùng thật), toàn bộ phiên bị từ chối kể cả
  điều hướng thông thường tới trang chủ. Điều này cho thấy hệ thống cũng giám sát **tốc độ
  và kiểu tương tác trong phiên**, không chỉ tần suất request.

**Kết luận cho việc triển khai thật:** `MSCBrowserSource` (`crawler/sources.py`) đã được cập
nhật để nhắm đúng trang tra cứu và tự động phân trang, nhưng khả năng chạy thành công phụ
thuộc mạnh vào **hạ tầng mạng nơi chạy** (mạng dân dụng có khả năng qua được WAF tốt hơn
mạng datacenter/cloud) và **phải giữ tốc độ thao tác chậm, giống người dùng thật** — xem
cảnh báo chi tiết trong docstring của `MSCBrowserSource`. Không nên tăng tốc độ phân trang
để lấy nhiều dữ liệu nhanh hơn — rủi ro bị chặn hoàn toàn cao hơn lợi ích.
