# Data Card — AutoTender-VN

## 1. Tổng quan các nguồn dữ liệu

| Bộ dữ liệu | Vị trí | Loại | Kích thước | Cách tạo |
|---|---|---|---|---|
| Bản ghi TBMT mẫu | `data/samples/tender_notices.jsonl` | Tổng hợp (synthetic) | 20 bản ghi | `scripts/build_samples.py` |
| Bản ghi TBMT thật (pilot, muasamcong) | `data/samples/real_pilot_sample.jsonl` | **Thật**, thu thập thủ công | 12 bản ghi | Xem Mục 8 bên dưới |
| Bản ghi TBMT thật (dauthau.asia) | `data/samples/real_dauthau_asia_sample.jsonl` | **Thật**, crawl tự động có rate-limit | 489 bản ghi | `scripts/crawl_dauthau_asia.py --enrich-details` — xem Mục 9 |
| Corpus mẫu HSMT + nguyên tắc pháp lý | `data/samples/corpus/*.md` | Tổng hợp/minh hoạ | 3 file, 13 chunk | Biên soạn thủ công |
| Dataset NER (distant supervision) | `data/processed/ner_dataset.jsonl` | Nhãn tự động (silver) | 20 bản ghi | `scripts/build_dataset.py` |
| Dataset Classifier | `data/processed/classifier_dataset.jsonl` | Nhãn tự động | 20 bản ghi | `scripts/build_dataset.py` |
| Tập test Compliance (M6) | `scripts/eval_utils.py::COMPLIANCE_TEST_SET` | Gán tay | 10 câu | Biên soạn thủ công |

**Không có dữ liệu cá nhân nào được thu thập.** Toàn bộ dữ liệu thật thu thập được (Mục 8,
9) là thông tin đấu thầu công khai (tên gói thầu, chủ đầu tư là tổ chức/cơ quan, không phải
cá nhân) — không chứa thông tin cá nhân.

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

Tương tự, `m4_retrieval.bm25_proxy_recall_at_5` (số liệu lịch sử, module `models/retriever.py`
sinh ra nó đã gỡ khỏi repo) dùng proxy "đúng nếu top-5 chứa chunk từ đúng file chương" thay
vì tập câu hỏi–chunk gán tay độc lập — xem Mục 12 để biết tập gán tay độc lập thay thế.

---

## 6. Tập test Compliance (M6)

10 câu được biên soạn thủ công, phủ đều 4 lớp (R1, R2, R3, OK), dùng để tính precision/
recall/F1 — số liệu ghi trong `docs/MODEL_CARD.md` là ảnh chụp cố định từ lần chạy cuối
của `scripts/evaluate.py` (đã gỡ khỏi repo khi dọn hệ Tier1/2/3 cũ đã bị thay thế bởi
RAG+LLM). Do bộ test do cùng người viết rule biên soạn, kết quả F1 cao (gần 1.0) không đại
diện cho hiệu năng trên câu văn E-HSMT thật đa dạng hơn — cần mở rộng tập test bằng dữ liệu
HSMT thật + phản hồi HITL (`hitl/feedback.py`) trước khi dùng số liệu này để so sánh với
Tier 1 thật.

---

## 7. Vòng lặp cải tiến dữ liệu (Data Flywheel)

`hitl/feedback.py` xuất `data/processed/feedback_generator_{date}.jsonl` (cặp
generated→edited) và `feedback_compliance_{date}.jsonl` (nhãn đúng/sai người dùng xác
nhận cho từng cờ). Đây là cơ chế để, khi hệ thống được dùng thật, dữ liệu chất lượng cao
hơn dữ liệu tổng hợp ở trên sẽ tích luỹ dần và dùng để huấn luyện lại M5/M6 — xem
`notebooks/04_train_generator.ipynb` để biết cách dữ liệu phản hồi này được đưa vào vòng
huấn luyện tiếp theo (notebook train M4-retriever cũ đã gỡ cùng `models/retriever.py`,
xem `docs/MODEL_CARD.md`).

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

---

## 9. Nguồn thứ hai: crawl dauthau.asia (489 bản ghi thật, đủ 6/8 trường NER)

**dauthau.asia (DauThau.info)** là sản phẩm phần mềm thương mại (gói trả phí VIP1/VIP5)
của Công ty CP Hệ sinh thái Đấu Thầu, tự nhận là tổng hợp lại dữ liệu công khai từ Hệ
thống mạng đấu thầu quốc gia và cung cấp thêm công cụ tìm kiếm/phân tích.

**Ràng buộc điều khoản sử dụng — đã đọc trước khi crawl:** Điều 2 và Điều 4 của
[Điều khoản sử dụng](https://dauthau.asia/siteterms/terms-and-conditions.html) quy định
không được sao chép/tái sử dụng nội dung ngoài **mục đích cá nhân, nội bộ, phi thương mại**
nếu không có sự cho phép bằng văn bản. Việc crawl dưới đây được thực hiện sau khi người
thực hiện đồ án xác nhận rõ ràng đây là mục đích học thuật/nội bộ, phi thương mại, phù hợp
với ngoại lệ nêu trong chính điều khoản đó. **Nếu tái sử dụng dữ liệu này cho mục đích khác
(đặc biệt là thương mại), cần liên hệ xin phép DauThau.info trước.**

**Khác biệt kỹ thuật so với muasamcong.mpi.gov.vn (Mục 2, 8):** trang này dùng phân trang
URL tĩnh (`https://dauthau.asia/thongbao/moithau/?page=N`), server-render HTML thuần,
KHÔNG cần CSRF token động hay trình duyệt — gọi thẳng bằng `httpx` qua
`MscHttpClient.request_text()` (dùng chung hạ tầng rate-limit/cache/robots.txt với M0).
`robots.txt` của trang chỉ chặn vài thư mục kỹ thuật (`/data/`, `/users/`,
`/statistics/`...), không chặn `/thongbao/moithau/`.

**Giai đoạn 1 (danh sách):** trang danh sách chỉ có `tbmt_id`, `package_name`, `investor`,
`publish_date`, `close_date`, `source_url`.

**Giai đoạn 2 (làm giàu từ trang chi tiết — `enrich_dauthau_asia_detail`):** mỗi trang chi
tiết (vd `.../thongbao/moithau/<slug>.html`) hiển thị CÔNG KHAI (không cần đăng nhập) thêm 4
trường quan trọng: `funding_source` ("Chi tiết nguồn vốn"), `selection_method` ("Hình thức
LCNT"), `contract_type` ("Loại hợp đồng"), và đặc biệt **`package_type`** từ trường "Lĩnh vực
MSC" — khớp thẳng với 5 lớp phân loại M3 cần (hàng hóa/xây lắp/tư vấn/phi tư vấn/hỗn hợp),
không phải suy đoán. `package_value` (giá gói thầu) và `execution_time` (thời gian thực
hiện) LUÔN hiển thị "Để xem đầy đủ thông tin mời bạn Đăng nhập hoặc Đăng ký" — bị khoá sau
đăng nhập, code phát hiện đúng marker này và giữ nguyên `None`, không suy đoán (Mục 2.2).

**Đã kiểm tra thêm:** trang "Kết quả lựa chọn nhà thầu" (`/ketqua/luachon-nhathau/`, kết quả
đấu thầu ĐÃ công bố) cũng khoá cột giá trúng thầu sau đăng nhập tương tự — xác nhận đây là
giới hạn nhất quán trên toàn site theo mô hình kinh doanh của họ (không phải do dữ liệu còn
"nóng"/nhạy cảm), không có cách nào lấy `package_value`/`execution_time` công khai từ site
này.

**Kết quả:** `scripts/crawl_dauthau_asia.py --max-pages 25 --enrich-details` lấy được **489
bản ghi thật, không trùng lặp**, không gặp lỗi/chặn nào trong toàn bộ 2 lần chạy (khác hẳn
trải nghiệm với muasamcong.mpi.gov.vn), rate-limit 2.5 giây/request (~500 request cho danh
sách + chi tiết, tổng ~20 phút chia 2 lần chạy, cache tránh tải lại các trang đã có). Lưu tại
`data/samples/real_dauthau_asia_sample.jsonl`. Độ phủ trường: `package_type` 100%,
`funding_source` 98%, `selection_method`/`contract_type` 100%, `package_value`/
`execution_time` 0% (khoá vĩnh viễn, xem trên).

**Tác động tới huấn luyện:** `scripts/build_dataset.py` mặc định gộp cả 3 nguồn (20 tổng
hợp + 12 muasamcong + 489 dauthau.asia = 521 bản ghi) để sinh `data/processed/ner_dataset.jsonl`
và `classifier_dataset.jsonl`. Đánh giá lại M2/M3 trên bộ 521 bản ghi này (`docs/MODEL_CARD.md`)
cho kết quả trung thực hơn hẳn, và ở quy mô này baseline TF-IDF+LogisticRegression đã vượt
Tier 3 rule-based cho M3 — đúng dự đoán lý thuyết, minh chứng cho lý do cần Tier 1.

---

## 10. Kho tri thức pháp luật thật cho RAG (`data/samples/legal_corpus/`) — bản redesign RAG+LLM

Theo đề cương mới (`de-cuong-hsmt-rag-cntt-phan-mem.pdf`), corpus RAG chuyển từ văn bản
**minh hoạ** (Mục 4) sang văn bản pháp luật **thật, nguyên văn**, lấy theo từng Điều bằng
`src/autotender/knowledge/legal_fetch.py` (Playwright, không dùng WebFetch để tránh tóm tắt
qua model nhỏ — xem docstring module này). Văn bản quy phạm pháp luật không thuộc đối tượng
bảo hộ quyền tác giả (Điều 15 Luật Sở hữu trí tuệ VN) nên lưu trữ nguyên văn là hợp lệ.

| Văn bản | File | Số Điều lấy được | Nguồn |
|---|---|---|---|
| Luật Đấu thầu 22/2023/QH15 (hợp nhất 57/2024, 90/2025) | `luat_22_2023_qh15.jsonl` | 90/93 (xem giới hạn dưới) | dauthau.gxd.vn |
| Nghị định 214/2025/NĐ-CP (thay NĐ 24/2024/NĐ-CP) | `nd_214_2025_ndcp.jsonl` | 145/146 | dauthau.gxd.vn |
| Thông tư 01/2024/TT-BKHĐT | `tt_01_2024_bkhdt.jsonl` | 22/~32 (xem Mục 10.3) | dauthau.gxd.vn (HTML) |
| Thông tư 22/2024/TT-BKHĐT | `tt_22_2024_bkhdt.jsonl` | 26/33 (xem Mục 10.3) | dauthau.gxd.vn (HTML) |
| Nghị định 45/2026/NĐ-CP (chuyên ngành CNTT) | `nd_45_2026_ndcp.jsonl` | 43/43 (xem Mục 10.4) | luatvietnam.vn (HTML, thay nguồn scan ảnh) |

**Phát hiện quan trọng — NĐ 24/2024/NĐ-CP đã hết hiệu lực:** lần fetch đầu tiên nhắm vào
Nghị định 24/2024/NĐ-CP (đúng như đề cương liệt kê), nhưng khi đối chiếu nội dung Điều 44
Luật (bản hợp nhất) đã đơn giản hoá đáng kể so với cấu trúc 7 thành phần cũ, tra cứu lại thì
NĐ 24/2024/NĐ-CP **đã được thay thế bởi Nghị định 214/2025/NĐ-CP (hiệu lực từ 04/8/2025)**.
Đã fetch lại đúng văn bản hiện hành, xoá file `nd_24_2024_ndcp.jsonl` cũ khỏi corpus. Đây là
lý do quan trọng phải luôn kiểm tra hiệu lực văn bản trước khi đưa vào RAG, không chỉ tin
theo tên văn bản nêu trong đề cương gốc.

**Giới hạn đã biết (không chặn tiến độ, ghi rõ để minh bạch):**
1. **Luật hợp nhất thiếu Điều 2, 5, 90-93**: Điều 90-93 bị ẩn/collapsed trên trang nguồn
   (chỉ còn ký tự phân cách "⋮", không có nội dung thật — `_is_meaningful_text()` phát hiện
   và loại bỏ đúng, không lưu bản ghi rỗng); Điều 2, 5 không xuất hiện trong khoảng
   start/end marker đã chọn (có thể đã bị bãi bỏ bởi luật sửa đổi — cần xác minh thêm nếu
   dùng cho báo cáo chính thức).
2. **NĐ 214/2025/NĐ-CP thiếu Điều 145 "Hiệu lực thi hành"**: heading này hoàn toàn không
   xuất hiện trong text hiển thị (`inner_text`) dù có xuất hiện dưới dạng link neo (anchor)
   trên trang — dấu hiệu của một phần tử DOM bị ẩn bởi CSS/JS (không phải lỗi parser). Đây là
   điều khoản về ngày hiệu lực, không phải nội dung nghiệp vụ cần cho việc soạn HSMT.
3. **Thông tư 01/2024/TT-BKHĐT và 22/2024/TT-BKHĐT — ĐÃ đưa vào corpus qua parser HTML
   riêng**: lần thử đầu (quét text phẳng qua `inner_text()`, giống Luật/Nghị định) thất
   bại vì trang có heading "Điều N." lặp/thiếu do lỗi biên soạn — xác minh trực tiếp qua
   HTML thô (không phải lỗi parser): (a) một số Điều có heading `<h3>` xuất hiện 2 lần
   liên tiếp, và ngay cả bản "sạch" hơn cũng lẫn nội dung Điều liền kề ở ranh giới; (b)
   một số Điều hoàn toàn không có heading `<h3>` trong nội dung thật (chỉ có trong
   sidebar điều hướng), khiến nội dung của nó (nếu có) tự động dồn vào Điều liền trước.
   Giải pháp: viết `parse_gxd_theme_articles()` (`knowledge/legal_fetch.py`) quét trực
   tiếp theo cấu trúc thẻ HTML (`<h2>`=Chương, `<h3>`=Điều, `<p>/<ul>/<ol>`=nội dung,
   heading thật lấy từ `a.header-anchor` để bỏ qua text nút "Copy đoạn/Ghi chú" — đây
   cũng là nguồn gốc thật của ký tự "⋮" từng thấy khi quét text phẳng, không phải lỗi
   trang), kèm 2 lớp bảo vệ an toàn: Điều có heading lặp bị LOẠI BỎ HOÀN TOÀN (không đủ
   căn cứ chọn đúng bản), Điều dài bất thường (>2000 từ, dấu hiệu trộn nội dung Điều
   thiếu heading liền sau) cũng bị loại — ưu tiên đúng nguyên tắc "không bịa đặt"/không
   trích dẫn sai hơn là đầy đủ. Kết quả: **22/32 Điều (TT 01/2024)** và **26/33 Điều (TT
   22/2024)** trích xuất sạch, đã kiểm tra thủ công không còn nội dung trộn lẫn. Các
   Điều còn thiếu là giới hạn nguồn có thật, không phải lỗi xử lý. Nội dung giá trị nhất
   của 2 thông tư (42+23 **mẫu hồ sơ** E-HSMT dạng Word/Excel) vẫn nằm ngoài phạm vi
   corpus RAG (không phù hợp pipeline trích dẫn theo Điều/Khoản), có thể xử lý riêng làm
   template tham khảo cho module sinh (M5) nếu còn thời gian.
4. **Nghị định 45/2026/NĐ-CP — ĐÃ lấy được qua nguồn thay thế, sau khi nguồn chính thức
   thất bại**: bản PDF chính thức duy nhất tại `datafiles.chinhphu.vn` (73 trang, mỗi trang
   1 ảnh TIFF) vẫn là **văn bản scan dạng ảnh**, `pypdf` trích xuất được 0 ký tự chữ — OCR
   nằm ngoài phạm vi đề cương nên không xử lý bản này. Tìm được bản **transcript dạng text**
   (server-render, không phải ảnh) của cùng văn bản tại luatvietnam.vn — đã kiểm tra
   `robots.txt` cho phép (không có disallow riêng cho bot, chỉ chặn vài path tìm kiếm nội
   bộ không liên quan). Viết parser riêng cho cấu trúc HTML của trang này
   (`parse_luatvietnam_articles`, `knowledge/legal_fetch.py`) — mỗi đơn vị nội dung nằm
   trong `div.mab2 > p`, nhận diện Chương/Điều bằng regex trên text (không dựa số thứ tự
   class CSS, không ổn định giữa các văn bản). Kết quả: **43/43 Điều** trích xuất sạch, đủ
   Chương I–VI, đã kiểm tra không lặp/thiếu heading. Căn cứ hợp lệ để lưu nguyên văn dù khác
   nguồn hiển thị: văn bản quy phạm pháp luật không thuộc đối tượng bảo hộ quyền tác giả
   (Điều 15 Luật Sở hữu trí tuệ VN) — nội dung pháp luật là như nhau bất kể trang nào biên
   tập lại, chỉ khác cách trình bày HTML.

**Kết luận:** corpus hiện tại gồm cả 5 văn bản (Luật + Nghị định 214/2025 + 2 Thông tư +
Nghị định 45/2026 chuyên ngành CNTT, 326 Điều thật, 684 chunk sau khi chia theo Khoản —
xem `scripts/build_legal_index.py`) đủ để xây dựng retrieval + compliance checker cho cả
phần lõi (Chương III năng lực-kinh nghiệm, nội dung HSMT), thủ tục đăng tải/nộp thầu qua
mạng (đề cương Mức 1-2), lẫn nội dung chuyên ngành CNTT (Mục 1.5 đề cương — quản lý đầu
tư, thiết kế, nghiệm thu, bảo hành phần mềm nội bộ) trước đây còn thiếu hoàn toàn.

---

## 11. Best-effort tải HSMT phần mềm thật đã duyệt — kết quả: bị chặn như dự kiến

Theo rủi ro đã liệt kê sẵn trong kế hoạch (Mục "Rủi ro cần lưu ý #1"), đã thử tải file
E-HSMT thật cho gói thầu phần mềm/CNTT, theo đúng quy trình:

1. **Lọc 489 bản ghi `real_dauthau_asia_sample.jsonl`** theo từ khoá phần mềm/CNTT
   (`phần mềm`, `CNTT`, `công nghệ thông tin`, `hệ thống thông tin`, `ứng dụng`, `máy chủ`...)
   trên `package_name`/`package_type` → **16 gói thầu phù hợp** (vd "Gói thầu TV08: Mua sắm
   trang thiết bị, phần mềm thương mại" — Cục CNTT Bộ Tư pháp; "Thuê hệ thống phần mềm quản
   lý tổng thể bệnh viện"...).
2. **Ghé trang chi tiết** (`MscHttpClient` httpx, dùng lại rate-limit/cache/robots.txt của
   crawler dauthau.asia Mục 9 — **không** dùng Playwright ở bước này vì Playwright độc lập
   bị Cloudflare chặn ngay khi thử, đúng như hành vi WAF đã ghi nhận với muasamcong ở Mục 8).
3. **Kết quả:** trang chi tiết hiển thị công khai **cấu trúc** hồ sơ (tên chương: "Chương I:
   Chỉ dẫn nhà thầu", "Chương II: Bảng dữ liệu", "Chương III: Tiêu chuẩn đánh giá HSDT"...,
   tên file đính kèm như "Yeu cau Chuong 3.pdf") nhưng **nút tải file yêu cầu đăng nhập**
   (`Đăng nhập`/`Đăng ký`), và ngay cả sau đăng nhập, trang nói rõ: tải trực tiếp trên Hệ
   thống Mua Sắm Công **chỉ chạy trên Windows + phần mềm Client Agent riêng** (không hỗ trợ
   Linux/macOS), hoặc cần tài khoản trả phí DauThau.info để bỏ qua bước này.

**Quyết định:** KHÔNG tạo tài khoản/đăng nhập để tải file (vi phạm nguyên tắc không tạo tài
khoản/nhập thông tin xác thực khi thu thập dữ liệu tự động). Việc này xác nhận đúng rủi ro đã
dự đoán trước — không phải lỗi kỹ thuật có thể sửa được trong phạm vi 15 ngày. Corpus RAG vẫn
chạy tốt chỉ với luật thật (Mục 10); phần "mẫu HSMT phần mềm thật" sẽ **không có** trong hệ
thống — nếu cần minh hoạ cấu trúc gói thầu phần mềm thật cho báo cáo, có thể trích riêng danh
sách 16 bản ghi này (tên gói, chủ đầu tư, cấu trúc chương — không phải nội dung file) làm ví
dụ định tính, không đưa vào corpus RAG dưới dạng trích dẫn có căn cứ.

---

## 12. Tập câu hỏi gán tay đánh giá retrieval (`data/eval/retrieval_queries.jsonl`)

**46 câu hỏi** biên soạn thủ công (không sinh tự động), mỗi câu gán nhãn `(law_id, dieu_so)`
— Điều đúng phải được truy xuất, dựa trực tiếp trên tiêu đề Điều thật trong corpus (Mục 10)
để đảm bảo nhãn đúng. Chỉ chọn các Điều có tiêu đề KHÔNG trùng lặp trong cùng văn bản (một số
Điều của Nghị định 214/2025 dùng lại tiêu đề "Lập hồ sơ mời thầu"/"Quy trình chi tiết" cho
từng loại gói thầu khác nhau — tránh dùng làm nhãn vàng vì nhiều đáp án đều "đúng", gây nhiễu
kết quả đo). Phủ đa dạng chủ đề: nguyên tắc chung, hình thức lựa chọn nhà thầu, nội dung/thời
gian HSMT, phương pháp đánh giá, hợp đồng, ưu đãi, xử lý vi phạm, và 8 câu bổ sung nhắm riêng
vào nội dung Nghị định 45/2026/NĐ-CP (thử nghiệm sản phẩm, báo cáo kinh tế-kỹ thuật, mô tả
yêu cầu kỹ thuật phần mềm nội bộ, bảo hành, nghiệm thu...) sau khi văn bản này được đưa vào
corpus (Mục 10.4) — đảm bảo phần mở rộng corpus thật sự được đánh giá, không chỉ nằm im.

Đây là bước thay thế đúng nghĩa cho proxy `bm25_proxy_recall_at_5` của bản cũ (Mục 5) —
`src/autotender/eval/retrieval_eval.py` cài đặt Recall@k, MRR, nDCG@k (`k ∈ {1,3,5,10}`),
tính trên "Điều đúng có nằm trong top-k hay không" (một Điều có thể bị chunk thành nhiều
mảnh theo Khoản — tính đúng nếu BẤT KỲ mảnh nào thuộc đúng Điều xuất hiện trong top-k).

**Kết quả chạy `scripts/run_retrieval_eval.py`** (model `vi_bi_encoder`, corpus 326
Điều/684 chunk, SAU 3 fix kỹ thuật ở Mục 12.1 bên dưới — xem `reports/retrieval_metrics.json`):

| Chế độ | Recall@5 | MRR | nDCG@5 | Thời gian (46 câu) |
|---|---|---|---|---|
| BM25 (sparse) | 0.565 | 0.385 | 0.426 | 0.7s |
| Dense (bi-encoder) | 0.696 | 0.546 | 0.580 | 13.4s |
| Hybrid RRF (dense+sparse) | 0.674 | 0.537 | 0.564 | 2.9s |
| Hybrid RRF + rerank (cross-encoder) | **0.761** | **0.587** | **0.627** | 351.2s |

**Nhận xét:** BM25 đơn lẻ vẫn yếu nhất (đúng như kỳ vọng — không hiểu ngữ nghĩa, chỉ khớp
từ); hybrid RRF vẫn thấp hơn dense-only trên MRR/nDCG@5 (BM25 đôi khi đẩy kết quả đúng ra
xa top-1 dù vẫn giữ trong top-5); rerank cross-encoder vẫn cải thiện nhiều nhất so với
không rerank nhưng **chi phí thời gian đã giảm 61%** so với lần đo trước (908s→351s cho 46
câu) sau khi sửa lỗi tải lại model mỗi lượt gọi (Mục 12.1).

### 12.1. Ba lỗi kỹ thuật tìm thấy khi rà soát kiến trúc RAG (nâng cấp theo yêu cầu)

Rà soát trực tiếp code (không chỉ đọc tài liệu) theo 6 hạng mục kỹ thuật RAG phát hiện 3
vấn đề thật, đã sửa và đo lại tác động:

1. **`CrossEncoder` bị tải lại mỗi lượt gọi rerank** (`rag/rerank.py`) — đo thực tế: tải
   model tốn ~6-7s, trong khi inference thật trên 50-80 ứng viên chỉ ~0.2-0.4s với văn bản
   ngắn (nhưng ~7-9s với văn bản luật dài thật — xem điểm 2). Vì `HybridLegalRetriever`
   luôn dùng ĐÚNG 1 tên model xuyên suốt vòng đời tiến trình, thêm cache theo `model_name`
   (cùng nguyên tắc lazy-cache đã áp dụng cho bi-encoder) loại bỏ hoàn toàn chi phí tải lại
   lặp lại. Tác động đo được: tổng thời gian rerank cho 46 câu giảm từ 908s xuống 391s
   (giảm 57%) TRƯỚC KHI áp dụng thêm điểm 2 bên dưới.
2. **`retrieve_reranked` đưa nhiều hơn `candidate_k` ứng viên vào cross-encoder** —
   `_fuse_rrf` trả về HỢP của dense+sparse (có thể lên tới 2×candidate_k nếu 2 nhánh không
   trùng ứng viên), nhưng code cũ không cắt về đúng `candidate_k` trước khi rerank. Đo thực
   tế trên corpus 684 chunk: 82 ứng viên thay vì 50 (tăng 64% khối lượng tính toán không
   cần thiết). Đã sửa bằng cách cắt `[:candidate_k]` trước khi gọi cross-encoder.
3. **65% chunk kho tri thức (447/684) bị CẮT ÂM THẦM khi embed** — phát hiện quan trọng
   nhất: `vietnamese-bi-encoder` có `max_position_embeddings=258` (giới hạn KIẾN TRÚC
   RoBERTa/PhoBERT nền tảng, không thể tăng qua cấu hình), nhưng chunk trong kho tri thức
   trung bình dài 310 token, tối đa 568 token — `SentenceTransformer.encode()` mặc định
   cắt phần vượt quá mà KHÔNG báo lỗi rõ ràng, nghĩa là embedding của phần lớn chunk chỉ
   phản ánh ĐOẠN ĐẦU, không phải toàn bộ nội dung Điều/Khoản. Đã viết
   `rag/embedding_models.py::encode_texts` — cắt văn bản dài thành các cửa sổ chồng lấn
   dựa trên tokenizer thật của model, embed từng cửa sổ, mean-pool + chuẩn hoá L2 lại (kỹ
   thuật chuẩn cho "long document embedding"), áp dụng nhất quán ở cả lúc build index
   (`scripts/build_legal_index.py`), phân tích embedding (`scripts/analyze_embeddings.py`)
   lẫn lúc encode câu truy vấn (`HybridLegalRetriever.retrieve_dense`).

   **Tác động đo được (số liệu KHÔNG bị pha trộn với 2 fix trên, vì chỉ ảnh hưởng bước
   embed, không ảnh hưởng bước rerank):**
   - Độ tách biệt không gian embedding (Mục 13 MODEL_CARD.md) tăng rõ rệt: `vi_bi_encoder`
     0.1595→**0.1836** (+15%), `multilingual_minilm` 0.1477→**0.1674** (+13%) — bằng chứng
     độc lập (không phụ thuộc tập câu hỏi gán tay) rằng embedding giờ phản ánh đúng hơn nội
     dung đầy đủ của từng Điều.
   - Dense-only và hybrid RRF (trước rerank) cải thiện nhẹ MRR/nDCG@5 (dense MRR
     0.543→0.546, hybrid RRF MRR 0.507→**0.537**, nDCG@5 0.541→**0.564**) trong khi Recall@5
     giữ nguyên.
   - **Kết quả SAU rerank lại giảm nhẹ** (Recall@5 0.804→0.761, MRR 0.611→0.587) — trái với
     kỳ vọng ban đầu. Giả thuyết hợp lý nhất: embedding chính xác hơn làm THAY ĐỔI tập ứng
     viên top-50 đưa vào cross-encoder cho một số câu hỏi (không còn giống hệt tập ứng viên
     cũ), và cross-encoder — vốn được đánh giá độc lập, không được tinh chỉnh lại cho tập
     ứng viên mới — có thể nhạy với thành phần tập ứng viên theo cách không đơn điệu. Với
     cỡ mẫu chỉ 46 câu, một vài câu đổi kết quả đúng/sai có thể xoay chuyển vài điểm % —
     chưa đủ để kết luận đây là xu hướng thật hay nhiễu thống kê. **Quyết định giữ fix**:
     việc cắt âm thầm 65% chunk là lỗi đúng-sai rõ ràng (không phải lựa chọn thiết kế có
     thể tranh luận), nên vẫn ưu tiên sửa dù số liệu rerank tổng hợp trên tập 46 câu chưa
     cho thấy lợi ích rõ — cần tập câu hỏi lớn hơn để đánh giá lại tác động lên rerank một
     cách đáng tin cậy hơn, ghi vào hướng phát triển (Mục "Giới hạn").

---

## 13. Faithfulness (LLM-as-judge) + bảng ablation LLM-only vs RAG

`src/autotender/eval/faithfulness_eval.py` chấm 2 chiều — **faithfulness** (mọi khẳng định
trong văn bản sinh ra có được trích đoạn căn cứ hỗ trợ không) và **completeness** (có tận
dụng đầy đủ thông tin sẵn có trong trích đoạn không) — bằng cách gọi Claude API làm giám
khảo, chấm theo rubric cố định, trả JSON có cấu trúc (điểm số + danh sách khẳng định không
có căn cứ + giải thích).

**Giới hạn phương pháp luận cần nêu rõ (đúng như đề cương Mục 7 gợi ý xác nhận lại với
giảng viên):** giám khảo và mô hình sinh CÙNG là Claude — có thể có thiên lệch tự ưu ái nhẹ
(self-preference bias), hiện tượng đã ghi nhận trong literature LLM-as-judge. Không có ngân
sách dùng judge khác họ (GPT-4, Gemini...) trong phạm vi 15 ngày.

`scripts/run_ablation_table.py` gộp 2 phần:
- **Phần A (Retrieval)** — đọc lại `reports/retrieval_metrics.json` (Mục 12): dense vs BM25
  vs hybrid vs hybrid+rerank.
- **Phần B (Generation — LLM-only vs RAG)** — với mỗi câu hỏi trong tập eval (Mục 12), gọi
  Claude 2 lần: KHÔNG kèm trích dẫn (LLM-only, chỉ dựa kiến thức đã huấn luyện sẵn) và CÓ
  kèm trích dẫn thật (RAG); cả 2 câu trả lời đều được chấm bằng CÙNG trích dẫn thật làm căn
  cứ đối chiếu — đo được LLM-only "đoán đúng nhờ kiến thức nền" hay "bịa" nhiều đến đâu so
  với khi có RAG.

**Kết quả chạy thật** (`python scripts/run_ablation_table.py --n-questions 8`, sau khi có
`ANTHROPIC_API_KEY`; xem `reports/ablation_table.json`):

| Điều kiện | Faithfulness (TB) | Completeness (TB) | Số câu chấm được |
|---|---|---|---|
| LLM-only (không RAG) | 0.41 | 0.44 | 6/8 |
| **RAG (có trích dẫn thật)** | **0.94** | **0.87** | 8/8 |

**Nhận xét:** RAG cải thiện faithfulness gấp hơn 2 lần (0.41 → 0.94) và completeness gần
gấp đôi (0.44 → 0.87) so với gọi thẳng LLM không có ngữ cảnh — đúng như giả thuyết chính
của toàn bộ hướng redesign: LLM có sẵn (Claude), dù có kiến thức nền rộng, KHÔNG đủ tin cậy
để trả lời câu hỏi pháp lý cụ thể (dễ nhầm lẫn giữa các quy định cũ/mới, bịa chi tiết nghe
hợp lý) nếu không có trích dẫn văn bản thật làm căn cứ — minh chứng định lượng cho đúng lỗi
tôi tự phát hiện ở Mục 10 khi ban đầu suýt dùng nhầm Nghị định 24/2024 đã hết hiệu lực.

**Vấn đề kỹ thuật phát hiện khi chạy live (đã sửa, xem `generation/claude_client.py`):**
1. Model `claude-sonnet-5` từ chối tham số `temperature` (lỗi 400) — sửa bằng cách không
   gửi tham số này nếu không cần override tường minh.
2. `claude-sonnet-5` bật **extended thinking** mặc định — thinking token tính vào
   `max_tokens`, có lúc tiêu hết ngân sách trước khi sinh ra text (response chỉ có block
   `thinking`, lỗi "không có nội dung text" khó hiểu nếu không biết nguyên nhân) — sửa bằng
   cách tắt tường minh (`thinking={"type": "disabled"}`) cho các tác vụ RAG-grounded này
   (không cần suy luận nhiều bước lộ ra ngoài).
3. `verify_numeric_consistency` (bộ dò số liệu R4 trong `models/generator.py`) được thiết
   kế cho luồng template-filling cũ (chèn nguyên văn trích dẫn nên xoá bằng string replace)
   — khi Claude (Tier 1) diễn giải và trích dẫn nội tuyến kiểu "(Điều 26 Nghị định
   214/2025/NĐ-CP)", các số Điều/Khoản/năm ban hành bị gắn cờ R4 oan uổng (quan sát thực tế:
   20+ cờ giả mỗi mục, hầu hết là số trích dẫn). Sửa bằng cách loại các cụm trích dẫn nội
   tuyến trước khi kiểm tra số liệu, cùng với việc bỏ qua số thứ tự mục kiểu "1.1"/"2.3".
4. `max_tokens=1536` (mục soạn) và `512` (LLM-only/RAG generation trong ablation) từng cắt
   cụt câu trả lời giữa chừng ở các mục dài — tăng lên `4096`/`1024` tương ứng.

Các vấn đề này chỉ lộ ra khi chạy THẬT với API key có credit — một minh chứng cụ thể cho
giá trị của việc kiểm thử end-to-end thay vì chỉ tin vào test đã mock.
