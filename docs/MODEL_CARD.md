# Model Card — AutoTender-VN

Mỗi module ML dùng cơ chế fallback 3 tầng (`autotender/models/base.py::BaseModule`,
xem Mục 2.1 SPEC). **Trạng thái hiện tại: cả 5 module đang chạy ở Tier 3 (rule-based)**
vì môi trường phát triển đồ án (laptop cá nhân, không GPU, không truy cập Google Colab)
không cho phép chạy `notebooks/01-05` để tạo checkpoint Tier 1 thật trong phạm vi 7 ngày.
Đây là hạn chế đã được lường trước và xử lý đúng theo Degraded Mode — phần mềm vẫn chạy
được và demo được đầy đủ luồng end-to-end.

Số liệu dưới đây trích từ lần chạy `scripts/evaluate.py` gần nhất — xem `reports/metrics.json`
để có số mới nhất và `docs/DATA_CARD.md` mục 5-6 để hiểu giới hạn của cách đánh giá.

---

## M2 — NER trích xuất trường (`models/ner.py`)

| Tier | Kiến trúc | Trạng thái |
|---|---|---|
| 1 | PhoBERT-base-v2 + token classification head | Chưa có checkpoint (`models/ner_phobert/` trống) |
| 2 | XLM-R zero-shot (QA-style prompting) | Chưa cài `transformers` trong môi trường demo |
| 3 | Regex + từ điển từ khoá | **Đang chạy** |

**Metric (Tier 3, distant supervision, 232 bản ghi = 20 tổng hợp + 12 + 200 THẬT — xem
DATA_CARD.md mục 8, 9):** entity-F1 = 0.965 (micro), per-entity: CONTRACT_TYPE 1.00, DURATION
1.00, FUNDING 0.949, INVESTOR 0.983, METHOD 0.971, PACKAGE_NAME 0.909, VALUE 1.00.

Đây là số liệu **có ý nghĩa hơn** kết quả F1=1.0 trước đó chạy trên riêng 20 mẫu tổng hợp
(vốn phản ánh tính nhất quán nội tại do nhãn và regex cùng dựa trên khớp chuỗi — xem DATA_CARD.md
mục 5). Đánh giá lại trên 232 bản ghi (212 bản ghi thật) từng phát hiện một lỗi thật: regex
DURATION khớp nhầm chính placeholder `[CẦN NGƯỜI DÙNG BỔ SUNG...]` khi trường `execution_time`
là `None` (212/232 bản ghi thật không có trường này) — precision khi đó chỉ 0.138. Đã sửa
trong `models/ner.py` (bỏ qua match trùng placeholder) — một minh chứng cụ thể cho giá trị của
việc đánh giá trên dữ liệu thật thay vì chỉ dữ liệu tổng hợp tự nhất quán.

**Việc cần làm để có Tier 1 thật:** chạy `notebooks/01_train_ner.ipynb` trên Colab với
`data/processed/ner_dataset.jsonl` (đã tự động dùng 232 bản ghi kết hợp) + 200 mẫu gán tay
làm test set độc lập.

---

## M3 — Phân loại gói thầu (`models/classifier.py`)

| Tier | Kiến trúc | Trạng thái |
|---|---|---|
| 1 | PhoBERT + classification head | Chưa có checkpoint |
| 2 | XLM-R zero-shot classification | Chưa cài `transformers` |
| 3 | Keyword matching | **Đang chạy** |

**Metric (232 bản ghi, 212 bản ghi THẬT — nhãn `package_type` lấy trực tiếp từ trường
"Lĩnh vực MSC" trên trang chi tiết dauthau.asia, không phải suy đoán):**
- Tier 3 (keyword): macro-F1 = 0.531
- Baseline TF-IDF + LogisticRegression (3 seed): macro-F1 = 0.465 ± 0.031

**Nhận xét:** trên tập dữ liệu thật đa dạng hơn nhiều (5 lớp, phân bố thật: xây lắp 83,
hàng hóa 68, phi tư vấn 25, hỗn hợp 15, tư vấn 9), cả 2 phương pháp đều thấp hơn hẳn số liệu
cũ đo trên 20 mẫu tổng hợp (0.610/0.394) — con số trước phản ánh dữ liệu quá nhỏ/dễ, con số
này trung thực hơn và cho thấy rõ **cần Tier 1 (fine-tune) thật sự** để vượt qua giới hạn của
keyword matching trên văn bản thật đa dạng.

---

## M4 — Retrieval / RAG (`models/retriever.py`, `rag/`)

| Tier | Kiến trúc | Trạng thái |
|---|---|---|
| 1 | Bi-encoder fine-tuned + FAISS + cross-encoder rerank | Chưa có checkpoint |
| 2 | `bkai-foundation-models/vietnamese-bi-encoder` zero-shot + FAISS | Chưa cài `sentence-transformers`/`faiss-cpu` |
| 3 | BM25 thuần Python (`rag/bm25.py`) | **Đang chạy** |

**Metric:** BM25 proxy Recall@5 = 1.0/8 truy vấn mẫu (xem giới hạn phương pháp proxy ở
DATA_CARD.md mục 5). Baseline chính thức theo Mục 10 SPEC (Recall@5/MRR@10/nDCG@10 so
BM25 vs bi-encoder fine-tuned) cần chạy `notebooks/03_train_retriever.ipynb` trên Colab.

**Corpus:** 13 chunk từ 3 file mẫu minh hoạ (`data/samples/corpus/`) — xem DATA_CARD.md
mục 4 về giới hạn dữ liệu tổng hợp.

---

## M5 — Sinh dự thảo (`models/generator.py`)

| Tier | Kiến trúc | Trạng thái |
|---|---|---|
| 1 | VietAI/vit5-base fine-tune | Chưa có checkpoint |
| 2 | LLM API ngoài (OpenAI-compatible), chỉ gọi nếu có `AUTOTENDER_LLM_API_KEY` | Không cấu hình trong môi trường demo (mặc định tắt để tránh phát sinh chi phí/rò rỉ dữ liệu ngoài ý muốn) |
| 3 | Template filling từ corpus mẫu + slot-filling số liệu | **Đang chạy** |

**Verifier số liệu (`verify_numeric_consistency`):** so khớp mọi con số trong phần văn bản
KHÔNG PHẢI trích dẫn nguyên văn (loại trừ số liệu đã có citation) với `ExtractedField` từ
KHLCNT — lệch thì gắn cờ `R4`. Đây là cơ chế bắt buộc theo Mục 2.2 SPEC, hoạt động độc lập
với tier đang chạy (áp dụng cho cả Tier 1/2 khi có).

**Ablation (bỏ retrieval):** số citation giảm từ 5 → 0, nội dung sinh ra chỉ còn phần
slot-filling, mất căn cứ tham chiếu — minh hoạ vai trò của M4 trong pipeline.

**Việc cần làm để có Tier 1 thật:** chạy `notebooks/04_train_generator.ipynb`, đo
ROUGE-L/BERTScore so với baseline template filling.

---

## M6 — Compliance Guard (`models/compliance.py`) — module trọng tâm

| Tier | Kiến trúc | Trạng thái |
|---|---|---|
| 1 | Cross-encoder XLM-R fine-tuned, 5 lớp | Chưa có checkpoint |
| 2 | XLM-R zero-shot classification | Chưa cài `transformers` |
| 3 | Từ điển nhãn hiệu (17 hãng) + regex ngưỡng bất hợp lý + phát hiện phủ định | **Đang chạy** |

**Metric (10 câu gán tay, xem giới hạn DATA_CARD.md mục 6):** precision = recall = F1 =
1.0 trên cả 4 lớp (OK, R1, R2, R3). R4 (số liệu sai lệch KHLCNT) được xử lý riêng trong
M5 verifier, không thuộc phạm vi test này.

**Đặc điểm kỹ thuật đáng chú ý:** Tier 3 có xử lý phủ định đơn giản (ví dụ câu "Không được
đưa ra thông số... duy nhất trên thị trường" mô tả NGUYÊN TẮC cấm, không bị gắn cờ nhầm
thành vi phạm) — phát hiện và sửa trong quá trình phát triển khi module tự gắn cờ chính
văn bản mô tả nguyên tắc mà nó tham chiếu.

**Ablation (bỏ M6):** số cờ phát hiện được trên tập test giảm từ 6 → 0 — toàn bộ vi phạm
tiềm ẩn (nhãn hiệu, doanh thu bất hợp lý, thông số may đo) sẽ không được cảnh báo cho
người dùng nếu thiếu module này.

**Việc cần làm để có Tier 1 thật:** cần dữ liệu gán nhãn quy mô lớn hơn (xem SPEC Mục 14:
"sinh dữ liệu tổng hợp bằng LLM rồi người kiểm duyệt"), chạy `notebooks/05_train_compliance.ipynb`
(chưa tạo trong đồ án này — ghi vào hạn chế/future work), dùng focal loss do dữ liệu mất
cân bằng nặng giữa lớp OK và các lớp vi phạm.

---

## Ablation tổng hợp (Mục 10 SPEC)

| Ablation | Kết quả |
|---|---|
| (a) Bỏ retrieval | Citation: 5 → 0 |
| (b) Bỏ hard negative (huấn luyện bi-encoder) | N/A — cần Colab, chưa thực hiện |
| (c) Bỏ M6 | Cờ phát hiện: 6 → 0 |
| (d) PhoBERT vs XLM-R | N/A — cần Colab, chưa thực hiện |

---

## So sánh với việc gọi thẳng ChatGPT/Claude (câu hỏi bảo vệ #2, Mục 15 SPEC)

Bảng trên cho thấy: dù mọi module đang ở Tier 3 (chưa có mô hình fine-tune thật), hệ
thống đã hoạt động end-to-end với kiến trúc sẵn sàng nâng cấp lên Tier 1 (mô hình nhỏ,
chạy offline, chi phí ~0, giải thích được qua attention/rule trace) mà không cần đổi
kiến trúc phần mềm. Đây là luận điểm chính khi bảo vệ: giá trị đóng góp nằm ở **kiến trúc
hệ thống và cơ chế fallback/HITL/compliance**, không phải ở việc gọi một API LLM có sẵn.
