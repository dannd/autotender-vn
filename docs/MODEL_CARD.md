# Model Card — AutoTender-VN

Mỗi module ML dùng cơ chế fallback 3 tầng (`autotender/models/base.py::BaseModule`,
xem Mục 2.1 SPEC). **Trạng thái hiện tại: cả 5 module đang chạy ở Tier 3 (rule-based)**
vì môi trường phát triển đồ án (laptop cá nhân, không GPU, không truy cập Google Colab)
không cho phép chạy `notebooks/01-05` để tạo checkpoint Tier 1 thật trong phạm vi 7 ngày.
Đây là hạn chế đã được lường trước và xử lý đúng theo Degraded Mode — phần mềm vẫn chạy
được và demo được đầy đủ luồng end-to-end.

Số liệu dưới đây là **ảnh chụp cố định** từ lần chạy cuối của `scripts/evaluate.py` — script
này (cùng `models/classifier.py`, `models/retriever.py`, `scripts/build_index.py` và 2
notebook train tương ứng) đã bị gỡ khỏi repo khi dọn dẹp code sau khi chuyển sang bản
redesign RAG+LLM (M3 Classifier và M4 BM25-proxy bên dưới không còn được dùng/tái tạo được
— xem Mục "M3-M4 (lịch sử" bên dưới và `docs/DATA_CARD.md` mục 5-6 để hiểu giới hạn của
cách đánh giá gốc).

---

## M2 — NER trích xuất trường (`models/ner.py`)

| Tier | Kiến trúc | Trạng thái |
|---|---|---|
| 1 | PhoBERT-base-v2 + token classification head | Chưa có checkpoint (`models/ner_phobert/` trống) |
| 2 | XLM-R zero-shot (QA-style prompting) | Chưa cài `transformers` trong môi trường demo |
| 3 | Regex + từ điển từ khoá | **Đang chạy** |

**Metric (Tier 3, distant supervision, 521 bản ghi = 20 tổng hợp + 12 + 489 THẬT — xem
DATA_CARD.md mục 8, 9):** entity-F1 = 0.956 (micro), per-entity: CONTRACT_TYPE 1.00, DURATION
0.985, FUNDING 0.936, INVESTOR 0.983, METHOD 0.958, PACKAGE_NAME 0.899, VALUE 1.00.

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

## M3-M4 (lịch sử) — Phân loại gói thầu & Retrieval BM25-only của bản đồ án solo gốc

Bản đồ án solo 7 ngày gốc có 2 module đã bị **thay thế hoàn toàn** khi chuyển sang bản
redesign RAG+LLM (nhóm 4 người, 15 ngày) — code, test, config và notebook train tương ứng
đã được gỡ khỏi repo (`models/classifier.py`, `models/retriever.py`, `scripts/evaluate.py`,
`scripts/build_index.py`, `notebooks/02_train_classifier.ipynb`,
`notebooks/03_train_retriever.ipynb`). Giữ lại tóm tắt số liệu lịch sử để tham chiếu:

- **M3 — Phân loại gói thầu:** Tier 3 (keyword) macro-F1 = 0.500 trên 521 bản ghi (501
  thật); baseline TF-IDF+LogisticRegression = 0.554 ± 0.008 — phát hiện đáng chú ý: baseline
  thống kê vượt rule-based khi đủ dữ liệu (với 20 mẫu ban đầu, Tier 3 vẫn vượt baseline).
  Module này bị loại khỏi pipeline mới vì phạm vi hệ thống đã khoá cứng "phần mềm/CNTT"
  (đề cương mới), phân loại loại gói thầu không còn ảnh hưởng đến luồng sinh HSMT.
- **M4 — Retrieval (BM25-only):** BM25 proxy Recall@5 = 1.0/8 truy vấn mẫu trên corpus minh
  hoạ 13 chunk (`data/samples/corpus/`, xem DATA_CARD.md mục 5 về giới hạn phương pháp
  proxy). Thay thế bởi `rag/hybrid_retriever.py` (dense+BM25 hybrid, corpus luật thật 283
  Điều/587 chunk) — xem `docs/DATA_CARD.md` mục 12 cho số liệu Recall@k/MRR/nDCG@k thật
  trên tập 38 câu hỏi gán tay.

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

Bảng trên là ablation của bản đồ án solo 7 ngày gốc (Tier 3 rule-based). Bảng ablation
LLM-only vs RAG **thật** (đo trực tiếp bằng Claude API, faithfulness/completeness) của bản
redesign RAG+LLM nằm ở `docs/DATA_CARD.md` mục 13.

---

## So sánh với việc gọi thẳng ChatGPT/Claude (câu hỏi bảo vệ #2, Mục 15 SPEC)

Bảng trên cho thấy: dù mọi module đang ở Tier 3 (chưa có mô hình fine-tune thật), hệ
thống đã hoạt động end-to-end với kiến trúc sẵn sàng nâng cấp lên Tier 1 (mô hình nhỏ,
chạy offline, chi phí ~0, giải thích được qua attention/rule trace) mà không cần đổi
kiến trúc phần mềm. Đây là luận điểm chính khi bảo vệ: giá trị đóng góp nằm ở **kiến trúc
hệ thống và cơ chế fallback/HITL/compliance**, không phải ở việc gọi một API LLM có sẵn.

**Lưu ý (bản redesign RAG+LLM, xem Mục dưới):** phần này viết cho bản đồ án solo 7 ngày
gốc (đối tượng: tự train Tier 1). Đề cương mới (nhóm 4 người, 15 ngày) đổi hướng — dùng
thẳng LLM có sẵn (Claude API) làm đường sinh CHÍNH, không bắt buộc tự train — nên luận điểm
"giá trị nằm ở kiến trúc fallback, không phải gọi LLM" không còn là câu trả lời chính cho
bản mới. Câu trả lời tương ứng cho bản mới: giá trị nằm ở **kho tri thức pháp luật thật +
kiến trúc RAG (retrieval có kiểm chứng, trích dẫn bắt buộc, compliance guard R1-R5)** — khác
với việc hỏi thẳng ChatGPT/Claude không có RAG, vốn không có cơ chế nào đảm bảo câu trả lời
dựa trên đúng văn bản pháp luật hiện hành (rủi ro "bịa" điều khoản hoặc dùng luật đã hết hiệu
lực — xem đúng lỗi này tôi tự phát hiện ở Mục 10 DATA_CARD.md khi ban đầu định dùng nhầm
Nghị định 24/2024 đã hết hiệu lực). Bảng ablation LLM-only vs RAG (`docs/DATA_CARD.md` Mục
13) đo trực tiếp sự khác biệt này khi có `ANTHROPIC_API_KEY`.

---

## Phân tích Deep Learning — so sánh không gian embedding (Giai đoạn 3, đề cương RAG+LLM)

Theo đúng hướng đề cương mới: "Deep Learning" ở đây thể hiện qua **phân tích/đánh giá** các
mạng nơ-ron pretrained (embedding, cross-encoder), KHÔNG qua việc tự huấn luyện. Chạy
`scripts/analyze_embeddings.py` trên 587 chunk của kho tri thức thật (`data/samples/legal_corpus/` —
Luật + Nghị định 214/2025 + 2 Thông tư, xem `docs/DATA_CARD.md` Mục 10), so sánh 2 model
embedding đã đăng ký (`rag/embedding_models.py`):

| Model | Kiến trúc/dữ liệu train | Chiều | intra-Điều (TB) | inter-Điều (TB) | Độ tách biệt |
|---|---|---|---|---|---|
| `bkai-foundation-models/vietnamese-bi-encoder` | SimCSE fine-tune trên PhoBERT/XLM-R, dữ liệu tiếng Việt | 768 | 0.4794 | 0.3341 | **0.1453** |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Đa ngôn ngữ (50+ ngôn ngữ), MiniLM distill | 384 | 0.6608 | 0.5228 | 0.1380 |

**"Độ tách biệt"** = cosine similarity trung bình giữa các chunk CÙNG một Điều (intra) trừ
đi similarity trung bình giữa các chunk KHÁC Điều (inter) — không gian biểu diễn "tốt cho
retrieval" phải có độ tách biệt cao (các đoạn cùng chủ đề pháp lý gần nhau hơn hẳn các đoạn
khác chủ đề). **Kết quả:** dù model đa ngôn ngữ có similarity tuyệt đối cao hơn hẳn ở cả 2
nhóm (0.66/0.52 so với 0.48/0.33 — dấu hiệu embedding "co cụm" hơn, kém phân biệt hơn theo
nghĩa tuyệt đối), model tiếng Việt chuyên biệt lại có **độ tách biệt tương đối cao hơn**
(0.1453 > 0.1380) — tức phân biệt tốt hơn giữa các chủ đề pháp lý khác nhau dù giá trị
similarity tuyệt đối thấp hơn. Kết quả này **khớp** với bảng Recall@k/MRR/nDCG đo trực tiếp
qua truy vấn thật (`docs/DATA_CARD.md` Mục 12, chạy trên `vi_bi_encoder`: Recall@5=0.658
dense-only) — cả 2 phép đo (độc lập, một dựa trên cấu trúc không gian embedding, một dựa
trên truy vấn thật) đều ủng hộ model tiếng Việt chuyên biệt phù hợp hơn cho corpus pháp
luật tiếng Việt so với model đa ngôn ngữ tổng quát.

**Trực quan hoá t-SNE/UMAP** (`reports/figures/embedding_{model}_{tsne,umap}.png`, tô màu
theo văn bản nguồn — Luật vs Nghị định): cả 2 phép chiếu đều cho thấy một khối trung tâm lớn
trộn lẫn giữa 2 màu (hợp lý — Nghị định là văn bản CHI TIẾT HOÁ của Luật, dùng chung nhiều
thuật ngữ/chủ đề) cùng một số cụm nhỏ tách biệt ở rìa — khớp với quan sát thủ công ở Mục 11
DATA_CARD.md rằng nhiều Điều của Nghị định dùng lại gần như nguyên văn cấu trúc/tiêu đề cho
các loại gói thầu khác nhau (hàng hóa/xây lắp/dịch vụ), tạo thành các nhóm văn bản gần giống
hệt nhau về mặt ngữ nghĩa.
