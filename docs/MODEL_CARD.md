# Model Card — AutoTender-VN

M5 (Generator)/QA (Mức 1) dùng cơ chế fallback 3 tầng (`autotender/models/base.py::BaseModule`,
xem Mục 2.1 SPEC), Tier 1 = Claude API. M2 (NER) và M6 (Compliance) từng dùng khung tương tự
(Tier 1 = checkpoint fine-tuned, Tier 2 = zero-shot pretrained) nhưng cả 2 tầng đó **chưa
từng chạy thật** — môi trường phát triển đồ án (laptop cá nhân, không GPU, không truy cập
Google Colab) không cho phép tạo checkpoint Tier 1 thật, và Tier 2 luôn rơi về rule-based —
nên khung 3-tier của 2 module này đã được đơn giản hoá, giữ lại đúng logic rule-based (Tier
3 cũ) làm hành vi trực tiếp, duy nhất. Đây là hạn chế đã được lường trước và xử lý đúng theo
Degraded Mode — phần mềm vẫn chạy được và demo được đầy đủ luồng end-to-end.

Số liệu dưới đây là **ảnh chụp cố định** từ lần chạy cuối của `scripts/evaluate.py` — script
này (cùng `models/classifier.py`, `models/retriever.py`, `scripts/build_index.py` và 2
notebook train tương ứng) đã bị gỡ khỏi repo khi dọn dẹp code sau khi chuyển sang bản
redesign RAG+LLM (M3 Classifier và M4 BM25-proxy bên dưới không còn được dùng/tái tạo được
— xem Mục "M3-M4 (lịch sử" bên dưới và `docs/DATA_CARD.md` mục 5-6 để hiểu giới hạn của
cách đánh giá gốc).

---

## M2 — NER trích xuất trường (`models/ner.py`)

Rule-based thuần (regex + từ điển từ khoá), luôn chạy trực tiếp. Module này từng có khung
`BaseModule` 3-tier (Tier 1 = PhoBERT-base-v2 fine-tune, Tier 2 = XLM-R zero-shot QA-style
prompting) nhưng cả 2 tầng đó **chưa từng chạy thật** trong đồ án (không có checkpoint đã
train — môi trường không GPU/Colab; Tier 2 còn có lỗi runtime dùng sai tên task pipeline
của `transformers`) nên khung đó đã được gỡ bỏ khỏi code (giữ nguyên số liệu Tier 3 thật
bên dưới). Muốn có mô hình fine-tune thật, xem `notebooks/01_train_ner.ipynb` (huấn luyện
độc lập, chưa nối lại vào `models/ner.py`).

**Metric (rule-based, distant supervision, 521 bản ghi = 20 tổng hợp + 12 + 489 THẬT — xem
DATA_CARD.md mục 8, 9):** entity-F1 = 0.956 (micro), per-entity: CONTRACT_TYPE 1.00, DURATION
0.985, FUNDING 0.936, INVESTOR 0.983, METHOD 0.958, PACKAGE_NAME 0.899, VALUE 1.00.

Đây là số liệu **có ý nghĩa hơn** kết quả F1=1.0 trước đó chạy trên riêng 20 mẫu tổng hợp
(vốn phản ánh tính nhất quán nội tại do nhãn và regex cùng dựa trên khớp chuỗi — xem DATA_CARD.md
mục 5). Đánh giá lại trên 232 bản ghi (212 bản ghi thật) từng phát hiện một lỗi thật: regex
DURATION khớp nhầm chính placeholder `[CẦN NGƯỜI DÙNG BỔ SUNG...]` khi trường `execution_time`
là `None` (212/232 bản ghi thật không có trường này) — precision khi đó chỉ 0.138. Đã sửa
trong `models/ner.py` (bỏ qua match trùng placeholder) — một minh chứng cụ thể cho giá trị của
việc đánh giá trên dữ liệu thật thay vì chỉ dữ liệu tổng hợp tự nhất quán.

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
  proxy). Thay thế bởi `rag/hybrid_retriever.py` (dense+BM25 hybrid, corpus luật thật 326
  Điều/684 chunk) — xem `docs/DATA_CARD.md` mục 12 cho số liệu Recall@k/MRR/nDCG@k thật
  trên tập 46 câu hỏi gán tay.

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

**2 nâng cấp thêm cho Tier 1 (Claude), sau khi rà soát đối chiếu với một báo cáo kiến trúc
RAG nâng cao tham khảo — xem `docs/DATA_CARD.md` Mục 15 để biết ý tưởng nào bị loại và vì
sao (đa số phần còn lại của báo cáo đó là template chung/số liệu không kiểm chứng được,
không đáng áp dụng nguyên trạng):**
1. **Chặn tiêu chí hạn chế cạnh tranh NGAY TỪ PROMPT sinh** (`_SYSTEM_PROMPT`), không chỉ
   dựa vào M6 rà soát SAU khi đã sinh — phòng thủ 2 lớp: cấm nêu nhãn hiệu/xuất xứ cụ thể,
   cấm yêu cầu năng lực/doanh thu vượt ngưỡng hợp lý, cấm mô tả thông số "may đo" ngay trong
   chỉ dẫn hệ thống gửi cho Claude.
2. **Gửi TRỌN Điều (không chỉ đoạn Khoản đã khớp truy hồi) làm ngữ cảnh cho Claude** —
   `HybridLegalRetriever.expand_to_parent_article`. Nhiều Điều dài bị chunk theo Khoản
   (487/684 chunk trong kho tri thức hiện tại, xem `rag/chunker.py`); trước đây Claude chỉ
   thấy đúng khoản đã khớp, có thể thiếu ngữ cảnh của cả Điều. Chỉ áp cho NGỮ CẢNH gửi LLM —
   không đổi hành vi retrieval/rerank/citation hiển thị UI hay số liệu Recall@k/MRR/nDCG đã
   đo (Mục 12 DATA_CARD.md).

---

## M6 — Compliance Guard (`models/compliance.py`) — module trọng tâm

Rule-based thuần: từ điển nhãn hiệu (17 hãng) + regex ngưỡng bất hợp lý + phát hiện phủ
định, luôn chạy trực tiếp. Module này từng có khung `BaseModule` 3-tier (Tier 1 =
cross-encoder XLM-R fine-tuned 5 lớp, Tier 2 = XLM-R zero-shot classification) nhưng cả 2
tầng đó **chưa từng chạy thật** trong đồ án (không có checkpoint đã train) nên khung đó đã
được gỡ bỏ khỏi code (giữ nguyên số liệu Tier 3 thật bên dưới).

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

**Muốn có mô hình fine-tune thật (không thuộc phạm vi đồ án):** cần dữ liệu gán nhãn quy mô
lớn hơn (xem SPEC Mục 14: "sinh dữ liệu tổng hợp bằng LLM rồi người kiểm duyệt"), notebook
train tương ứng (`notebooks/05_train_compliance.ipynb`) chưa từng được tạo, dùng focal loss
do dữ liệu mất cân bằng nặng giữa lớp OK và các lớp vi phạm.

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
redesign RAG+LLM nằm ở `docs/DATA_CARD.md` mục 13 (đối chiếu thuật ngữ với RAGAS ở mục 14).

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
`scripts/analyze_embeddings.py` trên 684 chunk của kho tri thức thật (`data/samples/legal_corpus/` —
Luật + Nghị định 214/2025 + 2 Thông tư + Nghị định 45/2026, xem `docs/DATA_CARD.md` Mục 10),
so sánh 3 model embedding đã đăng ký (`rag/embedding_models.py`). Số liệu dưới đây đo SAU khi
sửa lỗi mã hoá phát hiện khi rà soát kiến trúc RAG (`encode_texts`, xem `docs/DATA_CARD.md`
Mục 12.1, điểm 3): trước đó `SentenceTransformer.encode()` cắt âm thầm 65% chunk (447/684)
vượt quá `max_seq_length` của model, khiến embedding chỉ phản ánh đoạn đầu văn bản.

| Model | Kiến trúc/dữ liệu train | Chiều | intra-Điều (TB) | inter-Điều (TB) | Độ tách biệt |
|---|---|---|---|---|---|
| `bkai-foundation-models/vietnamese-bi-encoder` | SimCSE fine-tune trên PhoBERT/XLM-R, dữ liệu tiếng Việt | 768 | 0.5320 | 0.3483 | **0.1836** |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Đa ngôn ngữ (50+ ngôn ngữ), MiniLM distill | 384 | 0.7480 | 0.5806 | 0.1674 |
| `BAAI/bge-m3` | Đa ngôn ngữ, đa hạt (dense/sparse/ColBERT), ngữ cảnh dài (8192 token) | 1024 | 0.7467 | 0.6312 | 0.1155 |

**"Độ tách biệt"** = cosine similarity trung bình giữa các chunk CÙNG một Điều (intra) trừ
đi similarity trung bình giữa các chunk KHÁC Điều (inter) — không gian biểu diễn "tốt cho
retrieval" phải có độ tách biệt cao (các đoạn cùng chủ đề pháp lý gần nhau hơn hẳn các đoạn
khác chủ đề). **Kết quả:** dù model đa ngôn ngữ có similarity tuyệt đối cao hơn hẳn ở cả 2
nhóm (0.75/0.58 so với 0.53/0.35 — dấu hiệu embedding "co cụm" hơn, kém phân biệt hơn theo
nghĩa tuyệt đối), model tiếng Việt chuyên biệt lại có **độ tách biệt tương đối cao hơn**
(0.1836 > 0.1674) — tức phân biệt tốt hơn giữa các chủ đề pháp lý khác nhau dù giá trị
similarity tuyệt đối thấp hơn. So với số liệu đo TRƯỚC khi sửa lỗi cắt âm thầm (0.1595 và
0.1477 — xem lịch sử git), độ tách biệt của cả 2 model đều tăng rõ rệt (+15% và +13%) sau
khi embedding phản ánh đúng TOÀN BỘ nội dung chunk thay vì chỉ đoạn đầu — bằng chứng độc lập
(không phụ thuộc tập câu hỏi gán tay) cho thấy việc sửa lỗi mã hoá thực sự nâng chất lượng
không gian biểu diễn, không chỉ là thay đổi trung tính. Kết quả này **khớp một phần** với
bảng Recall@k/MRR/nDCG đo trực tiếp qua truy vấn thật (`docs/DATA_CARD.md` Mục 12: dense-only
MRR/nDCG@5 cải thiện nhẹ sau fix) — nhưng chỉ số SAU rerank lại giảm nhẹ, một phát hiện nuance
được thảo luận đầy đủ ở Mục 12.1 DATA_CARD.md thay vì bị bỏ qua.

**`bge-m3` — thêm sau, kết quả TRÁI VỚI KỲ VỌNG kiến trúc:** model này hỗ trợ ngữ cảnh tới
8192 token (so với 256/128 của 2 model trên), nên xử lý TRỌN VẸN mọi chunk trong MỘT lượt
(không cần sliding-window của `encode_texts`) — về lý thuyết nên loại bỏ hoàn toàn rủi ro cắt
âm thầm từng là nguyên nhân gốc của bug đã sửa ở trên. Nhưng độ tách biệt đo được
(**0.1155**) lại THẤP HƠN CẢ 2 model nhỏ hơn — giống `multilingual_minilm`, `bge-m3` cho
similarity tuyệt đối cao ở cả intra/inter (0.75/0.63, "co cụm"), nhưng khoảng cách tương đối
giữa 2 nhóm còn hẹp hơn cả `multilingual_minilm`. Diễn giải hợp lý nhất: `bge-m3` là model
tổng quát đa miền (không fine-tune riêng cho tiếng Việt hay văn bản pháp lý), nên dù giải
quyết được vấn đề kiến trúc (ngữ cảnh dài), không gian biểu diễn kết quả lại kém phân biệt
theo chủ đề pháp lý cụ thể hơn model chuyên biệt nhỏ hơn (`vi_bi_encoder`, chỉ 768 chiều).
**Kết luận: không đổi model embedding mặc định.** Đây là minh chứng thứ 3 trong đồ án (cùng
Mục 12.2/12.3 DATA_CARD.md) cho nguyên tắc đo thật trước khi kết luận: "giải quyết đúng vấn đề
kiến trúc đã biết" không tự động đồng nghĩa "cải thiện chất lượng thật" — cần đo trực tiếp
trên corpus/tác vụ cụ thể thay vì suy luận từ thông số kỹ thuật của model. (Không chạy lại
Recall@k/MRR/nDCG đầy đủ với `bge-m3` làm FAISS index chính — CPU-only, ~18 phút chỉ để embed
684 chunk một lần, ước tính vài giờ nếu build lại toàn bộ pipeline eval; độ tách biệt kém hơn
rõ rệt đã đủ căn cứ không ưu tiên bước đó trong phạm vi đồ án.)

**Trực quan hoá t-SNE/UMAP** (`reports/figures/embedding_{model}_{tsne,umap}.png`, tô màu
theo văn bản nguồn — Luật vs Nghị định): cả 2 phép chiếu đều cho thấy một khối trung tâm lớn
trộn lẫn giữa 2 màu (hợp lý — Nghị định là văn bản CHI TIẾT HOÁ của Luật, dùng chung nhiều
thuật ngữ/chủ đề) cùng một số cụm nhỏ tách biệt ở rìa — khớp với quan sát thủ công ở Mục 11
DATA_CARD.md rằng nhiều Điều của Nghị định dùng lại gần như nguyên văn cấu trúc/tiêu đề cho
các loại gói thầu khác nhau (hàng hóa/xây lắp/dịch vụ), tạo thành các nhóm văn bản gần giống
hệt nhau về mặt ngữ nghĩa.
