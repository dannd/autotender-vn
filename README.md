# AutoTender-VN

Phần mềm hỗ trợ **soạn thảo Hồ sơ mời thầu (E-HSMT) cho gói thầu phần mềm/CNTT** tại
Việt Nam bằng **RAG + LLM có sẵn (Claude API)** — đồ án cuối môn Deep Learning, bậc
Thạc sĩ, thực hiện trong 15 ngày.

**Tác giả:** Nguyễn Đình Đán, Nguyễn Văn Vũ, Triệu Việt Hoa, Hoàng Xuân Sơn, Nguyễn Thái
Thịnh — Trường Kinh doanh FSB, Đại học FPT. Xem báo cáo đầy đủ tại
[`docs/AutoTender-VN_Report_IEEE.docx`](docs/AutoTender-VN_Report_IEEE.docx) và slide
trình bày tại [`docs/AutoTender-VN_Slides.pptx`](docs/AutoTender-VN_Slides.pptx).

> **Lịch sử dự án:** bản đầu là đồ án solo 7 ngày với kiến trúc đa module tự train
> (NER/Classifier/Generator fine-tune). Đề cương môn học sau đó đổi hướng sang phạm vi
> hẹp hơn (chỉ gói phần mềm/CNTT) và cách tiếp cận RAG+LLM (dùng LLM có sẵn, không bắt
> buộc tự train) — README này mô tả **trạng thái hiện tại** theo hướng mới. Một phần code
> Phase 1 cũ (kiến trúc 3-tier tự train, dữ liệu TBMT tổng hợp/thật) vẫn còn trong repo
> (`models/ner.py`, `crawler/`) với vai trò giảm — trích trường từ KHLCNT upload; phần đã
> bị thay thế hoàn toàn (M3 Classifier, M4 BM25-only retriever cũ) đã được gỡ khỏi code khi
> dọn dẹp, chỉ còn số liệu lịch sử trong `docs/MODEL_CARD.md` — xem `docs/DATA_CARD.md` và
> `docs/MODEL_CARD.md` để phân biệt rõ nội dung nào thuộc bản cũ và nội dung nào thuộc bản
> redesign.

Xem đặc tả gốc tại [`docs/SPEC.md`](docs/SPEC.md) (Phase 1) — đề cương redesign nằm
ngoài repo (`de-cuong-hsmt-rag-cntt-phan-mem.pdf`, cung cấp bởi giảng viên).

> ⚠️ **Mọi nội dung do hệ thống sinh ra là dự thảo hỗ trợ soạn thảo** — bắt buộc thẩm
> định và phê duyệt theo quy định pháp luật trước khi phát hành chính thức. Xem
> [Nguyên tắc bắt buộc](#nguyên-tắc-thiết-kế-bắt-buộc) bên dưới.

---

## Cài đặt

Yêu cầu Python 3.10+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Dựng kho tri thức RAG (bắt buộc trước khi soạn/hỏi-đáp chất lượng cao):**

```bash
python scripts/fetch_legal_corpus.py --all      # tải luật thật (đã có sẵn trong data/samples/legal_corpus/)
python scripts/build_legal_index.py             # build FAISS cho 2 model embedding (vài phút, cần mạng)
```

**Cấu hình Claude API (tuỳ chọn nhưng khuyến nghị):**

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Windows: set ANTHROPIC_API_KEY=...
```

Không có key vẫn chạy được — hệ thống tự rơi xuống chế độ không-LLM (liệt kê trích dẫn
thật/template-filling, xem [Degraded Mode](#nguyên-tắc-thiết-kế-bắt-buộc)) — nhưng **Mức
1/Mức 2 sẽ không có câu trả lời/mục soạn tổng hợp bằng tự nhiên ngôn ngữ**, chỉ có trích
dẫn thô.

**Ghi chú cài đặt khác:**
- `paddleocr`/`paddlepaddle` (OCR, dùng cho upload KHLCNT dạng scan) có thể không cài
  được trên một số máy — không sao, hệ thống tự động bỏ qua OCR (xem `ingest/ocr.py`).
- `weasyprint` cần thư viện hệ thống GTK; nếu thiếu (phổ biến trên Windows), phần mềm
  **tự động fallback sang ReportLab** (xem `export/pdf.py`).
- `playwright`: sau khi `pip install`, chạy thêm `playwright install chromium` — cần
  cho việc fetch văn bản luật (`knowledge/legal_fetch.py`, các trang là SPA render JS).

## Chạy ứng dụng

```bash
streamlit run app/main.py
```

Mở trình duyệt tại `http://localhost:8501`. Bắt đầu ở **Trang 7 — Hỏi-đáp** (Mức 1,
không cần tài liệu nào) hoặc **Trang 2 — Nạp KHLCNT** rồi **Trang 3 — Soạn thảo HSMT**
(Mức 2) nếu muốn soạn theo một gói thầu cụ thể.

## Chạy test

```bash
pytest
```

~95 test. Phần lớn chạy nhanh (rule-based/mock); một số test (`test_orchestrator.py`,
Tier 1 mock trong `test_generator.py`/`test_legal_qa.py`) dùng real embedding model —
lần chạy đầu tải model từ HuggingFace nên chậm hơn (~2 phút), các lần sau dùng cache
nhanh hơn nhiều.

---

## Kiến trúc hệ thống (bản redesign RAG+LLM)

```
OFFLINE — dựng kho tri thức thật
  Luật Đấu thầu 22/2023/QH15 (hợp nhất) + Nghị định 214/2025/NĐ-CP + Nghị định 45/2026/NĐ-CP
  + Thông tư 01/2024 & 22/2024/TT-BKHĐT
        │  fetch verbatim (Playwright/httpx, KHÔNG dùng WebFetch — xem knowledge/legal_fetch.py)
        ▼  chunk theo Điều/Khoản (rag/chunker.py) — 684 chunk thật
        ▼  embed bằng 2 model (rag/embedding_models.py) — so sánh không gian biểu diễn
        ▼  index: FAISS (dense) + BM25 (sparse) — rag/hybrid_retriever.py

ONLINE
  Mức 1 (Hỏi-đáp)          Mức 2 (Soạn mục HSMT)
  models/legal_qa.py       models/generator.py + pipeline/orchestrator.py
        │                         │
        ▼                         ▼
  Hybrid retrieve (RRF) + rerank cross-encoder (rag/hybrid_retriever.py)
        │
        ▼
  Claude API sinh câu trả lời/nội dung mục, bắt buộc trích dẫn
  (generation/claude_client.py) — không có API key thì rơi xuống
  liệt kê trích dẫn thô/template-filling, KHÔNG BAO GIỜ lỗi
        │
        ▼
  Compliance Guard (models/compliance.py): R1-R3 hạn chế cạnh tranh (câu),
  R5 thiếu thành phần bắt buộc (tài liệu, đối chiếu Điều 26 NĐ 214/2025/NĐ-CP)
        │
        ▼
  HITL duyệt (hitl/store.py) → Export PDF/DOCX (export/*)

ĐÁNH GIÁ (eval/) — xem Trang 8 (Đánh giá) trên GUI
  retrieval_eval.py: Recall@k/MRR/nDCG trên 46 câu gán tay (data/eval/)
  faithfulness_eval.py: LLM-as-judge (Claude) cho faithfulness/completeness
  embedding_compare.py: t-SNE/UMAP + độ tách biệt intra/inter-Điều
```

Chi tiết: [`docs/DATA_CARD.md`](docs/DATA_CARD.md) (nguồn dữ liệu, Mục 10-13 cho phần
redesign) và [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) (kiến trúc/kết quả từng phần).

### Nguyên tắc thiết kế bắt buộc

1. **Luôn chạy được (Degraded Mode):** mọi module sinh/hỏi-đáp đều có 3 tầng —
   Tier 1 (Claude API + RAG) → Tier 2 (dự phòng, chưa dùng) → Tier 3 (không LLM, luôn
   thành công). Thiếu `ANTHROPIC_API_KEY` hay lỗi mạng đều tự rơi xuống Tier 3, không
   làm sập ứng dụng. UI hiển thị badge tier đang chạy.
2. **Không bịa đặt:** câu trả lời/nội dung sinh ra CHỈ dựa trên trích đoạn luật thật đã
   truy xuất (system prompt bắt buộc trích dẫn Điều/Khoản); số liệu gói thầu (giá, thời
   gian, nguồn vốn) chèn bằng slot-filling, verifier `verify_numeric_consistency` gắn cờ
   `R4` nếu phát hiện số liệu lạ.
3. **Văn bản pháp luật phải đúng hiệu lực:** phát hiện và sửa một trường hợp thật trong
   quá trình làm — NĐ 24/2024/NĐ-CP (đề cương gốc liệt kê) đã hết hiệu lực, thay bằng
   NĐ 214/2025/NĐ-CP — xem `docs/DATA_CARD.md` Mục 10.
4. **Human-in-the-loop:** không mục nào được coi là hoàn thành nếu chưa qua **Phê
   duyệt** (`hitl/store.py`).
5. **Thu thập dữ liệu có trách nhiệm:** tôn trọng `robots.txt` (kể cả disallow riêng cho
   `ClaudeBot`), rate-limit, cache toàn bộ response.

---

## Luồng sử dụng (8 trang)

1. **Thu thập dữ liệu** / 2. **Nạp KHLCNT** — (Phase 1) crawl/upload để lấy trường
   thông tin gói thầu tự động, dùng làm input cho Mức 2.
3. **Soạn thảo HSMT (Mức 2)** — sinh dự thảo từng mục Chương III/V bằng Claude API+RAG,
   sửa trực tiếp, phê duyệt/từ chối, kiểm tra đủ thành phần (R5).
4. **Kiểm tra tuân thủ** — rà soát cờ R1-R5.
5. **Xuất và In** — xuất PDF/DOCX.
6. **Bảng điều khiển Model** — tier/metric Phase 1.
7. **Hỏi-đáp (Mức 1)** — hỏi tự do, trả lời có trích dẫn luật thật.
8. **Đánh giá** — bảng Recall@k/MRR/nDCG, ablation LLM-only vs RAG, so sánh embedding
   (t-SNE/UMAP).

---

## Kết quả đánh giá (bản redesign)

Retrieval (46 câu hỏi gán tay, `scripts/run_retrieval_eval.py`, chi tiết
`docs/DATA_CARD.md` Mục 12):

| Chế độ | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| BM25 (sparse) | 0.565 | 0.385 | 0.426 |
| Dense (vi_bi_encoder) | 0.696 | 0.546 | 0.580 |
| Hybrid RRF | 0.674 | 0.537 | 0.564 |
| **Hybrid RRF + rerank** | **0.761** | **0.587** | **0.627** |

So sánh embedding (`scripts/analyze_embeddings.py`, `docs/MODEL_CARD.md`): model tiếng
Việt chuyên biệt (`vi_bi_encoder`, 768d) tách biệt intra/inter-Điều tốt hơn model đa
ngôn ngữ (`multilingual_minilm`, 384d) — 0.184 so với 0.167 — dù similarity tuyệt đối
thấp hơn; khớp với kết quả Recall@k đo được ở trên.

**Rà soát kỹ thuật RAG (6 hạng mục: kiến trúc, xử lý embedding, encode/decode vector, LLM
transformer, chunking, indexing branch)** phát hiện và sửa 3 lỗi thật, chi tiết đầy đủ +
số liệu trước/sau tại `docs/DATA_CARD.md` Mục 12.1: (1) cross-encoder rerank bị tải lại
model mỗi lượt gọi thay vì cache — sửa xong giảm 57% thời gian rerank; (2) đưa dư số ứng
viên vào rerank so với `candidate_k` khai báo; (3) **65% chunk kho tri thức bị cắt âm thầm
khi embed** vì vượt quá `max_seq_length` của model (giới hạn kiến trúc, không phải cấu
hình) — đã viết `encode_texts` xử lý bằng sliding-window mean-pooling, xác nhận cải thiện
độ tách biệt không gian embedding +15%, nhưng ghi nhận trung thực rằng chỉ số SAU rerank
trên tập 46 câu hỏi lại giảm nhẹ — chưa đủ dữ liệu để kết luận đây là xu hướng thật hay
nhiễu thống kê, xem thảo luận đầy đủ trong DATA_CARD.md.

Faithfulness (LLM-as-judge) + ablation LLM-only vs RAG (`scripts/run_ablation_table.py`,
8 câu hỏi): RAG cải thiện faithfulness từ **0.41 → 0.94** và completeness từ **0.44 →
0.87** so với gọi thẳng LLM không có ngữ cảnh — xem chi tiết + các lỗi kỹ thuật phát hiện
khi chạy live (temperature/extended thinking/số trích dẫn bị gắn cờ oan) tại
`docs/DATA_CARD.md` Mục 13 (đối chiếu thuật ngữ Faithfulness/Context Recall/Context
Precision với framework RAGAS ở Mục 14).

---

## Giới hạn đã biết (bản redesign)

- **Thông tư 01/2024 & 22/2024/TT-BKHĐT** — đã đưa vào corpus (22/32 và 26/33 Điều, xem
  `docs/DATA_CARD.md` Mục 10.3) qua parser HTML riêng viết để xử lý lỗi heading lặp/thiếu
  phía nguồn; các mẫu biểu Word/Excel đi kèm 2 thông tư vẫn nằm ngoài phạm vi corpus RAG
  (không phù hợp pipeline trích dẫn theo Điều/Khoản).
- **Nghị định 45/2026/NĐ-CP** — bản chính thức chỉ có scan ảnh (OCR nằm ngoài phạm vi đề
  cương); đã lấy được 43/43 Điều qua bản transcript text ở nguồn thay thế (luatvietnam.vn)
  — xem Mục 10.4.
- **Không tải được HSMT phần mềm thật đã duyệt** — xác nhận đúng rủi ro đã dự đoán:
  cần đăng nhập + Windows-only Client Agent trên hệ thống chính thức — Mục 11.
- Nội dung Phase 1 cũ (`docs/DATA_CARD.md` Mục 1-9, `docs/MODEL_CARD.md` phần đầu) vẫn
  giữ nguyên giới hạn đã ghi khi đó (chưa có checkpoint Tier 1 thật cho NER/Classifier...).

---

## Cấu trúc thư mục

Xem Mục 4 của [`docs/SPEC.md`](docs/SPEC.md) (Phase 1) — thư mục mới thêm cho redesign:
`src/autotender/knowledge/` (fetch luật), `src/autotender/generation/` (Claude client),
`src/autotender/eval/` (retrieval/faithfulness/embedding eval), `data/samples/legal_corpus/`,
`data/eval/`, `data/index/` (build artifact, gitignore).

## Tài liệu liên quan

- [`docs/SPEC.md`](docs/SPEC.md) — đặc tả Phase 1 gốc
- [`docs/DATA_CARD.md`](docs/DATA_CARD.md) — nguồn gốc/giới hạn dữ liệu (Mục 10-13 cho redesign)
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — kiến trúc, tier, metric (phần cuối cho redesign)
- [`docs/COLAB_GUIDE.md`](docs/COLAB_GUIDE.md) — hướng dẫn Colab (Phase 1, không bắt buộc cho redesign)
