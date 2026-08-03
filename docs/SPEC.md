# AutoTender-VN — Bản đặc tả bàn giao cho Claude Code

> **Cách dùng:** Mở Claude Code trong thư mục dự án trống, dán **Mục 13 (Prompt khởi động)** vào trước.
> Sau đó mỗi ngày dán prompt của ngày tương ứng trong **Mục 11**.
> File này chính là tài liệu tham chiếu — hãy lưu nó vào repo tại `docs/SPEC.md` để Claude Code đọc lại bất cứ lúc nào.

---

## 0. Bối cảnh & ràng buộc cứng

| Hạng mục | Giá trị |
|---|---|
| Đề tài | Áp dụng Deep Learning tự động soạn thảo Hồ sơ mời thầu (E-HSMT) tại Việt Nam |
| Cấp độ | Đồ án cuối môn Deep Learning, bậc Thạc sĩ |
| Thời gian | **7 ngày** (kể cả viết báo cáo + slide) |
| Nhân lực | 1 người |
| Hạ tầng train | Google Colab (T4/L4 free hoặc Pro) |
| Hạ tầng chạy app | Laptop cá nhân, CPU-only phải chạy được |
| Ngôn ngữ | Python 3.10+ |
| Sản phẩm bàn giao | Phần mềm có GUI + repo code + báo cáo + slide + demo video |

**Ràng buộc quan trọng nhất:** phần mềm **bắt buộc phải chạy được và demo được vào ngày 7**, kể cả khi mọi mô hình đều train thất bại. Xem Mục 2 (Degraded Mode).

---

## 1. Mục tiêu sản phẩm

Xây dựng phần mềm desktop/web cục bộ tên **AutoTender-VN**, cho phép cán bộ đấu thầu:

1. Thu thập tự động dữ liệu thông báo mời thầu và E-HSMT công khai từ Hệ thống mạng đấu thầu quốc gia.
2. Nạp một Quyết định phê duyệt Kế hoạch lựa chọn nhà thầu (KHLCNT) dạng PDF/DOCX/text.
3. Hệ thống tự trích xuất trường thông tin, phân loại gói thầu, truy xuất mẫu điều khoản tương tự và **sinh dự thảo các chương của E-HSMT**.
4. Mọi nội dung sinh ra đều được **gắn cờ tuân thủ** (compliance flag) và **trích dẫn nguồn** (citation).
5. Người dùng **review, sửa trực tiếp, phê duyệt hoặc từ chối từng mục** (human-in-the-loop).
6. Xuất **PDF đúng thể thức văn bản hành chính Việt Nam** và **in trực tiếp**.
7. Mọi thao tác sửa của người dùng được ghi lại thành dữ liệu phản hồi phục vụ huấn luyện lại.

---

## 2. Nguyên tắc bắt buộc (NON-NEGOTIABLE)

Claude Code phải tuân thủ tuyệt đối các nguyên tắc sau trong toàn bộ quá trình code:

### 2.1. Degraded Mode — phần mềm không bao giờ được chết
Mỗi module ML phải có **3 tầng fallback**, chọn tự động theo thứ tự:

```
Tier 1: Mô hình fine-tuned (checkpoint trong models/)
   ↓ nếu checkpoint không tồn tại hoặc load lỗi
Tier 2: Mô hình pretrained zero-shot / few-shot
   ↓ nếu không có GPU / không tải được model
Tier 3: Rule-based (regex, từ điển, template)
```

UI phải hiển thị rõ đang chạy ở tier nào bằng badge màu. **Không bao giờ raise exception ra tới người dùng.**

### 2.2. Không bịa đặt
- Không được sinh ra điều/khoản pháp luật mà không có trong corpus đã nạp.
- Mọi con số trong dự thảo (giá gói thầu, thời gian, nguồn vốn) phải **copy nguyên văn** từ KHLCNT đầu vào, không được mô hình sinh tự do. Dùng slot-filling cho số liệu, generation chỉ cho phần diễn giải.
- Nếu không đủ căn cứ, module sinh phải trả về placeholder `[CẦN NGƯỜI DÙNG BỔ SUNG: <mô tả>]` thay vì đoán.

### 2.3. Human-in-the-loop là mặc định, không phải tuỳ chọn
- Không mục nào được đánh dấu "hoàn thành" nếu chưa có người bấm **Phê duyệt**.
- Xuất PDF phải hiển thị cảnh báo nếu còn mục chưa phê duyệt.
- Trang bìa PDF phải có dòng: *"Dự thảo do hệ thống hỗ trợ tạo lập — bắt buộc thẩm định và phê duyệt theo quy định pháp luật trước khi phát hành."*

### 2.4. Thu thập dữ liệu có trách nhiệm
- Tôn trọng `robots.txt`, rate limit tối thiểu 1 request/2 giây, có `User-Agent` khai báo rõ mục đích nghiên cứu.
- Cache toàn bộ response xuống đĩa, không crawl lại dữ liệu đã có.
- Chỉ thu thập dữ liệu công khai. Không thu thập thông tin cá nhân.
- Có cờ `--max-records` để giới hạn.

---

## 3. Tech stack (đã chốt, không thay đổi)

| Lớp | Công nghệ | Lý do |
|---|---|---|
| GUI | **Streamlit** | Nhanh nhất cho 1 tuần, hỗ trợ sẵn editor, file upload, download |
| Crawler | `httpx` + `selectolax`, fallback `playwright` | Hệ thống MSC là SPA, cần fallback headless |
| Đọc PDF | `pymupdf` (fitz), fallback OCR `paddleocr`/`vietocr` | Nhanh, giữ layout |
| Đọc DOCX | `python-docx` | |
| NLP | `transformers`, `torch`, `sentence-transformers`, `seqeval` | |
| Vector store | `faiss-cpu` (hoặc `chromadb` nếu dễ hơn) | Chạy offline |
| Template | `jinja2` | Một template dùng chung cho preview + print + PDF |
| PDF | **`weasyprint`** (HTML→PDF), fallback `reportlab` | Xử lý tiếng Việt và layout tốt nhất |
| DOCX export | `python-docx` | HSMT thực tế được sửa trên Word |
| Lưu trạng thái | SQLite qua `sqlite3` / `sqlmodel` | Không cần server |
| Cấu hình | `pydantic-settings` + `configs/*.yaml` | |
| Test | `pytest` | |

**Không dùng:** LangChain, LlamaIndex (quá nặng, khó giải thích trong báo cáo), Docker (thừa cho 1 tuần).

---

## 4. Cấu trúc thư mục

```
autotender-vn/
├── README.md
├── requirements.txt
├── docs/
│   ├── SPEC.md                  # chính file này
│   ├── DATA_CARD.md             # mô tả dataset
│   └── MODEL_CARD.md            # mô tả từng model + metric
├── configs/
│   ├── app.yaml
│   ├── crawler.yaml
│   └── models.yaml
├── data/
│   ├── raw/                     # HTML/JSON/PDF crawl về
│   ├── interim/                 # đã parse thành text
│   ├── processed/               # dataset train/val/test dạng jsonl
│   └── samples/                 # 20 mẫu bundled, commit vào repo
├── models/                      # checkpoint (gitignore, tải từ Drive)
├── notebooks/
│   ├── 01_train_ner.ipynb
│   ├── 02_train_classifier.ipynb
│   ├── 03_train_retriever.ipynb
│   ├── 04_train_generator.ipynb
│   └── 05_train_compliance.ipynb
├── src/autotender/
│   ├── __init__.py
│   ├── schemas.py               # Pydantic models — HỢP ĐỒNG DỮ LIỆU
│   ├── crawler/
│   │   ├── msc_client.py
│   │   ├── parser.py
│   │   └── pipeline.py
│   ├── ingest/
│   │   ├── pdf_reader.py
│   │   ├── docx_reader.py
│   │   └── ocr.py
│   ├── models/
│   │   ├── base.py              # BaseModule với cơ chế 3-tier fallback
│   │   ├── ner.py               # M2
│   │   ├── classifier.py        # M3
│   │   ├── retriever.py         # M4
│   │   ├── generator.py         # M5
│   │   └── compliance.py        # M6
│   ├── rag/
│   │   ├── chunker.py
│   │   ├── index.py
│   │   └── rerank.py
│   ├── pipeline/
│   │   └── orchestrator.py      # ghép M1→M6
│   ├── hitl/
│   │   ├── store.py             # SQLite: section states, edits
│   │   └── feedback.py          # xuất feedback thành jsonl huấn luyện
│   ├── export/
│   │   ├── templates/
│   │   │   ├── hsmt.html.j2
│   │   │   └── hsmt.css
│   │   ├── pdf.py
│   │   ├── docx.py
│   │   └── print.py
│   └── utils/
│       ├── vn_text.py           # chuẩn hoá tiếng Việt, tách câu
│       └── logging.py
├── app/
│   ├── main.py                  # entrypoint Streamlit
│   └── pages/
│       ├── 1_Thu_thap_du_lieu.py
│       ├── 2_Nap_KHLCNT.py
│       ├── 3_Soan_thao_HSMT.py
│       ├── 4_Kiem_tra_tuan_thu.py
│       ├── 5_Xuat_va_In.py
│       └── 6_Bang_dieu_khien_Model.py
├── scripts/
│   ├── crawl.py
│   ├── build_dataset.py
│   ├── build_index.py
│   └── evaluate.py
└── tests/
```

---

## 5. Hợp đồng dữ liệu (`src/autotender/schemas.py`)

Định nghĩa TRƯỚC TIÊN bằng Pydantic. Mọi module giao tiếp qua các schema này.

```python
class TenderNotice(BaseModel):      # bản ghi crawl về
    tbmt_id: str
    package_name: str
    investor: str
    procuring_entity: str | None
    package_value: float | None       # VND
    currency: str = "VND"
    funding_source: str | None
    selection_method: str | None      # đấu thầu rộng rãi, chào hàng cạnh tranh...
    contract_type: str | None
    package_type: str | None          # hàng hóa | xây lắp | tư vấn | phi tư vấn
    execution_time: str | None
    publish_date: date | None
    close_date: date | None
    attachments: list[str] = []
    source_url: str

class ExtractedField(BaseModel):
    name: str
    value: str
    confidence: float
    char_start: int | None            # để highlight trong văn bản gốc
    char_end: int | None
    source: Literal["ner_model", "regex", "manual"]

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    source_doc: str                   # "Thông tư 79/2025/TT-BTC — Mẫu số 1" ...
    score: float

class ComplianceFlag(BaseModel):
    rule_code: Literal["R1","R2","R3","R4","OK"]
    severity: Literal["cao","trung_binh","thap"]
    sentence: str
    explanation: str
    evidence: list[RetrievedChunk]
    confidence: float

class HSMTSection(BaseModel):
    section_id: str                   # "chuong_III.muc_2.1"
    title: str
    generated_text: str
    edited_text: str | None
    status: Literal["draft","edited","approved","rejected"] = "draft"
    citations: list[RetrievedChunk]
    flags: list[ComplianceFlag]
    model_tier: Literal[1,2,3]
    generated_at: datetime
    approved_by: str | None
    approved_at: datetime | None

class HSMTDocument(BaseModel):
    doc_id: str
    package: TenderNotice
    fields: list[ExtractedField]
    sections: list[HSMTSection]
    created_at: datetime
    updated_at: datetime
```

---

## 6. Đặc tả từng module

### M0 — Crawler (`src/autotender/crawler/`)

**Nhiệm vụ:** thu thập thông báo mời thầu và file E-HSMT đính kèm từ `muasamcong.mpi.gov.vn`.

**Yêu cầu triển khai:**
1. Trước tiên, thử phát hiện **API JSON nội bộ** của trang (mở DevTools → Network, tìm endpoint dạng `/o/egp-portal-.../services/...`). Ưu tiên gọi API vì ổn định hơn parse HTML.
2. Nếu không được, dùng Playwright headless để render và scrape.
3. Thiết kế theo interface `TenderSource` với 3 implementation:
   - `MSCApiSource` — gọi API thật
   - `MSCBrowserSource` — Playwright
   - `LocalSampleSource` — đọc từ `data/samples/`, **luôn hoạt động**
4. CLI: `python scripts/crawl.py --from 2025-01-01 --to 2026-06-30 --max-records 3000 --out data/raw/`
5. Ghi log tiến độ, resume được sau khi ngắt (lưu cursor vào `data/raw/_state.json`).
6. Chuẩn hoá về schema `TenderNotice`, ghi ra `data/interim/notices.jsonl`.

**Đầu ra tối thiểu để dự án chạy:** 20 bản ghi mẫu đã commit sẵn trong `data/samples/`. Nếu crawl thật thất bại, dự án vẫn demo được.

### M1 — Ingestion (`src/autotender/ingest/`)
- `pdf_reader.py`: dùng PyMuPDF trích text theo trang, giữ toạ độ block. Nếu trang có < 50 ký tự → coi là scan → chuyển sang OCR.
- `ocr.py`: PaddleOCR với `lang='vi'`, fallback VietOCR. Bọc try/except, nếu không cài được thì trả về thông báo rõ ràng chứ không crash.
- Chuẩn hoá tiếng Việt: NFC normalization, gộp dòng bị ngắt, tách câu bằng regex có xử lý viết tắt tiếng Việt (`TT.`, `NĐ-CP`, `khoản`, số thứ tự).

### M2 — NER trích xuất trường

- **Tier 1:** `vinai/phobert-base-v2` + token classification head. Nhãn BIO cho 8 entity: `PACKAGE_NAME`, `VALUE`, `FUNDING`, `METHOD`, `CONTRACT_TYPE`, `DURATION`, `INVESTOR`, `LOCATION`.
- **Tier 2:** `xlm-roberta-base` zero-shot qua QA-style prompting.
- **Tier 3:** regex + keyword rules (ví dụ giá gói thầu: `(?:giá gói thầu|giá trị)[:\s]*([\d.,]+)\s*(?:đồng|VNĐ|VND)`).
- **Dữ liệu:** distant supervision — dùng metadata có cấu trúc từ crawler làm nhãn tự động bằng cách khớp chuỗi trong văn bản gốc. Gán tay 200 mẫu làm test set.
- **Metric:** entity-level F1 (seqeval), báo cáo per-entity.
- Notebook train: `notebooks/01_train_ner.ipynb`.

### M3 — Phân loại gói thầu
- Multi-class: `hàng hóa | xây lắp | tư vấn | phi tư vấn | hỗn hợp`.
- Tier 1: PhoBERT + classification head. Tier 3: keyword matching.
- Metric: macro-F1, confusion matrix.
- Nhãn miễn phí từ metadata crawler.

### M4 — Retrieval (RAG)
- **Corpus:** (a) mẫu E-HSMT theo Thông tư hiện hành, (b) trích đoạn Luật Đấu thầu 2023 + Nghị định 214/2025/NĐ-CP, (c) các HSMT lịch sử đã crawl.
- **Chunking:** theo điều/khoản/mục, giữ metadata `source_doc`, tối đa 512 token, overlap 64.
- **Bi-encoder:** `bkai-foundation-models/vietnamese-bi-encoder`, fine-tune bằng `MultipleNegativesRankingLoss` với hard negative từ BM25.
- **Cross-encoder rerank:** XLM-R base, top-50 → top-5.
- **Index:** FAISS `IndexFlatIP` (dataset nhỏ, không cần HNSW).
- **Metric:** Recall@5, MRR@10, nDCG@10. So sánh với baseline BM25.

### M5 — Sinh dự thảo
- **Phạm vi:** chỉ sinh 2 chương khả thi trong 1 tuần:
  - Chương III — Tiêu chuẩn đánh giá E-HSDT (năng lực, kinh nghiệm, kỹ thuật)
  - Chương V — Yêu cầu về kỹ thuật / Phạm vi cung cấp
- **Sinh theo từng mục (section-wise), KHÔNG sinh cả tài liệu một lượt.**
- **Prompt template:** `[trường trích xuất từ KHLCNT] + [top-5 chunk truy xuất] + [khung mục cần sinh]`.
- Tier 1: `VietAI/vit5-base` fine-tune (fit trên T4). Tier 2: Qwen2.5-7B-Instruct + QLoRA nếu có A100, hoặc gọi API nếu người dùng cấu hình key. Tier 3: template filling thuần từ mẫu Thông tư.
- **Bắt buộc:** số liệu chèn bằng slot, không để mô hình sinh. Sau khi sinh, chạy verifier so khớp mọi con số với `ExtractedField`; lệch → tự động gắn flag `R4`.
- **Metric:** ROUGE-L, BERTScore (`xlm-roberta`), + edit-rate do người dùng chấm.

### M6 — Compliance Guard (module trọng tâm)
- Cross-encoder XLM-R, input `[CLS] câu_sinh_ra [SEP] bằng_chứng_truy_xuất [SEP]`, 5 lớp:

| Mã | Ý nghĩa | Mức độ |
|---|---|---|
| R1 | Nêu nhãn hiệu / xuất xứ / catalogue cụ thể | Cao |
| R2 | Yêu cầu năng lực, kinh nghiệm, doanh thu bất hợp lý so với quy mô gói thầu | Cao |
| R3 | Thông số kỹ thuật "may đo" theo một sản phẩm duy nhất | Cao |
| R4 | Số liệu sai lệch so với KHLCNT được duyệt | Cao |
| OK | Hợp lệ | — |

- Loss: focal loss (dữ liệu mất cân bằng nặng).
- Tier 3: từ điển nhãn hiệu phổ biến (Cisco, Dell, HP, Samsung...) + regex ngưỡng bất hợp lý (ví dụ yêu cầu doanh thu > 3× giá gói thầu).
- **Explainability:** với mỗi flag, hiển thị top-5 token có attention cao nhất (attention rollout) — đưa vào UI.
- **Metric:** precision/recall/F1 từng lớp, đặc biệt báo cáo **recall** vì bỏ sót vi phạm nguy hiểm hơn báo động giả.

---

## 7. Đặc tả giao diện (Streamlit)

### Trang 1 — Thu thập dữ liệu
- Form: khoảng thời gian, loại gói thầu, số bản ghi tối đa.
- Nút **Bắt đầu thu thập** → progress bar + log realtime.
- Bảng kết quả, nút tải xuống JSONL/CSV.
- Thẻ thống kê: tổng bản ghi, phân bố loại gói thầu, biểu đồ giá trị gói thầu.

### Trang 2 — Nạp KHLCNT
- Upload PDF/DOCX/paste text.
- Panel trái: văn bản gốc, **highlight màu các entity đã trích xuất**.
- Panel phải: bảng trường thông tin — mỗi dòng gồm `tên trường | giá trị | độ tin cậy | nút Sửa`.
- Trường có confidence < 0.7 → tô nền vàng, bắt buộc người dùng xác nhận.
- Badge hiển thị model tier đang dùng.
- Nút **Xác nhận và chuyển sang soạn thảo**.

### Trang 3 — Soạn thảo HSMT (màn hình chính)
Layout 3 cột:

```
┌──────────────┬─────────────────────────────┬──────────────────┐
│ CÂY MỤC LỤC  │  TRÌNH SOẠN THẢO            │  CĂN CỨ & CỜ     │
│              │                             │                  │
│ ▸ Chương I   │  [Tiêu đề mục]              │  📎 Trích dẫn:   │
│ ▾ Chương III │  ┌───────────────────────┐  │  • TT 79/2025... │
│   ✅ 2.1     │  │ text_area có thể sửa  │  │    (score 0.87)  │
│   ⚠️ 2.2     │  │ trực tiếp             │  │  • Luật ĐT 2023  │
│   ⏳ 2.3     │  └───────────────────────┘  │                  │
│ ▸ Chương V   │                             │  🚩 Cờ tuân thủ: │
│              │  [Sinh lại] [Phê duyệt]     │  R1 - Cao        │
│ Tiến độ:     │  [Từ chối]  [Khôi phục]     │  "nêu nhãn hiệu" │
│ 8/24 mục     │                             │  [Bỏ qua][Sửa]   │
└──────────────┴─────────────────────────────┴──────────────────┘
```

- Icon trạng thái: ⏳ draft · ✏️ edited · ✅ approved · ❌ rejected
- Nút **Sinh toàn bộ** với progress bar theo mục.
- Mỗi lần sửa → lưu vào SQLite + ghi diff vào bảng `edit_log`.
- Nút **So sánh** hiện diff giữa `generated_text` và `edited_text` (dùng `difflib` + HTML màu).

### Trang 4 — Kiểm tra tuân thủ
- Bảng tổng hợp toàn bộ flag, sort theo severity.
- Bộ lọc theo mã quy tắc.
- Mỗi dòng mở rộng được: câu vi phạm, giải thích, bằng chứng, token attention.
- Nút **Chấp nhận cờ** / **Đánh dấu dương tính giả** → ghi vào feedback store.
- Biểu đồ: số flag theo loại, tỷ lệ mục sạch.

### Trang 5 — Xuất và In
- Preview HTML render đúng thể thức (dùng chính template Jinja2).
- Cảnh báo đỏ nếu còn mục chưa phê duyệt, kèm danh sách.
- Nút: **Xuất PDF** · **Xuất DOCX** · **In** (JS `window.print()` qua `st.components.v1.html`).
- Tuỳ chọn: có/không đánh số trang, có/không đóng dấu "DỰ THẢO" watermark.

### Trang 6 — Bảng điều khiển Model
- Bảng: từng module | tier đang chạy | đường dẫn checkpoint | metric trên test set.
- Nút tải checkpoint từ URL/Drive.
- Biểu đồ so sánh baseline vs fine-tuned (đọc từ `reports/metrics.json`).
- **Đây là trang để demo trước hội đồng** — cho thấy có đủ mô hình và có số liệu đánh giá.

---

## 8. Đặc tả xuất PDF và In

### Nguyên tắc: một template, ba đầu ra
`hsmt.html.j2` + `hsmt.css` → dùng chung cho: preview trên màn hình, in trực tiếp, và WeasyPrint.

### Thể thức (theo quy định về thể thức văn bản hành chính, cần đối chiếu lại Nghị định 30/2020/NĐ-CP)
```css
@page {
  size: A4;
  margin-top: 20mm; margin-bottom: 20mm;
  margin-left: 30mm; margin-right: 20mm;
  @bottom-center { content: counter(page) "/" counter(pages); }
}
body { font-family: "Times New Roman", "Tinos", serif; font-size: 13pt; line-height: 1.5; }
```

**Cảnh báo tiếng Việt:** phải nhúng font hỗ trợ đầy đủ dấu tiếng Việt. Nếu dùng ReportLab fallback, bắt buộc `pdfmetrics.registerFont(TTFont('Times', 'fonts/Tinos-Regular.ttf'))` — commit file font vào `src/autotender/export/fonts/`. **Test bắt buộc:** render chuỗi `"Nguyễn Thị Hường — gói thầu số 05: Mua sắm thiết bị"` và kiểm tra bằng mắt.

### Cấu trúc PDF xuất ra
1. Trang bìa: tên gói thầu, chủ đầu tư, số hiệu, ngày, **dòng cảnh báo dự thảo**
2. Mục lục tự sinh
3. Các chương theo thứ tự
4. Phụ lục: nhật ký phê duyệt (ai duyệt mục nào, lúc nào) — chi tiết này rất được đánh giá cao vì thể hiện tính giải trình

### In
Nút In render HTML vào iframe ẩn với `@media print` rồi gọi `window.print()`. Test trên Chrome.

---

## 9. Human-in-the-loop store

Bảng SQLite:
```sql
CREATE TABLE documents (doc_id TEXT PRIMARY KEY, package_json TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE sections (section_id TEXT, doc_id TEXT, title TEXT, generated_text TEXT,
                       edited_text TEXT, status TEXT, model_tier INT,
                       citations_json TEXT, flags_json TEXT,
                       approved_by TEXT, approved_at TEXT, PRIMARY KEY(doc_id, section_id));
CREATE TABLE edit_log (id INTEGER PRIMARY KEY, doc_id TEXT, section_id TEXT,
                       before_text TEXT, after_text TEXT, edited_at TEXT);
CREATE TABLE flag_feedback (id INTEGER PRIMARY KEY, doc_id TEXT, section_id TEXT,
                            rule_code TEXT, user_verdict TEXT, note TEXT, created_at TEXT);
```

`hitl/feedback.py` xuất ra `data/processed/feedback_{date}.jsonl` với format sẵn sàng để fine-tune lại M5 (cặp generated → edited) và M6 (nhãn đúng/sai do người dùng xác nhận). **Nêu rõ vòng lặp này trong báo cáo — đây là đóng góp về mặt hệ thống.**

---

## 10. Đánh giá & báo cáo số liệu

`scripts/evaluate.py` chạy toàn bộ và ghi ra `reports/metrics.json` + `reports/figures/*.png`:

| Module | Metric | Baseline bắt buộc |
|---|---|---|
| M2 NER | entity F1 (seqeval) | BiLSTM-CRF, regex |
| M3 Classifier | macro-F1, confusion matrix | TF-IDF + LogisticRegression |
| M4 Retrieval | Recall@5, MRR, nDCG@10 | BM25 |
| M5 Generator | ROUGE-L, BERTScore | template filling |
| M6 Compliance | P/R/F1 per class, PR-curve | keyword rules |
| End-to-end | edit-rate, thời gian soạn thảo, số flag đúng | thủ công |

**Ablation bắt buộc** (chạy được trong 1 giờ): (a) bỏ retrieval, (b) bỏ hard negatives, (c) bỏ M6, (d) PhoBERT vs XLM-R.

Chạy 3 seed, báo cáo mean ± std.

---

## 11. Kế hoạch 7 ngày — prompt cho từng ngày

### Ngày 1 — Khung sườn + Crawler
> Đọc `docs/SPEC.md`. Hôm nay làm: (1) khởi tạo repo theo cấu trúc Mục 4, (2) viết đầy đủ `schemas.py` theo Mục 5, (3) triển khai crawler theo Mục 6/M0 với cả 3 source implementation, (4) tạo 20 bản ghi mẫu trong `data/samples/`, (5) `scripts/crawl.py` chạy được. Viết test cho parser. Cuối ngày phải chạy được `python scripts/crawl.py --source local --max-records 20`.

### Ngày 2 — Ingestion + M2 + M3 (Tier 3 trước)
> Triển khai `ingest/` và `models/ner.py`, `models/classifier.py` **ở Tier 3 (rule-based) trước tiên**, cùng lớp `BaseModule` với cơ chế fallback 3 tầng. Viết `scripts/build_dataset.py` tạo dataset distant-supervision cho NER. Tạo `notebooks/01` và `02` sẵn sàng chạy trên Colab. Test: nạp 1 PDF mẫu → ra `list[ExtractedField]`.

### Ngày 3 — RAG + Generator Tier 3
> Xây corpus pháp lý + mẫu HSMT trong `data/samples/corpus/`. Triển khai `rag/` (chunker, FAISS index, rerank) và `models/retriever.py`. Triển khai `models/generator.py` Tier 3 (template filling) và verifier so khớp số liệu. `scripts/build_index.py` chạy được. Test end-to-end: KHLCNT mẫu → sinh được Chương III bằng template.

### Ngày 4 — GUI
> Dựng toàn bộ 6 trang Streamlit theo Mục 7. Ưu tiên Trang 3 (soạn thảo) làm thật kỹ: cây mục lục, editor, diff, trạng thái. Triển khai `hitl/store.py` với SQLite. Cuối ngày: `streamlit run app/main.py` chạy trọn luồng từ upload đến phê duyệt.

### Ngày 5 — Xuất PDF/DOCX/In + M6
> Triển khai `export/` theo Mục 8, test kỹ tiếng Việt. Triển khai `models/compliance.py` Tier 3 + Trang 4. Song song: chạy `notebooks/01,02,05` trên Colab để có checkpoint thật.

### Ngày 6 — Nâng lên Tier 1 + đánh giá
> Tải checkpoint từ Colab vào `models/`, kiểm tra Tier 1 hoạt động. Chạy `notebooks/03,04`. Viết và chạy `scripts/evaluate.py`, sinh toàn bộ biểu đồ. Chạy ablation. Điền `reports/metrics.json` và Trang 6.

### Ngày 7 — Hoàn thiện
> Viết `README.md` đầy đủ (cài đặt, chạy, kiến trúc, kết quả), `DATA_CARD.md`, `MODEL_CARD.md`. Dọn code, thêm docstring. Quay demo video 5 phút. Sửa mọi bug demo. **Không thêm tính năng mới.**

---

## 12. Tiêu chí nghiệm thu (Definition of Done)

Đánh dấu ✅ từng mục trước khi nộp:

- [ ] `pip install -r requirements.txt && streamlit run app/main.py` chạy được trên máy sạch
- [ ] Chạy được khi **không có** thư mục `models/` (Tier 3 hoạt động)
- [ ] Crawler chạy được ở chế độ local, không cần mạng
- [ ] Luồng đầy đủ: upload PDF → trích xuất → sinh → sửa → phê duyệt → xuất PDF
- [ ] PDF hiển thị đúng 100% dấu tiếng Việt
- [ ] Nút In mở hộp thoại in của trình duyệt
- [ ] Ít nhất 3/5 module có checkpoint Tier 1 thật, có số liệu đánh giá
- [ ] `reports/metrics.json` có đủ metric + baseline + ablation
- [ ] Mọi mục sinh ra đều có ít nhất 1 citation
- [ ] `pytest` pass toàn bộ
- [ ] README có ảnh chụp màn hình

---

## 13. Prompt khởi động cho Claude Code

Copy nguyên khối dưới đây vào Claude Code phiên đầu tiên:

```
Tôi đang làm đồ án cuối môn Deep Learning bậc thạc sĩ, đề tài: "Áp dụng Deep Learning
tự động soạn thảo Hồ sơ mời thầu (E-HSMT) tại Việt Nam". Tôi có ĐÚNG 7 NGÀY.

Tôi đã chuẩn bị bản đặc tả đầy đủ. Hãy tạo file docs/SPEC.md với nội dung tôi dán
kèm bên dưới, đọc kỹ, rồi bắt đầu Ngày 1.

CÁC NGUYÊN TẮC BẮT BUỘC, KHÔNG ĐƯỢC VI PHẠM:
1. Phần mềm phải chạy được kể cả khi mọi model chưa được train (cơ chế fallback 3 tầng).
2. Không bịa số liệu, không bịa điều khoản pháp luật. Số liệu dùng slot-filling từ
   văn bản gốc, không để model sinh tự do.
3. Human-in-the-loop bắt buộc: không mục nào tự động được coi là hoàn thành.
4. Ưu tiên CHẠY ĐƯỢC hơn là ĐẦY ĐỦ. Làm Tier 3 (rule-based) trước, Tier 1 sau.
5. Tiếng Việt trong PDF phải đúng dấu 100% — đây là lỗi kinh điển, phải test sớm.

CÁCH LÀM VIỆC TÔI MUỐN:
- Trước mỗi ngày, tóm tắt kế hoạch ngắn gọn rồi mới code.
- Viết schemas.py trước tiên, mọi module giao tiếp qua Pydantic schema.
- Commit sau mỗi module hoàn thành, message tiếng Anh, rõ ràng.
- Nếu một thư viện không cài được, chuyển sang phương án dự phòng ngay, đừng loay hoay.
- Cuối mỗi ngày, chạy thử toàn bộ luồng và báo cáo cái gì chạy được / chưa được.
- Comment code bằng tiếng Việt ở phần logic nghiệp vụ đấu thầu, tiếng Anh ở phần kỹ thuật.

Bắt đầu bằng cách đọc SPEC rồi hỏi tôi tối đa 3 câu nếu có chỗ chưa rõ.

--- NỘI DUNG SPEC ---
[dán toàn bộ file AutoTender-VN_Claude-Code-Spec.md vào đây]
```

---

## 14. Rủi ro và phương án dự phòng

| Rủi ro | Xác suất | Phương án |
|---|---|---|
| Crawl bị chặn / có captcha | Cao | Dùng `LocalSampleSource`, tải thủ công 50 file, ghi rõ trong báo cáo là giới hạn nghiên cứu |
| Colab hết quota GPU | Trung bình | Giảm về `phobert-base`, batch 8, max_len 256; hoặc dùng Kaggle 30h/tuần |
| PaddleOCR cài không được | Trung bình | Bỏ OCR, chỉ nhận PDF có text layer, ghi vào phần Giới hạn |
| WeasyPrint lỗi trên Windows (thiếu GTK) | Cao | Fallback ReportLab, hoặc dùng `playwright` in HTML→PDF |
| Không đủ dữ liệu gán nhãn cho M6 | Cao | Sinh dữ liệu tổng hợp bằng LLM rồi người kiểm duyệt; ghi rõ phương pháp trong Data Card |
| Hết thời gian | Cao | Cắt theo thứ tự: bỏ M1/OCR → bỏ DOCX export → bỏ M3 → giữ bằng mọi giá M4+M5+M6+GUI |

---

## 15. Ghi chú cho phần bảo vệ

Ba câu hỏi gần như chắc chắn sẽ bị hỏi, chuẩn bị trước:

1. **"Model sinh sai thì ai chịu trách nhiệm?"** → Hệ thống là trợ lý soạn thảo. Mọi mục có citation, có confidence, có cờ tuân thủ, và bắt buộc người có thẩm quyền phê duyệt. Nhật ký phê duyệt được in kèm trong PDF.
2. **"Vì sao không dùng thẳng ChatGPT/Claude?"** → Chỉ ra bảng so sánh trong `reports/metrics.json`: mô hình fine-tune nhỏ cho F1 tương đương ở tác vụ trích xuất và tuân thủ, chạy offline (dữ liệu đấu thầu nhạy cảm), chi phí gần bằng 0, và có thể giải thích được qua attention.
3. **"Deep learning ở chỗ nào, hay chỉ là gọi API?"** → Trang 6 (Bảng điều khiển Model) + bảng phủ kỹ thuật + ablation study. Nhấn mạnh M6 là mô hình tự huấn luyện, không phải prompt.
