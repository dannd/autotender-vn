# KẾ HOẠCH PHÂN CÔNG CHUYÊN SÂU & KHUNG LÝ THUYẾT BẢO VỆ ĐỒ ÁN DEEP LEARNING (5 THÀNH VIÊN)
## DỰ ÁN: AUTOTENDER-VN — HỆ THỐNG TRỢ LÝ AI SOẠN THẢO & RÀ SOÁT E-HSMT BẰNG HYBRID RAG + LLM

> **Chương trình:** Thạc sĩ Kỹ thuật (Master of Engineering) — Môn: Học sâu nâng cao (Deep Learning)  
> **Đề tài:** AutoTender-VN — Hệ thống Trợ lý AI Soạn thảo & Rà soát Hồ sơ Mời thầu cho Gói thầu Phần mềm/CNTT tại Việt Nam  
> **Mô hình phối hợp:** Mô hình Đội ngũ Kỹ thuật AI Ứng dụng Doanh nghiệp (Enterprise Applied AI Product Team)  
> **Quy mô nhóm:** 5 Thành viên  
> **Nguyên tắc phân vai:** **Đi từ Nghiệp vụ & Chuyển thể Bài toán Người $\rightarrow$ AI $\rightarrow$ Tiền xử lý Dữ liệu & Chunking $\rightarrow$ Không gian Vector & Truy xuất $\rightarrow$ Mô hình Ngôn ngữ Lớn & Orchestration $\rightarrow$ An toàn AI, Giao diện HITL & Đánh giá Toàn hệ thống.**

---

## 1. MA TRẬN PHÂN CÔNG TỔNG QUAN (RESPONSIBILITY MATRIX)

```
┌───────┬──────────────────────────────────┬──────────────────────────────────────────┬──────────────────────────────────────────┐
│ STT   │ Vai trò & Vị trí Phân công       │ Khối Công việc & Module Codebase         │ Trọng tâm Lý thuyết & Đóng góp Cốt lõi   │
├───────┼──────────────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────┤
│ **1** │ **AI Business Analyst & Legal**  │ • Nghiệp vụ 8 chương & 8 bộ luật        │ • Phân tích bài toán, Chuyển thể nghiệp vụ│
│       │ **Domain Architect**             │ • Data Contracts (`schemas.py`)          │   từ Con người $\rightarrow$ Trợ lý AI  │
│       │ *(Kiến trúc sư Nghiệp vụ & ĐT)*  │ • Định nghĩa bộ quy tắc Tuân thủ (R1-R5) │ • Thiết kế cấu trúc dữ liệu Pydantic     │
├───────┼──────────────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────┤
│ **2** │ **Data Curation, Benchmark &**   │ • `crawler/`, `ingest/`, OCR VietOCR     │ • Thu thập dữ liệu, gán nhãn Benchmark   │
│       │ **Legal Chunking Engineer**      │ • `rag/chunker.py` (LegalChunker)        │ • Token Classification (NER KHLCNT)      │
│       │ *(Kỹ sư Dữ liệu & Phân đoạn)*    │ • Dataset 46 câu hỏi Eval + Ground Truth │ • Hierarchical Legal Syntax Chunking     │
├───────┼──────────────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────┤
│ **3** │ **Representation Learning &**    │ • `rag/embedding_models.py`,             │ • Linear Attention GatedDeltaNet-2 (8K)  │
│       │ **Vector Database Engineer**     │ • `rag/qdrant_store.py`, `bm25.py`       │ • Dual-Tower Bi-Encoder, Cosine metric   │
│       │ *(Kỹ sư Biểu diễn & Vector DB)*  │ • `docker/embedding/` Microservice       │ • Qdrant HNSW Graph & Reciprocal Rank RRF│
├───────┼──────────────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────┤
│ **4** │ **Hybrid Reranking, Orchestrator**│ • `rag/rerank.py` (Cross-Encoder)       │ • Full Joint Self-Attention Reranking    │
│       │ **& Generative LLM Architect**   │ • `generation/llm_client.py` (Gateway)   │ • Deterministic Orchestration vs Agentic │
│       │ *(Kỹ sư Tái xếp hạng & LLM Gen)* │ • `models/generator.py` (Generator M5)   │ • In-Context Learning, 3-Tier Fallback   │
├───────┼──────────────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────┤
│ **5** │ **AI Safety, HITL Interface &**  │ • `models/compliance.py` (M6 Guard)      │ • Anti-Hallucination Numeric Verifiers   │
│       │ **System Evaluation Engineer**   │ • `hitl/`, `audit/`, `app/` (Streamlit)  │ • Human-In-The-Loop Safety & Audit Trail │
│       │ *(Kỹ sư An toàn AI & Đánh giá)*  │ • `export/docx.py` (NĐ 30/2020), `eval/` │ • LLM-as-a-judge Faithfulness Evaluation │
└───────┴──────────────────────────────────┴──────────────────────────────────────────┴──────────────────────────────────────────┘
```

---

## 2. LÀM RÕ CÁC CHUYÊN ĐỀ KỸ THUẬT QUAN TRỌNG (DEEP TECH SPOTLIGHTS)

### 📌 CHUYÊN ĐỀ 1: CHIẾN LƯỢC PHÂN ĐOẠN PHÁP LÝ (LEGAL CHUNKING STRATEGY)
*(Phụ trách chính: Thành viên 2 — Phối hợp: Thành viên 1)*

* **Vấn đề của các phương pháp Chunking thông thường:**
  * *Fixed-size Character Chunking (ví dụ: LangChain 500 chars, overlap 50):* Cắt ngẫu nhiên giữa câu, tách rời phần "chế tài/yêu cầu" ra khỏi "chủ thể/số Điều", làm loãng vector nhúng và khiến LLM trích dẫn sai số Điều.
  * *Standard Recursive Token Splitter:* Không hiểu biên giới pháp lý của một văn bản quy phạm Việt Nam (Chương $\rightarrow$ Mục $\rightarrow$ Điều $\rightarrow$ Khoản $\rightarrow$ Điểm).
* **Giải pháp `LegalChunker` của AutoTender-VN (Hierarchical Legal Syntax Chunking):**
  1. **Phân rã theo biên giới cú pháp tự nhiên:** Chỉ cắt tại ranh giới kết thúc một **Điều** hoặc một **Khoản**.
  2. **Cấu trúc Parent-Child Chunking:**
     * **Parent Chunk:** Toàn bộ nội dung của Điều luật (dành cho các điều ngắn $\le 1.000$ từ) $\rightarrow$ Đảm bảo tính toàn vẹn ngữ cảnh.
     * **Child Chunk:** Tách từng Khoản độc lập kèm tiêu đề Điều $\rightarrow$ Đạt độ đặc hiệu ngữ nghĩa cực cao khi truy xuất.
  3. **Bảo toàn Siêu dữ liệu (Metadata Preservation):** Mỗi vector point trong Qdrant được gắn chặt với Payload: `{law_id, law_name, doc_type, dieu_so, dieu_title, khoan_so, text, chunk_id}`. Khi RAG truy xuất, LLM luôn biết chính xác đoạn trích này nằm ở Điều nào, Nghị định nào.

---

### 📌 CHUYÊN ĐỀ 2: BẢN CHẤT KIẾN TRÚC RAG (ADVANCED MODULAR HYBRID RAG)
*(Phụ trách chính: Thành viên 3 & Thành viên 4)*

Hệ thống **không sử dụng Naive RAG** (chỉ nhúng vector đơn giản rồi đưa vào LLM) mà triển khai kiến trúc **Advanced Modular Hybrid RAG** gồm 5 giai đoạn nối tiếp:
```
[Câu hỏi/Mục HSMT] 
       │
       ├──► 1. Sparse Search: BM25 Inverted Index (bắt chính xác mã điều, con số tài chính)
       ├──► 2. Dense Search: Qdrant HNSW 1024d (bắt ngữ nghĩa sâu với DeepX Linear Attention)
       │
       ▼
[3. Rank-based Fusion: Reciprocal Rank Fusion (RRF k=60)] ──► Top 20 Candidates
       │
       ▼
[4. Deep Neural Reranking: Cross-Encoder Transformer] ──────► Top 5 Grounded Citations
       │
       ▼
[5. In-Context Grounded Prompt Generation (LLM Gateway)] ───► Bản thảo Markdown chuẩn xác
```

---

### 📌 CHUYÊN ĐỀ 3: THIẾT KẾ KIẾN TRÚC — ĐIỀU PHỐI TẤT ĐỊNH (ORCHESTRATION) vs AGENTIC TOOL CALLING
*(Phụ trách chính: Thành viên 4 & Thành viên 5)*

> **Câu hỏi trọng tâm của Hội đồng:** *"Hệ thống này có phải là một Autonomous Agentic LLM tự gọi Tool (Function Calling/ReAct) không? Tại sao nhóm lại thiết kế kiến trúc như hiện tại?"*

#### 1. Bản chất Thiết kế của AutoTender-VN:
AutoTender-VN được thiết kế theo mô hình **Deterministic Multi-Stage Orchestration Pipeline kết hợp Human-In-The-Loop (HITL) Guardrails**, chứ **KHÔNG PHẢI là một Autonomous Agentic LLM thả nổi tự do gọi tool**.

#### 2. So sánh Kiến trúc & Lý do Lựa chọn cho Hệ thống Công vụ:

| Tiêu chí | Autonomous Agentic LLM (ReAct / Tool-calling tự do) | Deterministic Orchestration + HITL (Kiến trúc AutoTender-VN) |
|---|---|---|
| **Mô hình hoạt động** | LLM tự suy luận vòng lặp (`Thought` $\rightarrow$ `Action` $\rightarrow$ `Tool Call` $\rightarrow$ `Observation`). | Orchestrator điều phối tuần tự 17 mục con của 8 chương theo kịch bản pháp định có sẵn. |
| **Tính tất định (Determinism)** | ❌ **Thấp / Khó đoán:** Dễ rơi vào vòng lặp gọi tool (loop), bỏ sót chương mục hoặc sinh thiếu cấu trúc pháp định. | ✅ **Tuyệt đối 100%:** Luôn đảm bảo sinh đủ 8 chương I–VIII theo đúng Điều 26 Nghị định 214/2025/NĐ-CP. |
| **Kiểm soát Chi phí & Ngân sách** | ❌ **Rủi ro cao:** Agent có thể gọi API liên tục gây cạn kiệt ngân sách (Token exhaustion). | ✅ **Kiểm soát chặt chẽ:** Tích hợp **Budget Guard** giới hạn trần $\$5.0$/session, Retry lũy thừa 3 lần qua `tenacity`. |
| **Độ trễ (Latency)** | ❌ Chậm do phải qua nhiều bước suy luận ReAct trung gian. | ✅ Nhanh và ổn định (xử lý song song / tuần tự có kiểm soát). |
| **Trách nhiệm Pháp lý & Kiểm toán** | ❌ **Hộp đen (Black box):** Khó giải trình lý do Agent đưa ra quyết định hoặc gọi tool sai. | ✅ **Bạch minh (Transparent):** Mọi trích dẫn luật, cờ vi phạm R1–R5 và thao tác duyệt của người dùng được lưu vết bất biến qua SQL Trigger. |
| **Khả năng tự phục hồi** | ❌ Sập toàn bộ khi Tool lỗi hoặc API bên thứ ba mất kết nối. | ✅ **3-Tier Graceful Degradation:** Tự động suy biến xuống Tier 3 template offline nếu mất mạng. |

* **Kết luận thiết kế:** Đối với nghiệp vụ đấu thầu công quyền của Nhà nước, **tính chính xác, tính ổn định, tuân thủ pháp luật và khả năng chịu trách nhiệm pháp lý** là ưu tiên số 1. Việc sử dụng **Deterministic Orchestrator** phối hợp với **Chuyên gia con người (HITL)** là sự lựa chọn kiến trúc kỹ thuật tối ưu nhất, vượt trội hoàn toàn so với việc thả nổi cho Autonomous Agent tự quyết định.

---

## 3. CHI TIẾT NHIỆM VỤ, LÝ THUYẾT & ĐÓNG GÓP CỦA TỪNG THÀNH VIÊN

---

### 👤 THÀNH VIÊN 1: AI BUSINESS ANALYST & LEGAL DOMAIN ARCHITECT
*(Kiến trúc sư Nghiệp vụ Đấu thầu & Chuyển thể Bài toán Con người $\rightarrow$ AI)*

#### 1. Phạm vi Trách nhiệm & Phần việc Đã làm
* **Nghiên cứu & Chuẩn hóa Nghiệp vụ Đấu thầu Công nghệ:**
  * Nắm trọn vẹn quy trình 7 bước con người lập, thẩm định và phê duyệt E-HSMT cho gói thầu phần mềm/CNTT sử dụng ngân sách nhà nước.
  * Phân tích và hệ thống hóa **8 bộ văn bản quy phạm pháp luật** (341 Điều): Luật Đấu thầu 22/2023, Nghị định 214/2025, Nghị định 73/2019 & 82/2024 (Quản lý đầu tư ứng dụng CNTT), Nghị định 85/2016 (5 Cấp độ ATTT), Thông tư 22/2024.
* **Chuyển thể Nghiệp vụ Con người sang Bài toán AI (Human-to-AI Translation):**
  * *Chuyển thể Cấu trúc Hồ sơ:* Thiết kế cấu trúc **8 Chương I–VIII** theo Điều 26 Nghị định 214/2025/NĐ-CP thành Data Contracts chuẩn Pydantic v2 trong `src/autotender/schemas.py`.
  * *Chuyển thể Tiêu chuẩn Kỹ thuật CNTT:* Định nghĩa các yêu cầu bài toán phần mềm: Đặc tả SRS, Kiến trúc mở, SLA hỗ trợ 24/7, An toàn thông tin Cấp độ 3 (NĐ 85) và Điều khoản bắt buộc bàn giao 100% mã nguồn (NĐ 73).
  * *Chuyển thể Quy tắc Kiểm soát Tuân thủ:* Thiết kế bộ logic 5 nhóm cờ vi phạm (R1: Chống chỉ định nhãn hiệu Điều 44, R2: Trần doanh thu $\le 3\times$, R3: Thông số may đo, R4: Sai lệch số liệu, R5: Đủ 8 chương).

#### 2. Tệp Mã Nguồn Phụ trách Trực tiếp
* `src/autotender/schemas.py` — Data contracts trung tâm (`HSMTDocument`, `HSMTSection`, `TenderNotice`, `ExtractedField`).
* `configs/models.yaml` — Cấu hình phạm vi 8 chương và danh mục nhãn hiệu cấm.
* `docs/SYSTEM_DESIGN.md` & `docs/DATA_CARD.md` — Tài liệu đặc tả bài toán và căn cứ pháp lý.

#### 3. Câu hỏi Vấn đáp Hội đồng Thường gặp & Gợi ý Trả lời
* **Q: Vai trò của bạn trong dự án là gì? Dự án đã giải quyết bài toán nghiệp vụ thực tế như thế nào?**
  * *Trả lời:* *"Thưa Thầy, em đóng vai trò là AI Business Analyst & Kiến trúc sư Nghiệp vụ. Em chịu trách nhiệm phân tích toàn bộ quy trình lập HSMT của cán bộ đấu thầu và khung pháp lý 8 bộ luật về đấu thầu và CNTT. Từ đó, em chuyển thể thành các đặc tả kỹ thuật cho nhóm kỹ sư: thiết kế cấu trúc dữ liệu `schemas.py`, định nghĩa 17 mục con của 8 chương HSMT, xác định 8 thực thể KHLCNT cần trích xuất, và xây dựng bộ tiêu chí rà soát tuân thủ R1–R5 cùng các điều kiện đặc thù về an toàn thông tin Cấp độ 3 và bàn giao mã nguồn."*

---

### 👤 THÀNH VIÊN 2: DATA CURATION, BENCHMARK DATASET & LEGAL CHUNKING ENGINEER
*(Kỹ sư Dữ liệu, Xây dựng Bộ Đánh giá & Phân đoạn Pháp luật)*

#### 1. Phạm vi Trách nhiệm & Phần việc Đã làm
* **Xây dựng Kho Dữ liệu Tri thức (Legal Corpus Curation):**
  * Thu thập, làm sạch và đóng gói 8 bộ văn bản luật chuẩn quốc gia thành các tệp `.jsonl` trong `data/samples/legal_corpus/`.
  * Xây dựng bộ công cụ Web Crawler (`crawler/`) thu thập thông báo mời thầu từ `muasamcong` và `dauthau.asia`.
  * Xây dựng pipeline đọc tệp PDF và OCR (VietOCR) xử lý văn bản scan ảnh.
* **Xây dựng Bộ Dữ liệu Kiểm thử Định lượng (Benchmark Evaluation Dataset):**
  * Xây dựng bộ dữ liệu **46 câu hỏi benchmark thực tế** kèm nhãn trích dẫn luật chuẩn xác (Ground Truth Golden Citations) trong `data/eval/` để phục vụ đo lường định lượng Recall, MRR và nDCG.
* **Thuật toán Phân đoạn Pháp lý (Legal Syntax Chunking):**
  * Xây dựng module `LegalChunker` bóc tách 341 Điều luật thành **699 Chunks** phân cấp theo cấu trúc Điều/Khoản kèm siêu dữ liệu đầy đủ.
* **Module Trích xuất Thực thể M2 NER:**
  * Xây dựng mô hình Token Classification kết hợp Rule-based Slot-filling để trích xuất 8 trường dữ liệu từ KHLCNT.

#### 2. Khung Lý thuyết Deep Learning Cần Nắm Vững
* **Named Entity Recognition (NER / Token Classification):** Gán nhãn token theo chuỗi với Pretrained Transformer kết hợp Regex Hybrid.
* **Hierarchical Legal Chunking:** Phân đoạn cú pháp tự nhiên theo Điều/Khoản bảo toàn ngữ cảnh và siêu dữ liệu.

#### 3. Tệp Mã Nguồn Phụ trách Trực tiếp
* `src/autotender/rag/chunker.py` — Thuật toán bóc tách phân đoạn luật Điều/Khoản.
* `src/autotender/models/ner.py` — Module bóc tách thực thể KHLCNT.
* `src/autotender/crawler/` & `src/autotender/ingest/` — Thu thập dữ liệu và OCR bản scan.
* `data/samples/legal_corpus/` — Kho tri thức 8 bộ luật (699 chunks).
* `data/eval/` — Tập dữ liệu 46 câu hỏi benchmark gán nhãn tay.

#### 4. Câu hỏi Vấn đáp Hội đồng Thường gặp & Gợi ý Trả lời
* **Q: Chiến lược Chunking của nhóm khác gì so với các thư viện Chunking thông thường?**
  * *Trả lời:* *"Thưa Thầy, các thư viện như LangChain CharacterTextSplitter cắt cố định theo số ký tự sẽ làm đứt gãy câu văn luật và tách rời số Điều khỏi nội dung quy định. Em đã phát triển `LegalChunker` bóc tách theo cấu trúc phân cấp Điều/Khoản tự nhiên, gắn siêu dữ liệu `law_id`, `dieu_so`, `khoan_so` vào từng chunk, tạo ra 699 chunks chuẩn mực giúp việc tìm kiếm vector đạt độ chính xác cao nhất."*

---

### 👤 THÀNH VIÊN 3: REPRESENTATION LEARNING & VECTOR DATABASE ENGINEER
*(Kỹ sư Không gian Biểu diễn Vector & Cơ sở Dữ liệu Vector Qdrant)*

#### 1. Phạm vi Trách nhiệm & Phần việc Đã làm
* **Nghiên cứu & Ứng dụng Mô hình Nhúng (Dense Representation Learning):**
  * Nghiên cứu và triển khai mô hình nhúng `dxtech-asia/deepx-embedding-v1` (1024 chiều) ứng dụng kiến trúc **Linear Attention (GatedDeltaNet-2)** hỗ trợ cửa sổ ngữ cảnh **8.192 tokens**.
  * So sánh thực nghiệm với `vietnamese-bi-encoder` (768 chiều) và phân tích không gian nhúng (t-SNE/UMAP).
* **Đóng gói Microservice Nhúng Container (Docker):**
  * Xây dựng container `docker/embedding/` tối ưu đa luồng CPU (12 worker threads PyTorch).
* **Cơ sở Dữ liệu Vector Qdrant (HNSW Graph):**
  * Thiết lập và tối ưu hóa Qdrant Vector DB với cấu trúc đồ thị **HNSW (Cosine Distance)**.
  * Cấu hình **Payload Indexing** để lọc kết hợp theo loại luật (`law_id`).
* **Sparse Lexical Search (BM25):**
  * Xây dựng chỉ mục từ khóa ngược **BM25 Inverted Index** phục vụ tìm kiếm từ khóa mã điều và số liệu chính xác.

#### 2. Khung Lý thuyết Deep Learning Cần Nắm Vững
* **Dual-Tower Bi-Encoder:** Mã hóa độc lập $\mathbf{u} = \text{Encoder}(q), \mathbf{v} = \text{Encoder}(d)$ qua Mean Pooling và Cosine Similarity.
* **Linear Attention / GatedDeltaNet-2:** Đưa độ phức tạp từ $O(N^2)$ về $O(N)$ bằng trạng thái bộ nhớ hồi quy, giải quyết giới hạn 512 tokens để nhúng trọn vẹn văn bản dài 8.192 tokens.
* **Cấu trúc Đồ thị HNSW:** Thuật toán duyệt đồ thị tìm kiếm láng giềng gần nhất xấp xỉ (ANN) $O(\log N)$.

#### 3. Tệp Mã Nguồn Phụ trách Trực tiếp
* `src/autotender/rag/embedding_models.py` — Quản lý mô hình nhúng và vector hóa.
* `src/autotender/rag/qdrant_store.py` — Giao tiếp Qdrant, tìm kiếm HNSW và payload filter.
* `src/autotender/rag/bm25.py` — Chỉ mục tìm kiếm từ khóa BM25.
* `docker/embedding/` & `docker-compose.yml` — Microservice containerization.
* `scripts/ingest_to_qdrant.py` & `scripts/analyze_embeddings.py`.

#### 4. Câu hỏi Vấn đáp Hội đồng Thường gặp & Gợi ý Trả lời
* **Q: Tại sao mô hình nhúng `deepx-embedding-v1` lại giải quyết được bài toán Điều luật dài?**
  * *Trả lời:* *"Thưa Thầy, các mô hình gốc BERT như `vietnamese-bi-encoder` bị giới hạn cứng 512 tokens do cơ chế Self-Attention bậc hai $O(N^2)$. Các văn bản quy phạm pháp luật thường có nhiều Điều luật rất dài ($> 1.000$ từ), khiến mô hình cũ bị cắt cụt văn bản ngầm. `deepx-embedding-v1` dùng kiến trúc Linear Attention (GatedDeltaNet-2) xử lý mượt mà 8.192 tokens, biểu diễn trọn vẹn ngữ nghĩa các Điều luật dài mà không bị mất đuôi."*

---

### 👤 THÀNH VIÊN 4: HYBRID RERANKING, ORCHESTRATOR & GENERATIVE LLM ARCHITECT
*(Kỹ sư Tái xếp hạng, Điều phối Hệ thống & Mô hình Ngôn ngữ Lớn Sinh Dự thảo)*

#### 1. Phạm vi Trách nhiệm & Phần việc Đã làm
* **Thuật toán Dung hợp Thứ hạng (Reciprocal Rank Fusion - RRF):**
  * Kết hợp kết quả từ Dense Search (Qdrant) và Sparse Search (BM25) với hằng số $k=60$ để tạo Top-20 ứng viên.
* **Mô hình Tái xếp hạng Chuyên sâu (Cross-Encoder Deep Reranking):**
  * Tích hợp `cross-encoder/ms-marco-MiniLM-L-6-v2` chấm điểm tương quan từng cặp $(q, d)$ qua cơ chế Full Joint Self-Attention để chọn Top-5 căn cứ chuẩn xác nhất.
* **Kiến trúc Điều phối Deterministic Orchestrator & LLM Gateway:**
  * Thiết kế Orchestrator điều phối tuần tự 17 mục con của 8 chương theo kịch bản pháp định, kiểm soát chi phí qua **Budget Guard** ($\$5$/session) và **Exponential Backoff Retry** (`tenacity`).
* **Bộ sinh Generator M5 (Mức 2 — 8 Chương HSMT):**
  * Thiết kế Prompt chuyên sâu và kỹ thuật **In-Context Learning (Grounded RAG)** sinh 8 chương HSMT chuẩn văn phong hành chính.
  * Xây dựng kiến trúc **3-Tier Graceful Degradation** (Cloud LLM $\rightarrow$ Local FAISS $\rightarrow$ Offline Template).

#### 2. Khung Lý thuyết Deep Learning Cần Nắm Vững
* **Cross-Encoder Full Joint Self-Attention:** Tương tác token-to-token trực tiếp giữa câu hỏi và tài liệu qua toàn bộ các tầng Transformer để lọc tinh.
* **Reciprocal Rank Fusion (RRF $k=60$):** Hợp nhất dựa trên thứ hạng loại bỏ chênh lệch thang đo điểm số.
* **Decoder-only Transformer & In-Context Learning:** Cơ chế Masked Multi-Head Attention, RoPE, KV-Caching và Grounded Context Injection.

#### 3. Tệp Mã Nguồn Phụ trách Trực tiếp
* `src/autotender/rag/hybrid_retriever.py` — Bộ điều phối Hybrid RAG.
* `src/autotender/rag/rerank.py` — Cross-Encoder Transformer Reranker.
* `src/autotender/generation/llm_client.py` — Universal LLM Gateway & Budget Guard.
* `src/autotender/models/generator.py` — Bộ sinh 8 Chương HSMT và System Prompts.
* `src/autotender/models/orchestrator.py` — Điều phối luồng sinh toàn văn bản.

#### 4. Câu hỏi Vấn đáp Hội đồng Thường gặp & Gợi ý Trả lời
* **Q: Tại sao nhóm không sử dụng kiến trúc Autonomous Agentic LLM tự gọi Tool (Function Calling/ReAct) mà dùng Deterministic Orchestrator?**
  * *Trả lời:* *"Thưa Thầy, đối với nghiệp vụ soạn thảo văn bản quy phạm và pháp lý công quyền, tính tất định (determinism), sự ổn định và kiểm soát rủi ro là yếu tố sống còn. Một Autonomous Agent thả nổi tự gọi tool rất dễ bị rơi vào vòng lặp (loop), sinh thiếu các chương mục pháp định theo Điều 26 NĐ 214 và làm bùng nổ chi phí API. Nhóm đã thiết kế Deterministic Orchestrator điều phối tuần tự 17 mục con có cấu trúc, mỗi mục được kiểm soát bởi RAG Top-5 và bộ cờ tuân thủ R1–R5, vừa đảm bảo 100% đúng thể thức luật định, vừa kiểm soát tuyệt đối ngân sách dưới \$5."*

---

### 👤 THÀNH VIÊN 5: AI SAFETY, HITL INTERFACE, THỂ THỨC NĐ 30 & SYSTEM EVALUATION
*(Kỹ sư An toàn AI, Giao diện Chuyên gia & Đánh giá Hệ thống Toàn diện)*

#### 1. Phạm vi Trách nhiệm & Phần việc Đã làm
* **Module M6 Compliance Guard (Rà soát Tuân thủ Pháp lý & CNTT):**
  * Hiện thực hóa các quy tắc nghiệp vụ thành code rà soát tự động 5 nhóm cờ vi phạm: **R1** (Cấm nhãn hiệu Điều 44), **R2** (Trần doanh thu $3\times$), **R3** (Thông số may đo & Thiếu ATTT Cấp độ 3 / Bàn giao mã nguồn), **R4** (Verifier số liệu tài chính), **R5** (Đủ 8 chương NĐ 214).
* **Quy trình Human-In-The-Loop (HITL) & 8 Trang Web UI:**
  * Xây dựng giao diện Streamlit hỗ trợ chuyên gia: **Focus Mode** (lọc cờ đỏ), **1-Click Quick Fix** (tự chèn *"hoặc tương đương"*), **Duyệt nhanh hàng loạt**.
* **Bảo mật & Nhật ký Kiểm toán Bất biến (Audit Log):**
  * Băm mật khẩu PBKDF2-HMAC-SHA256 (600.000 vòng lặp + Salt).
  * Cài đặt **SQL Triggers** trong SQLite chặn mọi hành vi xóa/sửa nhật ký kiểm toán.
* **Engine Xuất bản Văn bản Hành chính Công vụ:**
  * Xây dựng engine xuất tệp `.docx` đạt **100% chuẩn thể thức Nghị định 30/2020/NĐ-CP** (Quốc hiệu, Tiêu ngữ, Căn lề 30-20-20-20mm, Justified, Thụt lề 1.27cm, Header Shading, Đánh số trang động).
* **Đánh giá Toàn hệ thống & Test Suite:**
  * Thực hiện đánh giá Faithfulness & Completeness bằng phương pháp LLM-as-a-judge.
  * Quản lý bộ kiểm thử tự động **164/164 tests Passed (100%)** bằng `uv`.

#### 2. Khung Lý thuyết Deep Learning & AI Alignment Cần Nắm Vững
* **AI Safety & Anti-Hallucination Guardrails:** Numeric Consistency Verifier (R4) so khớp tập hợp số học với Whitelist thông minh triệt tiêu cảnh báo giả.
* **Đánh giá LLM-as-a-Judge:** Đo lường độ trung thực (Faithfulness = 0.94) và độ bao phủ (Completeness = 0.87) qua mô hình đánh giá độc lập.
* **Human-In-The-Loop (HITL) Design Pattern:** Nguyên lý an toàn phân định quyền hạn: AI hỗ trợ tạo bản thảo, con người chịu trách nhiệm pháp lý và phê duyệt.

#### 3. Tệp Mã Nguồn Phụ trách Trực tiếp
* `src/autotender/models/compliance.py` — Bộ quy tắc R1–R5 và `check_it_specific_compliance`.
* `src/autotender/hitl/store.py` — Quản lý trạng thái phê duyệt SQLite.
* `src/autotender/auth/` & `src/autotender/audit/` — Xác thực PBKDF2 và Audit log bất biến.
* `src/autotender/export/docx.py` & `pdf.py` — Xuất file DOCX chuẩn NĐ 30/2020.
* `src/autotender/eval/faithfulness_eval.py` — Đánh giá độ trung thực LLM-as-a-judge.
* `app/pages/` — 8 trang giao diện Streamlit.
* `tests/` — Bộ 164 bài kiểm thử tự động.

#### 4. Câu hỏi Vấn đáp Hội đồng Thường gặp & Gợi ý Trả lời
* **Q: Làm sao hệ thống đảm bảo an toàn pháp lý khi đưa vào cơ quan nhà nước thực tế?**
  * *Trả lời:* *"Thưa Thầy, hệ thống áp dụng nguyên tắc Human-In-The-Loop tuyệt đối: AI chỉ đóng vai trò trợ lý tạo dự thảo. Module Compliance Guard (M6) tự động rà soát 5 nhóm lỗi vi phạm nghiêm trọng (R1-R5). Cán bộ phụ trách sử dụng Focus Mode để sửa nhanh lỗi và bấm phê duyệt từng mục. Mọi thay đổi và danh tính người duyệt đều được ghi vết bất biến vào Audit Log (được khóa cứng bằng SQL Trigger), đảm bảo tính minh bạch và phục vụ công tác thanh tra."*

---

## 4. KỊCH BẢN PHỐI HỢP TRÌNH BÀY TRÊN SLIDE (TIMELINE 20 PHÚT)

```
┌─────────────────┬──────────────┬────────────────────────────────────────────────────────────────────────┐
│ Thời gian       │ Người trình  │ Nội dung trình bày chính (Storyline liền mạch)                         │
│                 │ bày          │                                                                        │
├─────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────┤
│ **00:00 - 04:00**│ **Người 1**  │ • Bài toán thực tế, Nỗi đau nghiệp vụ lập HSMT gói thầu CNTT/Phần mềm   │
│ (4 phút)        │ (BA/Domain)  │ • Chuyển thể Nghiệp vụ $\rightarrow$ AI: Data Contracts & Bộ quy tắc R1-R5│
├─────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────┤
│ **04:00 - 08:00**│ **Người 2**  │ • Data Engineering, Pipeline OCR VietOCR, Bộ 46 câu hỏi Benchmark     │
│ (4 phút)        │ (Data/Chunk) │ • Thuật toán LegalChunker: Bóc tách phân cấp Điều/Khoản (699 chunks)   │
├─────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────┤
│ **08:00 - 12:00**│ **Người 3**  │ • Representation Learning: Linear Attention GatedDeltaNet-2 (8K tokens)│
│ (4 phút)        │ (Vector/HNSW)│ • Qdrant Vector DB (HNSW Graph), Docker Microservice & BM25 Sparse     │
├─────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────┤
│ **12:00 - 16:00**│ **Người 4**  │ • Hybrid RRF (k=60) & Cross-Encoder Joint Attention Deep Reranking     │
│ (4 phút)        │ (LLM/Gen M5) │ • Universal LLM Gateway, In-Context Learning 8 Chương & 3-Tier Fallback│
│                 │              │ • So sánh Kiến trúc: Deterministic Orchestrator vs Agentic Tool-calling│
├─────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────┤
│ **16:00 - 20:00**│ **Người 5**  │ • Compliance Guard (R1-R5), Focus Mode HITL, Demo xuất Word NĐ 30/2020 │
│ (4 phút)        │ (Safety/Eval)│ • Đánh giá Faithfulness (0.94), Test Suite 164/164 tests & Kết luận    │
└─────────────────┴──────────────┴────────────────────────────────────────────────────────────────────────┘
```
