# CẨM NANG TOÀN DIỆN: PHÂN CÔNG NHIỆM VỤ, KHUNG LÝ THUYẾT & KỊCH BẢN BẢO VỆ ĐỒ ÁN DEEP LEARNING (5 THÀNH VIÊN)
## DỰ ÁN: AUTOTENDER-VN — HỆ THỐNG TRỢ LÝ AI SOẠN THẢO & RÀ SOÁT HỒ SƠ MỜI THẦU (E-HSMT) GÓI THẦU PHẦN MỀM/CNTT BẰNG HYBRID RAG + LLM

> **Học vị:** Thạc sĩ Kỹ thuật (Master of Engineering) — **Chuyên ngành:** Khoa học Dữ liệu & Học sâu (Deep Learning)  
> **Đề tài:** Ứng dụng Hybrid Retrieval-Augmented Generation (RAG) và Large Language Models (LLM) trong Soạn thảo và Kiểm soát Tuân thủ Hồ sơ Mời thầu Gói thầu Phần mềm/CNTT tại Việt Nam  
> **Quy mô nhóm:** 5 Thành viên  
> **Mục tiêu tài liệu:** Đây là cẩm nang chi tiết từ A-Z giúp từng thành viên trong nhóm hiểu rõ vai trò, nắm vững bản chất lý thuyết, làm chủ mã nguồn phụ trách, tự tin thuyết trình trên slide và trả lời xuất sắc mọi câu hỏi phản biện của Hội đồng chấm thi.

---

# MỤC LỤC TỔNG QUAN

1. [Sơ Đồ Kiến Trúc Toàn Diện & Luồng Xử Lý 5 Tầng (System Pipeline)](#1-sơ-đồ-kiến-trúc-toàn-diện--luồng-xử-lý-5-tầng)
2. [Bảng Ma Trận Phân Chia Trách Nhiệm 5 Thành Viên](#2-bảng-ma-trận-phân-chia-trách-nhiệm-5-thành-viên)
3. [Chi Tiết Thành Viên 1: AI Business Analyst & Legal Domain Architect](#3-chi-tiết-thành-viên-1-ai-business-analyst--legal-domain-architect)
4. [Chi Tiết Thành Viên 2: Data Curation, Benchmark Dataset & Legal Chunking Engineer](#4-chi-tiết-thành-viên-2-data-curation-benchmark-dataset--legal-chunking-engineer)
5. [Chi Tiết Thành Viên 3: Representation Learning & Vector Database (Qdrant) Engineer](#5-chi-tiết-thành-viên-3-representation-learning--vector-database-qdrant-engineer)
6. [Chi Tiết Thành Viên 4: Hybrid Reranking, Orchestrator & Generative LLM Architect](#6-chi-tiết-thành-viên-4-hybrid-reranking-orchestrator--generative-llm-architect)
7. [Chi Tiết Thành Viên 5: AI Safety Guardrails, HITL UI, Thể Thức NĐ 30 & System Evaluation](#7-chi-tiết-thành-viên-5-ai-safety-guardrails-hitl-ui-thể-thức-nđ-30--system-evaluation)
8. [Kịch Bản Thuyết Trình Slide Chi Tiết Từng Phút (Timeline 20 Phút)](#8-kịch-bản-thuyết-trình-slide-chi-tiết-từng-phút-timeline-20-phút)
9. [Chiến Thuật Phối Hợp & Ứng Biến Khi Trả Lời Phản Biện Của Hội Đồng](#9-chiến-thuật-phối-hợp--ứng-biến-khi-trả-lời-phản-biện-của-hội-đồng)

---

## 1. SƠ ĐỒ KIẾN TRÚC TOÀN DIỆN & LUỒNG XỬ LÝ 5 TẦNG

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              AUTOTENDER-VN: KIẾN TRÚC HỆ THỐNG 5 TẦNG                                  │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                        │
│ [TẦNG 1: NGHIỆP VỤ & DATA CONTRACTS] (Phụ trách: Thành viên 1)                                        │
│  ├── 8 Bộ Luật: Luật 22/2023, NĐ 214/2025, NĐ 73/2019, NĐ 82/2024, NĐ 85/2016, TT 22/2024 (341 Điều)   │
│  ├── 8 Chương E-HSMT pháp định theo Điều 26 NĐ 214 (17 mục con tiêu chuẩn)                             │
│  └── Data Contracts chuẩn hóa qua Pydantic v2: schemas.py (HSMTDocument, TenderNotice, ExtractedField) │
│                                                                                                        │
│                                           │ (Truyền cấu trúc & Quy tắc nghiệp vụ)                      │
│                                           ▼                                                            │
│ [TẦNG 2: DATA ENGINEERING & CHUNKING] (Phụ trách: Thành viên 2)                                        │
│  ├── Crawlers (dauthau.asia, muasamcong) + PDF Ingestion + VietOCR (xử lý bản scan)                   │
│  ├── Module M2 NER (Token Classification + Regex Hybrid) bóc tách 8 thực thể KHLCNT                    │
│  ├── Dataset Benchmark: 46 câu hỏi kiểm thử thực tế gán nhãn tay (Ground Truth Golden Citations)       │
│  └── Thuật toán LegalChunker: Phân cấp Parent-Child (Điều / Khoản), 699 Chunks kèm Payload Metadata   │
│                                                                                                        │
│                                           │ (Truyền 699 Chunks có cấu trúc)                            │
│                                           ▼                                                            │
│ [TẦNG 3: VECTOR DB & REPRESENTATION LEARNING] (Phụ trách: Thành viên 3)                                │
│  ├── Mô hình DeepX (dxtech-asia/deepx-embedding-v1): Linear Attention GatedDeltaNet-2, 1024d, 8K tokens│
│  ├── Vector DB Qdrant: HNSW Graph Index (Cosine Distance) + Payload Indexing (law_id, doc_type...)     │
│  └── Nhánh Lexical Search: BM25 Inverted Index (bắt chính xác mã điều, ngày tháng, số tiền)           │
│                                                                                                        │
│                                           │ (Truy vấn song song Sparse & Dense)                        │
│                                           ▼                                                            │
│ [TẦNG 4: HYBRID RERANKING & GENERATIVE LLM] (Phụ trách: Thành viên 4)                                  │
│  ├── Reciprocal Rank Fusion (RRF k=60): Dung hợp BM25 + Qdrant Dense ──► Top 20 Candidates            │
│  ├── Cross-Encoder Deep Reranking (ms-marco-MiniLM-L-6-v2): Full Joint Self-Attention ──► Top 5 Citations│
│  ├── Deterministic Orchestrator: Điều phối tuần tự 17 mục con (chống loop, kiểm soát Budget < $5)     │
│  └── Universal LLM Gateway: Claude 3.5 Sonnet / DeepSeek-V3 In-Context Grounded Generation + 3-Tier   │
│                                                                                                        │
│                                           │ (Dự thảo 8 Chương thô kèm Citations)                       │
│                                           ▼                                                            │
│ [TẦNG 5: SAFETY GUARDRAILS, HITL & EXPORT] (Phụ trách: Thành viên 5)                                   │
│  ├── Module M6 Compliance Guard: Bộ lọc 5 cờ (R1 cấm nhãn hiệu, R2 trần 3x, R3 ATTT/Mã nguồn, R4, R5)│
│  ├── Numeric Consistency Verifier (R4): Đối chiếu số liệu tài chính so với KHLCNT (Anti-Hallucination)│
│  ├── Human-In-The-Loop UI (Streamlit 8 trang): Focus Mode lọc cờ đỏ, 1-Click Quick Fix, Duyệt nhanh    │
│  ├── Bảo mật & Bất biến: PBKDF2 Password Hashing + Audit Log SQLite SQL Triggers (chống xóa/sửa)       │
│  └── Document Engine: Xuất file .DOCX đạt 100% Thể thức Nghị định 30/2020/NĐ-CP & Đánh giá (Faith=0.94│
│                                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. BẢNG MA TRẬN PHÂN CHIA TRÁCH NHIỆM 5 THÀNH VIÊN

| STT | Họ tên & Vị trí phân công | Module Codebase phụ trách | Trọng tâm Nghiệp vụ | Trọng tâm Lý thuyết Deep Learning / AI |
|:---:|---|---|---|---|
| **1** | **Thành viên 1**<br>AI Business Analyst & Legal Domain Architect | `src/autotender/schemas.py`<br>`configs/models.yaml`<br>`docs/SYSTEM_DESIGN.md`<br>`docs/DATA_CARD.md` | • Quy trình 7 bước lập HSMT của cán bộ.<br>• 8 Bộ văn bản quy phạm (341 Điều).<br>• Cấu trúc 8 chương E-HSMT Điều 26 NĐ 214.<br>• Quy định ATTT Cấp 3 (NĐ 85) & Bàn giao mã nguồn (NĐ 73). | • Problem Formulation in Applied NLP.<br>• Data Contracts Architecture (Pydantic v2).<br>• Chuyển thể quy tắc nghiệp vụ sang logic AI Guardrails (R1–R5). |
| **2** | **Thành viên 2**<br>Data Curation, Benchmark Dataset & Legal Chunking Engineer | `src/autotender/rag/chunker.py`<br>`src/autotender/models/ner.py`<br>`src/autotender/crawler/`<br>`src/autotender/ingest/`<br>`data/samples/legal_corpus/`<br>`data/eval/` | • Thu thập KHLCNT từ muasamcong & dauthau.asia.<br>• Xử lý PDF scan qua VietOCR.<br>• Xây dựng 46 câu hỏi benchmark thực tế kèm Ground Truth nhãn luật vàng. | • Token Classification (NER Transformer + Regex Hybrid).<br>• Hierarchical Syntax Chunking (Parent-Child Chunking theo Điều/Khoản).<br>• Metadata Preservation trong RAG. |
| **3** | **Thành viên 3**<br>Representation Learning & Vector Database (Qdrant) Engineer | `src/autotender/rag/embedding_models.py`<br>`src/autotender/rag/qdrant_store.py`<br>`src/autotender/rag/bm25.py`<br>`docker/embedding/`<br>`docker-compose.yml` | • Quản lý kho lưu trữ vector tập trung.<br>• Lập chỉ mục văn bản luật phục vụ tìm kiếm siêu tốc. | • Dual-Tower Bi-Encoder & Cosine Metric.<br>• Linear Attention / GatedDeltaNet-2 (8.192 tokens).<br>• Cấu trúc đồ thị HNSW (Hierarchical Navigable Small World Graph) trong Qdrant.<br>• Payload Indexing & Filtered Search. |
| **4** | **Thành viên 4**<br>Hybrid Reranking, Orchestrator & Generative LLM Architect | `src/autotender/rag/hybrid_retriever.py`<br>`src/autotender/rag/rerank.py`<br>`src/autotender/generation/llm_client.py`<br>`src/autotender/models/generator.py`<br>`src/autotender/models/orchestrator.py` | • Soạn thảo tự động 8 chương HSMT chuyên sâu CNTT.<br>• Viện dẫn chính xác điều khoản luật thật. | • Reciprocal Rank Fusion (RRF $k=60$).<br>• Cross-Encoder Full Joint Self-Attention Reranker.<br>• So sánh: Deterministic Orchestrator vs Agentic Tool-calling.<br>• In-Context Learning (Grounded RAG) & 3-Tier Degradation. |
| **5** | **Thành viên 5**<br>AI Safety Guardrails, HITL UI, Thể Thức NĐ 30 & System Evaluation | `src/autotender/models/compliance.py`<br>`src/autotender/hitl/store.py`<br>`src/autotender/auth/`<br>`src/autotender/audit/`<br>`src/autotender/export/docx.py`<br>`src/autotender/eval/faithfulness_eval.py`<br>`app/pages/` | • Kiểm soát rủi ro pháp lý (chống chỉ định nhãn hiệu, trần doanh thu, chống cài cắm).<br>• Trải nghiệm cán bộ thẩm định (HITL).<br>• Xuất file Word chuẩn 100% công vụ NĐ 30/2020. | • AI Safety, Guardrails & Anti-Hallucination.<br>• Numeric Consistency Verifier (R4) với Whitelist filtering.<br>• LLM-as-a-Judge (Faithfulness & Completeness).<br>• SQLite Trigger-based Immutability (Bảo mật kiểm toán). |

---

## 3. CHI TIẾT THÀNH VIÊN 1: AI BUSINESS ANALYST & LEGAL DOMAIN ARCHITECT
*(Kiến trúc sư Nghiệp vụ Đấu thầu & Chuyển thể Bài toán Con người $\rightarrow$ AI)*

### 3.1. Bối cảnh & "Nỗi đau" Nghiệp vụ Giải quyết
Khi một cơ quan nhà nước chuẩn bị đấu thầu một gói thầu CNTT (ví dụ: *Hệ thống Bệnh án Điện tử EMR, Cổng Dịch vụ công, Phần mềm Quản trị số*), chuyên viên đấu thầu phải đối mặt với:
1. **Khối lượng soạn thảo khổng lồ:** Bộ hồ sơ E-HSMT mẫu gồm **8 Chương, dài từ 100 đến 250 trang**.
2. **Giao thoa nhiều tầng pháp lý phức tạp:**
   * *Luật Đấu thầu số 22/2023/QH15* (có hiệu lực từ 01/01/2024, được sửa đổi bởi Luật 57/2024 và Luật 90/2025).
   * *Nghị định 214/2025/NĐ-CP* (có hiệu lực từ 04/08/2025, thay thế hoàn toàn Nghị định 24/2024/NĐ-CP đã hết hiệu lực). Quy định chi tiết nội dung 8 chương HSMT gói thầu mua sắm hàng hóa/phần mềm.
   * *Nghị định 73/2019/NĐ-CP & Nghị định 82/2024/NĐ-CP:* Quy định quản lý đầu tư ứng dụng CNTT sử dụng vốn ngân sách nhà nước, xác định phần mềm nội bộ, kiểm thử UAT và nghiệm thu.
   * *Nghị định 85/2016/NĐ-CP & TCVN 11930:2017:* Quy định bắt buộc về bảo đảm an toàn hệ thống thông tin theo 5 cấp độ.
   * *Thông tư 22/2024/TT-BKHĐT:* Mẫu hồ sơ mời thầu điện tử (E-HSMT) chuẩn quốc gia.
3. **Rủi ro hình sự & pháp lý nếu "cài cắm tiêu chí":**
   * Khoản 2 Điều 44 Luật Đấu thầu 22/2023 nghiêm cấm nêu nhãn hiệu cụ thể (*"Oracle", "Dell", "Cisco"...*) mà không kèm cụm từ *"hoặc tương đương"*.
   * Đưa ra yêu cầu doanh thu $> 3\times$ giá gói thầu sẽ bị coi là hành vi hạn chế sự tham gia của nhà thầu.
   * Quên điều khoản bàn giao mã nguồn sẽ dẫn đến nguy cơ nhà thầu chiếm giữ độc quyền công nghệ (Vendor Lock-in).

### 3.2. Công việc Cụ thể Đã Làm (Deliverables)
* **Xây dựng Data Contracts chuẩn (`src/autotender/schemas.py`):**
  * Định nghĩa cấu trúc `TenderNotice` gồm 14 trường thông tin trích xuất từ KHLCNT.
  * Định nghĩa `HSMTSection` quản lý vòng đời trạng thái của từng mục (`draft` $\rightarrow$ `edited` $\rightarrow$ `approved` $\rightarrow$ `rejected`).
  * Định nghĩa `HSMTDocument` đóng gói toàn bộ 8 chương và tiến độ duyệt (`approval_progress`).
  * Định nghĩa `ComplianceFlag` mô tả vi phạm kèm mức độ nghiêm trọng (`cao`, `trung_binh`, `thap`), câu vi phạm và chứng cứ luật.
* **Chuyển thể Cấu trúc 8 Chương Pháp định:**
  * Ánh xạ Điều 26 Nghị định 214/2025/NĐ-CP thành 17 mục con chi tiết trong `configs/models.yaml`:
    * *Chương I:* Chỉ dẫn nhà thầu (Quy định chung, hành vi bị cấm).
    * *Chương II:* Bảng dữ liệu đấu thầu (Giá gói thầu, phương thức lựa chọn, bảo đảm dự thầu).
    * *Chương III:* Tiêu chuẩn đánh giá (Năng lực tài chính, nhân sự chủ chốt CNTT: PMP, Solution Architect, CISSP).
    * *Chương IV:* Biểu mẫu dự thầu.
    * *Chương V:* Yêu cầu kỹ thuật phần mềm (Đặc tả SRS, API RESTful/gRPC, ATTT Cấp độ 3 theo NĐ 85, cam kết SLA 24/7).
    * *Chương VI & VII:* Điều kiện chung và Điều kiện cụ thể của Hợp đồng (Bắt buộc bàn giao 100% mã nguồn theo Điều 55 NĐ 73).
    * *Chương VIII:* Biểu mẫu hợp đồng.
* **Thiết kế Logic 5 Bộ Cờ Tuân thủ (R1–R5):**
  * R1: Kiểm tra nhãn hiệu thương mại độc quyền.
  * R2: Kiểm tra tỷ lệ doanh thu tài chính bất hợp lý.
  * R3: Kiểm tra thông số may đo độc quyền & thiếu điều khoản bắt buộc về CNTT.
  * R4: Kiểm tra tính nhất quán số học giữa dự thảo và KHLCNT.
  * R5: Kiểm tra tính đầy đủ của 8 chương pháp định.

### 3.3. Khung Lý thuyết Cần Nắm Vững
* **Problem Formulation in Applied NLP:** Cách phân rã một quy trình công việc phức tạp thành chuỗi bài toán Deep Learning (Extraction $\rightarrow$ Dense Retrieval $\rightarrow$ Constrained Generation $\rightarrow$ Rule-based Verification).
* **Data Contracts Architecture:** Nguyên lý thiết kế Schema bất biến (Pydantic v2) đảm bảo tính toàn vẹn dữ liệu xuyên suốt hệ thống microservices.

### 3.4. Mã Nguồn Cốt Lõi Phụ Trách
* [`src/autotender/schemas.py`](src/autotender/schemas.py): `TenderNotice`, `ExtractedField`, `ComplianceFlag`, `HSMTSection`, `HSMTDocument`.
* [`configs/models.yaml`](configs/models.yaml): `sections_supported`, `compliance.brand_dictionary`, `compliance.revenue_ratio_threshold`, `it_specifications`.

### 3.5. Bộ Câu Hỏi Vấn Đáp Phản Biện Từ Hội Đồng (Q&A)
* **Q1: Vai trò của một Business Analyst trong đồ án Deep Learning là gì, tại sao không để kỹ sư tự code hết?**
  * *Trả lời:* *"Thưa Thầy, một mô hình Deep Learning/LLM dù mạnh đến đâu nhưng nếu nạp vào dữ liệu rác hoặc không có ràng buộc nghiệp vụ chính xác thì kết quả sinh ra hoàn toàn vô giá trị trong thực tế (Garbage in, Garbage out). Vai trò của em là định nghĩa bài toán (Problem Formulation), thiết kế cấu trúc dữ liệu chuẩn Pydantic để các module giao tiếp không bị lỗi, và xây dựng bộ quy tắc rà soát pháp lý R1–R5. Nếu không có khâu này, các bạn kỹ sư sẽ không biết phải chunk văn bản luật ra sao, truy xuất điều gì và ràng buộc LLM như thế nào."*
* **Q2: Tại sao nhóm lại sử dụng Nghị định 214/2025/NĐ-CP mà không dùng Nghị định 24/2024/NĐ-CP như đề cương ban đầu?**
  * *Trả lời:* *"Thưa Thầy, đây là điểm cập nhật pháp lý cực kỳ quan trọng của nhóm. Nghị định 24/2024/NĐ-CP đã chính thức hết hiệu lực và được thay thế bởi Nghị định 214/2025/NĐ-CP (có hiệu lực từ 04/8/2025). Cấu trúc 8 chương E-HSMT hiện nay được quy định chi tiết tại Điều 26 Khoản 2 Nghị định 214. Nhóm đã chủ động cập nhật toàn bộ cơ sở pháp lý mới nhất để đảm bảo hệ thống sinh ra hồ sơ có giá trị pháp lý thực tế hiện hành."*
* **Q3: Các điều khoản đặc thù về CNTT trong Chương V và Chương VI được nhóm đưa vào dựa trên căn cứ pháp lý nào?**
  * *Trả lời:* *"Thưa Thầy, các gói thầu phần mềm nhà nước có 2 rủi ro lớn nhất: (1) Rủi ro mất an toàn thông tin: Nhóm căn cứ vào Nghị định 85/2016/NĐ-CP quy định bắt buộc phần mềm phải đạt An toàn thông tin Cấp độ 3, mã hóa TLS 1.3 và rà quét lỗ hổng trước nghiệm thu; (2) Rủi ro khóa nhà cung cấp (Vendor Lock-in): Nhóm căn cứ vào Điều 55 Nghị định 73/2019/NĐ-CP bắt buộc hợp đồng phải có điều khoản Chủ đầu tư sở hữu 100% mã nguồn sạch (Clean Source Code) và cơ sở dữ liệu sau khi nghiệm thu."*

---

## 4. CHI TIẾT THÀNH VIÊN 2: DATA CURATION, BENCHMARK DATASET & LEGAL CHUNKING ENGINEER
*(Kỹ sư Dữ liệu, Xây dựng Bộ Đánh giá & Phân đoạn Pháp luật)*

### 4.1. Bối cảnh & Thách thức Dữ liệu Pháp lý
1. **Văn bản pháp luật không phải là văn bản thông thường:** Văn bản luật có cấu trúc phân cấp nghiêm ngặt: Chương $\rightarrow$ Mục $\rightarrow$ Điều $\rightarrow$ Khoản $\rightarrow$ Điểm. Nếu dùng các công cụ cắt văn bản thông thường (cắt theo 500 ký tự), câu văn sẽ bị đứt đôi giữa chừng, mất liên kết giữa phần "chế tài" và "số Điều", khiến vector embedding bị sai lệch ngữ nghĩa.
2. **Thiếu tập dữ liệu đánh giá chuẩn (Evaluation Benchmark):** Để chứng minh hệ thống RAG hoạt động tốt, bắt buộc phải có một tập câu hỏi kiểm thử thực tế có gán nhãn tay căn cứ luật chuẩn (Ground Truth Golden Citations).
3. **Đa dạng định dạng đầu vào của KHLCNT:** Văn bản quyết định phê duyệt kế hoạch lựa chọn nhà thầu tải từ cổng thông tin có thể là PDF văn bản (native PDF) hoặc PDF quét ảnh (scanned PDF), đòi hỏi pipeline OCR tiếng Việt chuẩn xác.

### 4.2. Công việc Cụ thể Đã Làm (Deliverables)
* **Xây dựng Kho Tri thức 8 Bộ Luật (`data/samples/legal_corpus/`):**
  * Thu thập, làm sạch và chuẩn hóa NFC toàn bộ 341 Điều luật thành các bản ghi JSON Lines (`LegalArticle`):
    * `luat_22_2023_qh15.jsonl`: 90 Điều.
    * `nd_214_2025_ndcp.jsonl`: 145 Điều.
    * `nd_45_2026_ndcp.jsonl`: 43 Điều.
    * `nd_73_2019_ndcp.jsonl`: 8 Điều trọng điểm CNTT.
    * `nd_82_2024_ndcp.jsonl`: 3 Điều sửa đổi NĐ 73.
    * `nd_85_2016_ndcp.jsonl`: 4 Điều về 5 cấp độ ATTT.
    * `tt_01_2024_bkhdt.jsonl`: 22 Điều.
    * `tt_22_2024_bkhdt.jsonl`: 26 Điều.
* **Phát triển Thuật toán Phân đoạn Pháp lý Phân cấp (`LegalChunker`):**
  * Tạo ra **699 Chunks** theo cơ chế Parent-Child:
    * Điều luật ngắn ($\le 1.000$ từ): Giữ nguyên làm 1 Parent Chunk để bảo toàn trọn vẹn ngữ cảnh.
    * Điều luật dài ($> 1.000$ từ, ví dụ Điều 26 NĐ 214): Tách thành từng Khoản độc lập (Child Chunk) nhưng luôn ghép tiêu đề của Điều vào đầu chunk:
      $$\text{Chunk Text} = \text{"["} + \text{law\_name} + \text{ " - Điều "} + \text{dieu\_so} + \text{": "} + \text{dieu\_title} + \text{"]\nKhoản "} + \text{khoan\_so} + \text{..."}$$
  * Tính toán ID xác định bằng **UUID5 (Deterministic UUID)** dựa trên chuỗi `(law_id, dieu_so, khoan_so)` để đảm bảo tính bất biến (Idempotent), nạp lại dữ liệu không bao giờ bị trùng lặp.
* **Xây dựng Bộ Benchmark 46 Câu Hỏi Đánh Giá (`data/eval/eval_queries.jsonl`):**
  * 46 câu hỏi tình huống thực tế bao phủ 8 chương HSMT, gán nhãn tay danh sách các `chunk_id` vàng làm căn cứ pháp lý phục vụ tính toán Recall, MRR, nDCG.
* **Xây dựng Module M2 NER & Pipeline Ingestion (`src/autotender/models/ner.py`):**
  * Kết hợp PyMuPDF cho PDF điện tử và **VietOCR** (kiến trúc CNN ResNet + Transformer Seq2Seq) cho bản scan ảnh.
  * Trích xuất chính xác 8 trường thực thể: `PACKAGE_NAME`, `VALUE`, `FUNDING`, `METHOD`, `CONTRACT_TYPE`, `DURATION`, `INVESTOR`, `LOCATION`.

### 4.3. Khung Lý thuyết Cần Nắm Vững
* **Hierarchical Legal Syntax Chunking:** Bản chất của việc phân đoạn văn bản theo cú pháp luật học thay vì ranh giới token ngẫu nhiên.
* **Named Entity Recognition (NER) & Token Classification:** Gán nhãn chuỗi thực thể theo định dạng BIO/IOB bằng Pretrained Transformers kết hợp biểu thức chính quy (Regex Slot-filling).
* **Optical Character Recognition (OCR) cho Tiếng Việt:** Kiến trúc tích hợp giữa Convolutional Neural Network (trích xuất đặc trưng thị giác) và Transformer Decoder (sinh chuỗi ký tự tiếng Việt có dấu).

### 4.4. Mã Nguồn Cốt Lõi Phụ Trách
* [`src/autotender/rag/chunker.py`](src/autotender/rag/chunker.py): `LegalChunker`, `chunk_legal_article`, `chunk_legal_corpus_dir`.
* [`src/autotender/models/ner.py`](src/autotender/models/ner.py): `NerModule.extract_fields`, `_extract_by_regex`, `_extract_package_value`.
* [`src/autotender/ingest/pdf_reader.py`](src/autotender/ingest/pdf_reader.py) & [`ocr.py`](src/autotender/ingest/ocr.py): Pipeline xử lý PDF và VietOCR.

### 4.5. Bộ Câu Hỏi Vấn Đáp Phản Biện Từ Hội Đồng (Q&A)
* **Q1: Tại sao không dùng RecursiveCharacterTextSplitter của LangChain mà phải tự viết `LegalChunker`?**
  * *Trả lời:* *"Thưa Thầy, RecursiveCharacterTextSplitter chia cắt văn bản dựa trên độ dài ký tự và các dấu xuống dòng thuần túy. Trong văn bản luật, một Khoản có thể bị cắt làm đôi nếu vượt quá `chunk_size`, làm mất chủ ngữ pháp lý ở nửa sau. `LegalChunker` của nhóm phân đoạn theo biên giới ngữ nghĩa pháp lý thực sự: bóc tách chính xác từng Khoản, và luôn tự động gán nhãn tiêu đề của Điều vào đầu mỗi chunk. Nhờ đó, vector embedding luôn mang đầy đủ thông tin về chủ thể và chế tài pháp lý."*
* **Q2: Cơ chế sinh ID bằng UUID5 có ý nghĩa gì trong hệ thống?**
  * *Trả lời:* *"Thưa Thầy, UUID5 là giải thuật băm SHA-1 có không gian tên (namespace). Khi tạo chunk, ID được sinh ra từ chuỗi kết hợp `(law_id, dieu_so, khoan_so)`. Điều này đảm bảo tính tất định (Idempotency): dù nhóm có chạy lại script nạp dữ liệu hàng trăm lần thì ID của Điều 8 Khoản 3 Nghị định 85 vẫn luôn là duy nhất một mã cố định. Qdrant sẽ thực hiện thao tác Upsert (cập nhật nếu đã có, thêm mới nếu chưa) mà không bao giờ bị nhân đôi dữ liệu (Zero Duplication)."*
* **Q3: Bộ dữ liệu 46 câu hỏi benchmark được xây dựng theo tiêu chuẩn nào?**
  * *Trả lời:* *"Thưa Thầy, tập 46 câu hỏi được xây dựng mô phỏng các tình huống thực tế của cán bộ lập hồ sơ thầu, trải dài trên 4 nhóm nghiệp vụ chính: (1) Tư cách hợp lệ và cấm chỉ định thương hiệu; (2) Tiêu chuẩn năng lực tài chính và doanh thu; (3) Yêu cầu kỹ thuật, SLA và an toàn thông tin; (4) Điều khoản hợp đồng và quyền sở hữu mã nguồn. Mỗi câu hỏi đều được nhóm đối chiếu và gán nhãn danh sách các `chunk_id` bắt buộc phải tìm thấy làm Ground Truth để phục vụ đánh giá khoa học."*

---

## 5. CHI TIẾT THÀNH VIÊN 3: REPRESENTATION LEARNING & VECTOR DATABASE (QDRANT) ENGINEER
*(Kỹ sư Không gian Biểu diễn Vector & Cơ sở Dữ liệu Vector Qdrant)*

### 5.1. Bối cảnh & Thách thức Biểu Diễn Ngữ Nghĩa
1. **Hạn chế 512 tokens của các mô hình BERT truyền thống:** Các mô hình như `vietnamese-bi-encoder` (RoBERTa) có độ phức tạp tính toán Self-Attention là $O(N^2)$, do đó bị giới hạn cứng ở 512 tokens. Nhiều Điều luật dài 1.000–1.500 từ sẽ bị cắt cụt đuôi (truncation), làm mất các quy định quan trọng ở cuối Điều.
2. **Cần mô hình nhúng hỗ trợ ngữ cảnh dài (Long-Context Embedding):** Cần một mô hình nhúng hiện đại có khả năng nạp trọn vẹn văn bản dài mà vẫn duy trì tốc độ tìm kiếm mili-giây.
3. **Quản lý Vector DB ở quy mô lớn:** Cần một cơ sở dữ liệu vector chuyên dụng hỗ trợ tìm kiếm xấp xỉ siêu tốc (ANN Search) kết hợp lọc siêu dữ liệu (Payload Filtering).

### 5.2. Công việc Cụ thể Đã Làm (Deliverables)
* **Nghiên cứu & Triển khai Mô hình `dxtech-asia/deepx-embedding-v1` (1024 chiều):**
  * Ứng dụng kiến trúc **Linear Attention (GatedDeltaNet-2)** cho phép mở rộng cửa sổ ngữ cảnh lên tới **8.192 tokens** mà không làm bùng nổ bộ nhớ RAM/VRAM.
  * So sánh thực nghiệm với `vietnamese-bi-encoder` (768 chiều, giới hạn 512 tokens) và lưu trữ kết quả trong `reports/embedding_comparison.json`.
* **Đóng gói Microservice Nhúng Container (`docker/embedding/`):**
  * Xây dựng FastAPI service phục vụ mô hình nhúng độc lập tại cổng `8080`, tối ưu hóa đa luồng CPU (12 workers PyTorch).
* **Kiến trúc Lưu Trữ Vector DB Qdrant (`src/autotender/rag/qdrant_store.py`):**
  * Thiết kế Collection trung tâm `legal_chunks` chứa 699 vector points 1024 chiều.
  * Cấu hình đồ thị **HNSW (Hierarchical Navigable Small World)** với hàm khoảng cách **Cosine Distance**.
  * Thiết lập **Payload Indexing** trên các trường: `law_id`, `doc_type`, `dieu_so`, `khoan_so` để phục vụ cơ chế **Single-Stage Filtered Search** (lọc trực tiếp ngay trong quá trình duyệt đồ thị).
* **Xây dựng Nhánh Lexical Search BM25 (`src/autotender/rag/bm25.py`):**
  * Triển khai BM25 Inverted Index song song để bắt chính xác các từ khóa số hiệu luật, số tiền và mốc thời gian.

### 5.3. Khung Lý thuyết Cần Nắm Vững
* **Dual-Tower Bi-Encoder Architecture:**
  * Mã hóa độc lập câu truy vấn $\mathbf{u} = \text{Encoder}(q)$ và đoạn văn bản $\mathbf{v} = \text{Encoder}(d)$ qua phép Mean Pooling và chuẩn hóa $L_2$.
  * Hàm đo độ tương đồng Cosine Similarity:
    $$\text{Cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \sum_{i=1}^{1024} u_i v_i \quad (\text{khi } \|\mathbf{u}\| = \|\mathbf{v}\| = 1)$$
* **Linear Attention & GatedDeltaNet-2 trong DeepX:**
  * *Tại sao Vanilla Transformer bị nghẽn?* Công thức Attention chuẩn $\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ có độ phức tạp thời gian và không gian là $O(N^2)$, gây tràn bộ nhớ khi $N > 512$.
  * *Cơ chế Linear Attention:* Thay thế Softmax bằng phép phân rã hàm nhân tử (Kernel Feature Map) $\phi(Q)\phi(K)^T V$, cho phép đổi thứ tự nhân ma trận thành $\phi(Q) \left(\phi(K)^T V\right)$, đưa độ phức tạp về tuyến tính **$O(N)$** với trạng thái bộ nhớ hồi quy (Recurrent State), giúp nhúng trọn vẹn văn bản dài 8.192 tokens.
* **Cấu trúc Đồ thị HNSW (Hierarchical Navigable Small World Graph):**
  * Thuật toán tìm kiếm láng giềng gần nhất xấp xỉ (Approximate Nearest Neighbor - ANN) xây dựng đồ thị nhiều tầng (tương tự cấu trúc Skip-list). Tầng trên cùng thưa để nhảy bước lớn, tầng đáy dày để hội tụ chính xác, đạt độ phức tạp tìm kiếm **$O(\log N)$**.

### 5.4. Mã Nguồn Cốt Lõi Phụ Trách
* [`src/autotender/rag/embedding_models.py`](src/autotender/rag/embedding_models.py): `DeepXContainerClient`, `SentenceTransformerEncoder`, `encode_texts`.
* [`src/autotender/rag/qdrant_store.py`](src/autotender/rag/qdrant_store.py): `QdrantLegalStore`, `ensure_collection`, `upsert_chunks`, `search`.
* [`src/autotender/rag/bm25.py`](src/autotender/rag/bm25.py): `BM25Index`, `tokenize_vn`.
* [`docker/embedding/server.py`](docker/embedding/server.py) & [`docker-compose.yml`](docker-compose.yml).

### 5.5. Bộ Câu Hỏi Vấn Đáp Phản Biện Từ Hội Đồng (Q&A)
* **Q1: Tại sao nhóm lại gom tất cả 8 bộ luật vào 1 Collection trong Qdrant mà không tách mỗi luật thành 1 Collection riêng?**
  * *Trả lời:* *"Thưa Thầy, đây là quyết định thiết kế kiến trúc rất quan trọng: (1) Nghiệp vụ soạn thảo HSMT đòi hỏi truy xuất giao thoa liên văn bản (Cross-Law Retrieval) — ví dụ khi soạn Chương V phần mềm, hệ thống phải lấy đồng thời căn cứ từ NĐ 73 và NĐ 85. Nếu tách thành nhiều collections, hệ thống phải query phân tán 8 lần song song và điểm Cosine giữa các collections độc lập không thể so sánh trực tiếp với nhau được; (2) Gom vào 1 collection giúp duyệt đồ thị HNSW toàn cục chỉ mất $O(\log N)$; (3) Khi cần lọc riêng từng luật, nhóm sử dụng tính năng Payload Indexing của Qdrant để lọc ngay trong quá trình duyệt đồ thị với độ trễ dưới 2 mili-giây."*
* **Q2: Cơ chế Linear Attention của DeepX khác gì so với Attention truyền thống của BERT?**
  * *Trả lời:* *"Thưa Thầy, BERT sử dụng Softmax Self-Attention toàn cục, ma trận chú ý có kích thước $N \times N$, khi độ dài văn bản tăng gấp đôi thì chi phí bộ nhớ tăng gấp 4 lần ($O(N^2)$). DeepX sử dụng kiến trúc GatedDeltaNet-2 với Linear Attention: bỏ hàm Softmax và thay bằng phép nhân ma trận kết hợp cổng chọn lọc (gating), duy trì một ma trận trạng thái cố định (Hidden State Matrix). Nhờ đó, chi phí tính toán chỉ tăng tuyến tính $O(N)$, cho phép mô hình đọc mượt mà các Điều luật dài tới 8.192 tokens mà không bị tràn RAM."*
* **Q3: HNSW Index trong Qdrant đánh đổi điều gì để đạt tốc độ tìm kiếm mili-giây?**
  * *Trả lời:* *"Thưa Thầy, HNSW là thuật toán tìm kiếm xấp xỉ (ANN), đánh đổi một phần rất nhỏ độ chính xác tuyệt đối (Recall ~99% thay vì 100% của Exact Search) và tốn thêm dung lượng RAM để lưu các cạnh đồ thị đa tầng, đổi lại tốc độ tìm kiếm nhanh hơn từ 100 đến 1.000 lần so với việc quét tuần tự (Flat Index) qua toàn bộ vector."*

---

## 6. CHI TIẾT THÀNH VIÊN 4: HYBRID RERANKING, ORCHESTRATOR & GENERATIVE LLM ARCHITECT
*(Kỹ sư Tái xếp hạng, Điều phối Hệ thống & Mô hình Ngôn ngữ Lớn Sinh Dự thảo)*

### 6.1. Bối cảnh & Thách thức Tầng Sinh Ngôn Ngữ
1. **Hiện tượng ảo giác luật (Legal Hallucination):** Nếu để LLM sinh văn bản tự do không có căn cứ kẹp kèm (Ungrounded Generation), mô hình rất dễ bịa ra số Điều không có thật, viện dẫn văn bản đã hết hiệu lực (như NĐ 24 cũ), hoặc sinh sai các con số tài chính của gói thầu.
2. **Hạn chế của Dense Retrieval đơn lẻ:** Tìm kiếm vector chỉ bắt được ngữ nghĩa gần đúng, rất dễ bỏ sót các từ khóa số hiệu luật chính xác tuyệt đối (*"Điều 44"*, *"Nghị định 85"*, *"3 lần"*).
3. **Rủi ro của Agentic Tool-calling tự do trong khối công quyền:** Một Autonomous Agent tự động gọi tool theo vòng lặp ReAct rất dễ bị rơi vào vòng lặp vô tận (infinite loop), sinh thiếu các chương mục bắt buộc theo luật định, và làm cạn kiệt ngân sách API.

### 6.2. Công việc Cụ thể Đã Làm (Deliverables)
* **Xây dựng Thuật toán Dung hợp Thứ hạng Reciprocal Rank Fusion (RRF $k=60$):**
  * Hợp nhất kết quả từ nhánh Dense Search (Qdrant) và Sparse Search (BM25) để chọn ra Top 20 ứng viên tốt nhất.
* **Tích hợp Mô hình Cross-Encoder Deep Reranking (`src/autotender/rag/rerank.py`):**
  * Ứng dụng `cross-encoder/ms-marco-MiniLM-L-6-v2` chấm điểm tương quan cặp $(q, d)$ với cơ chế Full Joint Self-Attention để lọc tinh từ Top 20 xuống **Top 5 đoạn trích xác đáng nhất**.
* **Xây dựng Universal OpenAI-compatible LLM Gateway (`src/autotender/generation/llm_client.py`):**
  * Hỗ trợ kết nối linh hoạt đa nhà cung cấp (Claude 3.5 Sonnet, Claude 4.5, DeepSeek-V3, GPT-4o).
  * Tích hợp cơ chế **Budget Guard** (chặn trần chi phí $\$5.0$/session) và **Exponential Backoff Retry** (thư viện `tenacity` thử lại tối đa 3 lần khi mạng chập chờn).
* **Kiến trúc Điều phối Deterministic Orchestrator & Generator M5 (`src/autotender/models/generator.py`):**
  * Thiết kế cấu trúc điều phối tuần tự 17 mục con của 8 chương HSMT.
  * Kỹ thuật **In-Context Grounded Prompting**: Ép Top-5 trích dẫn luật thật vào prompt, kèm yêu cầu nghiêm ngặt trích dẫn rõ số Điều/Khoản.
  * Kỹ thuật **Slot-filling**: Tự động điền các thông số tài chính, thời gian, tên gói thầu từ `ExtractedField` vào các biểu mẫu.
* **Cơ chế Phục hồi Lỗi 3 Tầng (3-Tier Graceful Degradation):**
  * Tier 1: Cloud LLM Gateway + Hybrid RAG (Đường chính chất lượng cao nhất).
  * Tier 2: Local Dense RAG (Dự phòng tương lai).
  * Tier 3: Local BM25 + Offline Template Slot-filling (Luôn chạy được 100%, không cần kết nối mạng hay API key).

### 6.3. Khung Lý thuyết Cần Nắm Vững
* **Bản chất Khác biệt Bi-Encoder vs Cross-Encoder:**
  * *Bi-Encoder (Tầng quét thô):* Câu hỏi $q$ và văn bản $d$ đi qua 2 tháp Transformer riêng biệt. Không có sự tương tác token giữa $q$ và $d$. Tốc độ nhanh $O(1)$ nhờ Vector DB nhưng độ tương quan chỉ mang tính xấp xỉ không gian.
  * *Cross-Encoder (Tầng lọc tinh):* Đưa chuỗi ghép đôi $\text{[CLS]} \circ q \circ \text{[SEP]} \circ d \circ \text{[SEP]}$ vào cùng một mạng Transformer. Toàn bộ các tầng **Self-Attention** cho phép từng từ trong câu hỏi tương tác trực tiếp với từng từ trong tài liệu. Độ chính xác tương quan vượt trội hoàn toàn.
* **Thuật toán Reciprocal Rank Fusion (RRF):**
  * Công thức tổng hợp dựa trên thứ hạng (Rank-based Fusion) thay vì điểm số thô:
    $$RRF(d) = \sum_{m \in \{\text{Dense}, \text{BM25}\}} \frac{1}{k + r_m(d)} \quad (k = 60)$$
  * Khắc phục triệt để vấn đề lệch thang đo điểm (Scale Incompatibility) giữa điểm BM25 (dương vô cùng) và điểm Cosine ($[-1, 1]$).
* **So sánh Kiến trúc: Deterministic Orchestrator vs Autonomous Agentic Tool-calling:**
  * Giải thích tại sao đối với hệ thống soạn thảo văn bản hành chính công quyền, một quy trình điều phối tất định có kiểm soát (Deterministic Pipeline) kết hợp Human-In-The-Loop vượt trội hoàn toàn về độ tin cậy so với mô hình Agentic tự gọi tool ngẫu nhiên.

### 6.4. Mã Nguồn Cốt Lõi Phụ Trách
* [`src/autotender/rag/hybrid_retriever.py`](src/autotender/rag/hybrid_retriever.py): `HybridRetriever.retrieve`, `_fuse_rrf`.
* [`src/autotender/rag/rerank.py`](src/autotender/rag/rerank.py): `CrossEncoderReranker.rerank`.
* [`src/autotender/generation/llm_client.py`](src/autotender/generation/llm_client.py): `LLMGatewayClient.generate_text`, `BudgetGuard`.
* [`src/autotender/models/generator.py`](src/autotender/models/generator.py): `GeneratorModule`, `SECTION_DEFINITIONS`, `_SYSTEM_PROMPT`.

### 6.5. Bộ Câu Hỏi Vấn Đáp Phản Biện Từ Hội Đồng (Q&A)
* **Q1: Tại sao nhóm không dùng kiến trúc Agentic LLM (như LangChain ReAct Agent hay AutoGPT) để tự động gọi tool tìm kiếm?**
  * *Trả lời:* *"Thưa Thầy, đây là quyết định kiến trúc mang tính sống còn của dự án: (1) Tính tất định (Determinism): Văn bản pháp lý công quyền bắt buộc phải tuân thủ nghiêm ngặt 8 chương theo Điều 26 NĐ 214. Một Agent tự do rất dễ bị ảo giác (hallucinate), gọi tool lệch hướng, hoặc sinh thiếu các chương mục quan trọng; (2) Kiểm soát ngân sách & độ trễ: Agentic loop tiêu tốn số lượng token rất lớn và không dự đoán được chi phí. Nhóm thiết kế Deterministic Orchestrator kiểm soát chính xác 17 mục con, có Budget Guard chặn trần $5 và tích hợp sẵn 3-Tier Degradation giúp hệ thống luôn hoạt động ổn định và tin cậy tuyệt đối."*
* **Q2: Thuật toán Reciprocal Rank Fusion (RRF) hoạt động như thế nào và tại sao hằng số $k=60$ lại được chọn?**
  * *Trả lời:* *"Thưa Thầy, điểm số của BM25 phụ thuộc vào độ dài tài liệu và tần suất từ (không bị chặn trên), trong khi điểm Cosine của Qdrant nằm trong khoảng từ -1 đến 1. Nếu cộng điểm trực tiếp sẽ làm sai lệch kết quả. RRF chuyển đổi điểm số thành thứ hạng: tài liệu đứng thứ nhất nhận điểm $1/(60+1)$, tài liệu đứng thứ nhì nhận $1/(60+2)$. Hằng số $k=60$ là tham số chuẩn được chứng minh qua thực nghiệm bởi Cormack et al., giúp làm mịn phân phối thứ hạng, ngăn các tài liệu xếp cao ở một nhánh đơn lẻ lấn át các tài liệu xuất hiện đều ở cả 2 nhánh."*
* **Q3: Cơ chế chống ảo giác (Anti-Hallucination) trong prompt sinh văn bản được nhóm cài đặt ra sao?**
  * *Trả lời:* *"Thưa Thầy, nhóm áp dụng kỹ thuật Grounded In-Context Learning: (1) Trong System Prompt, ép buộc LLM đóng vai trò thẩm định viên và chỉ được phép sử dụng dữ kiện nằm trong khối `<context>` chứa Top-5 căn cứ luật thật từ Cross-Encoder; (2) Bắt buộc mọi câu khẳng định phải có trích dẫn kèm số Điều và tên văn bản; (3) Các số liệu tài chính của gói thầu được bảo vệ bằng cơ chế Slot-filling, không để LLM tự tính toán ngẫu nhiên. Nhờ đó, điểm độ trung thực (Faithfulness) đo được đạt mức xuất sắc 0.94."*

---

## 7. CHI TIẾT THÀNH VIÊN 5: AI SAFETY GUARDRAILS, HITL UI, THỂ THỨC NĐ 30 & SYSTEM EVALUATION
*(Kỹ sư An toàn AI, Giao diện Chuyên gia & Đánh giá Hệ thống Toàn diện)*

### 7.1. Bối cảnh & Thách thức An Toàn AI và Ứng Dụng Thực Tế
1. **AI không thể tự chịu trách nhiệm pháp lý:** Trong quản lý nhà nước, nếu hồ sơ mời thầu cài cắm tiêu chí vi phạm pháp luật dẫn đến khiếu nại, thanh tra hoặc hủy thầu thì cán bộ ký duyệt phải chịu trách nhiệm trước pháp luật, AI không thể ký thay con người.
2. **Hiện tượng ảo giác số liệu (Numeric Hallucination):** LLM có thể vô tình sinh nhầm số tiền bảo đảm dự thầu hoặc thời gian thực hiện hợp đồng so với quyết định KHLCNT gốc.
3. **Tiêu chuẩn thể thức văn bản hành chính khắt khe:** Tài liệu xuất ra để ban hành phải tuân thủ 100% quy chuẩn trình bày văn bản hành chính nhà nước theo **Nghị định số 30/2020/NĐ-CP** (từ font chữ, cỡ chữ, căn lề, thụt đầu dòng, quốc hiệu, tiêu ngữ đến bảng biểu).

### 7.2. Công việc Cụ thể Đã Làm (Deliverables)
* **Xây dựng Module M6 Compliance Guard (`src/autotender/models/compliance.py`):**
  * Tự động rà soát và gắn nhãn cờ vi phạm:
    * **Cờ R1:** Phát hiện nhãn hiệu thương mại độc quyền (danh mục 17 hãng: Cisco, Dell, Oracle...) mà thiếu cụm từ *"hoặc tương đương"*.
    * **Cờ R2:** Bắt lỗi yêu cầu doanh thu vượt quá trần $3.0\times$ giá gói thầu.
    * **Cờ R3:** Bắt lỗi thông số kỹ thuật mang tính may đo độc quyền (*"duy nhất trên thị trường"*).
    * **Cờ R3 (CNTT Mới):** Hàm `check_it_specific_compliance` quét bắt buộc Chương V phải có tiêu chuẩn An toàn thông tin Cấp độ 3 (NĐ 85) và Chương VI/VII phải có điều khoản bàn giao 100% mã nguồn (NĐ 73).
    * **Cờ R4:** Verifier số học so khớp tài chính.
    * **Cờ R5:** Kiểm tra tính hoàn thiện của toàn bộ 8 chương HSMT.
* **Thuật toán Xác Thực Tính Nhất Quán Số Học (R4 Numeric Consistency Verifier):**
  * Tách toàn bộ các số xuất hiện trong văn bản sinh ra, đối chiếu với tập hợp số trong KHLCNT.
  * Tích hợp **Whitelist thông minh (`_COMMON_SPEC_NUMBERS`)** loại trừ các thông số kỹ thuật CNTT hợp lệ (cổng mạng 443, 8080, 5432; chuẩn mã hóa 256, 4096; uptime 99.9%; số hiệu Nghị định 73, 82, 85) để triệt tiêu hoàn toàn báo động giả (False Positives).
* **Quy trình Human-In-The-Loop (HITL) & Giao diện Streamlit 8 Trang:**
  * Xây dựng luồng phê duyệt: `Draft` $\rightarrow$ `Edited` $\rightarrow$ `Approved`.
  * Tính năng **Focus Mode**: Chỉ hiển thị các mục có cờ đỏ cảnh báo để chuyên gia tập trung xử lý.
  * Tính năng **1-Click Quick Fix**: Tự động chèn cụm từ *"hoặc tương đương"* vào các vị trí vi phạm cờ R1.
* **Bảo Mật & Nhật Ký Kiểm Toán Bất Biến (Audit Trail):**
  * Xác thực người dùng bằng thuật toán băm **PBKDF2-HMAC-SHA256** (600.000 vòng lặp kèm Salt).
  * Bảo vệ bảng `audit_log` bằng **SQL Triggers** trong SQLite chặn đứng mọi hành vi sửa (`UPDATE`) hoặc xóa (`DELETE`) bản ghi nhật ký.
* **Engine Xuất Bản Văn Bản Công Vụ Chuẩn 100% Nghị định 30/2020/NĐ-CP (`src/autotender/export/docx.py`):**
  * Tạo tệp Word `.docx` hoàn chỉnh: Trang bìa trang trọng, Quốc hiệu, Tiêu ngữ, Căn lề chuẩn (Trái 30mm, Phải 20mm, Trên 20mm, Dưới 20mm), Căn đều 2 bên (Justified), Thụt đầu dòng đoạn văn 1.27cm, Header Shading bảng biểu, đánh số trang tự động `Trang X/Y`.
* **Đánh Giá Toàn Diện Hệ Thống (System Evaluation):**
  * Đánh giá độ trung thực qua **LLM-as-a-judge** đạt **Faithfulness = 0.94** và **Completeness = 0.87**.
  * Quản lý bộ kiểm thử tự động đạt **164/164 bài test PASSED (100%)** qua công cụ `uv`.

### 7.3. Khung Lý thuyết Cần Nắm Vững
* **AI Safety, Guardrails & Verification:** Các phương pháp kiểm soát đầu ra của LLM bằng thuật toán tiền định (Deterministic Verifiers) và biểu thức chính quy.
* **LLM-as-a-Judge Evaluation Framework:**
  * Phương pháp trích xuất nhận định (Claims Extraction) và đối chiếu với văn bản nguồn (Source Context):
    $$\text{Faithfulness} = \frac{|\text{Số nhận định được hỗ trợ bởi Context}|}{|\text{Tổng số nhận định trong văn bản sinh ra}|}$$
* **Nguyên tắc Thiết kế Human-In-The-Loop (HITL):** Phân định ranh giới trách nhiệm giữa máy và người trong các hệ thống pháp lý công quyền.

### 7.4. Mã Nguồn Cốt Lõi Phụ Trách
* [`src/autotender/models/compliance.py`](src/autotender/models/compliance.py): `ComplianceModule.check_text`, `check_document_completeness`, `check_it_specific_compliance`.
* [`src/autotender/models/generator.py`](src/autotender/models/generator.py): `verify_numeric_consistency`, `_COMMON_SPEC_NUMBERS`.
* [`src/autotender/hitl/store.py`](src/autotender/hitl/store.py) & [`src/autotender/audit/store.py`](src/autotender/audit/store.py): Quản lý phê duyệt và SQL Trigger bất biến.
* [`src/autotender/export/docx.py`](src/autotender/export/docx.py): Render file Word chuẩn Nghị định 30.
* [`src/autotender/eval/faithfulness_eval.py`](src/autotender/eval/faithfulness_eval.py): Pipeline đánh giá LLM-as-a-judge.
* [`app/pages/`](app/pages/): 8 trang giao diện người dùng.

### 7.5. Bộ Câu Hỏi Vấn Đáp Phản Biện Từ Hội Đồng (Q&A)
* **Q1: Thuật toán Numeric Verifier (R4) giải quyết vấn đề ảo giác số học như thế nào và làm sao để không báo cờ đỏ sai các thông số kỹ thuật?**
  * *Trả lời:* *"Thưa Thầy, LLM rất dễ sinh nhầm số tiền hoặc thời gian hợp đồng. R4 hoạt động bằng cách: (1) Quét toàn bộ các con số trong văn bản sinh ra; (2) So khớp với tập số trích xuất từ KHLCNT gốc. Để tránh báo cờ sai cho các thông số kỹ thuật hợp lệ (như RAM 64GB, bảo hành 24 tháng, cổng 443, SLA 99.9%, số hiệu NĐ 73, 85), em đã thiết kế bộ lọc Whitelist thông minh `_COMMON_SPEC_NUMBERS` và biểu thức chính quy nhận diện số thứ tự mục `1.1`, `2.3`. Nhờ đó, R4 chỉ bắt đúng các số liệu lạ chưa được kiểm chứng, triệt tiêu hoàn toàn báo động giả."*
* **Q2: Tính bất biến của Nhật ký kiểm toán (Audit Log) được đảm bảo như thế nào ở tầng cơ sở dữ liệu?**
  * *Trả lời:* *"Thưa Thầy, trong cơ quan nhà nước, nhật ký thao tác phải tuyệt đối không thể bị làm giả hoặc xóa dấu vết. Em đã cài đặt trực tiếp 2 SQL Triggers trong SQLite: `trg_audit_log_no_update` và `trg_audit_log_no_delete`. Bất kỳ câu lệnh SQL nào cố tình chỉnh sửa (`UPDATE`) hoặc xóa (`DELETE`) bản ghi trong bảng `audit_log` đều bị cơ sở dữ liệu từ chối với lệnh `ABORT`, đảm bảo tính toàn vẹn 100% phục vụ công tác thanh tra."*
* **Q3: Điểm số Faithfulness = 0.94 được đo lường cụ thể bằng phương pháp nào?**
  * *Trả lời:* *"Thưa Thầy, em sử dụng phương pháp LLM-as-a-Judge theo chuẩn framework RAGAS: Mô hình thẩm định độc lập sẽ phân tách văn bản do AI sinh ra thành từng nhận định đơn lẻ (atomic statements), sau đó đối chiếu từng nhận định với ngữ cảnh trích dẫn luật gốc. Tỷ lệ nhận định được chứng minh trực tiếp bởi luật đạt 94%, vượt trội hoàn toàn so với mức 0.41 của mô hình LLM sinh tự do không có RAG."*

---

## 8. KỊCH BẢN THUYẾT TRÌNH SLIDE CHI TIẾT TỪNG PHÚT (TIMELINE 20 PHÚT)

```
┌─────────────────┬──────────────┬────────────────────────────────────────────────────────────────────────┐
│ Thời gian       │ Người trình  │ Nội dung trình bày chi tiết trên từng Slide                            │
│                 │ bày          │                                                                        │
├─────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────┤
│ **00:00 - 04:00**│ **Người 1**  │ **SLIDE 1-3: ĐẶT VẤN ĐỀ, CĂN CỨ PHÁP LÝ & DATA CONTRACTS**             │
│ (4 phút)        │ (BA/Domain)  │ • Nỗi đau thực tế: Soạn 200 trang E-HSMT mất 2-3 tuần, rủi ro cài cắm  │
│                 │              │   tiêu chí bị thanh tra hủy thầu theo Điều 44 Luật Đấu thầu 22/2023.   │
│                 │              │ • Cơ sở pháp lý mới nhất: NĐ 214/2025 (8 chương), NĐ 73, 82, 85 CNTT.  │
│                 │              │ • Chuyển thể bài toán: Thiết kế Data Contracts Pydantic (`schemas.py`) │
│                 │              │   và định nghĩa bộ cờ tuân thủ R1–R5.                                  │
│                 │              │ ──► Chuyển giao: *"Để hệ thống AI xử lý được các văn bản luật này,     │
│                 │              │ bạn [Tên Người 2] đã xây dựng tầng dữ liệu và giải thuật Chunking..."*│
├─────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────┤
│ **04:00 - 08:00**│ **Người 2**  │ **SLIDE 4-6: DATA ENGINEERING, BENCHMARK & THUẬT TOÁN LEGAL CHUNKER**   │
│ (4 phút)        │ (Data/Chunk) │ • Pipeline nạp KHLCNT: PyMuPDF + VietOCR xử lý bản scan ảnh.           │
│                 │              │ • Xây dựng tập 46 câu hỏi benchmark gán nhãn tay (Ground Truth).       │
│                 │              │ • Thuật toán `LegalChunker`: Tại sao không cắt theo ký tự mà phải      │
│                 │              │   phân cấp Điều/Khoản (699 Chunks kèm UUID5 và Metadata Payload).      │
│                 │              │ ──► Chuyển giao: *"Sau khi đã có 699 Chunks chuẩn, bạn [Tên Người 3]   │
│                 │              │ sẽ trình bày cách đưa dữ liệu này vào không gian vector Qdrant..."*   │
├─────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────┤
│ **08:00 - 12:00**│ **Người 3**  │ **SLIDE 7-9: REPRESENTATION LEARNING & VECTOR DATABASE (QDRANT)**       │
│ (4 phút)        │ (Vector/HNSW)│ • Không gian biểu diễn vector 1024 chiều của mô hình DeepX.            │
│                 │              │ • Kiến trúc Linear Attention (GatedDeltaNet-2) giải quyết nút thắt     │
│                 │              │   512 tokens của BERT, mở rộng ngữ cảnh lên tới 8.192 tokens.          │
│                 │              │ • Qdrant Vector DB: Cấu trúc đồ thị HNSW và kỹ thuật Payload Indexing  │
│                 │              │   kết hợp nhánh từ khóa BM25.                                          │
│                 │              │ ──► Chuyển giao: *"Từ không gian vector này, bạn [Tên Người 4] sẽ      │
│                 │              │ giải thích cách dung hợp thứ hạng và sinh ra 8 chương hồ sơ..."*       │
├─────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────┤
│ **12:00 - 16:00**│ **Người 4**  │ **SLIDE 10-12: HYBRID RERANKING, ORCHESTRATION & TẦNG SINH LLM**       │
│ (4 phút)        │ (LLM/Gen M5) │ • Reciprocal Rank Fusion (RRF k=60) + Cross-Encoder Deep Reranking     │
│                 │              │   chọn Top-5 căn cứ luật chính xác nhất.                               │
│                 │              │ • Mổ xẻ kiến trúc: Tại sao dùng Deterministic Orchestrator có kiểm     │
│                 │              │   soát thay vì Agentic Tool-calling tự do trong khối công quyền?       │
│                 │              │ • Grounded In-Context Learning sinh 8 chương & 3-Tier Degradation.     │
│                 │              │ ──► Chuyển giao: *"Để đảm bảo dự thảo an toàn tuyệt đối và xuất bản ra │
│                 │              │ văn bản công vụ, bạn [Tên Người 5] sẽ trình bày tầng Safety & HITL..."*│
├─────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────┤
│ **16:00 - 20:00**│ **Người 5**  │ **SLIDE 13-15: AI SAFETY, HUMAN-IN-THE-LOOP & ĐÁNH GIÁ TOÀN DIỆN**     │
│ (4 phút)        │ (Safety/Eval)│ • Module Compliance Guard M6: Bắt lỗi cờ R1-R5, rà soát ATTT Cấp độ 3  │
│                 │              │   và điều khoản bắt buộc bàn giao 100% mã nguồn theo NĐ 73.            │
│                 │              │ • R4 Numeric Verifier chống ảo giác tài chính; Audit Log SQL Trigger.  │
│                 │              │ • Trình diễn Live Demo: Focus Mode, 1-Click Fix, Xuất file Word        │
│                 │              │   .DOCX đạt 100% chuẩn Nghị định 30/2020/NĐ-CP.                        │
│                 │              │ • Báo cáo kết quả: Faithfulness 0.94, Test Suite 164/164 tests Passed. │
│                 │              │ • Kết luận và lời cảm ơn Hội đồng chấm thi.                            │
└─────────────────┴──────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

## 9. CHIẾN THUẬT PHỐI HỢP & ỨNG BIẾN KHI TRẢ LỜI PHẢN BIỆN CỦA HỘI ĐỒNG

1. **Nguyên tắc "Đúng người — Đúng việc":**
   * Thầy hỏi về **Luật, Nghiệp vụ, 8 Chương, Cấu trúc Schema Pydantic** $\rightarrow$ **Người 1** trả lời.
   * Thầy hỏi về **Thu thập dữ liệu, OCR, Gán nhãn Benchmark, Thuật toán Chunking** $\rightarrow$ **Người 2** trả lời.
   * Thầy hỏi về **Vector, Linear Attention, HNSW Graph, Qdrant, BM25** $\rightarrow$ **Người 3** trả lời.
   * Thầy hỏi về **RRF, Cross-Encoder Reranker, LLM Prompting, Agentic vs Orchestrator** $\rightarrow$ **Người 4** trả lời.
   * Thầy hỏi về **Compliance Guard, HITL, Bất biến Audit Log, Xuất Word NĐ 30, Đánh giá Faithfulness** $\rightarrow$ **Người 5** trả lời.

2. **Chiến thuật Hỗ trợ Đồng đội (Bọc lót thông minh):**
   * Nếu Thầy hỏi một câu tổng thể bao trùm nhiều mảng: Người phụ trách mảng chính sẽ trả lời phần kỹ thuật của mình trước, sau đó nói: *"Về phần triển khai cụ thể ở tầng [X], em xin phép mời bạn [Tên] bổ sung thêm để làm rõ ý kiến của Thầy ạ."*
   * Tuyệt đối không cướp lời hoặc cắt ngang khi đồng đội đang trả lời. Hãy để bạn nói xong rồi mới xin phép bổ sung thêm một góc nhìn bổ trợ.

3. **Tư thế Tự tin & Học thuật:**
   * Luôn mở đầu bằng: *"Em xin cảm ơn câu hỏi rất sâu sắc của Thầy/Cô ạ."*
   * Khi trả lời luôn gắn liền với **dữ liệu định lượng thực tế** trong báo cáo (ví dụ: *Recall@5 đạt 0.761, nDCG đạt 0.627, Faithfulness đạt 0.94, bộ test 164 bài đều pass 100%*).

---

*Cẩm nang này đã hoàn thiện đầy đủ toàn bộ kiến thức, mã nguồn và kỹ năng bảo vệ đồ án. Chúc nhóm bạn phối hợp ăn ý và đạt điểm số Xuất sắc cao nhất trước Hội đồng!*
