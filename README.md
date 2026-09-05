# AutoTender-VN

**Hệ thống Trợ lý AI Soạn thảo & Rà soát Hồ sơ Mời thầu (E-HSMT) cho Gói thầu Phần mềm & Công nghệ Thông tin tại Việt Nam**

> Đồ án môn **Học sâu Nâng cao (Advanced Deep Learning)** — Bậc Thạc sĩ Kỹ thuật Phần mềm  
> **Nhóm tác giả:** Nguyễn Đình Đan, Nguyễn Văn Vũ, Trần Việt Hòa, Hoàng Xuân Sơn, Nguyễn Thái Thịnh — Trường Kinh doanh FSB, Đại học FPT.  
> **Báo cáo học thuật IEEE:** [`docs/AutoTender-VN_Report_IEEE.docx`](docs/AutoTender-VN_Report_IEEE.docx) | **Slide trình bày:** [`docs/AutoTender-VN_Slides.pptx`](docs/AutoTender-VN_Slides.pptx)  
> **Phân công nhiệm vụ & Lý thuyết 5 thành viên:** [`reports/PHAN_CONG_NHIEM_VU_VA_LY_THUYET_5_THANH_VIEN.md`](reports/PHAN_CONG_NHIEM_VU_VA_LY_THUYET_5_THANH_VIEN.md)

---

> [!IMPORTANT]
> **Khuyến cáo Pháp lý:** Mọi nội dung do hệ thống sinh ra là bản dự thảo hỗ trợ nghiệp vụ chuyên sâu cho Bên mời thầu và Tổ chuyên gia. Văn bản bắt buộc phải trải qua quy trình thẩm định độc lập và phê duyệt chính thức của Chủ đầu tư / Người có thẩm quyền trước khi phát hành lên Hệ thống Mạng đấu thầu Quốc gia (e-GP / VNEPS).

---

## 📑 Mục Lục
1. [Tổng Quan Hệ Thống](#-tổng-quan-hệ-thống)
2. [Cài Đặt Nhanh (Quick Start)](#-cài-đặt-nhanh-quick-start)
3. [Kiến Trúc Kỹ Thuật Cốt Lõi](#-kiến-trúc-kỹ-thuật-cốt-lõi)
4. [Kho Tri Thức Pháp Luật (Legal Knowledge Base)](#-kho-tri-thức-pháp-luật-legal-knowledge-base)
5. [Mô Hình Nhúng DeepX Embedding v1.0 (1024d)](#-mô-hình-nhúng-deepx-embedding-v10-1024d)
6. [Universal LLM Gateway & Quản Lý Chi Phí](#-universal-llm-gateway--quản-lý-chi-phí)
7. [Luồng Nghiệp Vụ 8 Trang Giao Diện Streamlit](#-luồng-nghiệp-vụ-8-trang-giao-diện-streamlit)
8. [Bộ Quy Tắc Rà Soát Tuân Thủ Gói Thầu CNTT (R1 - R5)](#-bộ-quy-tắc-rà-soát-tuân-thủ-gói-thầu-cntt-r1---r5)
9. [Kết Quả Đánh Giá Thực Nghiệm (Benchmark)](#-kết-quả-đánh-giá-thực-nghiệm-benchmark)
10. [Bộ Kiểm Thử Tự Động (164 Tests Passed)](#-bộ-kiểm-thử-tự-động-164-tests-passed)
11. [Danh Mục Scripts Vận Hành CLI](#-danh-mục-scripts-vận-hành-cli)
12. [Bảo Mật & Tính Toàn Vẹn Hệ Thống](#-bảo-mật--tính-toàn-vẹn-hệ-thống)
13. [Cấu Trúc Thư Mục Dự Án](#-cấu-trúc-thư-mục-dự-án)

---

## 🌟 Tổng Quan Hệ Thống

**AutoTender-VN** là giải pháp toàn diện giải quyết bài toán phức tạp trong công tác lập và rà soát E-HSMT cho các gói thầu Phần mềm và Mua sắm CNTT sử dụng vốn Ngân sách Nhà nước. Hệ thống kết hợp:
- **Advanced Modular Hybrid RAG:** Truy xuất kết hợp Sparse (BM25) + Dense (Qdrant Vector DB 1024d HNSW) qua thuật toán **Reciprocal Rank Fusion (RRF $k=60$)** và tinh chỉnh thứ hạng bằng **Cross-Encoder Reranker** (`mmarco-mMiniLMv2-L12-H384-v1`).
- **DeepX Embedding SOTA:** Sử dụng mô hình nhúng `dxtech-asia/deepx-embedding-v1` (772M tham số, kiến trúc Linear Attention Gated DeltaNet-2, ngữ cảnh dài 8.192 tokens native, đạt nDCG@10 = 0.8162 trên Zalo Legal Text Retrieval).
- **Universal LLM Gateway:** Kết nối linh hoạt tới Claude 3.5 Sonnet / DeepSeek-V3 / OpenAI GPT-4o với cơ chế phòng vệ chi phí tối đa `$5.0 USD/tiến trình` và suy biến 3 tầng (Graceful Degradation).
- **M6 Compliance Guard:** Hệ thống 5 bộ quy tắc rà soát vi phạm pháp lý đặc thù cho gói thầu phần mềm theo Nghị định 73/2019/NĐ-CP, Nghị định 82/2024/NĐ-CP, Nghị định 85/2016/NĐ-CP và Thông tư 22/2024/TT-BKHĐT.
- **Human-in-the-Loop (HITL) Workflow:** Quy trình tương tác người - máy phê duyệt từng mục, cho phép tùy biến nội dung, gắn cờ phản hồi và xuất bản văn bản chuẩn thể thức hành chính Nhà nước (DOCX chuẩn NĐ 30/2020 và PDF mẫu TT 22/2024/TT-BKHĐT).

---

## 🚀 Cài Đặt Nhanh (Quick Start)

### Yêu Cầu Môi Trường
- **Hệ điều hành:** macOS (Apple Silicon M-series khuyến nghị), Linux (Ubuntu 22.04+), hoặc Windows 11.
- **Python:** 3.11 hoặc 3.12.
- **Docker:** Docker Desktop hoặc Docker Engine để chạy Qdrant Vector DB.
- **Công cụ quản lý gói:** `uv` (khuyến nghị cho tốc độ tối ưu) hoặc `pip`.

### Quy Trình Khởi Động 5 Bước

```bash
# 1. Clone mã nguồn và khởi tạo môi trường ảo
git clone https://github.com/dannd/autotender-vn.git
cd autotender-vn
uv venv
source .venv/bin/activate    # Linux/macOS (hoặc .venv\Scripts\activate trên Windows)
uv pip install -e .

# 2. Khởi động Qdrant Vector DB (cổng 6333)
docker compose up -d qdrant

# 3. Khởi chạy DeepX Embedding Service (cổng 8080)
# (Chạy Native trên máy host để tối ưu 100% CPU multi-threading / GPU)
uv run python -m uvicorn docker.embedding.server:app --host 0.0.0.0 --port 8080

# 4. Nạp kho tri thức 8 bộ văn bản pháp luật vào Qdrant (thực hiện ở terminal mới)
uv run python scripts/ingest_to_qdrant.py --recreate-collection

# 5. Tạo tài khoản Quản trị viên và khởi chạy Web App
uv run python scripts/create_user.py --username admin --display-name "Quản trị viên" --role admin --password admin123
uv run python -m streamlit run app/main.py --server.port 8501
```

👉 Mở trình duyệt và truy cập: **`http://localhost:8501`**  
- **Tên đăng nhập:** `admin`  
- **Mật khẩu:** `admin123`

---

## 🏗 Kiến Trúc Kỹ Thuật Cốt Lõi

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 STREAMLIT HITL WEB APP                                 │
│  [1-Thu thập]  [2-Nạp KHLCNT]  [3-Soạn thảo]  [4-Tuân thủ]  [5-Xuất/In]  [6,7,8-Phân tích] │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                               AUTOTENDER CORE ENGINE                                   │
│                                                                                        │
│  ┌──────────────────────┐   ┌──────────────────────┐   ┌───────────────────────────┐   │
│  │   RAG RETRIEVER      │   │     M5 GENERATOR     │   │   M6 COMPLIANCE GUARD     │   │
│  │  • Sparse (BM25)     │   │  • System Prompts    │   │  • R1: Anti-Brand Flag    │   │
│  │  • Dense (Qdrant)    │──>│  • Slot-filling      │──>│  • R2: Turnover Cap       │   │
│  │  • RRF Fusion (k=60) │   │  • Citations Filter  │   │  • R3: Mandatory Clauses  │   │
│  │  • Cross-Encoder     │   │  • Graceful Fallback │   │  • R4: Numeric Verifier   │   │
│  └──────────────────────┘   └──────────────────────┘   │  • R5: Full 8-Section     │   │
│                                                        └───────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │               UNIVERSAL OPENAI-COMPATIBLE LLM GATEWAY (Tenacity)                │   │
│  │      Claude 3.5 Sonnet / Claude Sonnet 4.5  │  DeepSeek-V3  │  OpenAI GPT-4o    │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               │                                                         │
┌──────────────▼─────────────┐                            ┌──────────────▼─────────────┐
│    QDRANT VECTOR DB (:6333)│                            │  DEEPX EMBEDDING (:8080)   │
│ • legal_chunks (1024d)     │                            │ • Linear Attention (GDN-2) │
│ • khlcnt_chunks (1024d)    │                            │ • 8.192 Token Context      │
│ • HNSW + Payload Index     │                            │ • Matryoshka 1024d Output  │
└────────────────────────────┘                            └────────────────────────────┘
```

### Cơ chế Suy biến 3 Tầng (3-Tier Graceful Degradation)
Hệ thống đảm bảo tính sẵn sàng 100% trong mọi điều kiện hạ tầng:
- **Tier 1 (Đầy đủ):** Universal LLM Gateway (Claude 3.5 / DeepSeek-V3) + Advanced Hybrid RAG (Qdrant 1024d + BM25 + Rerank).
- **Tier 2 (Ngoại tuyến nâng cao):** Local Dense Embedding + BM25 + Trích dẫn nguyên văn Điều/Khoản luật.
- **Tier 3 (Phòng thủ tối đa):** Deterministic Slot-filling điền số liệu trực tiếp từ KHLCNT gốc vào biểu mẫu chuẩn, kết hợp trích dẫn luật thô, không phụ thuộc kết nối Internet hay API Key.

---

## 📚 Kho Tri Thức Pháp Luật (Legal Knowledge Base)

Hệ thống đã chuẩn hóa và nạp toàn bộ **8 bộ văn bản quy phạm pháp luật** đầu ngành (341 Điều luật, 699 Chunks phân cấp):

| STT | Mã định danh (`law_id`) | Tên văn bản quy phạm pháp luật | Số Điều | Số Chunks | Trọng tâm nghiệp vụ |
|:---:|---|---|:---:|:---:|---|
| **1** | `luat_22_2023_qh15` | **Luật Đấu thầu số 22/2023/QH15** | 96 | 196 | Nguyên tắc cạnh tranh, hình thức LCNT, tiêu chuẩn đánh giá E-HSDT. |
| **2** | `nd_214_2025_ndcp` | **Nghị định 214/2025/NĐ-CP** | 143 | 303 | Quy định chi tiết thi hành Luật Đấu thầu, cấu trúc 8 chương E-HSMT. |
| **3** | `nd_45_2026_ndcp` | **Nghị định 45/2026/NĐ-CP** | 18 | 42 | Quản lý đầu tư CNTT sử dụng nguồn vốn ngân sách nhà nước mới nhất. |
| **4** | `nd_73_2019_ndcp` | **Nghị định 73/2019/NĐ-CP** | 65 | 114 | Thiết kế phần mềm nội bộ, kiểm thử UAT, sở hữu 100% mã nguồn và bản quyền. |
| **5** | `nd_82_2024_ndcp` | **Nghị định 82/2024/NĐ-CP** | 4 | 10 | Sửa đổi, bổ sung NĐ 73/2019 về đề cương dự toán và chi phí phần mềm. |
| **6** | `nd_85_2016_ndcp` | **Nghị định 85/2016/NĐ-CP** | 23 | 44 | Quy định bảo đảm an toàn thông tin theo 5 cấp độ bắt buộc cho phần mềm. |
| **7** | `tt_01_2024_bkhdt` | **Thông tư 01/2024/TT-BKHĐT** | 16 | 38 | Hướng dẫn việc cung cấp thông tin và mẫu E-HSMT trên mạng. |
| **8** | `tt_22_2024_bkhdt` | **Thông tư 22/2024/TT-BKHĐT** | 16 | 42 | Hướng dẫn mẫu E-HSMT mua sắm hàng hóa, dịch vụ phi tư vấn CNTT. |
| **TỔNG**| **8 Văn bản** | **Toàn diện hệ thống pháp lý đấu thầu CNTT** | **341** | **699** | **100% nguyên văn Điều/Khoản kèm siêu dữ liệu phân cấp** |

---

## ⚡ Mô Hình Nhúng DeepX Embedding v1.0 (1024d)

Mô hình [`dxtech-asia/deepx-embedding-v1`](https://huggingface.co/dxtech-asia/deepx-embedding-v1) được chọn làm mô hình mặc định nhờ những ưu thế vượt trội:

- **Kiến trúc Linear Attention O(n):** Sử dụng cơ chế Gated DeltaNet-2 kết hợp Hyperloop weight sharing, giúp xử lý các đoạn văn bản dài 8.192 tokens với tốc độ ổn định và bộ nhớ VRAM tối ưu.
- **Matryoshka Representation Learning:** Hỗ trợ linh hoạt từ 256d đến 1536d (AutoTender-VN chọn chuẩn **1024 chiều** đạt độ chính xác ~99% so với full dimension).
- **Benchmark Top 1 SOTA:** Đạt **nDCG@10 = 0.8162** trên tập dữ liệu Zalo Legal Text Retrieval (vượt qua VietLegal-Harrier 0.7813 và Multilingual-E5 0.6660).
- **Đa nền tảng:** Hỗ trợ chạy Native (tự động phát hiện CUDA / Apple Silicon MPS / CPU multi-threading) hoặc đóng gói Docker Container độc lập.

### Các Endpoints Microservice:
- `GET /health` $\rightarrow$ Trạng thái dịch vụ và thiết bị tính toán (`cpu` / `mps` / `cuda`).
- `GET /info` $\rightarrow$ Thông tin kiến trúc, context length (8K) và số chiều vector mặc định (1024).
- `POST /embed` $\rightarrow$ Nhận danh sách văn bản và trả về ma trận embedding `float32[N][1024]` chuẩn hóa $L_2$.

---

## 🌐 Universal LLM Gateway & Quản Lý Chi Phí

Module [`src/autotender/generation/llm_client.py`](src/autotender/generation/llm_client.py) cung cấp cổng giao tiếp chuẩn hóa OpenAI-compatible:

```yaml
llm_gateway:
  base_url: "https://llm.wokushop.com/v1"      # Ghi đè qua LLM_BASE_URL
  default_model: "claude-sonnet-4-5-20250929"  # Ghi đè qua LLM_MODEL
  timeout_seconds: 60
  usd_cap_per_process: 5.0                     # Khống chế trần ngân sách an toàn
```

### Danh mục Mô hình Hỗ trợ:
1. `claude-3-5-sonnet-20241022` / `claude-sonnet-4-5-20250929`: Soạn thảo chính xác điều khoản kỹ thuật và pháp lý chuyên sâu.
2. `claude-haiku-4-5-20251001`: Tối ưu hóa truy vấn và phân loại nhanh.
3. `deepseek-chat` (DeepSeek-V3): Soạn thảo tốc độ cao, chi phí thấp ($0.14/Mtok input).
4. `gpt-4o` / `gpt-4o-mini`: Đánh giá trung thực độc lập (LLM-as-a-judge).

---

## 🖥 Luồng Nghiệp Vụ 8 Trang Giao Diện Streamlit

| Trang | Tên màn hình | Chức năng nghiệp vụ chi tiết |
|:---:|---|---|
| **1** | **Thu thập dữ liệu** | Crawler tự động dữ liệu Thông báo mời thầu (TBMT) từ Hệ thống mạng đấu thầu quốc gia (`muasamcong.mpi.gov.vn`) và `dauthau.asia`. |
| **2** | **Nạp KHLCNT** | Tiếp nhận tệp PDF/DOCX Quyết định phê duyệt KHLCNT; mô hình M2 NER tự động trích xuất 8 trường thông tin cốt lõi (Tên gói thầu, Chủ đầu tư, Giá gói thầu, Nguồn vốn, Phương thức LCNT, Bảo đảm dự thầu...). |
| **3** | **Soạn thảo HSMT** | **Trọng tâm hệ thống:** Sinh dự thảo toàn diện 8 chương I–VIII bằng Hybrid RAG + LLM. Hỗ trợ Focus Mode lọc các mục có cờ cảnh báo, Quick Fix 1-click và duyệt HITL từng chương. |
| **4** | **Kiểm tra tuân thủ** | Bộ lọc M6 quét tự động 5 nhóm cờ vi phạm pháp lý R1–R5, hiển thị báo cáo tổng quan Pass/Warning/Fail trước khi phê duyệt. |
| **5** | **Xuất và In** | Trình diễn thể thức văn bản hành chính theo Nghị định 30/2020/NĐ-CP; xuất bản tệp `.docx` và `.pdf` (mẫu chuẩn TT 22/2024/TT-BKHĐT) kèm phụ lục nhật ký phê duyệt. |
| **6** | **Bảng điều khiển (Admin)** | Giám sát trạng thái Vector DB, Embedding Service, chi phí LLM thời gian thực và tra cứu Nhật ký kiểm toán bất biến (Audit Log). |
| **7** | **Hỏi - Đáp pháp lý** | **Mức 1:** Trợ lý hỏi đáp chuyên sâu Luật Đấu thầu và các Nghị định hướng dẫn, đối chiếu và hiển thị nguyên văn từng Điều/Khoản luật. |
| **8** | **Đánh giá RAG** | Khung đo lường định lượng: Recall@k, MRR, nDCG@5, Faithfulness (độ trung thực), Completeness (độ đầy đủ) và biểu đồ không gian vector t-SNE/UMAP. |

---

## 🛡 Bộ Quy Tắc Rà Soát Tuân Thủ Gói Thầu CNTT (R1 - R5)

```
┌───────┬──────────────────────────────────┬────────────────────────────────────────────────────────────────────────────┐
│ Mã cờ │ Loại quy tắc kiểm tra            │ Căn cứ pháp lý & Ý nghĩa nghiệp vụ                                         │
├───────┼──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
│  R1   │ Chống chỉ định nhãn hiệu/xuất xứ │ Khoản 2 Điều 44 Luật Đấu thầu 2023: Cấm nêu nhãn hiệu độc quyền           │
│       │ (Brand Restriction)              │ (Oracle, Microsoft, Cisco, Dell...) nếu không có cụm từ "hoặc tương đương".│
├───────┼──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
│  R2   │ Khống chế trần doanh thu         │ Thông tư 22/2024/TT-BKHĐT: Yêu cầu doanh thu bình quân hàng năm của       │
│       │ (Excessive Turnover)             │ nhà thầu không được vượt quá 3 lần giá trị gói thầu.                       │
├───────┼──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
│  R3   │ Điều khoản bắt buộc gói thầu CNTT│ NĐ 85/2016/NĐ-CP (ATTT theo cấp độ), NĐ 73/2019/NĐ-CP (sở hữu 100% mã      │
│       │ (Mandatory IT Clauses)           │ nguồn, nghiệm thu UAT), cam kết bảo hành tối thiểu 12 tháng.               │
├───────┼──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
│  R4   │ Xác thực tính nhất quán số liệu  │ Chống ảo giác LLM: So khớp số tiền gói thầu, giá trị bảo đảm dự thầu      │
│       │ (Numeric Consistency Verifier)   │ và thời gian thực hiện hợp đồng khớp chính xác từng chữ số với KHLCNT gốc. │
├───────┼──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
│  R5   │ Cấu trúc 8 chương E-HSMT         │ Điều 26 Nghị định 214/2025/NĐ-CP: Đảm bảo đầy đủ trọn bộ 8 chương I-VIII  │
│       │ (Full 8-Section Completeness)    │ theo mẫu chuẩn của Bộ Kế hoạch và Đầu tư.                                  │
└───────┴──────────────────────────────────┴────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Kết Quả Đánh Giá Thực Nghiệm (Benchmark)

### 1. Hiệu năng Truy xuất Pháp luật (Retrieval Evaluation — 46 câu hỏi benchmark)
| Cấu hình thử nghiệm | Recall@5 | MRR | nDCG@5 |
|---|:---:|:---:|:---:|
| BM25 (Sparse Lexical Search) | 0.565 | 0.385 | 0.426 |
| Dense Search (`vietnamese-bi-encoder`, 768d) | 0.696 | 0.546 | 0.580 |
| Hybrid RRF ($k=60$) | 0.674 | 0.537 | 0.564 |
| **Hybrid RRF + Cross-Encoder Reranker** | **0.761** | **0.587** | **0.627** |

### 2. Độ Trung Thực & Đầy Đủ (Faithfulness & Completeness — LLM-as-a-judge)
| Chế độ thực thi | Độ trung thực (Faithfulness) | Độ đầy đủ (Completeness) |
|---|:---:|:---:|
| LLM thuần (Không RAG — Zero-shot) | 0.41 | 0.44 |
| **Hệ thống AutoTender-VN (Hybrid RAG + LLM)** | **0.94** | **0.87** |

---

## 🧪 Bộ Kiểm Thử Tự Động (164 Tests Passed)

Toàn bộ hệ thống được bảo vệ bởi **164 unit và integration tests** tự động:

```bash
# Chạy toàn bộ test suite:
uv run python -m pytest

# Chạy có đo lường độ bao phủ (Coverage):
uv run python -m pytest --cov=autotender
```

| Nhóm Kiểm Thử | Tệp Test | Số Tests | Nội Dung Kiểm Tra |
|---|---|:---:|---|
| **Hybrid RAG** | `test_hybrid_retriever.py`, `test_rerank.py` | **17** | Dense, Sparse BM25, RRF Fusion, Cross-Encoder, Mock Qdrant. |
| **LLM Gateway** | `test_claude_client.py`, `test_query_rewrite.py` | **9** | Quản lý ngân sách, retry tenacity, cost tracking, graceful fallback. |
| **M5 Generator** | `test_generator.py`, `test_orchestrator.py` | **13** | Sinh 8 chương, slot-filling, strip citation, verifier số liệu. |
| **M6 Compliance**| `test_compliance.py` | **8** | Rà soát 5 bộ quy tắc R1–R5, phát hiện sai phạm phần mềm. |
| **Pháp luật & QA**| `test_legal_qa.py`, `test_legal_fetch.py` | **20** | Parser Điều/Khoản, Q&A Mức 1, phân cấp văn bản quy phạm. |
| **HITL & Storage**| `test_hitl_store.py`, `test_schemas.py` | **12** | Vòng đời trạng thái Draft $\rightarrow$ Approved, Pydantic data contracts. |
| **Auth & Audit** | `test_auth_store.py`, `test_audit_store.py` | **11** | PBKDF2-HMAC-SHA256, SQL Trigger chống sửa/xóa audit log. |
| **Xuất bản & In** | `test_export.py` | **10** | Render HTML, xuất DOCX (NĐ 30/2020), PDF (TT 22/2024). |
| **Khác** | Ingest, Crawler, NER, Eval, VN Text | **64** | OCR, VietOCR, UMAP, Tokenizer, xử lý tiếng Việt NFC. |
| **TỔNG CỘNG** | **164 / 164 PASSED (100%)** | **164** | **Bảo đảm tính toàn vẹn 100% trước khi triển khai** |

---

## 🛠 Danh Mục Scripts Vận Hành CLI

| Kịch bản CLI | Chức năng thực hiện |
|---|---|
| [`scripts/ingest_to_qdrant.py`](scripts/ingest_to_qdrant.py) | Nạp 699 chunks văn bản luật vào Qdrant qua cổng 8080. |
| [`scripts/check_qdrant_schema.py`](scripts/check_qdrant_schema.py) | Kiểm tra schema, số lượng points và vector dimension trong Qdrant. |
| [`scripts/create_user.py`](scripts/create_user.py) | Tạo tài khoản người dùng an toàn (Admin / Editor). |
| [`scripts/smoke_test_retrieval.py`](scripts/smoke_test_retrieval.py) | Kiểm tra nhanh luồng truy xuất Hybrid RAG độc lập. |
| [`scripts/run_retrieval_eval.py`](scripts/run_retrieval_eval.py) | Chạy đánh giá Recall@k, MRR, nDCG trên 46 câu hỏi benchmark. |
| [`scripts/run_ablation_table.py`](scripts/run_ablation_table.py) | Xuất bảng so sánh đối đầu RAG vs LLM Zero-shot. |
| [`scripts/analyze_embeddings.py`](scripts/analyze_embeddings.py) | Trực quan hóa không gian nhúng t-SNE / UMAP theo loại điều khoản. |
| [`scripts/crawl_dauthau_asia.py`](scripts/crawl_dauthau_asia.py) | Thu thập tự động gói thầu CNTT từ dauthau.asia. |
| [`scripts/ask_legal_qa.py`](scripts/ask_legal_qa.py) | Giao diện dòng lệnh hỏi đáp pháp luật đấu thầu nhanh. |

---

## 🔒 Bảo Mật & Tính Toàn Vẹn Hệ Thống

1. **Xác thực PBKDF2-HMAC-SHA256:** Băm mật khẩu với 600.000 vòng lặp kèm muối ngẫu nhiên (salt) độc lập cho từng tài khoản, lưu trữ tại `data/processed/auth.db`.
2. **Nhật ký Kiểm toán Bất biến (Immutable Audit Log):** Bảng `audit_log` trong `data/processed/audit.db` được bảo vệ bằng SQL Trigger chặn hoàn toàn các thao tác `UPDATE` và `DELETE`.
3. **Đồng thời Đa người dùng:** SQLite kích hoạt chế độ `WAL (Write-Ahead Logging)` kết hợp `RLock` trong Python bảo đảm an toàn dữ liệu khi có nhiều người dùng thao tác cùng lúc.
4. **Kiểm soát Ngân sách:** Hệ thống tự động ngắt kết nối LLM khi đạt giới hạn ngân sách cấu hình và chuyển tiếp sang Tier 3 để bảo vệ tài khoản API.

---

## 📂 Cấu Trúc Thư Mục Dự Án

```
autotender-vn/
├── app/                          # Giao diện người dùng Streamlit
│   ├── main.py                   # Điểm khởi đầu ứng dụng & điều hướng menu
│   ├── auth_ui.py                # Giao diện xác thực & phân quyền
│   ├── common.py                 # Tiện ích giao diện dùng chung
│   └── pages/                    # 8 trang nghiệp vụ chuyên biệt
├── configs/                      # Cấu hình hệ thống (app.yaml, models.yaml)
├── docker/                       # Cấu hình Docker & Docker Compose
│   ├── docker-compose.yml        # Điều phối Qdrant & Embedding Service
│   └── embedding/                # Dockerfile & FastAPI server DeepX (server.py)
├── data/
│   ├── samples/legal_corpus/     # 8 bộ văn bản quy phạm pháp luật (699 chunks)
│   ├── eval/                     # Tập dữ liệu 46 câu hỏi benchmark RAG
│   └── processed/                # Cơ sở dữ liệu SQLite (hitl.db, auth.db, audit.db)
├── src/autotender/               # Mã nguồn lõi (Core Engine)
│   ├── schemas.py                # Data contracts Pydantic v2
│   ├── rag/                      # Pipeline Hybrid RAG (BM25 + Qdrant + RRF + Rerank)
│   ├── generation/               # Universal OpenAI-compatible LLM Gateway
│   ├── models/                   # Generator M5, Compliance M6, NER M2, Legal QA
│   ├── export/                   # Render Engine DOCX (NĐ 30/2020) & PDF (TT 22/2024)
│   ├── hitl/                     # Quản lý quy trình phê duyệt Human-In-The-Loop
│   ├── auth/ & audit/            # Xác thực người dùng & Nhật ký kiểm toán bất biến
│   └── crawler/ & ingest/        # Bộ thu thập dữ liệu thầu & Trích xuất văn bản/OCR
├── scripts/                      # 20 kịch bản CLI phục vụ vận hành & đánh giá
├── tests/                        # 164 bài kiểm thử tự động pytest
├── docs/                         # Báo cáo IEEE, Slide, DATA_CARD, MODEL_CARD, SPEC
└── reports/                      # Báo cáo chuyên sâu về Deep Learning & Nghiệp vụ đấu thầu
```

---

## 📄 Bản Quyền & Giấy Phép

Dự án được phát triển phục vụ mục đích nghiên cứu học thuật và ứng dụng thực tiễn trong quản lý đấu thầu công nghệ thông tin tại Việt Nam.  
Phát hành theo giấy phép **Apache License 2.0**.
