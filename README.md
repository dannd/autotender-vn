# AutoTender-VN

Phan mem ho tro **soan thao Ho so moi thau (E-HSMT) cho goi thau phan mem/CNTT** tai
Viet Nam bang **RAG + LLM (Universal OpenAI-compatible Gateway)** -- do an cuoi mon Deep
Learning, bac Thac si.

**Tac gia:** Nguyen Dinh Dan, Nguyen Van Vu, Tran Viet Hoa, Hoang Xuan Son, Nguyen Thai
Thinh -- Truong Kinh doanh FSB, Dai hoc FPT.
Bao cao day du: [`docs/AutoTender-VN_Report_IEEE.docx`](docs/AutoTender-VN_Report_IEEE.docx)
Slide trinh bay: [`docs/AutoTender-VN_Slides.pptx`](docs/AutoTender-VN_Slides.pptx)

> Moi noi dung do he thong sinh ra la du thao ho tro soan thao -- bat buoc tham dinh va phe
> duyet theo quy dinh phap luat truoc khi phat hanh chinh thuc.

---

## Cai dat nhanh (Quick Start)

```bash
# 1. Tao moi truong (dung uv)
uv venv && uv pip install -r requirements.txt

# 2. Khoi dong Docker services (Qdrant + Embedding)
docker compose up -d

# 3. Nap corpus phap luat vao Qdrant (1 lan)
python scripts/ingest_to_qdrant.py

# 4. Tao tai khoan admin (1 lan)
python scripts/create_user.py --username admin --display-name "Admin" --role admin

# 5. Chay app
streamlit run app/main.py
```

Mo trinh duyet tai `http://localhost:8501`.

---

## Kien truc He thong

```
Streamlit Web App (8 trang)
  1-Thu thap  2-KHLCNT  3-Soan thao HSMT  4-Kiem tra  5-Xuat ban
  6-Model Dashboard     7-Hoi-dap          8-Danh gia
        |
src/autotender/ -- Core Engine
  schemas.py       data contract Pydantic chung toan he thong
  rag/             Hybrid RAG pipeline
  generation/      LLM gateway (Universal OpenAI-compatible)
  models/          Generator M5 (8 chuong), Legal Q&A Muc 1
  export/          DOCX/PDF render (ND 30/2020)
  hitl/            Human-in-the-loop SQLite store
  auth/ / audit/   Auth PBKDF2, Audit log bat bien
  crawler/         Web crawlers dauthau.asia / muasamcong
  ingest/          PDF/DOCX/OCR ingestion
  knowledge/       Fetch van ban phap luat
  eval/            Retrieval/faithfulness eval
        |
  Docker: Qdrant DB :6333          Docker: Embedding Service :8080
  legal_chunks (dense 1024d)  <-->  deepx-embedding-v1
  khlcnt_chunks (dense 1024d)       1024d, 8K token, CPU-only
```

---

## Cai dat Chi tiet

**Yeu cau:** Python 3.10+, Docker Desktop, uv hoac pip.

### Buoc 0: Tao moi truong

```bash
# Dung uv (khuyen nghi):
uv venv
uv pip install -r requirements.txt

# Hoac pip:
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Buoc 1: Khoi dong Docker services

```bash
# Khoi dong Qdrant + Embedding Service:
docker compose up -d

# Kiem tra trang thai:
docker compose ps

# Dashboard Qdrant: http://localhost:6333/dashboard
# Embedding health: http://localhost:8080/health
# Embedding info:   http://localhost:8080/info
```

Lan dau chay: container `embedding-service` tu tai model `dxtech-asia/deepx-embedding-v1`
(~1.5 GB) tu HuggingFace vao Docker volume `hf_cache`. Cac lan sau dung cache.

### Buoc 2: Nap kho tri thuc phap luat vao Qdrant

```bash
# Nap 684 chunk (Luat Dau thau 22/2023 + ND 214/2025 + TT 01/2024 & 22/2024):
python scripts/ingest_to_qdrant.py

# Nap lai tu dau (xoa va tao lai collection):
python scripts/ingest_to_qdrant.py --recreate-collection

# Kiem tra schema collection:
python scripts/check_qdrant_schema.py

# Chay thu truy xuat (khong can LLM key):
python scripts/smoke_test_retrieval.py --skip-rerank
```

Data persist trong Docker volume `qdrant_storage` -- khong can nap lai qua moi lan restart.

### Buoc 3: Cau hinh LLM API (tuy chon, khuyen nghi)

```bash
cp .env.example .env
# Mo .env va them it nhat mot trong:
#   LLM_API_KEY=...              (uu tien 1, dung cho WokuShop / bat ky OpenAI-compat endpoint)
#   OPENAI_API_KEY=sk-...        (uu tien 2)
#   ANTHROPIC_API_KEY=sk-ant-... (uu tien 3, tuong thich nguoc)
#
# Doi endpoint neu dung gateway khac:
#   LLM_BASE_URL=https://llm.wokushop.com/v1
#   LLM_MODEL=claude-3-5-sonnet-20241022
```

Khong co key van chay duoc -- he thong tu roi xuong Tier 3 (template-filling + trich dan tho).

---

## Embedding Microservice (Docker)

Container rieng biet phuc vu vector hoa van ban.
Moi thanh vien chi can `docker compose up -d` de dung duoc model.

| Endpoint | Mo ta |
|---|---|
| `GET /health` | Kiem tra container dang chay |
| `GET /info` | Thong tin model, so chieu, tokenizer |
| `POST /embed` | Vector hoa danh sach van ban -> float32[][] |

| Thong so | Gia tri |
|---|---|
| Model | `dxtech-asia/deepx-embedding-v1` (GatedDeltaNet-2 / Linear Attention) |
| So chieu | 1024 (Matryoshka Representation Learning) |
| Ngu canh toi da | **8 192 token** (khong cat am tham nhu BERT-based) |
| Benchmark | nDCG@10 = **0.816** tren Zalo Legal Text Retrieval |
| Runtime | CPU-only, torch 2.5.1+cpu, transformers 4.44-4.47 |
| Multi-threading | 12 luong PyTorch (tan dung da nhan CPU) |
| Matryoshka | Ho tro truncation xuong 768d/512d/256d neu can |

```bash
# Test thu cong:
curl http://localhost:8080/health
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d "{\"texts\": [\"Ho so moi thau phan mem\", \"Dieu kien nang luc nha thau\"]}"
```

---

## Qdrant Collection Schema

Collection `legal_chunks`:
```
Vectors: { "dense": VectorParams(size=1024, distance=COSINE) }
Payload (indexed):
  law_id      : keyword   -- "nd_214_2025_nd_cp", "luat_22_2023_qh15"
  law_name    : text      -- "Nghi dinh 214/2025/ND-CP"
  doc_type    : keyword   -- "luat", "nghi_dinh", "thong_tu"
  dieu_so     : integer
  dieu_title  : text
  khoan_so    : keyword   -- "1", "2"... (None neu chunk = tron Dieu)
  source_doc  : text      -- nhan hien thi UI
  text        : text      -- noi dung nguyen van
  chunk_id    : keyword   -- UUID duy nhat
```

Collection `khlcnt_chunks`:
```
Vectors: { "dense": VectorParams(size=1024, distance=COSINE) }
Payload: doc_id, section_title, text, page_number, upload_time
```

---

## LLM Gateway (Universal OpenAI-compatible)

Module `src/autotender/generation/llm_client.py` ho tro moi endpoint tuong thich OpenAI:

| Model | Provider |
|---|---|
| `claude-3-5-sonnet-20241022` | WokuShop / Anthropic (mac dinh) |
| `claude-sonnet-4-5-20250929` | WokuShop |
| `claude-haiku-4-5-20251001` | WokuShop (dung cho query rewrite) |
| `gpt-4o`, `gpt-4o-mini` | OpenAI |
| `deepseek-chat` | DeepSeek |

Co che bao ve:
- Tran ngan sach `$5/process` (cau hinh trong `configs/app.yaml -> llm_gateway.usd_cap_per_process`)
- Exponential backoff retry (tenacity) khi gap loi rate-limit / 5xx
- `LLMUnavailableError` -> tu fallback Tier 3, khong crash app
- Thread-safe cost tracking

---

## Luong su dung (8 trang)

| Trang | Chuc nang |
|---|---|
| 1 -- Thu thap du lieu | Crawl thong bao moi thau tu muasamcong, dauthau.asia (489 ban ghi) |
| 2 -- Nap KHLCNT | Upload PDF/DOCX ke hoach lua chon nha thau, trich truong tu dong |
| 3 -- Soan thao HSMT | **Muc 2**: sinh du thao 8 chuong I-VIII bang LLM + RAG, sua, phe duyet HITL |
| 4 -- Kiem tra tuan thu | Ra soat co R1-R5 (tieu chi han che canh tranh, thieu thanh phan bat buoc) |
| 5 -- Xuat va In | Xuat DOCX/PDF theo dinh dang ND 30/2020 |
| 6 -- Model Dashboard | Dieu khien model, lich su kiem toan (admin only) |
| 7 -- Hoi-dap | **Muc 1**: hoi tu do ve luat dau thau, tra loi co trich dan Dieu/Khoan that |
| 8 -- Danh gia | Recall@k, MRR, nDCG, ablation LLM-only vs RAG, so sanh embedding |

---

## Ket qua Danh gia

### Retrieval (46 cau hoi gan tay)

| Che do | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| BM25 (sparse only) | 0.565 | 0.385 | 0.426 |
| Dense (vi_bi_encoder, 768d) | 0.696 | 0.546 | 0.580 |
| Hybrid RRF | 0.674 | 0.537 | 0.564 |
| **Hybrid RRF + rerank** | **0.761** | **0.587** | **0.627** |

### Faithfulness (LLM-as-judge, 8 cau hoi)

| Che do | Faithfulness | Completeness |
|---|---|---|
| LLM thuan (khong RAG) | 0.41 | 0.44 |
| **RAG + LLM** | **0.94** | **0.87** |

### 5 Nang cap Kien truc RAG (do that)

| Nang cap | Trang thai | Ket qua |
|---|---|---|
| Chan tieu chi han che canh tranh tu system prompt | Bat | Phong thu 2 lop cung M6 |
| Gui tron Dieu (parent chunk) thay vi chi Khoan khop | Bat | 487/684 chunk |
| Query rewriting (HyDE-lite, Claude Haiku) | Tat mac dinh | Hai: nDCG@5 0.627 -> 0.511 |
| Metadata filtering theo law_id | Tat mac dinh | Oracle tot (0.734) nhung classifier that hai (0.531) |
| deepx-embedding-v1 (1024d, 8K token, container) | **Mac dinh moi** | Thay vi_bi_encoder 768d; 8K token native |

---

## Test Suite

```bash
pytest                                       # Toan bo 162 tests
pytest -v tests/test_hybrid_retriever.py     # Chi RAG
pytest -v tests/test_claude_client.py        # Chi LLM gateway
```

**Ket qua: 161 passed, 1 skipped, 0 failed** (89 giay)

| File test | Tests | Pham vi |
|---|---|---|
| test_hybrid_retriever.py | 15 | Dense/Sparse/RRF/Rerank voi mock Qdrant |
| test_claude_client.py | 6 | LLM Gateway: budget, retry, cost tracking |
| test_legal_qa.py | 6 | Muc 1: query rewrite, law_id filter, fallback |
| test_generator.py | 10 | M5: Tier 1/3, R4 verifier, slot-fill |
| test_compliance.py | 8 | R1-R5 rule checker |
| test_chunker_legal.py | 3 | Chunker Dieu/Khoan voi metadata |
| test_export.py | 10 | DOCX/PDF export |
| test_hitl_store.py | 8 | HITL SQLite store |
| test_auth_store.py | 7 | Auth PBKDF2 |
| test_audit_store.py | 4 | Audit append-only |

---

## Scripts Tien ich

| Script | Mo ta |
|---|---|
| `scripts/ingest_to_qdrant.py` | Nap corpus phap luat vao Qdrant (embed qua container) |
| `scripts/check_qdrant_schema.py` | Kiem tra collection, point count, payload breakdown |
| `scripts/smoke_test_retrieval.py` | Test truy xuat end-to-end voi cau hoi mau |
| `scripts/fetch_legal_corpus.py` | Tai van ban phap luat tu nguon chinh thuc |
| `scripts/run_retrieval_eval.py` | Do Recall@k/MRR/nDCG tren 46 cau hoi gan tay |
| `scripts/run_ablation_table.py` | So sanh RAG vs LLM-only |
| `scripts/analyze_embeddings.py` | Truc quan hoa embedding (t-SNE/UMAP) |
| `scripts/crawl_dauthau_asia.py` | Crawl thong bao moi thau |
| `scripts/create_user.py` | Tao tai khoan nguoi dung |
| `scripts/ask_legal_qa.py` | Hoi-dap phap luat CLI |

---

## Bao mat va Van hanh

- **Dang nhap:** PBKDF2-HMAC-SHA256 (600k vong), salt rieng tung user, khong co tai khoan mac dinh
- **Nhat ky kiem toan bat bien:** trigger SQL chan UPDATE/DELETE, xem tai Trang 6 (admin only)
- **Tran ngan sach LLM:** mac dinh $5/process, tu dong fallback Tier 3 khi cham tran
- **Dong thoi nhieu nguoi dung:** SQLite WAL + RLock Python cho tat ca store
- **Log JSON:** `AUTOTENDER_LOG_FORMAT=json` cho tich hop ELK/Loki

---

## Gioi han Da biet

- **Ingest corpus lan dau mat ~15 phut** (684 chunk x ~1.3s/chunk, CPU-only). Data persist qua volume, chi can lam 1 lan.
- **Thong tu 01/2024 & 22/2024/TT-BKHDT:** 22/32 va 26/33 Dieu trong corpus; mau bieu Word/Excel chua vao RAG.
- **Nghi dinh 45/2026/ND-CP:** ban chinh thuc chi co scan anh; lay duoc 43/43 Dieu tu luatvietnam.vn.
- **Metadata filtering law_id:** Oracle nDCG@5=0.734 nhung classifier that hai (0.531) -- giu tat mac dinh.
- **OCR:** PaddleOCR 3.x khong tuong thich Windows/CPU; VietOCR la engine that dang hoat dong.
- **WeasyPrint:** can GTK; neu thieu tu fallback sang ReportLab.

---

## Cau truc Thu muc

```
autotender-vn/
app/                      Streamlit pages (8 trang)
  pages/1-8_*.py
configs/
  app.yaml                Qdrant, embedding, LLM gateway, PDF export
  models.yaml             Section definitions, prompts
docker/embedding/
  Dockerfile              torch 2.5.1+cpu, transformers<4.47
  server.py               FastAPI /embed /health /info
docker-compose.yml        qdrant + embedding-service
scripts/                  CLI tools (ingest, eval, crawl...)
src/autotender/
  schemas.py              Data contracts Pydantic
  config.py               Settings tu app.yaml + env vars
  api.py                  FastAPI REST server (tuy chon)
  rag/                    Hybrid RAG pipeline
  generation/             LLM gateway + query rewrite
  models/                 Generator M5, Legal Q&A Muc 1
  export/                 DOCX/PDF render
  hitl/                   Human-in-the-loop SQLite store
  auth/ audit/            Auth PBKDF2, Audit log bat bien
  crawler/ ingest/        Web crawlers, PDF/DOCX/OCR ingestion
  knowledge/              Fetch van ban phap luat
  eval/                   Retrieval/faithfulness eval
tests/                    162 tests (161 pass, 1 skip)
data/samples/legal_corpus/ 684 chunk van ban phap luat
data/eval/                46 cau hoi gan tay
docs/                     Tai lieu ky thuat
reports/                  Bao cao he thong
```

---

## Tai lieu Lien quan

- [`docs/SPEC.md`](docs/SPEC.md) -- dac ta Phase 1 goc
- [`docs/DATA_CARD.md`](docs/DATA_CARD.md) -- nguon goc/gioi han du lieu
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) -- kien truc, tier, metric
- [`reports/DEEP_LEARNING_VA_LUONG_HE_THONG_AUTOTENDER.md`](reports/DEEP_LEARNING_VA_LUONG_HE_THONG_AUTOTENDER.md)
- [`reports/NGHIEP_VU_DAU_THAU_VA_AUTOTENDER.md`](reports/NGHIEP_VU_DAU_THAU_VA_AUTOTENDER.md)
- [`reports/HUONG_DAN_CHAY_VA_KIEM_THU_HE_THONG.md`](reports/HUONG_DAN_CHAY_VA_KIEM_THU_HE_THONG.md)

---

## Nguyen tac Thiet ke Bat buoc

1. **Luon chay duoc (Degraded Mode):** M5 va Muc 1 co 3 tang -- Tier 1 (LLM Gateway + RAG) -> Tier 3 (template-filling, khong can API key). Thieu key hoac loi mang -> tu roi xuong Tier 3, khong crash.
2. **Khong bia dat:** cau tra loi CHI dua tren trich doan luat that da truy xuat; so lieu goi thau chen bang slot-filling; verifier R4 gan co neu phat hien so lieu la.
3. **Van ban phap luat phai dung hieu luc:** ND 24/2024 (de cuong goc) da het hieu luc -> thay bang ND 214/2025/ND-CP.
4. **Human-in-the-loop:** khong muc nao duoc coi la hoan thanh neu chua qua phe duyet.
5. **Thu thap du lieu co trach nhiem:** ton trong robots.txt, rate-limit, cache toan bo response.
