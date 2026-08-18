"""Đăng ký các model embedding dùng để so sánh (Mục "Phân tích DL" — đề cương RAG+LLM).

Chọn 1 model tiếng Việt chuyên biệt và 1 model đa ngôn ngữ để có sự đối lập rõ về kiến
trúc/dữ liệu huấn luyện — phục vụ so sánh Recall@k/MRR/nDCG và trực quan hoá t-SNE/UMAP
không gian embedding theo loại điều khoản (Giai đoạn 3 của kế hoạch).

`bge_m3` (BAAI/bge-m3) thêm sau làm ứng viên thứ 3 — nhắm thẳng vào giới hạn KIẾN TRÚC đã
phát hiện ở `encode_texts` (bên dưới): `vi_bi_encoder` chỉ nhận 256 token (65% chunk kho tri
thức từng bị cắt âm thầm trước khi có sliding-window mean-pooling), trong khi `bge-m3` hỗ trợ
tới 8192 token — giải quyết tận gốc thay vì chỉ giảm nhẹ triệu chứng bằng windowing. Dùng qua
`sentence-transformers` (chỉ lấy nhánh dense embedding tiêu chuẩn của model, KHÔNG dùng tính
năng sparse/ColBERT multi-vector riêng của `bge-m3` — cần thư viện `FlagEmbedding` riêng,
ngoài phạm vi so sánh embedding đơn thuần ở đây).

## Chiến lược load model (theo thứ tự ưu tiên)

1. **Embedding Service Container** (`deepx_v1`): Nếu container đang chạy tại
   `EMBEDDING_SERVICE_URL` (mặc định `http://localhost:8080`), dùng HTTP client để gọi
   `/embed` — không cần cài deepx-embed trong môi trường Python app.
   Container phục vụ `deepx-embedding-v1` (1024d, 8K context, Linear Attention).

2. **Cách B (local deepx_embed)**: Nếu thư viện `deepx_embed` được cài trực tiếp
   (không cần container), load model native.

3. **SentenceTransformer**: Cho các model tiêu chuẩn (vi_bi_encoder, multilingual_minilm,
   bge_m3) hoặc khi deepx không khả dụng.
"""

from __future__ import annotations

import os

EMBEDDING_MODELS: dict[str, str] = {
    # --- MODEL MẶC ĐỊNH ---
    # Kiến trúc Linear Attention (Gated DeltaNet-2, O(n) complexity), train trên văn bản
    # pháp lý tiếng Việt, hỗ trợ 8K token native (không cần sliding-window workaround).
    # Benchmark: nDCG@10 = 0.816 trên Zalo Legal Text Retrieval — cao nhất trong 4 model.
    # Matryoshka Embeddings: hỗ trợ 256d → 1536d linh hoạt, mặc định dùng 1024d.
    # Phục vụ qua Docker container: http://localhost:8080/embed
    "deepx_v1": "dxtech-asia/deepx-embedding-v1",
    # --- CÁC MODEL GIỮ LẠI ĐỂ SO SÁNH ABLATION (Trang 8 — Đánh giá) ---
    # Fine-tune trên dữ liệu tiếng Việt (SimCSE trên nền PhoBERT/XLM-R) — 768 chiều.
    # Giới hạn: max 256 token, 65% chunk kho tri thức bị cắt âm thầm (bug đã ghi nhận).
    "vi_bi_encoder": "bkai-foundation-models/vietnamese-bi-encoder",
    # Đa ngôn ngữ tổng quát (paraphrase mining, 50+ ngôn ngữ trong đó có tiếng Việt) — 384 chiều.
    "multilingual_minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # Đa ngôn ngữ, ngữ cảnh dài (multi-granularity: dense/sparse/ColBERT) — 1024 chiều.
    # Kết quả thực tế: độ tách biệt embedding thấp hơn vi_bi_encoder (xem MODEL_CARD.md).
    "bge_m3": "BAAI/bge-m3",
}

# deepx_v1 là model mặc định — phục vụ qua embedding-service container.
DEFAULT_EMBEDDING_MODEL_KEY = "deepx_v1"

# Chiều vector mặc định tương ứng với DEFAULT_EMBEDDING_MODEL_KEY.
DEFAULT_VECTOR_SIZE = 1024

# URL của embedding service container — ghi đè bằng biến môi trường EMBEDDING_SERVICE_URL.
EMBEDDING_SERVICE_URL = os.environ.get("EMBEDDING_SERVICE_URL", "http://localhost:8080")


class EmbeddingServiceClient:
    """HTTP client gọi embedding-service container tại EMBEDDING_SERVICE_URL.

    Tương thích với interface của SentenceTransformer để dùng được trong encode_texts().
    Lazy-connect: chỉ kiểm tra /health khi có yêu cầu encode đầu tiên.
    """

    def __init__(self, base_url: str = EMBEDDING_SERVICE_URL, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_seq_length = 8192   # deepx_v1 native — encode_texts không cần sliding window
        self._info: dict | None = None

    def _get_info(self) -> dict:
        """Lấy thông tin model từ /info một lần duy nhất."""
        if self._info is None:
            import urllib.request, json
            with urllib.request.urlopen(f"{self.base_url}/info", timeout=5) as r:
                self._info = json.loads(r.read())
        return self._info

    def encode(
        self,
        texts: list[str],
        batch_size: int = 16,
        show_progress_bar: bool = False,
        **_kwargs,
    ):
        """Gửi texts theo batch tới /embed và nhận về numpy array (N, dim)."""
        import json, urllib.request
        import numpy as np

        all_embeddings = []
        try:
            from tqdm import tqdm
            iterator = range(0, len(texts), batch_size)
            if show_progress_bar and len(texts) > batch_size:
                iterator = tqdm(iterator, desc="Container Embedding", total=(len(texts) + batch_size - 1) // batch_size)
        except ImportError:
            iterator = range(0, len(texts), batch_size)

        for i in iterator:
            batch = texts[i : i + batch_size]
            payload = json.dumps({"texts": batch}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read())
                    all_embeddings.extend(resp["embeddings"])
            except Exception as e:
                raise RuntimeError(
                    f"Không thể gọi embedding-service tại {self.base_url}/embed (batch offset {i}): {e}\n"
                    "Hãy chạy: docker compose up -d embedding-service"
                ) from e

        return np.asarray(all_embeddings, dtype="float32")

    def is_available(self) -> bool:
        """Kiểm tra container đang chạy không (không raise)."""
        try:
            import urllib.request
            urllib.request.urlopen(f"{self.base_url}/health", timeout=3)
            return True
        except Exception:
            return False


def load_embedding_model(model_key_or_name: str):
    """Tải embedding model theo key hoặc model name.

    Thứ tự ưu tiên:
    1. deepx_v1 → EmbeddingServiceClient (gọi container HTTP) nếu container đang chạy
    2. deepx_v1 → DeepXEmbed (cài local, Cách B) nếu thư viện có sẵn
    3. SentenceTransformer cho vi_bi_encoder / multilingual_minilm / bge_m3

    Lý do dùng container làm ưu tiên 1:
    - Không cần cài deepx-embed trong môi trường Python app (nặng, phụ thuộc git)
    - Mọi người chỉ cần `docker compose up -d embedding-service` là dùng được
    - Fallback an toàn về vi_bi_encoder nếu container chưa chạy
    """
    from autotender.utils.logging import get_logger
    logger = get_logger(__name__)

    model_name = EMBEDDING_MODELS.get(model_key_or_name, model_key_or_name)
    is_deepx = model_key_or_name == "deepx_v1" or "deepx" in model_name.lower()

    # --- Ưu tiên 1: Gọi embedding-service container (HTTP) ---
    if is_deepx:
        client = EmbeddingServiceClient(base_url=EMBEDDING_SERVICE_URL)
        if client.is_available():
            logger.info(
                "Dùng embedding-service container tại %s (deepx-embedding-v1, 1024d, 8K context).",
                EMBEDDING_SERVICE_URL,
            )
            return client
        else:
            logger.warning(
                "Embedding-service container KHÔNG available tại %s. "
                "Thử load deepx_embed local...",
                EMBEDDING_SERVICE_URL,
            )

    # --- Ưu tiên 2: Load deepx_embed local (Cách B — cài từ GitHub) ---
    if is_deepx:
        try:
            import torch
            from deepx_embed import DeepXEmbed
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Dùng deepx_embed local (device=%s).", device)
            return DeepXEmbed.from_pretrained(model_name, device=device)
        except Exception as e:
            logger.warning(
                "Không load được deepx_embed local (%s). "
                "Fallback sang vi_bi_encoder (768d) — KẾT QUẢ SẼ KHÁC VÌ DIM KHÁC!",
                e,
            )

    # --- Ưu tiên 3: SentenceTransformer ---
    # Nếu config đang dùng deepx_v1 nhưng cả container lẫn local đều không chạy,
    # fallback về vi_bi_encoder để app không crash (Degraded Mode).
    if is_deepx and model_key_or_name == "deepx_v1":
        fallback_name = EMBEDDING_MODELS["vi_bi_encoder"]
        logger.warning(
            "Cả container lẫn deepx_embed local đều không khả dụng. "
            "Fallback về vi_bi_encoder (%s) — collection có thể không khớp dim.",
            fallback_name,
        )
        model_name = fallback_name

    from sentence_transformers import SentenceTransformer
    logger.info("Tải SentenceTransformer: %s", model_name)
    return SentenceTransformer(model_name)


def encode_texts(model, texts: list[str], batch_size: int = 32, show_progress_bar: bool = False):
    """Embed `texts` bằng `model` (EmbeddingServiceClient, SentenceTransformer hoặc DeepXEmbed).

    Đối với EmbeddingServiceClient và DeepXEmbed (8K context native): gọi trực tiếp `model.encode`.
    Đối với SentenceTransformer có context ngắn (vi_bi_encoder 256 token): áp dụng
    sliding-window mean-pooling để tránh bị cắt âm thầm.
    """
    import numpy as np

    # EmbeddingServiceClient hoặc model có max_seq_length >= 4096 (deepx, bge-m3)
    if not hasattr(model, "tokenizer") or getattr(model, "max_seq_length", 8192) >= 4096:
        vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
        return np.asarray(vecs, dtype="float32")

    # SentenceTransformer ngắn (vi_bi_encoder 256 token) — sliding-window mean-pooling
    max_tokens = max(model.max_seq_length - 2, 1)  # chừa chỗ cho [CLS]/[SEP]
    overlap_tokens = min(max(16, max_tokens // 8), max_tokens - 1) if max_tokens > 1 else 0

    windows_per_text: list[list[str]] = []
    for text in texts:
        token_ids = model.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            windows_per_text.append([text])
            continue
        parts: list[str] = []
        start = 0
        while start < len(token_ids):
            end = min(start + max_tokens, len(token_ids))
            parts.append(model.tokenizer.decode(token_ids[start:end], skip_special_tokens=True))
            if end == len(token_ids):
                break
            start = end - overlap_tokens
        windows_per_text.append(parts)

    flat_windows = [w for windows in windows_per_text for w in windows]
    flat_embeddings = np.asarray(model.encode(flat_windows, show_progress_bar=show_progress_bar, batch_size=batch_size))

    results = []
    offset = 0
    for windows in windows_per_text:
        n = len(windows)
        if n == 1:
            results.append(flat_embeddings[offset])
        else:
            pooled = np.mean(flat_embeddings[offset : offset + n], axis=0)
            norm = np.linalg.norm(pooled)
            results.append(pooled / norm if norm > 0 else pooled)
        offset += n
    return np.asarray(results, dtype="float32")


CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
