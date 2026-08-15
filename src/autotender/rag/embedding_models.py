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
"""

from __future__ import annotations

EMBEDDING_MODELS: dict[str, str] = {
    # --- MODEL MẶC ĐỊNH MỚI ---
    # Kiến trúc Linear Attention (Gated DeltaNet-2, O(n) complexity), train trên văn bản
    # pháp lý tiếng Việt, hỗ trợ 8K token native (không cần sliding-window workaround).
    # Benchmark: nDCG@10 = 0.816 trên Zalo Legal Text Retrieval — cao nhất trong 4 model.
    # Matryoshka Embeddings: hỗ trợ 256d → 1536d linh hoạt, mặc định dùng 1024d.
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

# vi_bi_encoder là model tiếng Việt ổn định mặc định.
DEFAULT_EMBEDDING_MODEL_KEY = "vi_bi_encoder"

# Chiều vector mặc định tương ứng với DEFAULT_EMBEDDING_MODEL_KEY.
DEFAULT_VECTOR_SIZE = 768


def load_embedding_model(model_key_or_name: str):
    """Tải embedding model theo key hoặc model name.

    Hỗ trợ cả:
    - Cách B (Đầy đủ): Dùng `DeepXEmbed` từ thư viện `deepx-embed` nếu là deepx_v1
    - SentenceTransformer: Cho các model tiêu chuẩn (vi_bi_encoder, multilingual_minilm, bge_m3).
    """
    model_name = EMBEDDING_MODELS.get(model_key_or_name, model_key_or_name)

    if model_key_or_name == "deepx_v1" or "deepx" in model_name.lower():
        try:
            from deepx_embed import DeepXEmbed
            return DeepXEmbed.from_pretrained(model_name)
        except ImportError:
            pass  # Fallback to SentenceTransformer

    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def encode_texts(model, texts: list[str], batch_size: int = 32, show_progress_bar: bool = False):
    """Embed `texts` bằng `model` (SentenceTransformer hoặc DeepXEmbed).

    Đối với DeepXEmbed (8K context native): gọi trực tiếp `model.encode`.
    Đối với SentenceTransformer có context ngắn (vi_bi_encoder 256 token): áp dụng
    sliding-window mean-pooling để tránh bị cắt âm thầm.
    """
    import numpy as np

    # Nếu là DeepXEmbed hoặc model hỗ trợ 8K native (không có tokenizer giới hạn 256)
    if not hasattr(model, "tokenizer") or getattr(model, "max_seq_length", 8192) >= 4096:
        vecs = model.encode(texts)
        return np.asarray(vecs, dtype="float32")

    max_tokens = max(model.max_seq_length - 2, 1)  # chừa chỗ cho token đặc biệt [CLS]/[SEP]
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

