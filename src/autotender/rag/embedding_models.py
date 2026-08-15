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

# deepx_v1 là mặc định mới — đọc từ config/env trong get_app_settings().embedding.model_key.
# Giữ hằng số này cho các script/test dùng trực tiếp không qua config.
DEFAULT_EMBEDDING_MODEL_KEY = "deepx_v1"

# Chiều vector mặc định tương ứng với DEFAULT_EMBEDDING_MODEL_KEY.
# Phải đồng bộ với `configs/app.yaml::embedding.vector_size` và Qdrant collection.
DEFAULT_VECTOR_SIZE = 1024


def encode_texts(model, texts: list[str], batch_size: int = 32, show_progress_bar: bool = False):
    """Embed `texts` bằng `model` (SentenceTransformer), xử lý đúng cho văn bản DÀI HƠN
    `model.max_seq_length`.

    Mặc định `SentenceTransformer.encode` CẮT ÂM THẦM phần vượt quá giới hạn — embedding
    khi đó chỉ phản ánh phần ĐẦU văn bản, không báo lỗi rõ ràng. Xác nhận thực tế trên kho
    tri thức luật: 447/684 chunk (65%) vượt quá 256 token của `vi_bi_encoder`
    (`max_position_embeddings=258` — giới hạn KIẾN TRÚC của RoBERTa/PhoBERT nền tảng model
    này, không thể tăng qua cấu hình như có thể làm với model BERT như
    `multilingual_minilm`, dù checkpoint đó cũng bị cấu hình giới hạn thấp hơn kiến trúc
    gốc, 128 so với 512 khả dụng).

    Văn bản dài được cắt thành các cửa sổ chồng lấn dựa trên TOKENIZER THẬT của model
    (chính xác hơn cách xấp xỉ theo số từ ở `rag/chunker.py`, vốn chỉ nhằm mục đích chunk
    theo ranh giới Khoản chứ không nhằm khớp đúng ngân sách token của một model cụ thể),
    embed từng cửa sổ trong 1 lượt batch, rồi mean-pool + chuẩn hoá L2 lại — kỹ thuật
    chuẩn cho "long document embedding" khi văn bản vượt quá cửa sổ ngữ cảnh của encoder.
    """
    import numpy as np

    max_tokens = max(model.max_seq_length - 2, 1)  # chừa chỗ cho token đặc biệt [CLS]/[SEP]
    # `overlap_tokens` PHẢI nhỏ hơn `max_tokens` — nếu không, mỗi bước trượt cửa sổ
    # (`start = end - overlap_tokens`) có thể ra số ÂM thay vì tăng dần, khiến vòng lặp
    # không bao giờ đạt điều kiện dừng (`start < len(token_ids)` luôn đúng khi start giảm
    # dần) — vòng lặp vô hạn, phát hiện qua test với `max_seq_length` giả lập rất nhỏ.
    overlap_tokens = min(max(16, max_tokens // 8), max_tokens - 1) if max_tokens > 1 else 0

    windows_per_text: list[list[str]] = []
    for text in texts:
        # Gọi `tokenizer.encode` trên text GỐC (chưa cắt) để ĐO độ dài — HuggingFace tự in
        # cảnh báo "Token indices sequence length is longer than..." ngay tại lệnh gọi này
        # nếu text dài hơn giới hạn model, KỂ CẢ KHI sau đó không đưa thẳng text gốc vào
        # model (đã xác nhận: từng cửa sổ tách ra bên dưới luôn nằm trong ngân sách token
        # sau khi decode/encode lại) — cảnh báo này vô hại, chỉ là tác dụng phụ của bước đo.
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
    return np.asarray(results)

# Cross-encoder rerank (Mục "Rerank cross-encoder" — GĐ2-N7). mMARCO là bộ MS MARCO dịch
# sang 14 ngôn ngữ (gồm tiếng Việt) — cross-encoder train trên đó là lựa chọn hợp lý nhất
# hiện có cho rerank tiếng Việt (không có cross-encoder train riêng cho tiếng Việt/pháp
# luật VN sẵn có công khai).
CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
