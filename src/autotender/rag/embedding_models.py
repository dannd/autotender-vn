"""Đăng ký 2 model embedding dùng để so sánh (Mục "Phân tích DL" — đề cương RAG+LLM).

Chọn 1 model tiếng Việt chuyên biệt và 1 model đa ngôn ngữ để có sự đối lập rõ về kiến
trúc/dữ liệu huấn luyện — phục vụ so sánh Recall@k/MRR/nDCG và trực quan hoá t-SNE/UMAP
không gian embedding theo loại điều khoản (Giai đoạn 3 của kế hoạch).
"""

from __future__ import annotations

EMBEDDING_MODELS: dict[str, str] = {
    # Fine-tune trên dữ liệu tiếng Việt (SimCSE trên nền PhoBERT/XLM-R) — 768 chiều.
    "vi_bi_encoder": "bkai-foundation-models/vietnamese-bi-encoder",
    # Đa ngôn ngữ tổng quát (paraphrase mining, 50+ ngôn ngữ trong đó có tiếng Việt) — 384 chiều.
    "multilingual_minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

DEFAULT_EMBEDDING_MODEL_KEY = "vi_bi_encoder"

# Cross-encoder rerank (Mục "Rerank cross-encoder" — GĐ2-N7). mMARCO là bộ MS MARCO dịch
# sang 14 ngôn ngữ (gồm tiếng Việt) — cross-encoder train trên đó là lựa chọn hợp lý nhất
# hiện có cho rerank tiếng Việt (không có cross-encoder train riêng cho tiếng Việt/pháp
# luật VN sẵn có công khai).
CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
