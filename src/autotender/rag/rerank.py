"""Cross-encoder rerank (Tier 1/2 của M4): top-50 (bi-encoder) -> top-5 (Mục 6/M4).

Import `sentence_transformers` được trì hoãn — chỉ cần khi Tier 1/2 khả dụng.
"""

from __future__ import annotations

# Cache theo model_name — nạp `CrossEncoder` (tải trọng số + gọi HF Hub kiểm tra phiên
# bản) tốn ~6-7s mỗi lần, đo thực tế khi khảo sát hiệu năng rerank (chiếm phần lớn trong
# ~20s/câu quan sát ở `docs/DATA_CARD.md` Mục 12, so với inference thật chỉ ~0.4s cho 82
# ứng viên). Trước đây `HybridLegalRetriever.retrieve_reranked` tạo mới `CrossEncoder` ở
# MỖI lượt gọi dù chỉ dùng đúng 1 tên model trong suốt vòng đời tiến trình — cache module-
# level loại bỏ chi phí tải lại này, cùng cơ chế lazy-cache đã áp dụng cho bi-encoder
# (`HybridLegalRetriever._get_encoder`).
_MODEL_CACHE: dict[str, object] = {}


def _get_cross_encoder(model_name: str):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder

        _MODEL_CACHE[model_name] = CrossEncoder(model_name)
    return _MODEL_CACHE[model_name]


def rerank_with_cross_encoder(model_name: str, query: str, candidates: list[str], top_k: int) -> list[tuple[int, float]]:
    """Trả về list[(index_trong_candidates, score)] đã sắp xếp giảm dần, lấy top_k."""
    model = _get_cross_encoder(model_name)
    pairs = [[query, c] for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(i, float(s)) for i, s in ranked[:top_k]]
