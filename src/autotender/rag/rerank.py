"""Cross-encoder rerank (Tier 1/2 của M4): top-50 (bi-encoder) -> top-5 (Mục 6/M4).

Import `sentence_transformers` được trì hoãn — chỉ cần khi Tier 1/2 khả dụng.
"""

from __future__ import annotations


def rerank_with_cross_encoder(model_name: str, query: str, candidates: list[str], top_k: int) -> list[tuple[int, float]]:
    """Trả về list[(index_trong_candidates, score)] đã sắp xếp giảm dần, lấy top_k."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    pairs = [[query, c] for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(i, float(s)) for i, s in ranked[:top_k]]
