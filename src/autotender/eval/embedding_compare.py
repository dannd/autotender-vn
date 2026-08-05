"""Phân tích Deep Learning: so sánh không gian biểu diễn (embedding space) của 2 model —
đúng hướng đề cương RAG+LLM ("Deep Learning thể hiện qua phân tích/đánh giá mạng nơ-ron
pretrained... không phải qua việc tự train"). Không huấn luyện lại — chỉ trích xuất
embedding có sẵn từ 2 model đã đăng ký (`rag/embedding_models.py`) rồi phân tích.

Hai phép đo:
1. `intra_inter_article_similarity` — cosine similarity trung bình giữa các chunk CÙNG một
   Điều (intra) so với KHÁC Điều (inter). Không gian biểu diễn "tốt" cho retrieval phải có
   intra > inter rõ rệt — các đoạn cùng chủ đề pháp lý (cùng Điều) phải gần nhau hơn các
   đoạn khác chủ đề.
2. `reduce_dimensions` — t-SNE/UMAP xuống 2 chiều để trực quan hoá, tô màu theo `law_id`
   (Luật vs Nghị định) hoặc theo khoảng số Điều (proxy thô cho Chương, vì metadata chunk
   hiện không lưu trực tiếp `chuong_so`).
"""

from __future__ import annotations

import numpy as np


def intra_inter_article_similarity(embeddings: np.ndarray, article_ids: list[tuple[str, int]]) -> dict:
    """`article_ids[i]` = (law_id, dieu_so) của chunk thứ i — dùng để nhóm intra/inter.
    Trả về {"intra_mean": ..., "inter_mean": ..., "separation": intra_mean - inter_mean}."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings / norms
    sim_matrix = normed @ normed.T  # cosine similarity, N x N

    n = len(article_ids)
    # Mã hoá (law_id, dieu_so) thành 1 số nguyên duy nhất để so sánh bằng broadcasting
    # thay vì vòng lặp lồng O(n^2) thuần Python.
    id_to_code = {aid: i for i, aid in enumerate(sorted(set(article_ids)))}
    codes = np.array([id_to_code[aid] for aid in article_ids])
    same_article = codes[:, None] == codes[None, :]
    np.fill_diagonal(same_article, False)  # không tính chunk với chính nó

    intra_sims = sim_matrix[same_article]
    inter_sims = sim_matrix[~same_article & ~np.eye(n, dtype=bool)]

    intra_mean = float(intra_sims.mean()) if intra_sims.size else float("nan")
    inter_mean = float(inter_sims.mean()) if inter_sims.size else float("nan")
    return {
        "intra_mean": intra_mean,
        "inter_mean": inter_mean,
        "separation": intra_mean - inter_mean,
        "n_intra_pairs": int(intra_sims.size),
        "n_inter_pairs": int(inter_sims.size),
    }


def reduce_dimensions(embeddings: np.ndarray, method: str = "tsne", random_state: int = 42) -> np.ndarray:
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, max(5, len(embeddings) // 10))
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, init="pca")
    elif method == "umap":
        import umap

        reducer = umap.UMAP(n_components=2, random_state=random_state)
    else:
        raise ValueError(f"method '{method}' không hỗ trợ — chỉ nhận 'tsne' hoặc 'umap'.")
    return reducer.fit_transform(embeddings)
