import numpy as np

from autotender.eval.embedding_compare import intra_inter_article_similarity


def test_intra_similarity_higher_when_same_article_vectors_are_close():
    # 2 chunk của Điều 1 gần nhau, 2 chunk của Điều 2 gần nhau, nhưng 2 nhóm cách xa nhau.
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],  # gần chunk đầu (cùng Điều 1)
            [0.0, 1.0],
            [0.01, 0.99],  # gần chunk thứ 3 (cùng Điều 2)
        ]
    )
    article_ids = [("x", 1), ("x", 1), ("x", 2), ("x", 2)]

    result = intra_inter_article_similarity(embeddings, article_ids)

    assert result["intra_mean"] > result["inter_mean"]
    assert result["separation"] > 0
    # 2 cặp cùng Điều 1 (0,1)+(1,0), 2 cặp cùng Điều 2 (2,3)+(3,2) -> 4 cặp intra (có hướng)
    assert result["n_intra_pairs"] == 4
    assert result["n_inter_pairs"] == 4 * 4 - 4 - 4  # tổng ngoài đường chéo trừ đi intra


def test_intra_inter_pair_counts_match_matrix_size():
    embeddings = np.eye(4)  # 4 vector trực giao
    article_ids = [("x", 1), ("x", 1), ("x", 2), ("x", 2)]

    result = intra_inter_article_similarity(embeddings, article_ids)

    total_off_diagonal = 4 * 4 - 4  # bỏ đường chéo
    assert result["n_intra_pairs"] + result["n_inter_pairs"] == total_off_diagonal


def test_no_intra_pairs_when_every_chunk_is_a_different_article():
    embeddings = np.random.RandomState(0).rand(5, 8)
    article_ids = [("x", i) for i in range(5)]

    result = intra_inter_article_similarity(embeddings, article_ids)

    assert result["n_intra_pairs"] == 0
    assert math_isnan(result["intra_mean"])


def math_isnan(x: float) -> bool:
    return x != x
