import math

from autotender.eval.retrieval_eval import (
    evaluate_retrieval_fn,
    load_eval_queries,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from autotender.schemas import RetrievedChunk


def _chunk(law_id: str, dieu_so: int, chunk_id: str = "c") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text="x", source_doc="x", score=1.0, law_id=law_id, dieu_so=dieu_so)


def test_recall_at_k_hit_within_k():
    retrieved = [_chunk("x", 1), _chunk("x", 44), _chunk("x", 2)]
    assert recall_at_k(retrieved, {("x", 44)}, k=3) == 1.0
    assert recall_at_k(retrieved, {("x", 44)}, k=1) == 0.0  # đúng ở vị trí 2, ngoài top-1


def test_recall_at_k_miss():
    retrieved = [_chunk("x", 1), _chunk("x", 2)]
    assert recall_at_k(retrieved, {("x", 999)}, k=5) == 0.0


def test_reciprocal_rank_first_hit_position():
    retrieved = [_chunk("x", 1), _chunk("x", 44), _chunk("x", 2)]
    assert reciprocal_rank(retrieved, {("x", 44)}) == 0.5  # rank 2 -> 1/2


def test_reciprocal_rank_zero_when_not_found():
    retrieved = [_chunk("x", 1)]
    assert reciprocal_rank(retrieved, {("x", 999)}) == 0.0


def test_ndcg_at_k_perfect_at_rank_one():
    retrieved = [_chunk("x", 44)]
    assert ndcg_at_k(retrieved, {("x", 44)}, k=5) == 1.0  # 1/log2(2) = 1


def test_ndcg_at_k_decreases_with_rank():
    retrieved = [_chunk("x", 1), _chunk("x", 44)]
    expected = 1.0 / math.log2(3)  # rank 2
    assert ndcg_at_k(retrieved, {("x", 44)}, k=5) == expected


def test_ndcg_at_k_zero_outside_k():
    retrieved = [_chunk("x", 1), _chunk("x", 2), _chunk("x", 44)]
    assert ndcg_at_k(retrieved, {("x", 44)}, k=2) == 0.0


def test_load_eval_queries_reads_jsonl(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"query": "câu hỏi 1", "law_id": "luat_22_2023_qh15", "dieu_so": 44}\n'
        '{"query": "câu hỏi 2", "law_id": "nd_214_2025_ndcp", "dieu_so": 26}\n',
        encoding="utf-8",
    )
    queries = load_eval_queries(path)
    assert len(queries) == 2
    assert queries[0].query == "câu hỏi 1"
    assert queries[0].relevant == {("luat_22_2023_qh15", 44)}


def test_evaluate_retrieval_fn_aggregates_across_queries():
    queries = load_eval_queries_from_rows(
        [("q1", "x", 1), ("q2", "x", 2)]
    )

    def fake_retrieve(query: str, top_k: int) -> list[RetrievedChunk]:
        # "q1" luôn đúng ở rank 1, "q2" không bao giờ tìm thấy
        if query == "q1":
            return [_chunk("x", 1)]
        return [_chunk("x", 999)]

    result = evaluate_retrieval_fn(fake_retrieve, queries, k_values=[1, 5])

    assert result["aggregate"]["n_queries"] == 2
    assert result["aggregate"]["recall@1"] == 0.5  # 1 trong 2 câu đúng
    assert result["aggregate"]["mrr"] == 0.5


def load_eval_queries_from_rows(rows):
    from autotender.eval.retrieval_eval import EvalQuery

    return [EvalQuery(query=q, relevant={(law_id, dieu_so)}) for q, law_id, dieu_so in rows]
