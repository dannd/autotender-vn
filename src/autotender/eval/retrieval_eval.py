"""Đánh giá retrieval bằng Recall@k, MRR, nDCG@k trên tập câu hỏi gán tay
(`data/eval/retrieval_queries.jsonl`) — Giai đoạn 3 (đề cương RAG+LLM), thay thế proxy
"top-5 chứa chunk từ đúng file chương" của bản cũ (`docs/DATA_CARD.md` Mục 5) bằng đánh
giá đúng nghĩa: mỗi câu hỏi gán nhãn (law_id, dieu_so) — Điều đúng phải trả về.

Một Điều có thể bị chunk thành nhiều mảnh (theo Khoản, `rag/chunker.py`) — "đúng" nghĩa là
BẤT KỲ chunk nào trong top-k thuộc đúng (law_id, dieu_so), không cần khớp đúng Khoản.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from autotender.schemas import RetrievedChunk

RelevantSet = set[tuple[str, int]]


@dataclass
class EvalQuery:
    query: str
    relevant: RelevantSet  # thường chỉ 1 phần tử (law_id, dieu_so), nhưng để dạng set để mở rộng sau này


def load_eval_queries(path: str | Path) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            queries.append(EvalQuery(query=row["query"], relevant={(row["law_id"], row["dieu_so"])}))
    return queries


def _is_relevant(chunk: RetrievedChunk, relevant: RelevantSet) -> bool:
    return (chunk.law_id, chunk.dieu_so) in relevant


def recall_at_k(retrieved: list[RetrievedChunk], relevant: RelevantSet, k: int) -> float:
    """Với 1 Điều liên quan (trường hợp phổ biến của tập eval này): 1.0 nếu Điều đó xuất
    hiện trong top-k, ngược lại 0.0 — tương đương "hit rate@k" cho truy vấn 1-đáp-án đúng."""
    top_k = retrieved[:k]
    return 1.0 if any(_is_relevant(c, relevant) for c in top_k) else 0.0


def reciprocal_rank(retrieved: list[RetrievedChunk], relevant: RelevantSet) -> float:
    for rank, chunk in enumerate(retrieved, start=1):
        if _is_relevant(chunk, relevant):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[RetrievedChunk], relevant: RelevantSet, k: int) -> float:
    """Relevance nhị phân (0/1) — DCG = 1/log2(rank+1) tại vị trí đầu tiên đúng (bỏ qua lần
    thứ 2 trở đi vì cùng 1 Điều, tránh cộng dồn ảo khi Điều bị chunk thành nhiều mảnh liền
    nhau ở đầu danh sách). IDCG = 1 (tốt nhất có thể là đúng ngay vị trí 1)."""
    top_k = retrieved[:k]
    for rank, chunk in enumerate(top_k, start=1):
        if _is_relevant(chunk, relevant):
            return 1.0 / math.log2(rank + 1)
    return 0.0


def evaluate_retrieval_fn(retrieve_fn, queries: list[EvalQuery], k_values: list[int]) -> dict:
    """`retrieve_fn(query: str) -> list[RetrievedChunk]` — đã tự quyết định top_k bên trong
    (truyền `max(k_values)` để đủ dữ liệu cho k lớn nhất cần tính)."""
    max_k = max(k_values)
    per_query: list[dict] = []
    for eq in queries:
        retrieved = retrieve_fn(eq.query, max_k)
        row = {
            "query": eq.query,
            "mrr": reciprocal_rank(retrieved, eq.relevant),
        }
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(retrieved, eq.relevant, k)
            row[f"ndcg@{k}"] = ndcg_at_k(retrieved, eq.relevant, k)
        per_query.append(row)

    n = len(per_query)
    aggregate = {"n_queries": n, "mrr": sum(r["mrr"] for r in per_query) / n if n else 0.0}
    for k in k_values:
        aggregate[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in per_query) / n if n else 0.0
        aggregate[f"ndcg@{k}"] = sum(r[f"ndcg@{k}"] for r in per_query) / n if n else 0.0

    return {"aggregate": aggregate, "per_query": per_query}
