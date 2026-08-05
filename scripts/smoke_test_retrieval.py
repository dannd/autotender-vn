"""Smoke test truy xuất (Mục "Xác minh" GĐ1 của kế hoạch) — chạy vài câu hỏi mẫu qua
dense-only, sparse-only (BM25), và hybrid (RRF) để xác nhận index build đúng và có sự
khác biệt hợp lý giữa 3 chế độ (đặt nền cho bảng ablation ở Giai đoạn 3).

Ví dụ: python scripts/smoke_test_retrieval.py
       python scripts/smoke_test_retrieval.py --model multilingual_minilm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.rag.embedding_models import DEFAULT_EMBEDDING_MODEL_KEY, EMBEDDING_MODELS  # noqa: E402
from autotender.rag.hybrid_retriever import HybridLegalRetriever  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402

ensure_utf8_console()

QUERIES = [
    "Hồ sơ mời thầu gồm những nội dung gì?",
    "Điều kiện năng lực, kinh nghiệm của nhà thầu được đánh giá như thế nào?",
    "Thời gian nhà thầu chuẩn bị hồ sơ dự thầu tối thiểu là bao lâu?",
    "Hành vi nào bị coi là hạn chế sự tham gia của nhà thầu trong hồ sơ mời thầu?",
    "Bảo đảm dự thầu được quy định như thế nào?",
]


def print_results(label: str, results) -> None:
    print(f"  [{label}]")
    for r in results:
        print(f"    ({r.score:.4f}) {r.source_doc}")
        print(f"       {r.text[:120].strip()}...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(EMBEDDING_MODELS), default=DEFAULT_EMBEDDING_MODEL_KEY)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    retriever = HybridLegalRetriever(model_key=args.model)
    print(f"Đã nạp index: {retriever.num_chunks} chunk, model={args.model} ({EMBEDDING_MODELS[args.model]})\n")

    for query in QUERIES:
        print(f"Câu hỏi: {query}")
        dense = [retriever._to_retrieved_chunk(i, s) for i, s in retriever.retrieve_dense(query, args.top_k)]
        sparse = [retriever._to_retrieved_chunk(i, s) for i, s in retriever.retrieve_sparse(query, args.top_k)]
        hybrid = retriever.retrieve(query, top_k=args.top_k)

        print_results("dense", dense)
        print_results("sparse (BM25)", sparse)
        print_results("hybrid (RRF)", hybrid)
        print()


if __name__ == "__main__":
    main()
