"""Chạy đánh giá retrieval (Recall@k/MRR/nDCG) trên tập câu hỏi gán tay
(`data/eval/retrieval_queries.jsonl`, 39 câu) cho 4 chế độ truy xuất — nền tảng cho bảng
ablation Giai đoạn 3 (dense-only vs hybrid, có/không rerank).

Ví dụ: python scripts/run_retrieval_eval.py
       python scripts/run_retrieval_eval.py --model multilingual_minilm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.config import resolve_path  # noqa: E402
from autotender.eval.retrieval_eval import evaluate_retrieval_fn, load_eval_queries  # noqa: E402
from autotender.rag.embedding_models import DEFAULT_EMBEDDING_MODEL_KEY, EMBEDDING_MODELS  # noqa: E402
from autotender.rag.hybrid_retriever import HybridLegalRetriever  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402
from autotender.utils.logging import get_logger  # noqa: E402

ensure_utf8_console()
logger = get_logger(__name__)

K_VALUES = [1, 3, 5, 10]
QUERIES_PATH = resolve_path("data/eval/retrieval_queries.jsonl")
OUT_PATH = resolve_path("reports/retrieval_metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(EMBEDDING_MODELS), default=DEFAULT_EMBEDDING_MODEL_KEY)
    parser.add_argument("--skip-rerank", action="store_true", help="Bỏ chế độ hybrid+rerank (nhanh hơn nhiều)")
    args = parser.parse_args()

    queries = load_eval_queries(QUERIES_PATH)
    logger.info("Đã nạp %d câu hỏi eval từ %s.", len(queries), QUERIES_PATH)

    retriever = HybridLegalRetriever(model_key=args.model)
    logger.info("Đã nạp retriever: %d chunk, model=%s.", retriever.num_chunks, args.model)

    modes = {
        "dense_only": lambda q, k: [retriever._to_retrieved_chunk(i, s) for i, s in retriever.retrieve_dense(q, k)],
        "sparse_only_bm25": lambda q, k: [retriever._to_retrieved_chunk(i, s) for i, s in retriever.retrieve_sparse(q, k)],
        "hybrid_rrf": lambda q, k: retriever.retrieve(q, top_k=k, candidate_k=max(k, 50)),
    }
    if not args.skip_rerank:
        modes["hybrid_rrf_rerank"] = lambda q, k: retriever.retrieve_reranked(q, top_k=k, candidate_k=50)

    results: dict[str, dict] = {}
    for mode_name, fn in modes.items():
        logger.info("Đang đánh giá chế độ '%s'...", mode_name)
        t0 = time.time()
        result = evaluate_retrieval_fn(fn, queries, K_VALUES)
        elapsed = time.time() - t0
        results[mode_name] = result["aggregate"]
        results[mode_name]["elapsed_seconds"] = round(elapsed, 1)
        logger.info(
            "  %s: Recall@5=%.3f MRR=%.3f nDCG@5=%.3f (%.1fs)",
            mode_name, result["aggregate"]["recall@5"], result["aggregate"]["mrr"],
            result["aggregate"]["ndcg@5"], elapsed,
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"embedding_model": args.model, "n_queries": len(queries), "modes": results}, f, ensure_ascii=False, indent=2)
    logger.info("Đã ghi kết quả vào %s", OUT_PATH)


if __name__ == "__main__":
    main()
