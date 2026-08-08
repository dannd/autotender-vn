"""Chạy đánh giá retrieval (Recall@k/MRR/nDCG) trên tập câu hỏi gán tay
(`data/eval/retrieval_queries.jsonl`, 39 câu) cho 4 chế độ truy xuất — nền tảng cho bảng
ablation Giai đoạn 3 (dense-only vs hybrid, có/không rerank).

Ví dụ: python scripts/run_retrieval_eval.py
       python scripts/run_retrieval_eval.py --model multilingual_minilm
       python scripts/run_retrieval_eval.py --rewrite          # đo tác động HyDE-lite (Mục 41 nâng cấp)
       python scripts/run_retrieval_eval.py --oracle-filter    # đo TRẦN lợi ích metadata filtering (Mục 42)
       python scripts/run_retrieval_eval.py --classify-filter  # đo bộ phân loại law_id THẬT (Mục 42)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.config import resolve_path  # noqa: E402
from autotender.eval.retrieval_eval import (  # noqa: E402
    evaluate_retrieval_fn,
    load_eval_queries,
    ndcg_at_k,
    reciprocal_rank,
    recall_at_k,
)
from autotender.generation.claude_client import is_configured as is_claude_configured  # noqa: E402
from autotender.generation.law_classifier import classify_relevant_law_ids  # noqa: E402
from autotender.generation.query_rewrite import rewrite_query  # noqa: E402
from autotender.rag.embedding_models import DEFAULT_EMBEDDING_MODEL_KEY, EMBEDDING_MODELS  # noqa: E402
from autotender.rag.hybrid_retriever import HybridLegalRetriever  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402
from autotender.utils.logging import get_logger  # noqa: E402

ensure_utf8_console()
logger = get_logger(__name__)

K_VALUES = [1, 3, 5, 10]
QUERIES_PATH = resolve_path("data/eval/retrieval_queries.jsonl")


def _evaluate_with_law_ids_filter(retriever: HybridLegalRetriever, queries, k_values: list[int], law_ids_fn, label: str) -> dict:
    """Đo Recall@k/MRR/nDCG@k khi lọc candidate theo `law_id` TRƯỚC khi xếp hạng, với
    `law_ids_fn(eq) -> set[str] | None` quyết định tập lọc cho từng câu hỏi (`None` = không
    lọc). Dùng chung cho 2 chế độ đo: `--oracle-filter` (biết trước đáp án — đo TRẦN lợi ích)
    và `--classify-filter` (bộ phân loại Claude Haiku thật — đo lợi ích THỰC TẾ đạt được, xem
    `docs/DATA_CARD.md` Mục 12.3). Chỉ đo 2 chế độ đang dùng thật cho Mức 1/2."""
    max_k = max(k_values)
    modes = {
        "hybrid_rrf": lambda q, k, law_ids: retriever.retrieve(q, top_k=k, candidate_k=max(k, 50), law_ids=law_ids),
        "hybrid_rrf_rerank": lambda q, k, law_ids: retriever.retrieve_reranked(q, top_k=k, candidate_k=50, law_ids=law_ids),
    }
    results: dict[str, dict] = {}
    for mode_name, retrieve in modes.items():
        logger.info("Đang đánh giá chế độ '%s' (%s)...", mode_name, label)
        per_query: list[dict] = []
        t0 = time.time()
        for eq in queries:
            law_ids = law_ids_fn(eq)
            retrieved = retrieve(eq.query, max_k, law_ids)
            row = {"mrr": reciprocal_rank(retrieved, eq.relevant)}
            for k in k_values:
                row[f"recall@{k}"] = recall_at_k(retrieved, eq.relevant, k)
                row[f"ndcg@{k}"] = ndcg_at_k(retrieved, eq.relevant, k)
            per_query.append(row)
        elapsed = time.time() - t0
        agg = {key: sum(r[key] for r in per_query) / len(per_query) for key in per_query[0]}
        agg["elapsed_seconds"] = round(elapsed, 1)
        results[mode_name] = agg
        logger.info(
            "  %s: Recall@5=%.3f MRR=%.3f nDCG@5=%.3f (%.1fs)",
            mode_name, agg["recall@5"], agg["mrr"], agg["ndcg@5"], elapsed,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(EMBEDDING_MODELS), default=DEFAULT_EMBEDDING_MODEL_KEY)
    parser.add_argument("--skip-rerank", action="store_true", help="Bỏ chế độ hybrid+rerank (nhanh hơn nhiều)")
    parser.add_argument(
        "--rewrite", action="store_true",
        help="Viết lại mỗi câu hỏi bằng Claude Haiku (HyDE-lite) trước khi truy hồi — cần ANTHROPIC_API_KEY; "
        "ghi kết quả ra reports/retrieval_metrics_rewrite.json thay vì đè file gốc.",
    )
    parser.add_argument(
        "--oracle-filter", action="store_true",
        help="Đo TRẦN lợi ích metadata filtering theo loại văn bản, lọc bằng law_id ĐÚNG (oracle) — "
        "ghi kết quả ra reports/retrieval_metrics_oracle_filter.json.",
    )
    parser.add_argument(
        "--classify-filter", action="store_true",
        help="Đo metadata filtering với bộ phân loại law_id THẬT (Claude Haiku, không biết trước đáp "
        "án) — cần ANTHROPIC_API_KEY; ghi kết quả ra reports/retrieval_metrics_classify_filter.json.",
    )
    args = parser.parse_args()
    filter_flags = [args.rewrite, args.oracle_filter, args.classify_filter]
    if sum(filter_flags) > 1:
        raise SystemExit("--rewrite / --oracle-filter / --classify-filter không kết hợp được — chạy riêng từng cái.")

    if args.oracle_filter:
        out_path = resolve_path("reports/retrieval_metrics_oracle_filter.json")
    elif args.classify_filter:
        out_path = resolve_path("reports/retrieval_metrics_classify_filter.json")
    elif args.rewrite:
        out_path = resolve_path("reports/retrieval_metrics_rewrite.json")
    else:
        out_path = resolve_path("reports/retrieval_metrics.json")

    queries = load_eval_queries(QUERIES_PATH)
    logger.info("Đã nạp %d câu hỏi eval từ %s.", len(queries), QUERIES_PATH)

    if args.rewrite:
        if not is_claude_configured():
            raise SystemExit("--rewrite cần ANTHROPIC_API_KEY (query rewrite gọi Claude Haiku).")
        logger.info("Đang viết lại %d câu hỏi bằng Claude Haiku (HyDE-lite)...", len(queries))
        for q in queries:
            rewritten = rewrite_query(q.query)
            if rewritten != q.query:
                logger.info("  [rewrite] %r -> %r", q.query, rewritten)
            q.query = rewritten

    retriever = HybridLegalRetriever(model_key=args.model)
    logger.info("Đã nạp retriever: %d chunk, model=%s.", retriever.num_chunks, args.model)

    if args.oracle_filter:
        results = _evaluate_with_law_ids_filter(
            retriever, queries, K_VALUES, lambda eq: {law_id for law_id, _ in eq.relevant}, "oracle-filter"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"embedding_model": args.model, "n_queries": len(queries), "modes": results}, f, ensure_ascii=False, indent=2)
        logger.info("Đã ghi kết quả vào %s", out_path)
        return

    if args.classify_filter:
        if not is_claude_configured():
            raise SystemExit("--classify-filter cần ANTHROPIC_API_KEY (bộ phân loại gọi Claude Haiku).")
        classify_cache: dict[str, set[str] | None] = {}

        def _classify(eq):
            if eq.query not in classify_cache:
                predicted = classify_relevant_law_ids(eq.query)
                classify_cache[eq.query] = predicted
                true_law_ids = {law_id for law_id, _ in eq.relevant}
                hit = bool(predicted) and true_law_ids.issubset(predicted)
                logger.info("  [classify] %r -> %s (đúng=%s)", eq.query, predicted, hit)
            return classify_cache[eq.query]

        results = _evaluate_with_law_ids_filter(retriever, queries, K_VALUES, _classify, "classify-filter")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"embedding_model": args.model, "n_queries": len(queries), "modes": results}, f, ensure_ascii=False, indent=2)
        logger.info("Đã ghi kết quả vào %s", out_path)
        return

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"embedding_model": args.model, "n_queries": len(queries), "query_rewrite": args.rewrite, "modes": results},
            f, ensure_ascii=False, indent=2,
        )
    logger.info("Đã ghi kết quả vào %s", out_path)


if __name__ == "__main__":
    main()
