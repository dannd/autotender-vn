"""Tổng hợp bảng ablation Giai đoạn 3 (đề cương RAG+LLM):
  A. Retrieval: dense-only vs BM25-only vs hybrid RRF vs hybrid+rerank
     (đọc lại reports/retrieval_metrics.json — chạy scripts/run_retrieval_eval.py trước).
  B. Generation: LLM-only (không RAG) vs RAG (có trích dẫn thật) — chấm faithfulness/
     completeness bằng LLM-as-judge (Claude, xem eval/faithfulness_eval.py) trên cùng bộ
     câu hỏi trích từ data/eval/retrieval_queries.jsonl.

Phần B CẦN `ANTHROPIC_API_KEY` — nếu thiếu, script vẫn chạy xong phần A và ghi rõ phần B là
"N/A (cần ANTHROPIC_API_KEY)" thay vì giả lập điểm số hay bỏ qua âm thầm.

Ví dụ: python scripts/run_ablation_table.py
       python scripts/run_ablation_table.py --n-questions 5   (giảm số câu để tiết kiệm chi phí API)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.config import resolve_path  # noqa: E402
from autotender.eval.faithfulness_eval import FaithfulnessJudgment, judge_faithfulness  # noqa: E402
from autotender.eval.retrieval_eval import load_eval_queries  # noqa: E402
from autotender.generation.claude_client import ClaudeUnavailableError, call_claude, is_configured  # noqa: E402
from autotender.rag.hybrid_retriever import HybridLegalRetriever  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402
from autotender.utils.logging import get_logger  # noqa: E402

ensure_utf8_console()
logger = get_logger(__name__)

RETRIEVAL_METRICS_PATH = resolve_path("reports/retrieval_metrics.json")
QUERIES_PATH = resolve_path("data/eval/retrieval_queries.jsonl")
OUT_PATH = resolve_path("reports/ablation_table.json")

_NO_RAG_SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý về đấu thầu tại Việt Nam. Trả lời câu hỏi bằng kiến thức bạn đã "
    "có sẵn (KHÔNG có tài liệu tham chiếu nào được cung cấp). Trả lời ngắn gọn, tiếng Việt."
)


def load_retrieval_ablation() -> dict:
    if not RETRIEVAL_METRICS_PATH.exists():
        logger.warning("Chưa có %s — chạy scripts/run_retrieval_eval.py trước để có phần A.", RETRIEVAL_METRICS_PATH)
        return {"status": "missing", "detail": "Chạy scripts/run_retrieval_eval.py trước."}
    with open(RETRIEVAL_METRICS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {"status": "ok", "modes": data["modes"], "n_queries": data["n_queries"]}


def _judgment_to_dict(j: FaithfulnessJudgment) -> dict:
    return {"faithfulness": j.faithfulness, "completeness": j.completeness, "unsupported_claims": j.unsupported_claims}


def run_generation_ablation(n_questions: int, model: str) -> dict:
    if not is_configured():
        logger.warning("ANTHROPIC_API_KEY chưa cấu hình — bỏ qua phần B (LLM-only vs RAG).")
        return {"status": "N/A", "detail": "Cần ANTHROPIC_API_KEY để chạy generation + LLM-as-judge."}

    queries = load_eval_queries(QUERIES_PATH)[:n_questions]
    retriever = HybridLegalRetriever()

    no_rag_scores, rag_scores = [], []
    for i, eq in enumerate(queries, start=1):
        logger.info("[%d/%d] %s", i, len(queries), eq.query)
        citations = retriever.retrieve_reranked(eq.query, top_k=5, candidate_k=50)
        if not citations:
            logger.warning("  Không có trích dẫn liên quan, bỏ qua câu này.")
            continue

        try:
            # max_tokens=1024 (không phải 512): phát hiện thực tế khi chạy live — 512 hay
            # cắt cụt câu trả lời giữa chừng, khiến response bị rỗng hoặc thiếu ý.
            no_rag_answer = call_claude(system=_NO_RAG_SYSTEM_PROMPT, user_prompt=eq.query, model=model, max_tokens=1024)
            no_rag_judgment = judge_faithfulness(eq.query, citations, no_rag_answer, model=model)
            no_rag_scores.append(_judgment_to_dict(no_rag_judgment))
        except ClaudeUnavailableError as e:
            logger.warning("  Lỗi ở nhánh no-RAG: %s", e)

        try:
            context_str = "\n\n".join(f"[{c.source_doc}]\n{c.text}" for c in citations)
            rag_answer = call_claude(
                system="Bạn là trợ lý pháp lý về đấu thầu. Trả lời CHỈ dựa vào trích đoạn sau, trích dẫn nguồn:",
                user_prompt=f"Trích đoạn:\n{context_str}\n\nCâu hỏi: {eq.query}",
                model=model, max_tokens=1024,
            )
            rag_judgment = judge_faithfulness(eq.query, citations, rag_answer, model=model)
            rag_scores.append(_judgment_to_dict(rag_judgment))
        except ClaudeUnavailableError as e:
            logger.warning("  Lỗi ở nhánh RAG: %s", e)

    def _avg(scores: list[dict], key: str) -> float:
        return sum(s[key] for s in scores) / len(scores) if scores else 0.0

    return {
        "status": "ok",
        "n_questions": len(queries),
        "no_rag": {"n_scored": len(no_rag_scores), "avg_faithfulness": _avg(no_rag_scores, "faithfulness"), "avg_completeness": _avg(no_rag_scores, "completeness")},
        "rag": {"n_scored": len(rag_scores), "avg_faithfulness": _avg(rag_scores, "faithfulness"), "avg_completeness": _avg(rag_scores, "completeness")},
        "caveat": "Judge dùng cùng họ model (Claude) với model sinh — có thể thiên lệch tự ưu ái, xem docstring eval/faithfulness_eval.py.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-questions", type=int, default=10, help="Số câu dùng cho phần B (giảm để tiết kiệm chi phí API)")
    parser.add_argument("--model", default="claude-sonnet-5")
    args = parser.parse_args()

    logger.info("Phần A — Retrieval ablation (đọc lại kết quả đã chạy)...")
    retrieval_ablation = load_retrieval_ablation()

    logger.info("Phần B — Generation ablation (LLM-only vs RAG)...")
    t0 = time.time()
    generation_ablation = run_generation_ablation(args.n_questions, args.model)
    logger.info("Phần B xong sau %.1fs.", time.time() - t0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"a_retrieval": retrieval_ablation, "b_generation": generation_ablation}, f, ensure_ascii=False, indent=2)
    logger.info("Đã ghi bảng ablation vào %s", OUT_PATH)


if __name__ == "__main__":
    main()
