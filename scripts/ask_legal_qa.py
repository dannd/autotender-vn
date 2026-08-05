"""CLI thử Mức 1 — Hỏi-đáp có trích dẫn (đề cương RAG+LLM). Cần biến môi trường
`ANTHROPIC_API_KEY` để dùng Tier 1 (Claude); nếu không có, tự động rơi xuống Tier 3
(liệt kê trích dẫn không qua LLM) — vẫn chạy được, chỉ không có câu trả lời tổng hợp.

Ví dụ: export ANTHROPIC_API_KEY=sk-ant-...
       python scripts/ask_legal_qa.py "Hồ sơ mời thầu gồm những nội dung gì?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.models.legal_qa import LegalQAModule  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402

ensure_utf8_console()


def main() -> None:
    if len(sys.argv) < 2:
        print('Cách dùng: python scripts/ask_legal_qa.py "câu hỏi của bạn"')
        sys.exit(1)
    question = sys.argv[1]

    module = LegalQAModule()
    answer = module.ask(question)

    print(f"Câu hỏi: {answer.question}")
    print(f"Tier dùng: {module.active_tier} (model: {answer.model_used})")
    print(f"\nTrả lời:\n{answer.answer}")
    print(f"\nTrích dẫn ({len(answer.citations)}):")
    for c in answer.citations:
        print(f"  - {c.source_doc} (score={c.score:.4f})")


if __name__ == "__main__":
    main()
