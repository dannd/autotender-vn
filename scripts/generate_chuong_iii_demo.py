"""Demo Mức 2 (đề cương RAG+LLM) — soạn trọn Chương III cho 1 gói thầu phần mềm mẫu.

Dùng Claude API nếu có `ANTHROPIC_API_KEY`, tự động rơi xuống template-filling (Tier 3,
không cần API key) nếu không — luôn chạy được để demo/kiểm tra nhanh.

Ví dụ: python scripts/generate_chuong_iii_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.pipeline.orchestrator import Orchestrator  # noqa: E402
from autotender.schemas import ExtractedField  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402

ensure_utf8_console()

_FIELDS = [
    ExtractedField(name="PACKAGE_NAME", value="Xây dựng phần mềm quản lý văn bản điều hành", confidence=0.95, source="manual"),
    ExtractedField(name="INVESTOR", value="Sở Thông tin và Truyền thông tỉnh Y", confidence=0.95, source="manual"),
    ExtractedField(name="VALUE", value="6.800.000.000", confidence=0.95, source="manual"),
    ExtractedField(name="FUNDING", value="Ngân sách nhà nước năm 2026", confidence=0.95, source="manual"),
    ExtractedField(name="DURATION", value="180 ngày", confidence=0.95, source="manual"),
]


def main() -> None:
    orch = Orchestrator()
    sections = orch.generate_chuong_iii(_FIELDS)

    print(f"Tier dùng: {orch.generator.active_tier}\n")
    for s in sections:
        print("=" * 70)
        print(f"{s.section_id} — {s.title}")
        print("=" * 70)
        print(s.generated_text)
        print(f"\nTrích dẫn ({len(s.citations)}):")
        for c in s.citations:
            print(f"  - {c.source_doc}")
        if s.flags:
            print(f"\nCờ tuân thủ ({len(s.flags)}):")
            for f in s.flags:
                print(f"  - [{f.rule_code}/{f.severity}] {f.explanation}")
        print()


if __name__ == "__main__":
    main()
