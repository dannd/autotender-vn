"""Xuất dữ liệu phản hồi HITL thành jsonl sẵn sàng huấn luyện lại M5/M6 (Mục 9).

- M5 (generator): cặp (generated_text -> edited_text) từ các mục người dùng đã sửa.
- M6 (compliance): nhãn đúng/sai do người dùng xác nhận trên từng cờ tuân thủ.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from autotender.hitl.store import HitlStore


def export_generator_feedback(store: HitlStore, doc_ids: list[str]) -> list[dict]:
    """Cặp (prompt_context, generated_text, edited_text) cho các mục có sửa thật sự."""
    records = []
    for doc_id in doc_ids:
        for section in store.list_sections(doc_id):
            if section.edited_text is not None and section.edited_text != section.generated_text:
                records.append(
                    {
                        "doc_id": doc_id,
                        "section_id": section.section_id,
                        "generated_text": section.generated_text,
                        "edited_text": section.edited_text,
                        "model_tier": section.model_tier,
                        "status": section.status,
                    }
                )
    return records


def export_compliance_feedback(store: HitlStore, doc_ids: list[str] | None = None) -> list[dict]:
    """Nhãn đúng/sai người dùng xác nhận cho từng cờ tuân thủ — dùng huấn luyện lại M6."""
    records = []
    for doc_id in doc_ids or [d["doc_id"] for d in store.list_documents()]:
        for row in store.list_flag_feedback(doc_id):
            records.append(row)
    return records


def export_feedback(store: HitlStore, out_dir: str | Path, run_date: date | None = None) -> tuple[Path, Path]:
    """Ghi 2 file jsonl vào `data/processed/`: feedback_generator_{date}.jsonl và
    feedback_compliance_{date}.jsonl. Trả về đường dẫn 2 file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_date = run_date or date.today()

    doc_ids = [d["doc_id"] for d in store.list_documents()]
    generator_records = export_generator_feedback(store, doc_ids)
    compliance_records = export_compliance_feedback(store, doc_ids)

    gen_path = out_dir / f"feedback_generator_{run_date.isoformat()}.jsonl"
    comp_path = out_dir / f"feedback_compliance_{run_date.isoformat()}.jsonl"

    with open(gen_path, "w", encoding="utf-8") as f:
        for r in generator_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(comp_path, "w", encoding="utf-8") as f:
        for r in compliance_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return gen_path, comp_path
