"""Tạo dataset distant-supervision cho NER (M2) từ metadata có cấu trúc của crawler (Mục 6/M2).

Chiến lược: với mỗi `TenderNotice` đã crawl, sinh văn bản KHLCNT tổng hợp
(`ingest/synth_document.py`), rồi khớp chuỗi các trường đã biết (package_name,
investor, package_value...) vào văn bản để tự động gán nhãn BIO — không cần
gán tay cho tập train. Ghi rõ trong DATA_CARD.md: đây là nhãn tự động (silver
label), khác với 200 mẫu gán tay dùng làm test set (việc gán tay nằm ngoài
phạm vi tự động hoá của script này).

Output: data/processed/ner_dataset.jsonl (mỗi dòng: {"tokens": [...], "tags": [...]})
        data/processed/classifier_dataset.jsonl (mỗi dòng: {"text": ..., "label": ...})
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.ingest.synth_document import build_synthetic_khlcnt_text  # noqa: E402
from autotender.schemas import TenderNotice  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402
from autotender.utils.logging import get_logger  # noqa: E402

ensure_utf8_console()
logger = get_logger(__name__)

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
# Mặc định kết hợp cả dữ liệu tổng hợp lẫn dữ liệu THẬT đã crawl được (xem DATA_CARD.md
# mục 8, 9) — cho dataset huấn luyện đại diện hơn hẳn so với chỉ 20 mẫu tổng hợp. Notebooks
# 01/02 gọi script này không kèm tham số nên hưởng lợi trực tiếp từ việc đổi mặc định này.
DEFAULT_INPUT_FILES = [
    SAMPLES_DIR / "tender_notices.jsonl",
    SAMPLES_DIR / "real_pilot_sample.jsonl",
    SAMPLES_DIR / "real_dauthau_asia_sample.jsonl",
]
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

_PACKAGE_TYPE_TO_LABEL = {
    "hàng hóa": "hang_hoa",
    "xây lắp": "xay_lap",
    "tư vấn": "tu_van",
    "phi tư vấn": "phi_tu_van",
    "hỗn hợp": "hon_hop",
}


def _tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _field_spans(notice: TenderNotice, text: str) -> list[tuple[int, int, str]]:
    """Tìm vị trí (char_start, char_end, label) của các trường đã biết trong văn bản sinh ra."""
    value_str = f"{notice.package_value:,.0f}".replace(",", ".") if notice.package_value else None
    candidates = [
        ("PACKAGE_NAME", notice.package_name),
        ("INVESTOR", notice.investor),
        ("VALUE", value_str),
        ("FUNDING", notice.funding_source),
        ("METHOD", notice.selection_method),
        ("CONTRACT_TYPE", notice.contract_type),
        ("DURATION", notice.execution_time),
    ]
    spans: list[tuple[int, int, str]] = []
    for label, value in candidates:
        if not value or "[CẦN NGƯỜI DÙNG BỔ SUNG" in value:
            continue
        idx = text.find(value)
        if idx >= 0:
            spans.append((idx, idx + len(value), label))
    return spans


def _bio_tags(tokens: list[tuple[str, int, int]], spans: list[tuple[int, int, str]]) -> list[str]:
    tags = ["O"] * len(tokens)
    for start, end, label in spans:
        first = True
        for i, (_tok, tstart, tend) in enumerate(tokens):
            if tstart >= start and tend <= end:
                tags[i] = ("B-" if first else "I-") + label
                first = False
    return tags


def build_ner_dataset(notices: list[TenderNotice]) -> list[dict]:
    records = []
    for notice in notices:
        text = build_synthetic_khlcnt_text(notice)
        tokens_with_spans = _tokenize_with_spans(text)
        spans = _field_spans(notice, text)
        tags = _bio_tags(tokens_with_spans, spans)
        records.append({"tokens": [t[0] for t in tokens_with_spans], "tags": tags, "tbmt_id": notice.tbmt_id})
    return records


def build_classifier_dataset(notices: list[TenderNotice]) -> list[dict]:
    records = []
    for notice in notices:
        if not notice.package_type:
            continue
        label = _PACKAGE_TYPE_TO_LABEL.get(notice.package_type)
        if not label:
            continue
        text = build_synthetic_khlcnt_text(notice)
        records.append({"text": text, "label": label, "tbmt_id": notice.tbmt_id})
    return records


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _resolve_input_path(p: Path) -> Path:
    """Cho phép truyền tên file trần (vd `real_dauthau_asia_sample.jsonl`) mà không cần
    gõ đường dẫn đầy đủ — tự tìm trong `data/samples/` nếu không thấy ở đường dẫn đã cho."""
    if p.exists():
        return p
    candidate = SAMPLES_DIR / p.name
    return candidate if candidate.exists() else p


def load_notices(paths: list[Path]) -> list[TenderNotice]:
    notices: list[TenderNotice] = []
    for path in paths:
        if not path.exists():
            # Bỏ qua thay vì lỗi cứng (Degraded Mode, Mục 2.1) — vd real_dauthau_asia_sample.jsonl
            # chưa crawl thì vẫn build được dataset từ các nguồn còn lại.
            logger.warning("Bỏ qua %s (không tồn tại). Chạy scripts/build_samples.py hoặc crawl_dauthau_asia.py để có thêm dữ liệu.", path)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    notices.append(TenderNotice.model_validate_json(line))
        logger.info("Đã nạp %s", path.name)
    if not notices:
        raise FileNotFoundError(
            f"Không nạp được bản ghi nào từ {[str(p) for p in paths]} — chạy scripts/build_samples.py trước."
        )
    return notices


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo dataset distant-supervision cho M2/M3 từ 1 hoặc nhiều file TenderNotice.")
    parser.add_argument(
        "--input", nargs="+", type=Path, default=DEFAULT_INPUT_FILES,
        help="1 hoặc nhiều file .jsonl chứa TenderNotice (vd: tender_notices.jsonl real_dauthau_asia_sample.jsonl "
        "real_pilot_sample.jsonl). Mặc định chỉ dùng dữ liệu tổng hợp.",
    )
    parser.add_argument("--out-ner", type=Path, default=OUT_DIR / "ner_dataset.jsonl")
    parser.add_argument("--out-classifier", type=Path, default=OUT_DIR / "classifier_dataset.jsonl")
    args = parser.parse_args()

    input_paths = [_resolve_input_path(p) for p in args.input]
    notices = load_notices(input_paths)

    ner_records = build_ner_dataset(notices)
    clf_records = build_classifier_dataset(notices)

    _write_jsonl(ner_records, args.out_ner)
    _write_jsonl(clf_records, args.out_classifier)

    logger.info("Đã ghi %d bản ghi NER vào %s", len(ner_records), args.out_ner)
    logger.info("Đã ghi %d bản ghi classifier vào %s", len(clf_records), args.out_classifier)
    logger.info(
        "LƯU Ý: đây là nhãn tự động (distant supervision) từ %d bản ghi nguồn (%s). "
        "Tập test 200 mẫu gán tay theo Mục 6/M2 của SPEC cần thực hiện thủ công trên dữ liệu crawl thật.",
        len(notices),
        ", ".join(p.name for p in input_paths),
    )


if __name__ == "__main__":
    main()
