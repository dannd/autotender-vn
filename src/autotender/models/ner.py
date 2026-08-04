"""M2 — NER trích xuất trường thông tin từ văn bản KHLCNT (Mục 6/M2).

Tier 1: PhoBERT fine-tuned (BIO tagging), checkpoint trong `models/ner_phobert`.
Tier 2: XLM-R zero-shot qua QA-style prompting (mỗi entity là 1 câu hỏi).
Tier 3: regex + từ điển từ khoá — LUÔN THÀNH CÔNG.
"""

from __future__ import annotations

import re
from pathlib import Path

from autotender.config import get_models_settings, resolve_path
from autotender.models.base import BaseModule, TierUnavailableError
from autotender.schemas import ExtractedField

ENTITY_LABELS = [
    "PACKAGE_NAME", "VALUE", "FUNDING", "METHOD", "CONTRACT_TYPE", "DURATION", "INVESTOR", "LOCATION",
]

_QA_QUESTIONS = {
    "PACKAGE_NAME": "Tên gói thầu là gì?",
    "VALUE": "Giá gói thầu là bao nhiêu?",
    "FUNDING": "Nguồn vốn của gói thầu là gì?",
    "METHOD": "Hình thức lựa chọn nhà thầu là gì?",
    "CONTRACT_TYPE": "Loại hợp đồng là gì?",
    "DURATION": "Thời gian thực hiện hợp đồng là bao lâu?",
    "INVESTOR": "Chủ đầu tư là ai?",
    "LOCATION": "Địa điểm thực hiện gói thầu ở đâu?",
}

# Tier 3 — regex/từ điển. Mỗi entry: (label, regex, group_index_giá_trị).
_TIER3_PATTERNS: list[tuple[str, re.Pattern, int]] = [
    (
        "VALUE",
        re.compile(r"(?:giá gói thầu|giá trị gói thầu|giá trị)[:\s]*([\d.,]+)\s*(?:đồng|VNĐ|VND)", re.IGNORECASE),
        1,
    ),
    (
        "FUNDING",
        re.compile(r"(?:nguồn vốn)[:\s]*([^\n.;]+)", re.IGNORECASE),
        1,
    ),
    (
        "METHOD",
        re.compile(
            r"(đấu thầu rộng rãi[^\n.;]*|chào hàng cạnh tranh[^\n.;]*|chỉ định thầu[^\n.;]*|"
            r"tự thực hiện[^\n.;]*|mua sắm trực tiếp[^\n.;]*)",
            re.IGNORECASE,
        ),
        1,
    ),
    (
        "CONTRACT_TYPE",
        re.compile(r"(?:loại hợp đồng)[:\s]*([^\n.;]+)", re.IGNORECASE),
        1,
    ),
    (
        "DURATION",
        re.compile(r"(?:thời gian thực hiện(?: hợp đồng)?)[:\s]*([^\n.;]+)", re.IGNORECASE),
        1,
    ),
    (
        "INVESTOR",
        re.compile(r"(?:chủ đầu tư)[:\s]*([^\n.;]+)", re.IGNORECASE),
        1,
    ),
    (
        "PACKAGE_NAME",
        re.compile(r"(?:tên gói thầu|gói thầu số\s*\d+[:\s]*)[:\s]*([^\n.;]+)", re.IGNORECASE),
        1,
    ),
]


class NERModule(BaseModule[list[ExtractedField]]):
    module_name = "M2-NER"

    def __init__(self) -> None:
        super().__init__()
        self._cfg = get_models_settings().ner
        self._tier1_pipeline = None
        self._tier2_pipeline = None

    def extract(self, text: str) -> list[ExtractedField]:
        return self.run(text)

    # -- Tier 1 -----------------------------------------------------------
    def _try_tier1(self, text: str) -> list[ExtractedField]:
        checkpoint = resolve_path(self._cfg.get("tier1_checkpoint", "models/ner_phobert"))
        if not Path(checkpoint).exists():
            raise TierUnavailableError(f"Không tìm thấy checkpoint tại {checkpoint}")
        try:
            from transformers import pipeline
        except ImportError as e:
            raise TierUnavailableError("Thư viện `transformers` chưa cài đặt") from e

        if self._tier1_pipeline is None:
            try:
                self._tier1_pipeline = pipeline(
                    "token-classification", model=str(checkpoint), aggregation_strategy="simple"
                )
            except Exception as e:  # noqa: BLE001
                raise TierUnavailableError(f"Load checkpoint Tier 1 lỗi: {e}") from e

        results = self._tier1_pipeline(text)
        fields = []
        for r in results:
            fields.append(
                ExtractedField(
                    name=r["entity_group"],
                    value=r["word"],
                    confidence=float(r["score"]),
                    char_start=int(r["start"]),
                    char_end=int(r["end"]),
                    source="ner_model",
                )
            )
        return fields

    # -- Tier 2 -------------------------------------------------------------
    def _try_tier2(self, text: str) -> list[ExtractedField]:
        try:
            from transformers import pipeline
        except ImportError as e:
            raise TierUnavailableError("Thư viện `transformers` chưa cài đặt") from e

        if self._tier2_pipeline is None:
            try:
                self._tier2_pipeline = pipeline(
                    "question-answering", model=self._cfg.get("tier2_model", "xlm-roberta-base")
                )
            except Exception as e:  # noqa: BLE001 — thường do không tải được model từ HF Hub (mạng/GPU)
                raise TierUnavailableError(f"Không tải được model Tier 2: {e}") from e

        fields = []
        for label, question in _QA_QUESTIONS.items():
            try:
                result = self._tier2_pipeline(question=question, context=text)
            except Exception as e:  # noqa: BLE001
                raise TierUnavailableError(f"Suy luận Tier 2 lỗi ở nhãn {label}: {e}") from e
            if result.get("score", 0) < 0.05:
                continue
            fields.append(
                ExtractedField(
                    name=label,
                    value=result["answer"],
                    confidence=float(result["score"]),
                    char_start=int(result["start"]),
                    char_end=int(result["end"]),
                    source="ner_model",
                )
            )
        return fields

    # -- Tier 3 (bắt buộc luôn thành công) -----------------------------------
    def _try_tier3(self, text: str) -> list[ExtractedField]:
        fields: list[ExtractedField] = []
        for label, pattern, group_idx in _TIER3_PATTERNS:
            match = pattern.search(text)
            if match:
                value = match.group(group_idx).strip()
                if "[CẦN NGƯỜI DÙNG BỔ SUNG" in value:
                    # Placeholder báo thiếu thông tin (Mục 2.2) — không phải giá trị thật,
                    # không được coi là đã trích xuất được (tránh khớp nhầm chính placeholder).
                    continue
                fields.append(
                    ExtractedField(
                        name=label,
                        value=value,
                        confidence=0.6,
                        char_start=match.start(group_idx),
                        char_end=match.end(group_idx),
                        source="regex",
                    )
                )
        return fields
