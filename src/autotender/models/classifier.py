"""M3 — Phân loại gói thầu: hàng hóa | xây lắp | tư vấn | phi tư vấn | hỗn hợp (Mục 6/M3).

Tier 1: PhoBERT fine-tuned classification head, checkpoint trong `models/classifier_phobert`.
Tier 2: XLM-R zero-shot classification (`zero-shot-classification` pipeline).
Tier 3: keyword matching — LUÔN THÀNH CÔNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autotender.config import get_models_settings, resolve_path
from autotender.models.base import BaseModule, TierUnavailableError

LABELS = ["hang_hoa", "xay_lap", "tu_van", "phi_tu_van", "hon_hop"]

_LABEL_DISPLAY = {
    "hang_hoa": "hàng hóa",
    "xay_lap": "xây lắp",
    "tu_van": "tư vấn",
    "phi_tu_van": "phi tư vấn",
    "hon_hop": "hỗn hợp",
}

_KEYWORDS: dict[str, list[str]] = {
    "xay_lap": ["thi công", "xây dựng", "xây lắp", "cải tạo", "sửa chữa công trình", "lắp đặt hệ thống"],
    "hang_hoa": ["mua sắm", "cung cấp thiết bị", "mua thiết bị", "vật tư", "hàng hóa"],
    "tu_van": ["tư vấn", "lập báo cáo", "giám sát thi công", "thiết kế", "khảo sát"],
    "phi_tu_van": ["dịch vụ bảo trì", "dịch vụ vệ sinh", "dịch vụ bảo vệ", "vận hành"],
    "hon_hop": ["cung cấp và lắp đặt", "cung cấp, lắp đặt", "epc", "chìa khóa trao tay"],
}


@dataclass
class ClassificationResult:
    label: str
    label_display: str
    confidence: float


class ClassifierModule(BaseModule[ClassificationResult]):
    module_name = "M3-Classifier"

    def __init__(self) -> None:
        super().__init__()
        self._cfg = get_models_settings().classifier
        self._tier1_pipeline = None
        self._tier2_pipeline = None

    def classify(self, text: str) -> ClassificationResult:
        return self.run(text)

    # -- Tier 1 -----------------------------------------------------------
    def _try_tier1(self, text: str) -> ClassificationResult:
        checkpoint = resolve_path(self._cfg.get("tier1_checkpoint", "models/classifier_phobert"))
        if not Path(checkpoint).exists():
            raise TierUnavailableError(f"Không tìm thấy checkpoint tại {checkpoint}")
        try:
            from transformers import pipeline
        except ImportError as e:
            raise TierUnavailableError("Thư viện `transformers` chưa cài đặt") from e

        if self._tier1_pipeline is None:
            try:
                self._tier1_pipeline = pipeline("text-classification", model=str(checkpoint))
            except Exception as e:  # noqa: BLE001
                raise TierUnavailableError(f"Load checkpoint Tier 1 lỗi: {e}") from e

        result = self._tier1_pipeline(text[:2000])[0]
        label = result["label"]
        return ClassificationResult(label=label, label_display=_LABEL_DISPLAY.get(label, label), confidence=result["score"])

    # -- Tier 2 -------------------------------------------------------------
    def _try_tier2(self, text: str) -> ClassificationResult:
        try:
            from transformers import pipeline
        except ImportError as e:
            raise TierUnavailableError("Thư viện `transformers` chưa cài đặt") from e

        if self._tier2_pipeline is None:
            try:
                self._tier2_pipeline = pipeline(
                    "zero-shot-classification", model=self._cfg.get("tier2_model", "xlm-roberta-base")
                )
            except Exception as e:  # noqa: BLE001
                raise TierUnavailableError(f"Không tải được model Tier 2: {e}") from e

        candidate_labels = [_LABEL_DISPLAY[l] for l in LABELS]
        try:
            result = self._tier2_pipeline(text[:2000], candidate_labels=candidate_labels)
        except Exception as e:  # noqa: BLE001
            raise TierUnavailableError(f"Suy luận Tier 2 lỗi: {e}") from e

        top_label_display = result["labels"][0]
        reverse_map = {v: k for k, v in _LABEL_DISPLAY.items()}
        label = reverse_map.get(top_label_display, top_label_display)
        return ClassificationResult(label=label, label_display=top_label_display, confidence=float(result["scores"][0]))

    # -- Tier 3 (bắt buộc luôn thành công) -----------------------------------
    def _try_tier3(self, text: str) -> ClassificationResult:
        lowered = text.lower()
        scores = {label: 0 for label in LABELS}
        for label, keywords in _KEYWORDS.items():
            for kw in keywords:
                if kw in lowered:
                    scores[label] += 1

        best_label = max(scores, key=lambda l: scores[l])
        total_hits = sum(scores.values())
        if total_hits == 0:
            # Không khớp từ khoá nào — mặc định "hàng hóa" (loại phổ biến nhất) với confidence thấp
            return ClassificationResult(label="hang_hoa", label_display=_LABEL_DISPLAY["hang_hoa"], confidence=0.2)

        confidence = min(0.5 + 0.1 * scores[best_label], 0.9)
        return ClassificationResult(label=best_label, label_display=_LABEL_DISPLAY[best_label], confidence=confidence)
