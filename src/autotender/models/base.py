"""BaseModule — cơ chế fallback 3 tầng dùng chung cho mọi module ML (Mục 2.1).

```
Tier 1: Mô hình fine-tuned (checkpoint trong models/)
   ↓ nếu checkpoint không tồn tại hoặc load lỗi
Tier 2: Mô hình pretrained zero-shot / few-shot
   ↓ nếu không có GPU / không tải được model
Tier 3: Rule-based (regex, từ điển, template)
```

NGUYÊN TẮC: Tier 3 KHÔNG BAO GIỜ được phép raise exception — đây là tầng đảm bảo
phần mềm luôn chạy được kể cả khi không có mạng/GPU/checkpoint nào cả.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

from autotender.utils.logging import get_logger

logger = get_logger(__name__)

ModelTier = Literal[1, 2, 3]
T = TypeVar("T")


class TierUnavailableError(Exception):
    """Tầng này không dùng được (thiếu checkpoint, thiếu thư viện, thiếu GPU, lỗi mạng...).

    Không phải lỗi nghiêm trọng — BaseModule sẽ bắt exception này và thử tầng kế tiếp.
    """


class BaseModule(ABC, Generic[T]):
    """Lớp cha cho mọi module ML (NER, Classifier, Retriever, Generator, Compliance).

    Subclass implement `_try_tier1`, `_try_tier2`, `_try_tier3`. Tier 3 phải luôn
    thành công (rule-based thuần, không phụ thuộc mạng/GPU/thư viện nặng).
    """

    module_name: str = "module"

    def __init__(self) -> None:
        self._active_tier: ModelTier | None = None
        self._tier1_loaded = False
        self._tier2_loaded = False

    @property
    def active_tier(self) -> ModelTier | None:
        """Tier đã dùng ở lần gọi `run()` gần nhất — hiển thị badge trên UI."""
        return self._active_tier

    def run(self, *args: object, **kwargs: object) -> T:
        for tier, fn in ((1, self._try_tier1), (2, self._try_tier2), (3, self._try_tier3)):
            try:
                result = fn(*args, **kwargs)
            except TierUnavailableError as e:
                logger.info("[%s] Tier %d không dùng được: %s", self.module_name, tier, e)
                continue
            except Exception as e:  # noqa: BLE001 — không để lỗi bất ngờ ở tier 1/2 làm sập app
                if tier == 3:
                    raise  # Tier 3 lỗi là lỗi cấu hình/code thật sự, phải biết ngay
                logger.warning(
                    "[%s] Tier %d lỗi không mong đợi, coi như không dùng được: %s", self.module_name, tier, e
                )
                continue
            self._active_tier = tier  # type: ignore[assignment]
            logger.info("[%s] Chạy ở Tier %d", self.module_name, tier)
            return result
        raise RuntimeError(f"[{self.module_name}] Cả 3 tầng đều thất bại — không thể xảy ra nếu Tier 3 đúng.")

    @abstractmethod
    def _try_tier1(self, *args: object, **kwargs: object) -> T:
        """Checkpoint fine-tuned. Raise `TierUnavailableError` nếu checkpoint không tồn tại/lỗi load."""

    @abstractmethod
    def _try_tier2(self, *args: object, **kwargs: object) -> T:
        """Pretrained zero/few-shot. Raise `TierUnavailableError` nếu thiếu thư viện/GPU/mạng."""

    @abstractmethod
    def _try_tier3(self, *args: object, **kwargs: object) -> T:
        """Rule-based thuần — PHẢI LUÔN THÀNH CÔNG."""
