import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autotender.models.base import TierUnavailableError  # noqa: E402


def force_tier3(monkeypatch, module) -> None:
    """Ép một `BaseModule` chạy Tier 3 (rule-based) bất kể môi trường có cài
    torch/transformers/sentence-transformers hay không — các test `*_tier3_*` kiểm tra
    hành vi rule-based cụ thể, không nên phụ thuộc vào việc máy chạy test có cài sẵn thư
    viện ML nặng hay chưa (Tier 1/2 có thể "tình cờ" khả dụng và cho kết quả khác Tier 3)."""

    def _raise(*args: object, **kwargs: object):
        raise TierUnavailableError("forced tier3 in test")

    monkeypatch.setattr(module, "_try_tier1", _raise)
    monkeypatch.setattr(module, "_try_tier2", _raise)
