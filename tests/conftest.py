import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autotender.models.base import TierUnavailableError  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_env_api_keys(monkeypatch):
    """Mặc định trong test suite, cô lập môi trường mạng/API Key LLM.

    Tránh các unit test (vd: test_orchestrator) vô tình gọi HTTP request ra Gateway
    nếu người dùng có file .env cục bộ. Các test kiểm tra LLM sẽ tự monkeypatch.setenv trong thân hàm.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def force_tier3(monkeypatch, module) -> None:
    """Ép một `BaseModule` chạy Tier 3 (rule-based) bất kể môi trường có cài
    torch/transformers/sentence-transformers hay không."""

    def _raise(*args: object, **kwargs: object):
        raise TierUnavailableError("forced tier3 in test")

    monkeypatch.setattr(module, "_try_tier1", _raise)
    monkeypatch.setattr(module, "_try_tier2", _raise)
