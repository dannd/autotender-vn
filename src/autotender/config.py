"""Nạp cấu hình từ configs/*.yaml bằng pydantic-settings.

Dùng chung cho toàn bộ hệ thống để tránh hard-code đường dẫn/tham số rải rác.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIGS_DIR / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class PdfExportConfig(BaseModel):
    page_size: str = "A4"
    margin_top_mm: int = 20
    margin_bottom_mm: int = 20
    margin_left_mm: int = 30
    margin_right_mm: int = 20
    font_family: str = "Times New Roman"
    font_fallback: str = "Tinos"
    font_size_pt: int = 13
    draft_watermark: bool = True
    page_numbers: bool = True


class AppConfig(BaseModel):
    name: str = "AutoTender-VN"
    data_dir: str = "data"
    db_path: str = "data/processed/hitl.db"
    auth_db_path: str = "data/processed/auth.db"
    audit_db_path: str = "data/processed/audit.db"
    reports_dir: str = "reports"


class ClaudeModelPricing(BaseModel):
    input_usd_per_mtok: float
    output_usd_per_mtok: float


class ClaudeBudgetConfig(BaseModel):
    # Trần chi tiêu Claude API tính từ lúc process Python hiện tại khởi động (không phải
    # per-request Streamlit session, không bền qua restart) — bộ đếm nằm hẳn trong bộ nhớ
    # tiến trình (xem generation/claude_client.py). Đủ để chặn 1 vòng lặp lỗi/bug tiêu tiền
    # không kiểm soát ở quy mô bản beta; KHÔNG phải hệ thống kế toán chi phí đa người dùng
    # thật — muốn vậy cần theo dõi ở tầng nhà cung cấp (Anthropic Console budget alert) hoặc
    # 1 bảng chi phí bền vững riêng, ngoài phạm vi bản beta này.
    usd_cap_per_process: float = 5.0
    # Giá tham khảo (USD / 1 triệu token) — CẦN đối chiếu lại với trang giá chính thức của
    # Anthropic trước khi dùng số này để ra quyết định ngân sách thật; để trong config (không
    # hard-code) chính là để sửa nhanh khi giá đổi mà không cần sửa code.
    pricing_usd_per_mtok: dict[str, ClaudeModelPricing] = {
        "claude-sonnet-5": ClaudeModelPricing(input_usd_per_mtok=3.0, output_usd_per_mtok=15.0),
        "claude-haiku-4-5-20251001": ClaudeModelPricing(input_usd_per_mtok=1.0, output_usd_per_mtok=5.0),
    }


class QdrantConfig(BaseModel):
    """Cấu hình kết nối Qdrant Vector DB.

    Các giá trị đọc từ biến môi trường QDRANT_HOST / QDRANT_PORT / QDRANT_COLLECTION nếu có
    (ưu tiên hơn app.yaml) — cho phép override nhanh trong CI/CD hoặc khi chạy trên máy chủ
    khác mà không cần sửa config file.
    """

    host: str = "localhost"
    port: int = 6333
    collection: str = "legal_chunks"
    timeout: int = 10

    @classmethod
    def from_env_or_yaml(cls, yaml_data: dict[str, Any]) -> "QdrantConfig":
        """Tạo QdrantConfig ưu tiên biến môi trường, fallback sang yaml_data."""
        base = cls(**yaml_data) if yaml_data else cls()
        return cls(
            host=os.environ.get("QDRANT_HOST", base.host),
            port=int(os.environ.get("QDRANT_PORT", base.port)),
            collection=os.environ.get("QDRANT_COLLECTION", base.collection),
            timeout=base.timeout,
        )


class EmbeddingConfig(BaseModel):
    """Cấu hình embedding model dùng cho index và retrieval.

    model_key phải khớp với key trong `rag/embedding_models.py::EMBEDDING_MODELS`.
    vector_size phải khớp với chiều output thực tế của model — dùng để tạo Qdrant collection.
    """

    model_key: str = "vi_bi_encoder"
    vector_size: int = 768   # vi_bi_encoder default
    batch_size: int = 32

    @classmethod
    def from_env_or_yaml(cls, yaml_data: dict[str, Any]) -> "EmbeddingConfig":
        """Ưu tiên biến môi trường EMBEDDING_MODEL_KEY nếu có."""
        base = cls(**yaml_data) if yaml_data else cls()
        model_key = os.environ.get("EMBEDDING_MODEL_KEY", base.model_key)
        return cls(model_key=model_key, vector_size=base.vector_size, batch_size=base.batch_size)


class LLMGatewayConfig(BaseModel):
    """Cấu hình kết nối Universal OpenAI-compatible LLM Gateway (WokuShop / OpenAI / vLLM)."""

    base_url: str = "https://llm.wokushop.com/v1"
    default_model: str = "claude-3-5-sonnet-20241022"
    timeout_seconds: int = 60
    usd_cap_per_process: float = 5.0
    pricing_usd_per_mtok: dict[str, ClaudeModelPricing] = {
        "claude-3-5-sonnet-20241022": ClaudeModelPricing(input_usd_per_mtok=3.0, output_usd_per_mtok=15.0),
        "claude-sonnet-5": ClaudeModelPricing(input_usd_per_mtok=3.0, output_usd_per_mtok=15.0),
        "claude-haiku-4-5-20251001": ClaudeModelPricing(input_usd_per_mtok=1.0, output_usd_per_mtok=5.0),
        "deepseek-chat": ClaudeModelPricing(input_usd_per_mtok=0.14, output_usd_per_mtok=0.28),
        "gpt-4o": ClaudeModelPricing(input_usd_per_mtok=2.5, output_usd_per_mtok=10.0),
        "gpt-4o-mini": ClaudeModelPricing(input_usd_per_mtok=0.15, output_usd_per_mtok=0.60),
    }

    @classmethod
    def from_env_or_yaml(cls, yaml_data: dict[str, Any]) -> "LLMGatewayConfig":
        base = cls(**yaml_data) if yaml_data else cls()
        base_url = os.environ.get("LLM_BASE_URL", base.base_url)
        default_model = os.environ.get("LLM_MODEL", base.default_model)
        return cls(
            base_url=base_url,
            default_model=default_model,
            timeout_seconds=base.timeout_seconds,
            usd_cap_per_process=base.usd_cap_per_process,
            pricing_usd_per_mtok=base.pricing_usd_per_mtok,
        )


class AppSettings(BaseModel):
    app: AppConfig = AppConfig()
    sections_scope: list[str] = [
        "chuong_I", "chuong_II", "chuong_III", "chuong_IV",
        "chuong_V", "chuong_VI", "chuong_VII", "chuong_VIII",
    ]
    pdf_export: PdfExportConfig = PdfExportConfig()
    claude_budget: ClaudeBudgetConfig = ClaudeBudgetConfig()
    llm_gateway: LLMGatewayConfig = LLMGatewayConfig()
    qdrant: QdrantConfig = QdrantConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()


class CrawlerConfig(BaseModel):
    base_url: str = "https://muasamcong.mpi.gov.vn"
    user_agent: str = "AutoTenderVN-ResearchBot/0.1"
    min_request_interval_seconds: float = 2.0
    respect_robots_txt: bool = True
    timeout_seconds: int = 30
    max_retries: int = 3
    cache_dir: str = "data/raw"
    state_file: str = "data/raw/_state.json"
    default_source: str = "local"
    samples_dir: str = "data/samples"


class CrawlerSettings(BaseModel):
    crawler: CrawlerConfig = CrawlerConfig()


class ModelsSettings(BaseModel):
    ner: dict[str, Any] = {}
    generator: dict[str, Any] = {}
    compliance: dict[str, Any] = {}
    qa: dict[str, Any] = {}


def _build_app_settings() -> AppSettings:
    """Tạo AppSettings từ app.yaml với override từ biến môi trường cho Qdrant/Embedding/LLM."""
    raw = _load_yaml("app.yaml")
    # Qdrant, Embedding, LLM Gateway có logic đọc env var trước yaml
    qdrant_cfg = QdrantConfig.from_env_or_yaml(raw.pop("qdrant", {}))
    embedding_cfg = EmbeddingConfig.from_env_or_yaml(raw.pop("embedding", {}))
    llm_cfg = LLMGatewayConfig.from_env_or_yaml(raw.pop("llm_gateway", {}))
    settings = AppSettings.model_validate(raw)
    settings.qdrant = qdrant_cfg
    settings.embedding = embedding_cfg
    settings.llm_gateway = llm_cfg
    return settings


@lru_cache
def get_app_settings() -> AppSettings:
    return _build_app_settings()


@lru_cache
def get_crawler_settings() -> CrawlerSettings:
    return CrawlerSettings.model_validate(_load_yaml("crawler.yaml"))


@lru_cache
def get_models_settings() -> ModelsSettings:
    return ModelsSettings.model_validate(_load_yaml("models.yaml"))


def resolve_path(relative: str) -> Path:
    """Trả về đường dẫn tuyệt đối tính từ gốc dự án cho một đường dẫn tương đối trong config."""
    return PROJECT_ROOT / relative

