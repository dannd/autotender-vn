"""Nạp cấu hình từ configs/*.yaml bằng pydantic-settings.

Dùng chung cho toàn bộ hệ thống để tránh hard-code đường dẫn/tham số rải rác.
"""

from __future__ import annotations

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


class AppSettings(BaseModel):
    app: AppConfig = AppConfig()
    sections_scope: list[str] = ["chuong_III", "chuong_V"]
    pdf_export: PdfExportConfig = PdfExportConfig()


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


@lru_cache
def get_app_settings() -> AppSettings:
    return AppSettings.model_validate(_load_yaml("app.yaml"))


@lru_cache
def get_crawler_settings() -> CrawlerSettings:
    return CrawlerSettings.model_validate(_load_yaml("crawler.yaml"))


@lru_cache
def get_models_settings() -> ModelsSettings:
    return ModelsSettings.model_validate(_load_yaml("models.yaml"))


def resolve_path(relative: str) -> Path:
    """Trả về đường dẫn tuyệt đối tính từ gốc dự án cho một đường dẫn tương đối trong config."""
    return PROJECT_ROOT / relative
