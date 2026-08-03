"""Hợp đồng dữ liệu (data contract) dùng chung cho toàn bộ hệ thống AutoTender-VN.

Mọi module (crawler, ingestion, NER, RAG, generator, compliance, HITL, export)
giao tiếp với nhau DUY NHẤT qua các model Pydantic định nghĩa ở đây.
Không module nào được tự ý định nghĩa lại các trường dữ liệu này.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class TenderNotice(BaseModel):
    """Bản ghi một thông báo mời thầu, thu thập từ Hệ thống mạng đấu thầu quốc gia."""

    tbmt_id: str
    package_name: str
    investor: str
    procuring_entity: str | None = None
    package_value: float | None = None  # VND
    currency: str = "VND"
    funding_source: str | None = None
    selection_method: str | None = None  # đấu thầu rộng rãi, chào hàng cạnh tranh...
    contract_type: str | None = None
    package_type: str | None = None  # hàng hóa | xây lắp | tư vấn | phi tư vấn
    execution_time: str | None = None
    publish_date: date | None = None
    close_date: date | None = None
    attachments: list[str] = Field(default_factory=list)
    source_url: str


class ExtractedField(BaseModel):
    """Một trường thông tin được trích xuất từ văn bản KHLCNT (bởi NER model hoặc regex)."""

    name: str
    value: str
    confidence: float
    char_start: int | None = None  # để highlight trong văn bản gốc
    char_end: int | None = None
    source: Literal["ner_model", "regex", "manual"]


class RetrievedChunk(BaseModel):
    """Một đoạn văn bản được truy xuất từ corpus RAG (mẫu điều khoản, văn bản pháp luật)."""

    chunk_id: str
    text: str
    source_doc: str  # "Thông tư 79/2025/TT-BTC — Mẫu số 1" ...
    score: float


class ComplianceFlag(BaseModel):
    """Cờ tuân thủ do module M6 (Compliance Guard) gắn vào một câu sinh ra."""

    rule_code: Literal["R1", "R2", "R3", "R4", "OK"]
    severity: Literal["cao", "trung_binh", "thap"]
    sentence: str
    explanation: str
    evidence: list[RetrievedChunk] = Field(default_factory=list)
    confidence: float


class HSMTSection(BaseModel):
    """Một mục/khoản trong E-HSMT, sinh bởi M5 và chờ người dùng phê duyệt (HITL)."""

    section_id: str  # "chuong_III.muc_2.1"
    title: str
    generated_text: str
    edited_text: str | None = None
    status: Literal["draft", "edited", "approved", "rejected"] = "draft"
    citations: list[RetrievedChunk] = Field(default_factory=list)
    flags: list[ComplianceFlag] = Field(default_factory=list)
    model_tier: Literal[1, 2, 3]
    generated_at: datetime
    approved_by: str | None = None
    approved_at: datetime | None = None

    @property
    def current_text(self) -> str:
        """Văn bản hiện hành: ưu tiên bản người dùng đã sửa, nếu chưa sửa thì dùng bản sinh ra."""
        return self.edited_text if self.edited_text is not None else self.generated_text


class HSMTDocument(BaseModel):
    """Toàn bộ hồ sơ E-HSMT đang soạn thảo cho một gói thầu."""

    doc_id: str
    package: TenderNotice
    fields: list[ExtractedField] = Field(default_factory=list)
    sections: list[HSMTSection] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    @property
    def is_fully_approved(self) -> bool:
        return bool(self.sections) and all(s.status == "approved" for s in self.sections)

    @property
    def approval_progress(self) -> tuple[int, int]:
        approved = sum(1 for s in self.sections if s.status == "approved")
        return approved, len(self.sections)
