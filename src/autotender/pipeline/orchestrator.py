"""Orchestrator — ghép M1 (ingest) -> M2 (NER) -> M3 (classify) -> M4 (RAG) -> M5 (generate)
-> M6 (compliance) thành một luồng xử lý dùng chung cho GUI (Mục 4, Mục 7).

Mỗi bước dùng đúng module 3-tier tương ứng — orchestrator không tự quyết định tier,
chỉ đọc `active_tier` sau khi module chạy xong để hiển thị badge trên UI.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from autotender.ingest.docx_reader import extract_docx_text
from autotender.ingest.ocr import ocr_image_bytes
from autotender.ingest.pdf_reader import extract_pdf_text
from autotender.models.classifier import ClassificationResult, ClassifierModule
from autotender.models.compliance import ComplianceModule, check_document_completeness
from autotender.models.generator import SECTION_DEFINITIONS, GeneratorModule
from autotender.models.ner import NERModule
from autotender.rag.hybrid_retriever import HybridLegalRetriever
from autotender.schemas import ComplianceFlag, ExtractedField, HSMTDocument, HSMTSection, TenderNotice
from autotender.utils.vn_text import normalize_document_text


class Orchestrator:
    def __init__(self, retriever: HybridLegalRetriever | None = None):
        self.ner = NERModule()
        self.classifier = ClassifierModule()
        self.retriever = retriever or HybridLegalRetriever()
        self.generator = GeneratorModule(self.retriever)
        self.compliance = ComplianceModule()

    # -- M1 Ingestion ---------------------------------------------------------
    def ingest_file(self, path: str | Path) -> str:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            result = extract_pdf_text(path, ocr_fn=ocr_image_bytes)
            return result.full_text
        if suffix == ".docx":
            return extract_docx_text(path)
        if suffix == ".txt":
            return normalize_document_text(path.read_text(encoding="utf-8"))
        raise ValueError(f"Định dạng không hỗ trợ: {suffix} (chỉ nhận .pdf/.docx/.txt)")

    def ingest_text(self, raw_text: str) -> str:
        return normalize_document_text(raw_text)

    # -- M2 NER -----------------------------------------------------------------
    def extract_fields(self, text: str) -> list[ExtractedField]:
        return self.ner.extract(text)

    # -- M3 Classifier ------------------------------------------------------------
    def classify_package(self, text: str) -> ClassificationResult:
        return self.classifier.classify(text)

    # -- Tạo tài liệu HSMT rỗng (chưa sinh mục nào) ----------------------------
    def create_document(self, doc_id: str, package: TenderNotice, fields: list[ExtractedField]) -> HSMTDocument:
        now = datetime.now(timezone.utc)
        return HSMTDocument(doc_id=doc_id, package=package, fields=fields, sections=[], created_at=now, updated_at=now)

    # -- M4+M5+M6: sinh 1 mục, kèm compliance check --------------------------
    def generate_section(self, section_id: str, fields: list[ExtractedField]) -> HSMTSection:
        generated = self.generator.generate_section(section_id, fields)
        compliance_flags = self.compliance.check_text(generated.text, generated.citations)
        all_flags = generated.flags + compliance_flags
        return HSMTSection(
            section_id=generated.section_id,
            title=generated.title,
            generated_text=generated.text,
            status="draft",
            citations=generated.citations,
            flags=all_flags,
            model_tier=self.generator.active_tier or 3,
            generated_at=datetime.now(timezone.utc),
        )

    def generate_all_sections(self, fields: list[ExtractedField]) -> list[HSMTSection]:
        return [self.generate_section(section_id, fields) for section_id in SECTION_DEFINITIONS]

    # -- Mức 2 (đề cương RAG+LLM): soạn trọn Chương III end-to-end -------------
    def generate_chuong_iii(self, fields: list[ExtractedField]) -> list[HSMTSection]:
        """Sinh đủ 4 mục của Chương III — Tiêu chuẩn đánh giá E-HSDT (năng lực-kinh nghiệm,
        kỹ thuật, giá, nhãn hiệu/xuất xứ), mỗi mục kèm trích dẫn + cờ tuân thủ."""
        section_ids = [sid for sid in SECTION_DEFINITIONS if sid.startswith("chuong_III.")]
        return [self.generate_section(section_id, fields) for section_id in section_ids]

    def generate_chuong_v(self, fields: list[ExtractedField]) -> list[HSMTSection]:
        """Sinh đủ 4 mục của Chương V — Yêu cầu về kỹ thuật (phạm vi cung cấp, thông số kỹ
        thuật, bảo hành/bảo trì, tiến độ), mỗi mục kèm trích dẫn + cờ tuân thủ."""
        section_ids = [sid for sid in SECTION_DEFINITIONS if sid.startswith("chuong_V.")]
        return [self.generate_section(section_id, fields) for section_id in section_ids]

    # -- Cờ R5: kiểm tra đủ thành phần bắt buộc (Điều 26 Khoản 2 NĐ 214/2025/NĐ-CP,
    # chi tiết hoá Điều 44 Luật Đấu thầu) trong phạm vi Chương III + Chương V --------
    def check_completeness(self, sections: list[HSMTSection]) -> list[ComplianceFlag]:
        return check_document_completeness(sections)
