"""Trích xuất text từ PDF (Mục 6/M1).

Dùng PyMuPDF (fitz) để giữ layout theo trang/block. Nếu một trang có ít hơn
`MIN_CHARS_TEXT_LAYER` ký tự, coi như trang scan (không có text layer) và
chuyển sang OCR (xem `ocr.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autotender.utils.logging import get_logger
from autotender.utils.vn_text import normalize_document_text

logger = get_logger(__name__)

MIN_CHARS_TEXT_LAYER = 50


@dataclass
class PageResult:
    page_number: int  # 1-indexed
    text: str
    used_ocr: bool


@dataclass
class PdfExtractionResult:
    pages: list[PageResult]
    full_text: str
    any_ocr_used: bool


def extract_pdf_text(pdf_path: str | Path, ocr_fn=None) -> PdfExtractionResult:
    """Trích text từng trang. `ocr_fn(page_image_bytes) -> str` được gọi khi trang là scan.

    Nếu `ocr_fn` là None và gặp trang scan, trả về placeholder rõ ràng thay vì bịa nội dung.
    """
    import fitz  # PyMuPDF — import trễ để module này import được kể cả khi chưa cài lib

    pdf_path = Path(pdf_path)
    pages: list[PageResult] = []
    any_ocr = False

    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            used_ocr = False
            if len(text.strip()) < MIN_CHARS_TEXT_LAYER:
                used_ocr = True
                any_ocr = True
                if ocr_fn is not None:
                    pix = page.get_pixmap(dpi=200)
                    image_bytes = pix.tobytes("png")
                    try:
                        text = ocr_fn(image_bytes)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("OCR lỗi ở trang %d: %s", i, e)
                        text = f"[CẦN NGƯỜI DÙNG BỔ SUNG: trang {i} không đọc được (OCR lỗi)]"
                else:
                    text = f"[CẦN NGƯỜI DÙNG BỔ SUNG: trang {i} là bản scan, chưa có OCR]"
            pages.append(PageResult(page_number=i, text=normalize_document_text(text), used_ocr=used_ocr))
    finally:
        doc.close()

    full_text = "\n\n".join(p.text for p in pages)
    return PdfExtractionResult(pages=pages, full_text=full_text, any_ocr_used=any_ocr)
