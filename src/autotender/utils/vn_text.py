"""Chuẩn hoá và xử lý văn bản tiếng Việt dùng chung (Mục 6/M1).

- NFC normalization: nhiều PDF/DOCX xuất ra tổ hợp Unicode chưa chuẩn hoá (NFD),
  dẫn đến lỗi tìm kiếm/so khớp dấu tiếng Việt nếu không đưa về NFC trước.
- Gộp dòng bị ngắt: PyMuPDF trích theo layout nên câu dài bị xuống dòng giữa chừng.
- Tách câu: xử lý các viết tắt phổ biến trong văn bản pháp lý/đấu thầu tiếng Việt
  (không tách câu ngay sau các viết tắt này dù có dấu chấm).
"""

from __future__ import annotations

import re
import unicodedata

# Các viết tắt không được coi là kết thúc câu khi theo sau bởi dấu chấm.
_ABBREVIATIONS = [
    "TT", "NĐ", "NĐ-CP", "QĐ", "TW", "UBND", "HĐND", "TNHH", "CP", "BTC",
    "KHLCNT", "HSMT", "HSDT", "E-HSMT", "E-HSDT", "STT", "ThS", "TS", "PGS",
    "GS", "TP", "Q", "P", "Đ", "khoản", "điểm", "mục", "chương", "Điều",
]

_ABBR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.(\s*)(?=[a-zà-ỹ0-9])",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ỵ0-9])")


def normalize_nfc(text: str) -> str:
    """Chuẩn hoá Unicode về dạng NFC (tổ hợp sẵn) — bắt buộc trước khi xử lý tiếp."""
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Đưa non-breaking space (U+00A0, hay gặp khi PDF/DOCX xuất ra) về space thường."""
    return text.replace(" ", " ")


def merge_broken_lines(text: str) -> str:
    """Gộp các dòng bị ngắt giữa câu do layout PDF (dòng không kết thúc bằng dấu câu)."""
    lines = text.split("\n")
    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            merged.append("")
            continue
        if merged and merged[-1] and not re.search(r"[.!?:;•\-]\s*$", merged[-1]):
            merged[-1] = merged[-1].rstrip() + " " + stripped
        else:
            merged.append(stripped)
    return "\n".join(merged)


def split_sentences(text: str) -> list[str]:
    """Tách câu, tránh tách nhầm sau các viết tắt tiếng Việt thường gặp."""
    protected = _ABBR_PATTERN.sub(lambda m: m.group(1) + "<DOT>" + m.group(2), text)
    raw_sentences = _SENTENCE_SPLIT_RE.split(protected)
    return [s.replace("<DOT>", ".").strip() for s in raw_sentences if s.strip()]


def normalize_document_text(raw_text: str) -> str:
    """Pipeline chuẩn hoá đầy đủ: NFC -> chuẩn hoá khoảng trắng -> gộp dòng vỡ.

    Dùng trước khi đưa vào NER/RAG.
    """
    return merge_broken_lines(normalize_whitespace(normalize_nfc(raw_text)))
