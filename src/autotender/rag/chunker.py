"""Chunking corpus RAG theo điều/khoản/mục (Mục 6/M4).

Quy ước file corpus: mỗi heading `## ` đánh dấu biên giới một mục — chunk theo mục,
nếu mục quá dài (> max_words) thì cắt tiếp theo cửa sổ trượt có overlap.

Ghi chú: spec quy định "tối đa 512 token, overlap 64" — ở đây xấp xỉ bằng số từ
(word count) thay vì token thật của tokenizer, vì việc chunk xảy ra trước khi biết
sẽ dùng tokenizer nào (PhoBERT/XLM-R có subword tokenizer khác nhau). Sai số này
được chấp nhận cho phạm vi đồ án 7 ngày.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from autotender.schemas import LegalArticle

MAX_WORDS = 400  # xấp xỉ 512 token
OVERLAP_WORDS = 50  # xấp xỉ 64 token

_HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_KHOAN_RE = re.compile(r"^(\d+)\.\s+", re.MULTILINE)


@dataclass
class RawChunk:
    chunk_id: str
    text: str
    source_doc: str
    # Metadata pháp lý (tuỳ chọn) — rỗng cho corpus markdown minh hoạ cũ, có giá trị
    # cho corpus luật thật (xem chunk_legal_article).
    law_id: str | None = None
    dieu_so: int | None = None


def _make_chunk_id(source_doc: str, index: int) -> str:
    key = f"{source_doc}::{index}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _split_long_section(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    parts = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        parts.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_words
    return parts


def chunk_markdown_file(path: str | Path, doc_label: str | None = None) -> list[RawChunk]:
    """Chunk một file corpus markdown theo heading `## `.

    `doc_label` là tên nguồn hiển thị cho người dùng (vd "[MINH HỌA] Mẫu Chương III").
    Nếu không truyền, dùng dòng `# ` đầu file (title) làm nhãn.
    """
    path = Path(path)
    content = path.read_text(encoding="utf-8")

    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    label = doc_label or (title_match.group(1) if title_match else path.stem)

    headings = list(_HEADING_RE.finditer(content))
    chunks: list[RawChunk] = []

    if not headings:
        sections = [(label, content)]
    else:
        sections = []
        for i, m in enumerate(headings):
            start = m.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
            heading_title = m.group(1).strip()
            sections.append((f"{label} — {heading_title}", content[start:end].strip()))

    idx = 0
    for section_label, section_text in sections:
        if not section_text.strip():
            continue
        for part in _split_long_section(section_text, MAX_WORDS, OVERLAP_WORDS):
            chunks.append(
                RawChunk(chunk_id=_make_chunk_id(section_label, idx), text=part.strip(), source_doc=section_label)
            )
            idx += 1
    return chunks


def chunk_corpus_dir(corpus_dir: str | Path) -> list[RawChunk]:
    """Chunk toàn bộ file `.md` trong thư mục corpus (bỏ qua README.md)."""
    corpus_dir = Path(corpus_dir)
    all_chunks: list[RawChunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        all_chunks.extend(chunk_markdown_file(path))
    return all_chunks


def _split_by_khoan(text: str, max_words: int, overlap_words: int) -> list[tuple[str | None, str]]:
    """Cắt nội dung một Điều theo ranh giới Khoản (`"1. ..."`, `"2. ..."` ở đầu dòng),
    gộp các khoản liên tiếp lại tới gần `max_words` để tránh chunk quá vụn. Nếu bản thân
    một khoản đã vượt `max_words` (vd Điều nhiều điểm a/b/c lồng bên trong), cắt tiếp
    bằng cửa sổ trượt như `_split_long_section` cho riêng khoản đó.

    Trả về list[(nhãn_khoản | None, text)] — nhãn khoản là số khoản ĐẦU TIÊN trong phần
    đó (vd "1" nếu gộp khoản 1-2, chỉ ghi "1" để không gây hiểu nhầm phạm vi).
    """
    matches = list(_KHOAN_RE.finditer(text))
    if not matches:
        # Điều không có cấu trúc khoản đánh số rõ (hiếm) — cắt bằng cửa sổ trượt thường.
        return [(None, part) for part in _split_long_section(text, max_words, overlap_words)]

    segments: list[tuple[str, str]] = []  # (so_khoan, text)
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segments.append((m.group(1), text[start:end].strip()))

    parts: list[tuple[str | None, str]] = []
    buf_khoan: str | None = None
    buf_text = ""
    for khoan_so, seg_text in segments:
        seg_words = len(seg_text.split())
        if seg_words > max_words:
            if buf_text:
                parts.append((buf_khoan, buf_text))
                buf_khoan, buf_text = None, ""
            for sub in _split_long_section(seg_text, max_words, overlap_words):
                parts.append((khoan_so, sub))
            continue
        candidate = f"{buf_text}\n\n{seg_text}".strip() if buf_text else seg_text
        if len(candidate.split()) > max_words and buf_text:
            parts.append((buf_khoan, buf_text))
            buf_khoan, buf_text = khoan_so, seg_text
        else:
            buf_text = candidate
            if buf_khoan is None:
                buf_khoan = khoan_so
    if buf_text:
        parts.append((buf_khoan, buf_text))
    return parts


def chunk_legal_article(article: LegalArticle) -> list[RawChunk]:
    """Chunk một `LegalArticle` — mặc định 1 Điều = 1 chunk (đủ ngữ cảnh cho trích dẫn
    chính xác); nếu Điều quá dài (> MAX_WORDS, hay gặp ở các Điều liệt kê nhiều khoản/điểm)
    thì cắt theo ranh giới Khoản thay vì cửa sổ trượt mù, để mỗi chunk vẫn là một đơn vị
    pháp lý trọn vẹn (Khoản) thay vì bị cắt giữa câu.
    """
    base_label = f"{article.law_name} — Điều {article.dieu_so}. {article.dieu_title}"
    word_count = len(article.text.split())

    if word_count <= MAX_WORDS:
        return [
            RawChunk(
                chunk_id=_make_chunk_id(f"{article.law_id}:{article.dieu_so}", 0),
                text=article.text,
                source_doc=base_label,
                law_id=article.law_id,
                dieu_so=article.dieu_so,
            )
        ]

    chunks: list[RawChunk] = []
    for idx, (khoan_so, part_text) in enumerate(_split_by_khoan(article.text, MAX_WORDS, OVERLAP_WORDS)):
        label = f"{base_label}, Khoản {khoan_so}" if khoan_so else base_label
        chunks.append(
            RawChunk(
                chunk_id=_make_chunk_id(f"{article.law_id}:{article.dieu_so}", idx),
                text=part_text.strip(),
                source_doc=label,
                law_id=article.law_id,
                dieu_so=article.dieu_so,
            )
        )
    return chunks


def load_legal_articles(path: str | Path) -> list[LegalArticle]:
    """Đọc file `.jsonl` do `scripts/fetch_legal_corpus.py` sinh ra thành `list[LegalArticle]`."""
    articles: list[LegalArticle] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            articles.append(LegalArticle(**json.loads(line)))
    return articles


def chunk_legal_corpus_dir(legal_corpus_dir: str | Path) -> list[RawChunk]:
    """Chunk toàn bộ file `.jsonl` (mỗi dòng 1 `LegalArticle`) trong thư mục corpus luật thật."""
    legal_corpus_dir = Path(legal_corpus_dir)
    all_chunks: list[RawChunk] = []
    for path in sorted(legal_corpus_dir.glob("*.jsonl")):
        for article in load_legal_articles(path):
            all_chunks.extend(chunk_legal_article(article))
    return all_chunks
