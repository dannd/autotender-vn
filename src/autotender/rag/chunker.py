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
import re
from dataclasses import dataclass
from pathlib import Path

MAX_WORDS = 400  # xấp xỉ 512 token
OVERLAP_WORDS = 50  # xấp xỉ 64 token

_HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


@dataclass
class RawChunk:
    chunk_id: str
    text: str
    source_doc: str


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
