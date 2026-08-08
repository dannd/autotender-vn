"""BM25 thuần Python — dùng làm Tier 3 (rule-based) cho M4 Retrieval, không phụ thuộc
thư viện ML nặng nào, LUÔN HOẠT ĐỘNG kể cả khi không có mạng/GPU (Mục 2.1).

Cũng dùng làm baseline bắt buộc để so sánh với bi-encoder đã fine-tune (Mục 10).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25Index:
    doc_tokens: list[list[str]]
    doc_freqs: dict[str, int]  # số văn bản chứa từ
    avg_doc_len: float
    k1: float = 1.5
    b: float = 0.75

    def score(self, query: str, doc_index: int) -> float:
        query_tokens = tokenize(query)
        tokens = self.doc_tokens[doc_index]
        doc_len = len(tokens)
        term_counts = Counter(tokens)
        n_docs = len(self.doc_tokens)
        score = 0.0
        for term in query_tokens:
            if term not in term_counts:
                continue
            df = self.doc_freqs.get(term, 0)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            tf = term_counts[term]
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int, allowed_indices: set[int] | None = None) -> list[tuple[int, float]]:
        """`allowed_indices` (tuỳ chọn): chỉ tính điểm/xếp hạng trong tập chỉ số này — dùng
        cho metadata filtering theo loại văn bản (`HybridLegalRetriever`), lọc TRƯỚC khi xếp
        hạng thay vì lọc top-k đã cắt sẵn (tránh bỏ sót kết quả đúng nằm ngoài top-k không lọc
        nhưng lẽ ra phải vào top-k SAU KHI lọc)."""
        indices = range(len(self.doc_tokens)) if allowed_indices is None else sorted(allowed_indices)
        scores = [(i, self.score(query, i)) for i in indices]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def build_bm25_index(documents: list[str]) -> BM25Index:
    doc_tokens = [tokenize(d) for d in documents]
    doc_freqs: dict[str, int] = defaultdict(int)
    for tokens in doc_tokens:
        for term in set(tokens):
            doc_freqs[term] += 1
    avg_len = sum(len(t) for t in doc_tokens) / max(len(doc_tokens), 1)
    return BM25Index(doc_tokens=doc_tokens, doc_freqs=dict(doc_freqs), avg_doc_len=avg_len or 1.0)
