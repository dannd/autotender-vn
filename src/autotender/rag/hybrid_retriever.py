"""Hybrid retrieval (dense FAISS + sparse BM25, hợp nhất bằng Reciprocal Rank Fusion) cho
kho tri thức luật thật — đường truy xuất CHÍNH của bản redesign RAG+LLM (khác bản cũ, nơi
BM25 chỉ là Tier 3 dự phòng — xem `models/retriever.py`).

RRF được chọn thay vì weighted-sum vì không cần chuẩn hoá thang điểm giữa 2 hệ thống khác
hẳn nhau (cosine similarity của FAISS vs BM25 score không cùng đơn vị) — chỉ dùng thứ hạng
(rank), công thức chuẩn: score(d) = sum_{hệ thống} 1 / (k + rank_hệ_thống(d)), k=60 (giá trị
phổ biến trong literature, ít nhạy với k trong khoảng 10-100).
"""

from __future__ import annotations

import json
from pathlib import Path

from autotender.config import resolve_path
from autotender.rag.bm25 import BM25Index, build_bm25_index
from autotender.rag.embedding_models import CROSS_ENCODER_MODEL, DEFAULT_EMBEDDING_MODEL_KEY, EMBEDDING_MODELS
from autotender.rag.index import FaissChunkIndex
from autotender.rag.rerank import rerank_with_cross_encoder
from autotender.schemas import RetrievedChunk

RRF_K = 60


class HybridLegalRetriever:
    def __init__(self, model_key: str = DEFAULT_EMBEDDING_MODEL_KEY, index_dir: str | Path | None = None):
        if model_key not in EMBEDDING_MODELS:
            raise ValueError(f"model_key '{model_key}' không có trong EMBEDDING_MODELS: {list(EMBEDDING_MODELS)}")
        self.model_key = model_key
        self.model_name = EMBEDDING_MODELS[model_key]
        self._index_dir = Path(index_dir) if index_dir else resolve_path("data/index")

        self._chunks = self._load_chunk_metadata(self._index_dir / "chunks.jsonl")
        if not self._chunks:
            raise RuntimeError(
                f"Không tìm thấy chunk metadata tại {self._index_dir / 'chunks.jsonl'} — "
                "chạy scripts/build_legal_index.py trước."
            )

        model_dir = self._index_dir / model_key
        dim = int((model_dir / "dim.txt").read_text(encoding="utf-8").strip())
        self._faiss_index = FaissChunkIndex.load(model_dir / "index.faiss", dim=dim)

        self._bm25_index: BM25Index | None = None
        self._encoder = None  # tải trễ — chỉ cần khi có truy vấn thật
        self._cross_encoder_name = CROSS_ENCODER_MODEL

    @staticmethod
    def _load_chunk_metadata(path: Path) -> list[dict]:
        chunks = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def _get_bm25(self) -> BM25Index:
        if self._bm25_index is None:
            self._bm25_index = build_bm25_index([c["text"] for c in self._chunks])
        return self._bm25_index

    def _to_retrieved_chunk(self, idx: int, score: float) -> RetrievedChunk:
        c = self._chunks[idx]
        return RetrievedChunk(
            chunk_id=c["chunk_id"], text=c["text"], source_doc=c["source_doc"],
            score=score, law_id=c.get("law_id"), dieu_so=c.get("dieu_so"),
        )

    def retrieve_dense(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        encoder = self._get_encoder()
        query_vec = encoder.encode([query], show_progress_bar=False)[0]
        indices, scores = self._faiss_index.search(query_vec, min(top_k, len(self._chunks)))
        return [(i, s) for i, s in zip(indices, scores) if i >= 0]

    def retrieve_sparse(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        return self._get_bm25().search(query, min(top_k, len(self._chunks)))

    def _fuse_rrf(self, query: str, candidate_k: int) -> list[tuple[int, float]]:
        dense = self.retrieve_dense(query, candidate_k)
        sparse = self.retrieve_sparse(query, candidate_k)

        rrf_scores: dict[int, float] = {}
        for rank, (idx, _score) in enumerate(dense):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (idx, _score) in enumerate(sparse):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, query: str, top_k: int = 10, candidate_k: int = 50) -> list[RetrievedChunk]:
        """Hợp nhất dense + sparse bằng RRF. `candidate_k`: số ứng viên lấy từ MỖI hệ thống
        trước khi hợp nhất (mặc định 50, theo kế hoạch "top-50" trước rerank)."""
        ranked = self._fuse_rrf(query, candidate_k)[:top_k]
        return [self._to_retrieved_chunk(idx, score) for idx, score in ranked]

    def retrieve_reranked(self, query: str, top_k: int = 5, candidate_k: int = 50) -> list[RetrievedChunk]:
        """Hybrid RRF (top `candidate_k`) rồi rerank bằng cross-encoder xuống `top_k`
        (mặc định top-50 -> top-5, đúng theo kế hoạch). Chậm hơn `retrieve` nhiều lần
        (cross-encoder chạy N lần forward pass, N = candidate_k) — chỉ dùng khi cần độ
        chính xác cao nhất (Mức 2 soạn mục HSMT), không dùng cho việc so sánh tốc độ."""
        candidates = self._fuse_rrf(query, candidate_k)
        if not candidates:
            return []

        texts = [self._chunks[idx]["text"] for idx, _ in candidates]
        reranked = rerank_with_cross_encoder(self._cross_encoder_name, query, texts, top_k)
        return [self._to_retrieved_chunk(candidates[i][0], score) for i, score in reranked]
