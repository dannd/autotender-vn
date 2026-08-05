"""Hybrid retrieval (dense FAISS + sparse BM25, hợp nhất bằng Reciprocal Rank Fusion) cho
kho tri thức luật thật — đường truy xuất CHÍNH của bản redesign RAG+LLM (khác bản cũ, nơi
BM25 chỉ là Tier 3 dự phòng — xem `models/retriever.py`).

RRF được chọn thay vì weighted-sum vì không cần chuẩn hoá thang điểm giữa 2 hệ thống khác
hẳn nhau (cosine similarity của FAISS vs BM25 score không cùng đơn vị) — chỉ dùng thứ hạng
(rank), công thức chuẩn: score(d) = sum_{hệ thống} 1 / (k + rank_hệ_thống(d)), k=60 (giá trị
phổ biến trong literature, ít nhạy với k trong khoảng 10-100).

Metadata chunk (`chunks.jsonl`) và index FAISS đều là ARTIFACT build được từ
`scripts/build_legal_index.py`, không commit vào git (`.gitignore`). Nếu chưa build,
`__init__` tự chunk lại trực tiếp từ `data/samples/legal_corpus/*.jsonl` (đã commit, chunk
nhanh — không cần ML) để BM25/`retrieve_sparse` LUÔN dùng được ngay cả khi chưa chạy script
build — giữ đúng nguyên tắc "Tier 3 luôn thành công" cho các module gọi retriever này
(`models/generator.py`, `models/legal_qa.py`). Chỉ riêng FAISS (dense) mới thật sự cần
`scripts/build_legal_index.py` — thiếu thì `retrieve_dense`/`retrieve`/`retrieve_reranked`
raise lỗi rõ ràng, các module gọi (Tier 1) tự rơi xuống Tier 3 (`BaseModule.run`).
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

        chunks_path = self._index_dir / "chunks.jsonl"
        if chunks_path.exists():
            self._chunks = self._load_chunk_metadata(chunks_path)
        else:
            self._chunks = self._chunk_legal_corpus_directly()
        if not self._chunks:
            raise RuntimeError("Kho tri thức luật thật rỗng — kiểm tra data/samples/legal_corpus/*.jsonl.")

        self._faiss_index: FaissChunkIndex | None = None
        model_dir = self._index_dir / model_key
        dim_path, index_path = model_dir / "dim.txt", model_dir / "index.faiss"
        if dim_path.exists() and index_path.exists():
            self._faiss_index = FaissChunkIndex.load(index_path, dim=int(dim_path.read_text(encoding="utf-8").strip()))

        self._bm25_index: BM25Index | None = None
        self._encoder = None  # tải trễ — chỉ cần khi có truy vấn thật
        self._cross_encoder_name = CROSS_ENCODER_MODEL

    @staticmethod
    def _chunk_legal_corpus_directly() -> list[dict]:
        from autotender.rag.chunker import chunk_legal_corpus_dir

        raw_chunks = chunk_legal_corpus_dir(resolve_path("data/samples/legal_corpus"))
        return [
            {"chunk_id": c.chunk_id, "text": c.text, "source_doc": c.source_doc, "law_id": c.law_id, "dieu_so": c.dieu_so}
            for c in raw_chunks
        ]

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

    @property
    def has_dense_index(self) -> bool:
        """True nếu đã build FAISS (`scripts/build_legal_index.py`) cho `model_key` này —
        dùng để hiển thị trạng thái trên GUI (Trang 6) mà không cần đọc thuộc tính private."""
        return self._faiss_index is not None

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
        if self._faiss_index is None:
            raise RuntimeError(
                f"Chưa có FAISS index cho model '{self.model_key}' tại {self._index_dir / self.model_key} — "
                "chạy `python scripts/build_legal_index.py` trước."
            )
        encoder = self._get_encoder()
        query_vec = encoder.encode([query], show_progress_bar=False)[0]
        indices, scores = self._faiss_index.search(query_vec, min(top_k, len(self._chunks)))
        return [(i, s) for i, s in zip(indices, scores) if i >= 0]

    def retrieve_sparse(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        return self._get_bm25().search(query, min(top_k, len(self._chunks)))


    def _fuse_rrf(self, query: str, candidate_k: int) -> list[tuple[int, float]]:
        # Dense chỉ chạy được nếu đã build FAISS (`scripts/build_legal_index.py`); nếu chưa,
        # coi như không có kết quả dense thay vì raise — để `retrieve`/`retrieve_reranked`
        # luôn thành công (BM25-only), giữ nguyên tắc "Tier 3 luôn chạy được" cho các module
        # gọi retriever này mà không cần biết trước FAISS đã sẵn sàng hay chưa.
        dense = self.retrieve_dense(query, candidate_k) if self._faiss_index is not None else []
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
