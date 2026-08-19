"""Hybrid retrieval (dense Qdrant + sparse BM25, hợp nhất bằng Reciprocal Rank Fusion) cho
kho tri thức luật thật — phiên bản v2 dùng Qdrant thay vì FAISS flat file.

Thay đổi so với v1 (FAISS):
- Dense retrieval: gọi QdrantLegalStore.search() thay vì FaissChunkIndex.search()
- Metadata filtering: thực hiện IN Qdrant DB (không phải lọc Python sau query)
- Chunk metadata: đọc trực tiếp từ Qdrant payload (không cần maintain chunks.jsonl riêng)
- Fallback: nếu Qdrant không available, tự động dùng BM25-only (giữ nguyên Degraded Mode)

Những phần KHÔNG thay đổi:
- BM25 vẫn build in-memory từ chunk text (không cần lưu vào Qdrant)
- RRF fusion logic (k=60) giữ nguyên
- Cross-encoder reranking giữ nguyên
- expand_to_parent_article (parent chunk lookup) giữ nguyên
"""

from __future__ import annotations

from pathlib import Path

from autotender.config import get_app_settings, resolve_path
from autotender.rag.bm25 import BM25Index, build_bm25_index
from autotender.rag.embedding_models import (
    CROSS_ENCODER_MODEL,
    EMBEDDING_MODELS,
    encode_texts,
)
from autotender.rag.qdrant_store import QdrantLegalStore, QdrantUnavailableError
from autotender.rag.rerank import rerank_with_cross_encoder
from autotender.schemas import RetrievedChunk
from autotender.utils.logging import get_logger

logger = get_logger(__name__)

RRF_K = 60


class HybridLegalRetriever:
    """Hybrid retriever: Dense (Qdrant) + Sparse (BM25) + Rerank (Cross-encoder).

    Khởi tạo:
        retriever = HybridLegalRetriever()         # dùng model từ config
        retriever = HybridLegalRetriever("deepx_v1")
        retriever = HybridLegalRetriever("vi_bi_encoder")  # để so sánh ablation

    Fallback tự động: nếu Qdrant không available (chưa chạy docker compose), hệ thống
    tự dùng BM25-only. Không crash, chỉ log warning.
    """

    def __init__(
        self,
        model_key: str | Path | None = None,
        qdrant_store: QdrantLegalStore | None = None,
        index_dir: Path | str | None = None,
        **kwargs,
    ) -> None:
        cfg = get_app_settings()
        if isinstance(model_key, (Path, str)) and ("/" in str(model_key) or "\\" in str(model_key) or Path(str(model_key)).exists()):
            index_dir = index_dir or Path(model_key)
            model_key = kwargs.get("model_name") or cfg.embedding.model_key

        self.model_key = (model_key if isinstance(model_key, str) and model_key in EMBEDDING_MODELS else None) or cfg.embedding.model_key
        self.model_name = EMBEDDING_MODELS.get(self.model_key, self.model_key)
        self.index_dir = Path(index_dir) if index_dir else None
        self._faiss_index = None

        # QdrantLegalStore — dùng config từ app.yaml / env vars
        if qdrant_store is not None:
            self._qdrant = qdrant_store
        else:
            self._qdrant = QdrantLegalStore(cfg=cfg.qdrant, vector_size=cfg.embedding.vector_size)

        # Load chunk list để dùng cho BM25 (text-only, không cần vector).
        self._chunks: list[dict] = self._load_chunks()
        if not self._chunks:
            raise RuntimeError("Kho tri thức luật thật rỗng — kiểm tra data/samples/legal_corpus/*.jsonl.")

        self._bm25_index: BM25Index | None = None
        self._encoder = None        # lazy-load: tải model khi có query thật
        self._cross_encoder_name = CROSS_ENCODER_MODEL
        self._article_lookup: dict[tuple[str, int], str] | None = None  # lazy

    def _load_chunks(self) -> list[dict]:
        """Tải danh sách chunk cho BM25.

        Thứ tự ưu tiên:
        1. Từ index_dir / "chunks.jsonl" nếu có cung cấp (trong test hoặc custom build).
        2. Scroll tất cả points từ Qdrant (payload có đủ text).
        3. Fallback: chunk trực tiếp từ corpus file.
        """
        if self.index_dir and (self.index_dir / "chunks.jsonl").exists():
            import json
            chunks = []
            with open(self.index_dir / "chunks.jsonl", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
            if chunks:
                return chunks

        if self._qdrant.is_available() and self._qdrant.collection_exists():
            try:
                chunks = self._scroll_all_chunks_from_qdrant()
                if chunks:
                    logger.info("Đã tải %d chunks từ Qdrant collection '%s'.", len(chunks), self._qdrant._cfg.collection)
                    return chunks
            except Exception as e:  # noqa: BLE001
                logger.warning("Không đọc được chunks từ Qdrant (%s) — fallback về corpus file.", e)

        logger.info("Qdrant chưa có dữ liệu hoặc offline — chunk trực tiếp từ corpus (BM25-only mode).")
        return self._chunk_legal_corpus_directly()

    def _scroll_all_chunks_from_qdrant(self) -> list[dict]:
        """Scroll tất cả points từ Qdrant để lấy payload (text + metadata) cho BM25."""
        client = self._qdrant._get_client()
        collection = self._qdrant._cfg.collection
        chunks = []
        offset = None
        while True:
            result, next_offset = client.scroll(
                collection_name=collection,
                with_payload=True,
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for point in result:
                p = point.payload or {}
                chunks.append({
                    "chunk_id": p.get("chunk_id", str(point.id)),
                    "text": p.get("text", ""),
                    "source_doc": p.get("source_doc", ""),
                    "law_id": p.get("law_id"),
                    "dieu_so": p.get("dieu_so"),
                    # Giữ qdrant_point_id để map kết quả dense search về đúng chunk
                    "_qdrant_id": str(point.id),
                })
            if next_offset is None:
                break
            offset = next_offset
        return chunks

    @staticmethod
    def _chunk_legal_corpus_directly() -> list[dict]:
        from autotender.rag.chunker import chunk_legal_corpus_dir

        raw_chunks = chunk_legal_corpus_dir(resolve_path("data/samples/legal_corpus"))
        return [
            {"chunk_id": c.chunk_id, "text": c.text, "source_doc": c.source_doc,
             "law_id": c.law_id, "dieu_so": c.dieu_so}
            for c in raw_chunks
        ]

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def has_dense_index(self) -> bool:
        """True nếu Qdrant đang chạy và collection đã có dữ liệu, hoặc FAISS index có sẵn."""
        if self._qdrant.is_available() and self._qdrant.collection_exists():
            return True
        if self.index_dir and (self.index_dir / self.model_key / "index.faiss").exists():
            return True
        return False

    def _get_encoder(self):
        if self._encoder is None:
            from autotender.rag.embedding_models import load_embedding_model
            self._encoder = load_embedding_model(self.model_key)
        return self._encoder

    def _get_bm25(self) -> BM25Index:
        if self._bm25_index is None:
            self._bm25_index = build_bm25_index([c["text"] for c in self._chunks])
        return self._bm25_index


    def _chunk_index_by_id(self) -> dict[str, int]:
        """Map chunk_id → vị trí trong self._chunks — dùng để map kết quả Qdrant về index."""
        if not hasattr(self, "_chunk_id_map"):
            self._chunk_id_map = {c["chunk_id"]: i for i, c in enumerate(self._chunks)}
        return self._chunk_id_map

    def _to_retrieved_chunk(self, idx: int, score: float) -> RetrievedChunk:
        c = self._chunks[idx]
        return RetrievedChunk(
            chunk_id=c["chunk_id"], text=c["text"], source_doc=c["source_doc"],
            score=score, law_id=c.get("law_id"), dieu_so=c.get("dieu_so"),
        )

    def retrieve_dense(
        self, query: str, top_k: int = 50, law_ids: set[str] | None = None
    ) -> list[tuple[int, float]]:
        """Dense search qua Qdrant (hoặc FAISS fallback nếu có index_dir)."""
        if self._qdrant.is_available() and self._qdrant.collection_exists():
            encoder = self._get_encoder()
            query_vec = encode_texts(encoder, [query])[0]
            raw_results = self._qdrant.search(
                query_vector=query_vec,
                top_k=top_k,
                filter_law_ids=law_ids,
            )
            id_map = self._chunk_index_by_id()
            results: list[tuple[int, float]] = []
            for _, score, payload in raw_results:
                cid = payload.get("chunk_id", "")
                idx = id_map.get(cid)
                if idx is not None:
                    results.append((idx, score))
            return results

        # Fallback sang FAISS nếu có index_dir (trong unit test hoặc môi trường offline không Docker)
        if self.index_dir:
            faiss_path = self.index_dir / self.model_key / "index.faiss"
            if faiss_path.exists():
                if self._faiss_index is None:
                    from autotender.rag.index import FaissChunkIndex
                    dim_file = self.index_dir / self.model_key / "dim.txt"
                    dim = int(dim_file.read_text(encoding="utf-8").strip()) if dim_file.exists() else 768
                    self._faiss_index = FaissChunkIndex.load(faiss_path, dim=dim)
                encoder = self._get_encoder()
                query_vec = encode_texts(encoder, [query])[0]
                allowed = None
                if law_ids is not None:
                    allowed = {i for i, c in enumerate(self._chunks) if c.get("law_id") in law_ids}
                indices, scores = self._faiss_index.search(query_vec, min(top_k, len(self._chunks)))
                results = [(idx, score) for idx, score in zip(indices, scores) if idx >= 0]
                if allowed is not None:
                    results = [(idx, score) for idx, score in results if idx in allowed]
                return results

        raise RuntimeError("Chưa có dense index — chạy `python scripts/build_legal_index.py` hoặc khởi động Qdrant.")

    def retrieve_sparse(
        self, query: str, top_k: int = 50, law_ids: set[str] | None = None
    ) -> list[tuple[int, float]]:
        allowed = None
        if law_ids is not None:
            allowed = {i for i, c in enumerate(self._chunks) if c.get("law_id") in law_ids}
        return self._get_bm25().search(query, min(top_k, len(self._chunks)), allowed_indices=allowed)

    def _fuse_rrf(
        self, query: str, candidate_k: int, law_ids: set[str] | None = None
    ) -> list[tuple[int, float]]:
        """Fuse dense + sparse bằng RRF. Dense bị bỏ qua nếu Qdrant không available."""
        try:
            dense = self.retrieve_dense(query, candidate_k, law_ids=law_ids)
        except (RuntimeError, QdrantUnavailableError):
            logger.warning("Dense retrieval không available — dùng BM25-only (Degraded Mode).")
            dense = []

        sparse = self.retrieve_sparse(query, candidate_k, law_ids=law_ids)

        rrf_scores: dict[int, float] = {}
        for rank, (idx, _score) in enumerate(dense):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (idx, _score) in enumerate(sparse):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(
        self, query: str, top_k: int = 10, candidate_k: int = 50, law_ids: set[str] | None = None
    ) -> list[RetrievedChunk]:
        """Hybrid RRF. `candidate_k` = ứng viên từ mỗi hệ thống trước fusion."""
        ranked = self._fuse_rrf(query, candidate_k, law_ids=law_ids)[:top_k]
        return [self._to_retrieved_chunk(idx, score) for idx, score in ranked]

    def _get_article_lookup(self) -> dict[tuple[str, int], str]:
        """`(law_id, dieu_so) -> nguyên văn TRỌN Điều` — dùng cho expand_to_parent_article."""
        if self._article_lookup is None:
            from autotender.rag.chunker import load_legal_articles

            lookup: dict[tuple[str, int], str] = {}
            corpus_dir = resolve_path("data/samples/legal_corpus")
            if corpus_dir.exists():
                for path in sorted(corpus_dir.glob("*.jsonl")):
                    for article in load_legal_articles(path):
                        lookup[(article.law_id, article.dieu_so)] = article.text
            self._article_lookup = lookup
        return self._article_lookup

    def expand_to_parent_article(self, chunk: RetrievedChunk) -> str:
        """Trả về nguyên văn TRỌN Điều chứa chunk (Parent-Child Chunking pattern).

        Giữ nguyên hành vi từ v1 — chỉ dùng cho LLM context, không ảnh hưởng
        retrieval/rerank/đánh giá Recall@k.
        """
        if chunk.law_id is None or chunk.dieu_so is None:
            return chunk.text
        return self._get_article_lookup().get((chunk.law_id, chunk.dieu_so), chunk.text)

    def retrieve_reranked(
        self, query: str, top_k: int = 5, candidate_k: int = 50, law_ids: set[str] | None = None
    ) -> list[RetrievedChunk]:
        """Hybrid RRF (top candidate_k) + cross-encoder rerank xuống top_k."""
        candidates = self._fuse_rrf(query, candidate_k, law_ids=law_ids)[:candidate_k]
        if not candidates:
            return []
        texts = [self._chunks[idx]["text"] for idx, _ in candidates]
        reranked = rerank_with_cross_encoder(self._cross_encoder_name, query, texts, top_k)
        return [self._to_retrieved_chunk(candidates[i][0], score) for i, score in reranked]

    def indices_for_law_ids(self, law_ids: set[str]) -> set[int]:
        """Backward compat — dùng bởi scripts/run_retrieval_eval.py cho ablation oracle filter."""
        return {i for i, c in enumerate(self._chunks) if c.get("law_id") in law_ids}
