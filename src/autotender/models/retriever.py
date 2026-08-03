"""M4 — Retrieval (RAG) (Mục 6/M4).

Tier 1: bi-encoder fine-tuned (`models/retriever_bi_encoder`) + FAISS IndexFlatIP,
        rerank bằng cross-encoder fine-tuned.
Tier 2: bi-encoder pretrained (`bkai-foundation-models/vietnamese-bi-encoder`, zero-shot)
        + FAISS, rerank bằng cross-encoder pretrained.
Tier 3: BM25 (rag/bm25.py) — LUÔN THÀNH CÔNG, không cần mạng/GPU/thư viện ML nặng.
"""

from __future__ import annotations

from pathlib import Path

from autotender.config import get_models_settings, resolve_path
from autotender.models.base import BaseModule, TierUnavailableError
from autotender.rag.bm25 import build_bm25_index
from autotender.rag.chunker import RawChunk, chunk_corpus_dir
from autotender.schemas import RetrievedChunk


class RetrieverModule(BaseModule[list[RetrievedChunk]]):
    module_name = "M4-Retriever"

    def __init__(self, corpus_dir: str | Path | None = None):
        super().__init__()
        self._cfg = get_models_settings().retriever
        self._corpus_dir = Path(corpus_dir) if corpus_dir else resolve_path("data/samples/corpus")
        self._chunks: list[RawChunk] = chunk_corpus_dir(self._corpus_dir)
        self._bm25_index = None
        self._tier1_bi_encoder = None
        self._tier1_faiss = None
        self._tier2_bi_encoder = None
        self._tier2_faiss = None

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.run(query, top_k)

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    # -- Tier 1 -------------------------------------------------------------
    def _try_tier1(self, query: str, top_k: int) -> list[RetrievedChunk]:
        checkpoint = resolve_path(self._cfg.get("tier1_checkpoint", "models/retriever_bi_encoder"))
        if not Path(checkpoint).exists():
            raise TierUnavailableError(f"Không tìm thấy checkpoint tại {checkpoint}")
        return self._embed_search(str(checkpoint), query, top_k, tier=1)

    # -- Tier 2 ---------------------------------------------------------------
    def _try_tier2(self, query: str, top_k: int) -> list[RetrievedChunk]:
        model_name = self._cfg.get("bi_encoder_base", "bkai-foundation-models/vietnamese-bi-encoder")
        return self._embed_search(model_name, query, top_k, tier=2)

    def _embed_search(self, model_name: str, query: str, top_k: int, tier: int) -> list[RetrievedChunk]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise TierUnavailableError("Thư viện `sentence-transformers` chưa cài đặt") from e

        cache_attr = f"_tier{tier}_bi_encoder"
        faiss_attr = f"_tier{tier}_faiss"
        encoder = getattr(self, cache_attr)
        if encoder is None:
            try:
                encoder = SentenceTransformer(model_name)
                setattr(self, cache_attr, encoder)
            except Exception as e:  # noqa: BLE001 — thường do không tải được model (mạng)
                raise TierUnavailableError(f"Không tải được bi-encoder '{model_name}': {e}") from e

        faiss_index = getattr(self, faiss_attr)
        if faiss_index is None:
            try:
                from autotender.rag.index import FaissChunkIndex

                embeddings = encoder.encode([c.text for c in self._chunks], show_progress_bar=False)
                faiss_index = FaissChunkIndex(dim=embeddings.shape[1])
                faiss_index.add(embeddings)
                setattr(self, faiss_attr, faiss_index)
            except Exception as e:  # noqa: BLE001
                raise TierUnavailableError(f"Không dựng được FAISS index: {e}") from e

        try:
            query_vec = encoder.encode([query], show_progress_bar=False)[0]
            top_n = min(self._cfg.get("top_k_retrieve", 50), len(self._chunks))
            indices, scores = faiss_index.search(query_vec, top_n)
        except Exception as e:  # noqa: BLE001
            raise TierUnavailableError(f"Suy luận Tier {tier} lỗi: {e}") from e

        candidates = [self._chunks[i] for i in indices if 0 <= i < len(self._chunks)]
        try:
            from autotender.rag.rerank import rerank_with_cross_encoder

            cross_model = self._cfg.get("cross_encoder_base", "xlm-roberta-base")
            reranked = rerank_with_cross_encoder(cross_model, query, [c.text for c in candidates], top_k)
            return [
                RetrievedChunk(chunk_id=candidates[i].chunk_id, text=candidates[i].text, source_doc=candidates[i].source_doc, score=score)
                for i, score in reranked
            ]
        except Exception:  # noqa: BLE001 — rerank thất bại: vẫn trả kết quả bi-encoder, không rơi cả tier
            return [
                RetrievedChunk(chunk_id=c.chunk_id, text=c.text, source_doc=c.source_doc, score=float(s))
                for c, s in zip(candidates[:top_k], scores[:top_k])
            ]

    # -- Tier 3 (bắt buộc luôn thành công) -----------------------------------
    def _try_tier3(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._bm25_index is None:
            self._bm25_index = build_bm25_index([c.text for c in self._chunks])
        results = self._bm25_index.search(query, top_k)
        return [
            RetrievedChunk(chunk_id=self._chunks[i].chunk_id, text=self._chunks[i].text, source_doc=self._chunks[i].source_doc, score=score)
            for i, score in results
        ]
