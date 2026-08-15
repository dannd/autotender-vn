"""Quản lý Kho tri thức Pháp luật (Legal Knowledge Base CRUD & Ingestion/Chunking Pipeline).

Cung cấp đầy đủ các thao tác:
- C (Create / Ingest): Thêm mới văn bản luật, chunking theo Điều/Khoản, vector hóa và nạp vào Qdrant.
- R (Read / Inspect): Liệt kê danh mục văn bản, kiểm tra số lượng Điều/Chunks, trạng thái index trong Qdrant.
- U (Update / Re-chunk / Re-index): Cập nhật nội dung văn bản, chunking lại và nạp vector mới.
- D (Delete / Purge): Xóa vector của văn bản khỏi Qdrant collection và xóa file lưu trữ cục bộ.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autotender.config import get_app_settings, resolve_path
from autotender.rag.chunker import RawChunk, chunk_legal_article, load_legal_articles
from autotender.rag.embedding_models import EMBEDDING_MODELS, encode_texts
from autotender.rag.qdrant_store import QdrantLegalStore
from autotender.schemas import LegalArticle
from autotender.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CORPUS_DIR = resolve_path("data/samples/legal_corpus")


class KnowledgeManager:
    """Quản lý toàn diện vòng đời của Kho tri thức (Knowledge Base)."""

    def __init__(self, corpus_dir: Path | str | None = None, qdrant_store: QdrantLegalStore | None = None) -> None:
        self.corpus_dir = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS_DIR
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        
        cfg = get_app_settings()
        self.qdrant_store = qdrant_store or QdrantLegalStore(cfg=cfg.qdrant, vector_size=cfg.embedding.vector_size)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from autotender.rag.embedding_models import load_embedding_model
            cfg = get_app_settings()
            self._encoder = load_embedding_model(cfg.embedding.model_key)
        return self._encoder


    def list_documents(self) -> list[dict[str, Any]]:
        """Liệt kê toàn bộ các văn bản luật trong kho tri thức kèm số liệu thống kê."""
        docs = []
        is_qdrant_up = self.qdrant_store.is_available()

        for file_path in sorted(self.corpus_dir.glob("*.jsonl")):
            try:
                articles = load_legal_articles(file_path)
                if not articles:
                    continue
                first = articles[0]
                law_id = first.law_id
                law_name = first.law_name
                
                # Đếm số chunk lý thuyết khi phân tách theo Điều/Khoản
                total_chunks = sum(len(chunk_legal_article(a)) for a in articles)
                
                # Đếm số vector thực tế trong Qdrant
                qdrant_count = self.qdrant_store.count_by_law_id(law_id) if is_qdrant_up else 0
                
                fetched_at = first.fetched_at
                file_stat = file_path.stat()
                
                docs.append({
                    "law_id": law_id,
                    "law_name": law_name,
                    "file_name": file_path.name,
                    "file_size_kb": round(file_stat.st_size / 1024, 1),
                    "total_articles": len(articles),
                    "total_chunks": total_chunks,
                    "indexed_vectors_in_qdrant": qdrant_count,
                    "is_indexed": qdrant_count > 0,
                    "fetched_at": fetched_at.isoformat() if fetched_at else None,
                    "last_modified": datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc).isoformat(),
                })
            except Exception as e:
                logger.warning("Lỗi khi đọc file %s: %s", file_path.name, e)
                continue

        return docs

    def get_document(self, law_id: str) -> dict[str, Any] | None:
        """Lấy chi tiết văn bản và toàn bộ các Điều/Khoản theo `law_id`."""
        file_path = self.corpus_dir / f"{law_id}.jsonl"
        if not file_path.exists():
            return None

        articles = load_legal_articles(file_path)
        chunks: list[dict[str, Any]] = []
        for art in articles:
            raw_chunks = chunk_legal_article(art)
            for rc in raw_chunks:
                chunks.append({
                    "chunk_id": rc.chunk_id,
                    "dieu_so": rc.dieu_so,
                    "source_doc": rc.source_doc,
                    "text": rc.text,
                    "char_count": len(rc.text),
                    "word_count": len(rc.text.split()),
                })

        return {
            "law_id": law_id,
            "law_name": articles[0].law_name if articles else "",
            "file_name": file_path.name,
            "articles_count": len(articles),
            "chunks_count": len(chunks),
            "articles": [
                {
                    "dieu_so": a.dieu_so,
                    "dieu_title": a.dieu_title,
                    "chuong_so": a.chuong_so,
                    "chuong_title": a.chuong_title,
                    "text": a.text,
                }
                for a in articles
            ],
            "chunks": chunks,
        }

    def ingest_document(
        self,
        law_id: str,
        law_name: str,
        articles: list[LegalArticle],
        embed_and_upsert: bool = True,
    ) -> dict[str, Any]:
        """Tạo/cập nhật văn bản mới vào file JSONL và nạp vào Qdrant (Create/Update)."""
        file_path = self.corpus_dir / f"{law_id}.jsonl"
        
        # 1. Lưu file JSONL
        with open(file_path, "w", encoding="utf-8") as f:
            for art in articles:
                f.write(json.dumps(art.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Đã lưu %d Điều của văn bản '%s' vào %s", len(articles), law_id, file_path)

        # 2. Chunking theo Điều/Khoản
        all_chunks: list[RawChunk] = []
        for art in articles:
            all_chunks.extend(chunk_legal_article(art))

        # 3. Vectorization & Upsert vào Qdrant
        upserted_count = 0
        if embed_and_upsert and all_chunks:
            self.qdrant_store.ensure_collection(recreate=False)
            encoder = self._get_encoder()
            texts = [c.text for c in all_chunks]
            embeddings = encode_texts(encoder, texts, batch_size=32)
            self.qdrant_store.upsert_chunks(all_chunks, embeddings)
            upserted_count = len(all_chunks)
            logger.info("Đã vector hoá và nạp %d chunks vào Qdrant cho '%s'", upserted_count, law_id)

        return {
            "status": "success",
            "law_id": law_id,
            "law_name": law_name,
            "articles_count": len(articles),
            "chunks_count": len(all_chunks),
            "upserted_to_qdrant": upserted_count,
        }

    def delete_document(self, law_id: str) -> dict[str, Any]:
        """Xóa hoàn toàn văn bản khỏi Qdrant và xóa file lưu trữ cục bộ (Delete/Purge)."""
        file_path = self.corpus_dir / f"{law_id}.jsonl"
        file_deleted = False
        if file_path.exists():
            file_path.unlink()
            file_deleted = True

        # Xóa vectors trong Qdrant
        vectors_deleted = 0
        if self.qdrant_store.is_available() and self.qdrant_store.collection_exists():
            vectors_deleted = self.qdrant_store.delete_by_law_id(law_id)

        return {
            "status": "success",
            "law_id": law_id,
            "file_deleted": file_deleted,
            "vectors_deleted_from_qdrant": vectors_deleted,
        }

    def reindex_document(self, law_id: str) -> dict[str, Any]:
        """Xóa vector cũ và nạp lại toàn bộ chunk của văn bản vào Qdrant."""
        file_path = self.corpus_dir / f"{law_id}.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file {file_path}")

        articles = load_legal_articles(file_path)
        if not articles:
            raise ValueError(f"File {file_path} không có Điều nào.")

        # 1. Xóa vector cũ trong Qdrant
        if self.qdrant_store.is_available():
            self.qdrant_store.delete_by_law_id(law_id)

        # 2. Chunking & Ingest lại
        all_chunks: list[RawChunk] = []
        for art in articles:
            all_chunks.extend(chunk_legal_article(art))

        encoder = self._get_encoder()
        texts = [c.text for c in all_chunks]
        embeddings = encode_texts(encoder, texts, batch_size=32)
        self.qdrant_store.upsert_chunks(all_chunks, embeddings)

        return {
            "status": "success",
            "law_id": law_id,
            "reindexed_chunks": len(all_chunks),
        }
