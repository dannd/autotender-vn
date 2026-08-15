"""QdrantLegalStore — lưu trữ và truy vấn vector embedding pháp luật trên Qdrant.

Thay thế FaissChunkIndex (FAISS flat file) bằng Qdrant Vector DB:
- Payload đầy đủ gắn liền với từng vector (law_id, dieu_so, source_doc, text, khoan_so)
- Metadata filtering trực tiếp trong DB, không cần lọc thủ công bằng Python sau khi query
- Dashboard UI tại http://localhost:6333/dashboard để xem collection, vector, payload
- Khởi động Qdrant qua `docker compose up -d qdrant` (xem docker-compose.yml)

Collection schema:
  - Vector: cosine similarity, dim = EmbeddingConfig.vector_size (mặc định 1024d cho deepx_v1)
  - Payload fields được index để filter nhanh:
      - law_id (keyword)      — lọc theo văn bản luật cụ thể (Luật/NĐ/TT)
      - dieu_so (integer)     — lọc theo số điều
      - source_doc (text)     — nhãn hiển thị UI (vd "Luật Đấu thầu 22/2023 — Điều 43")
      - text (text)           — nội dung chunk để BM25/cross-encoder dùng (không embed ở đây)
      - khoan_so (keyword)    — khoản thứ mấy trong Điều (None nếu chunk = trọn Điều)
"""

from __future__ import annotations

import uuid
from typing import Any

from autotender.config import QdrantConfig
from autotender.rag.chunker import RawChunk
from autotender.schemas import RetrievedChunk
from autotender.utils.logging import get_logger

logger = get_logger(__name__)

# Tên field payload — dùng hằng số để tránh lỗi typo rải rác trong code.
PAYLOAD_TEXT = "text"
PAYLOAD_SOURCE_DOC = "source_doc"
PAYLOAD_LAW_ID = "law_id"
PAYLOAD_DIEU_SO = "dieu_so"
PAYLOAD_KHOAN_SO = "khoan_so"
PAYLOAD_CHUNK_ID = "chunk_id"


class QdrantUnavailableError(RuntimeError):
    """Không kết nối được Qdrant — Tier 1 dense retrieval sẽ tự fallback về BM25 (sparse)."""


class QdrantLegalStore:
    """Wrapper quanh qdrant-client cho kho tri thức pháp luật.

    Lazy-connect: kết nối Qdrant chỉ được thực hiện lần đầu có thao tác thật
    (`upsert_chunks` hoặc `search`) — tránh crash ứng dụng ngay lúc import khi
    Docker chưa khởi động xong (đặc biệt quan trọng cho Degraded Mode).
    """

    def __init__(self, cfg: QdrantConfig, vector_size: int = 1024) -> None:
        self._cfg = cfg
        self._vector_size = vector_size
        self._client = None  # lazy-connect

    def _get_client(self):
        """Kết nối Qdrant lần đầu khi cần — raise QdrantUnavailableError nếu thất bại."""
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise QdrantUnavailableError("Thư viện `qdrant-client` chưa được cài đặt.") from e

        try:
            client = QdrantClient(
                host=self._cfg.host,
                port=self._cfg.port,
                timeout=self._cfg.timeout,
            )
            # Ping để xác nhận Qdrant đang chạy ngay lúc connect (không đợi tới lúc query)
            client.get_collections()
            self._client = client
            logger.info("Đã kết nối Qdrant tại %s:%s", self._cfg.host, self._cfg.port)
            return self._client
        except Exception as e:
            raise QdrantUnavailableError(
                f"Không kết nối được Qdrant tại {self._cfg.host}:{self._cfg.port} — "
                "hãy chạy `docker compose up -d qdrant` và thử lại. "
                f"Chi tiết lỗi: {e}"
            ) from e

    def is_available(self) -> bool:
        """Kiểm tra nhanh xem Qdrant có đang chạy không (không raise exception)."""
        try:
            self._get_client()
            return True
        except QdrantUnavailableError:
            return False

    def collection_exists(self) -> bool:
        """Kiểm tra collection đã tồn tại trong Qdrant chưa."""
        try:
            client = self._get_client()
            return client.collection_exists(self._cfg.collection)
        except QdrantUnavailableError:
            return False

    def collection_info(self) -> dict[str, Any]:
        """Thông tin collection dùng cho Trang 6 (Bảng điều khiển).

        Trả về dict rỗng nếu Qdrant không available, không raise exception.
        """
        try:
            client = self._get_client()
            if not client.collection_exists(self._cfg.collection):
                return {"status": "collection_not_found", "collection": self._cfg.collection}
            info = client.get_collection(self._cfg.collection)
            return {
                "status": "ok",
                "collection": self._cfg.collection,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "vector_size": self._vector_size,
                "host": f"{self._cfg.host}:{self._cfg.port}",
                "dashboard_url": f"http://{self._cfg.host}:{self._cfg.port}/dashboard",
            }
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": str(e)}

    def ensure_collection(self, recreate: bool = False) -> None:
        """Tạo collection nếu chưa có. `recreate=True` xóa và tạo lại (dùng khi đổi model/dim).

        Các payload fields được tạo index để filter nhanh:
        - law_id, khoan_so: keyword index (exact match)
        - dieu_so: integer index (range filter)
        - source_doc, text: không index (text search thuần BM25, không dùng Qdrant full-text)
        """
        from qdrant_client.models import (
            Distance,
            FieldCondition,
            PayloadSchemaType,
            VectorParams,
        )

        client = self._get_client()

        if recreate and client.collection_exists(self._cfg.collection):
            logger.warning("Xóa collection '%s' để tạo lại (recreate=True).", self._cfg.collection)
            client.delete_collection(self._cfg.collection)

        if not client.collection_exists(self._cfg.collection):
            client.create_collection(
                collection_name=self._cfg.collection,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )
            # Index payload fields để filter nhanh
            client.create_payload_index(self._cfg.collection, PAYLOAD_LAW_ID, PayloadSchemaType.KEYWORD)
            client.create_payload_index(self._cfg.collection, PAYLOAD_DIEU_SO, PayloadSchemaType.INTEGER)
            client.create_payload_index(self._cfg.collection, PAYLOAD_KHOAN_SO, PayloadSchemaType.KEYWORD)
            logger.info(
                "Đã tạo collection '%s' (dim=%d, cosine) với payload index cho law_id/dieu_so/khoan_so.",
                self._cfg.collection, self._vector_size,
            )
        else:
            logger.info("Collection '%s' đã tồn tại, bỏ qua bước tạo.", self._cfg.collection)

    def upsert_chunks(self, chunks: list[RawChunk], embeddings) -> None:
        """Đẩy danh sách chunk cùng embedding vào Qdrant collection.

        Dùng chunk_id làm deterministic UUID (upsert idempotent — chạy lại không tạo duplicate).
        """
        import numpy as np
        from qdrant_client.models import PointStruct

        client = self._get_client()
        vecs = np.asarray(embeddings, dtype="float32")

        points = []
        for chunk, vec in zip(chunks, vecs):
            # Tạo UUID deterministic từ chunk_id (hex 12 ký tự) để upsert idempotent
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"autotender:{chunk.chunk_id}"))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload={
                        PAYLOAD_CHUNK_ID: chunk.chunk_id,
                        PAYLOAD_TEXT: chunk.text,
                        PAYLOAD_SOURCE_DOC: chunk.source_doc,
                        PAYLOAD_LAW_ID: chunk.law_id,
                        PAYLOAD_DIEU_SO: chunk.dieu_so,
                        PAYLOAD_KHOAN_SO: None,  # chunker không expose khoan_so riêng hiện tại
                    },
                )
            )

        # Upsert theo batch 256 để tránh request quá lớn
        batch_size = 256
        total = len(points)
        for i in range(0, total, batch_size):
            batch = points[i : i + batch_size]
            client.upsert(collection_name=self._cfg.collection, points=batch)
            logger.info("Đã upsert %d/%d points vào collection '%s'.", min(i + batch_size, total), total, self._cfg.collection)

        logger.info("Hoàn tất upsert %d chunks vào Qdrant collection '%s'.", total, self._cfg.collection)

    def search(
        self,
        query_vector,
        top_k: int = 50,
        filter_law_ids: set[str] | None = None,
    ) -> list[tuple[int, float, dict]]:
        """Dense search trả về list[(index_trong_points, score, payload)].

        `filter_law_ids`: nếu không None, chỉ tìm trong các chunk thuộc các văn bản này
        (metadata filter được thực hiện IN DB, không phải sau khi query — hiệu quả hơn FAISS).

        Trả về list rỗng (không raise) nếu Qdrant không available.
        """
        import numpy as np

        try:
            from qdrant_client.models import FieldCondition, Filter, MatchAny

            client = self._get_client()
        except QdrantUnavailableError:
            logger.warning("Qdrant không available — bỏ qua dense search, chỉ dùng BM25.")
            return []

        vec = np.asarray(query_vector, dtype="float32").tolist()

        query_filter = None
        if filter_law_ids:
            query_filter = Filter(
                must=[FieldCondition(key=PAYLOAD_LAW_ID, match=MatchAny(any=list(filter_law_ids)))]
            )

        try:
            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=self._cfg.collection,
                    query=vec,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
                results = response.points
            else:
                results = client.search(
                    collection_name=self._cfg.collection,
                    query_vector=vec,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Lỗi khi search Qdrant: %s — fallback về BM25.", e)
            return []

        # Trả về (position placeholder=0, score, payload) — caller dùng payload trực tiếp
        return [(0, hit.score, hit.payload) for hit in results]


    def count_by_law_id(self, law_id: str) -> int:
        """Đếm số vector của một văn bản luật cụ thể trong Qdrant collection."""
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            client = self._get_client()
            res = client.count(
                collection_name=self._cfg.collection,
                count_filter=Filter(
                    must=[FieldCondition(key=PAYLOAD_LAW_ID, match=MatchValue(value=law_id))]
                ),
            )
            return res.count
        except Exception as e:  # noqa: BLE001
            logger.warning("Không đếm được vectors cho law_id '%s': %s", law_id, e)
            return 0

    def delete_by_law_id(self, law_id: str) -> int:
        """Xóa toàn bộ vector và payload của một văn bản luật khỏi Qdrant collection (CRUD Delete)."""
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            client = self._get_client()
            before_count = self.count_by_law_id(law_id)
            client.delete(
                collection_name=self._cfg.collection,
                points_selector=Filter(
                    must=[FieldCondition(key=PAYLOAD_LAW_ID, match=MatchValue(value=law_id))]
                ),
            )
            logger.info("Đã xóa %d vectors của law_id '%s' khỏi collection '%s'.", before_count, law_id, self._cfg.collection)
            return before_count
        except Exception as e:
            logger.error("Lỗi khi xóa vectors cho law_id '%s': %s", law_id, e)
            raise

    def payload_to_retrieved_chunk(self, payload: dict, score: float) -> RetrievedChunk:
        """Chuyển payload Qdrant thành RetrievedChunk để tương thích với phần còn lại của hệ thống."""
        return RetrievedChunk(
            chunk_id=payload.get(PAYLOAD_CHUNK_ID, ""),
            text=payload.get(PAYLOAD_TEXT, ""),
            source_doc=payload.get(PAYLOAD_SOURCE_DOC, ""),
            score=score,
            law_id=payload.get(PAYLOAD_LAW_ID),
            dieu_so=payload.get(PAYLOAD_DIEU_SO),
        )

