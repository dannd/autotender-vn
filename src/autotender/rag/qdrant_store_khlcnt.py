"""QdrantKhlcntStore — lưu trữ và truy vấn vector embedding tài liệu KHLCNT.

Collection riêng cho tài liệu Kế hoạch lựa chọn nhà thầu (KHLCNT) mà người dùng upload
ở Trang 2 — phục vụ RAG context cho Mức 2 (Soạn thảo HSMT).

Tách biệt khỏi `legal_chunks` vì:
- KHLCNT là tài liệu của từng gói thầu cụ thể, không phải corpus pháp luật chung
- Vòng đời khác: có thể xóa/thay thế theo từng phiên làm việc
- Payload khác: có document_id, uploaded_by, project_name, contract_value

Collection schema (khlcnt_chunks):
  - Named Vector "dense": cosine similarity, cùng dim với legal_chunks
  - Payload fields:
      - chunk_id (keyword)      — ID chunk
      - document_id (keyword)   — ID tài liệu KHLCNT (nhóm chunks của cùng 1 file)
      - content (text)          — nội dung text chunk
      - source_doc (text)       — tên file / nguồn
      - page_number (integer)   — trang PDF (nếu có)
      - doc_type (keyword)      — luôn là "KHLCNT"
      - uploaded_by (keyword)   — username người upload
      - uploaded_at (text)      — ISO timestamp
      - project_name (text)     — tên dự án/gói thầu
      - contract_value (float)  — giá trị hợp đồng (nếu NER parse được)
      - word_count (integer)    — số từ trong chunk
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from autotender.config import QdrantConfig
from autotender.utils.logging import get_logger

logger = get_logger(__name__)

KHLCNT_COLLECTION_NAME = "khlcnt_chunks"
VECTOR_DENSE = "dense"

# Payload field names
PAYLOAD_CHUNK_ID = "chunk_id"
PAYLOAD_DOCUMENT_ID = "document_id"
PAYLOAD_CONTENT = "content"
PAYLOAD_SOURCE_DOC = "source_doc"
PAYLOAD_PAGE_NUMBER = "page_number"
PAYLOAD_DOC_TYPE = "doc_type"
PAYLOAD_UPLOADED_BY = "uploaded_by"
PAYLOAD_UPLOADED_AT = "uploaded_at"
PAYLOAD_PROJECT_NAME = "project_name"
PAYLOAD_CONTRACT_VALUE = "contract_value"
PAYLOAD_WORD_COUNT = "word_count"


class QdrantKhlcntStore:
    """Wrapper Qdrant cho kho tài liệu KHLCNT — tách biệt với legal_chunks.

    Hỗ trợ:
    - Upsert chunks từ một tài liệu KHLCNT (theo document_id)
    - Search semantic trong kho KHLCNT (dùng cho Mức 2 RAG context)
    - Delete toàn bộ chunks của một document_id (khi người dùng upload lại)
    - List tất cả document_id đã nạp (Dashboard quản lý)
    """

    def __init__(self, cfg: QdrantConfig, vector_size: int = 768) -> None:
        self._cfg = cfg
        self._vector_size = vector_size
        self._collection = KHLCNT_COLLECTION_NAME
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise RuntimeError("Thư viện `qdrant-client` chưa được cài đặt.") from e

        try:
            client = QdrantClient(
                host=self._cfg.host,
                port=self._cfg.port,
                timeout=self._cfg.timeout,
            )
            client.get_collections()
            self._client = client
            logger.info("QdrantKhlcntStore: kết nối Qdrant tại %s:%s", self._cfg.host, self._cfg.port)
            return self._client
        except Exception as e:
            raise RuntimeError(
                f"Không kết nối được Qdrant tại {self._cfg.host}:{self._cfg.port} — "
                f"hãy chạy `docker compose up -d qdrant`. Chi tiết: {e}"
            ) from e

    def is_available(self) -> bool:
        try:
            self._get_client()
            return True
        except RuntimeError:
            return False

    def collection_exists(self) -> bool:
        try:
            return self._get_client().collection_exists(self._collection)
        except RuntimeError:
            return False

    def ensure_collection(self, recreate: bool = False) -> None:
        """Tạo collection khlcnt_chunks nếu chưa có."""
        from qdrant_client.models import (
            Distance,
            VectorParams,
            PayloadSchemaType,
        )

        client = self._get_client()

        if recreate and client.collection_exists(self._collection):
            logger.warning("Xóa collection '%s' để tạo lại (recreate=True).", self._collection)
            client.delete_collection(self._collection)

        if not client.collection_exists(self._collection):
            client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    VECTOR_DENSE: VectorParams(
                        size=self._vector_size,
                        distance=Distance.COSINE,
                    )
                },
            )
            # Index payload để filter và quản lý theo document_id
            for field, schema_type in [
                (PAYLOAD_DOCUMENT_ID, PayloadSchemaType.KEYWORD),
                (PAYLOAD_DOC_TYPE, PayloadSchemaType.KEYWORD),
                (PAYLOAD_UPLOADED_BY, PayloadSchemaType.KEYWORD),
                (PAYLOAD_PAGE_NUMBER, PayloadSchemaType.INTEGER),
            ]:
                client.create_payload_index(self._collection, field, schema_type)

            logger.info(
                "Đã tạo collection '%s' (named vector 'dense', dim=%d, cosine).",
                self._collection, self._vector_size,
            )
        else:
            logger.info("Collection '%s' đã tồn tại.", self._collection)

    def upsert_document_chunks(
        self,
        document_id: str,
        chunks_text: list[str],
        embeddings,
        source_doc: str = "",
        uploaded_by: str = "system",
        project_name: str = "",
        contract_value: float | None = None,
        page_numbers: list[int | None] | None = None,
    ) -> None:
        """Nạp tất cả chunks của 1 tài liệu KHLCNT vào Qdrant.

        Idempotent: xóa chunks cũ của document_id trước khi upsert mới.
        """
        import numpy as np
        from qdrant_client.models import PointStruct

        # Xóa chunks cũ của document này (nếu upload lại)
        self.delete_document(document_id)

        client = self._get_client()
        self.ensure_collection()

        vecs = np.asarray(embeddings, dtype="float32")
        uploaded_at = datetime.now(timezone.utc).isoformat()
        if page_numbers is None:
            page_numbers = [None] * len(chunks_text)

        points = []
        for idx, (text, vec, page_no) in enumerate(zip(chunks_text, vecs, page_numbers)):
            chunk_key = f"khlcnt:{document_id}:{idx}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_key))
            points.append(
                PointStruct(
                    id=point_id,
                    vector={VECTOR_DENSE: vec.tolist()},
                    payload={
                        PAYLOAD_CHUNK_ID: chunk_key,
                        PAYLOAD_DOCUMENT_ID: document_id,
                        PAYLOAD_CONTENT: text,
                        PAYLOAD_SOURCE_DOC: source_doc,
                        PAYLOAD_PAGE_NUMBER: page_no,
                        PAYLOAD_DOC_TYPE: "KHLCNT",
                        PAYLOAD_UPLOADED_BY: uploaded_by,
                        PAYLOAD_UPLOADED_AT: uploaded_at,
                        PAYLOAD_PROJECT_NAME: project_name,
                        PAYLOAD_CONTRACT_VALUE: contract_value,
                        PAYLOAD_WORD_COUNT: len(text.split()),
                    },
                )
            )

        batch_size = 128
        total = len(points)
        for i in range(0, total, batch_size):
            client.upsert(collection_name=self._collection, points=points[i : i + batch_size])
            logger.info(
                "KHLCNT upsert %d/%d chunks cho document_id='%s'.",
                min(i + batch_size, total), total, document_id,
            )
        logger.info("Hoàn tất upsert %d KHLCNT chunks (document_id='%s').", total, document_id)

    def search(
        self,
        query_vector,
        top_k: int = 10,
        document_id: str | None = None,
    ) -> list[tuple[float, dict]]:
        """Semantic search trong kho KHLCNT. Trả về list[(score, payload)].

        `document_id`: nếu không None, chỉ tìm trong tài liệu này.
        """
        import numpy as np
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not self.is_available() or not self.collection_exists():
            logger.warning("QdrantKhlcntStore: collection không available — bỏ qua search.")
            return []

        vec = np.asarray(query_vector, dtype="float32").tolist()
        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[FieldCondition(key=PAYLOAD_DOCUMENT_ID, match=MatchValue(value=document_id))]
            )

        try:
            response = self._get_client().query_points(
                collection_name=self._collection,
                query=vec,
                using=VECTOR_DENSE,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            return [(hit.score, hit.payload or {}) for hit in response.points]
        except Exception as e:  # noqa: BLE001
            logger.warning("Lỗi search KHLCNT: %s", e)
            return []

    def delete_document(self, document_id: str) -> int:
        """Xóa toàn bộ chunks của 1 tài liệu KHLCNT (khi upload lại)."""
        if not self.collection_exists():
            return 0
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            client = self._get_client()
            client.delete(
                collection_name=self._collection,
                points_selector=Filter(
                    must=[FieldCondition(key=PAYLOAD_DOCUMENT_ID, match=MatchValue(value=document_id))]
                ),
            )
            logger.info("Đã xóa chunks của document_id='%s' khỏi '%s'.", document_id, self._collection)
            return 0  # Qdrant delete không trả về count trực tiếp
        except Exception as e:
            logger.error("Lỗi xóa KHLCNT document '%s': %s", document_id, e)
            raise

    def list_documents(self) -> list[dict[str, Any]]:
        """Liệt kê tất cả document_id đã nạp (dùng cho Dashboard quản lý KHLCNT)."""
        if not self.collection_exists():
            return []
        try:
            client = self._get_client()
            # Scroll lấy 1 point từ mỗi document_id để lấy metadata
            # (Qdrant chưa có GROUP BY native, dùng scroll + dedup)
            seen: dict[str, dict] = {}
            offset = None
            while True:
                result, next_offset = client.scroll(
                    collection_name=self._collection,
                    with_payload=True,
                    with_vectors=False,
                    limit=500,
                    offset=offset,
                )
                for point in result:
                    p = point.payload or {}
                    doc_id = p.get(PAYLOAD_DOCUMENT_ID, "")
                    if doc_id and doc_id not in seen:
                        seen[doc_id] = {
                            "document_id": doc_id,
                            "source_doc": p.get(PAYLOAD_SOURCE_DOC, ""),
                            "project_name": p.get(PAYLOAD_PROJECT_NAME, ""),
                            "uploaded_by": p.get(PAYLOAD_UPLOADED_BY, ""),
                            "uploaded_at": p.get(PAYLOAD_UPLOADED_AT, ""),
                            "contract_value": p.get(PAYLOAD_CONTRACT_VALUE),
                        }
                if next_offset is None:
                    break
                offset = next_offset
            return list(seen.values())
        except Exception as e:  # noqa: BLE001
            logger.warning("Lỗi list KHLCNT documents: %s", e)
            return []

    def collection_info(self) -> dict[str, Any]:
        """Thông tin collection khlcnt_chunks (cho Dashboard)."""
        try:
            client = self._get_client()
            if not client.collection_exists(self._collection):
                return {"status": "collection_not_found", "collection": self._collection}
            info = client.get_collection(self._collection)
            return {
                "status": "ok",
                "collection": self._collection,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "host": f"{self._cfg.host}:{self._cfg.port}",
                "dashboard_url": f"http://{self._cfg.host}:{self._cfg.port}/dashboard",
            }
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": str(e)}
