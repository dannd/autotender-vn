"""Nạp kho tri thức pháp luật vào Qdrant Vector DB.

Script này thay thế `scripts/build_legal_index.py` (FAISS) cho pipeline Qdrant.

Luồng:
1. Chunk corpus luật theo Điều/Khoản (chunker.py) — 684 chunk thật
2. Embed bằng model cấu hình trong app.yaml (mặc định: deepx-embedding-v1)
3. Upsert vào Qdrant collection `legal_chunks` với payload đầy đủ

Idempotent: chạy lại không tạo duplicate (upsert theo chunk_id deterministic UUID).

Ví dụ:
    python scripts/ingest_to_qdrant.py
    python scripts/ingest_to_qdrant.py --model deepx_v1
    python scripts/ingest_to_qdrant.py --model vi_bi_encoder --recreate-collection
    python scripts/ingest_to_qdrant.py --dry-run   # chỉ đếm chunk, không embed/upsert
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.config import get_app_settings, resolve_path  # noqa: E402
from autotender.rag.chunker import chunk_legal_corpus_dir  # noqa: E402
from autotender.rag.embedding_models import EMBEDDING_MODELS, encode_texts  # noqa: E402
from autotender.rag.qdrant_store import QdrantLegalStore  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402
from autotender.utils.logging import get_logger  # noqa: E402

ensure_utf8_console()
logger = get_logger(__name__)

LEGAL_CORPUS_DIR = resolve_path("data/samples/legal_corpus")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        choices=list(EMBEDDING_MODELS),
        default=None,
        help="Model embedding (mặc định: đọc từ configs/app.yaml → embedding.model_key)",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        default=False,
        help="Xóa và tạo lại collection (dùng khi đổi model hoặc vector_size)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Chỉ đếm chunk và kiểm tra kết nối Qdrant, không embed/upsert",
    )
    args = parser.parse_args()

    cfg = get_app_settings()

    # Model key: CLI argument > biến môi trường > config
    model_key = args.model or cfg.embedding.model_key
    model_name = EMBEDDING_MODELS[model_key]
    vector_size = cfg.embedding.vector_size
    batch_size = cfg.embedding.batch_size

    logger.info("=== AutoTender-VN — Ingest vào Qdrant ===")
    logger.info("Qdrant:     %s:%s", cfg.qdrant.host, cfg.qdrant.port)
    logger.info("Collection: %s", cfg.qdrant.collection)
    logger.info("Model:      %s (%s)", model_key, model_name)
    logger.info("Vector dim: %d", vector_size)
    logger.info("Corpus:     %s", LEGAL_CORPUS_DIR)

    # --- Bước 1: Chunk ---
    logger.info("\n[1/4] Chunking corpus pháp luật...")
    t0 = time.time()
    chunks = chunk_legal_corpus_dir(LEGAL_CORPUS_DIR)
    if not chunks:
        logger.error("Không tìm thấy chunk nào — hãy kiểm tra %s", LEGAL_CORPUS_DIR)
        sys.exit(1)
    logger.info("    Đã tạo %d chunk trong %.1fs.", len(chunks), time.time() - t0)

    if args.dry_run:
        logger.info("\n[DRY RUN] %d chunks sẵn sàng để ingest.", len(chunks))
        # Kiểm tra kết nối Qdrant nhưng không upsert
        store = QdrantLegalStore(cfg=cfg.qdrant, vector_size=vector_size)
        if store.is_available():
            info = store.collection_info()
            logger.info("[DRY RUN] Qdrant available. Collection info: %s", info)
        else:
            logger.warning("[DRY RUN] Qdrant KHÔNG available tại %s:%s", cfg.qdrant.host, cfg.qdrant.port)
        return

    # --- Bước 2: Kết nối và chuẩn bị collection ---
    logger.info("\n[2/4] Kết nối Qdrant và chuẩn bị collection...")
    store = QdrantLegalStore(cfg=cfg.qdrant, vector_size=vector_size)
    try:
        store.ensure_collection(recreate=args.recreate_collection)
    except Exception as e:
        logger.error(
            "Không thể kết nối/tạo collection Qdrant: %s\n"
            "Hãy chạy `docker compose up -d qdrant` và thử lại.", e
        )
        sys.exit(1)

    # --- Bước 3: Embed ---
    logger.info("\n[3/4] Đang tải embedding model '%s'...", model_name)
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    encoder = SentenceTransformer(model_name)
    logger.info("    Tải model xong sau %.1fs.", time.time() - t0)

    logger.info("    Đang embed %d chunks (batch_size=%d)...", len(chunks), batch_size)
    t0 = time.time()
    texts = [c.text for c in chunks]
    embeddings = encode_texts(encoder, texts, batch_size=batch_size, show_progress_bar=True)
    logger.info("    Embed xong sau %.1fs. Shape: %s", time.time() - t0, embeddings.shape)

    # Kiểm tra dim khớp với collection
    actual_dim = embeddings.shape[1]
    if actual_dim != vector_size:
        logger.warning(
            "Vector dim thực tế (%d) khác với vector_size trong config (%d). "
            "Hãy chạy lại với --recreate-collection hoặc cập nhật configs/app.yaml.",
            actual_dim, vector_size,
        )
        # Tái tạo collection với dim đúng
        logger.info("Tái tạo collection với dim=%d...", actual_dim)
        store2 = QdrantLegalStore(cfg=cfg.qdrant, vector_size=actual_dim)
        store2.ensure_collection(recreate=True)
        store = store2

    # --- Bước 4: Upsert vào Qdrant ---
    logger.info("\n[4/4] Upsert vào Qdrant collection '%s'...", cfg.qdrant.collection)
    t0 = time.time()
    store.upsert_chunks(chunks, embeddings)
    logger.info("    Upsert xong sau %.1fs.", time.time() - t0)

    # Báo cáo kết quả
    info = store.collection_info()
    logger.info("\n=== Hoàn tất ===")
    logger.info("Collection : %s", info.get("collection"))
    logger.info("Vectors    : %s", info.get("vectors_count"))
    logger.info("Dashboard  : %s", info.get("dashboard_url", f"http://{cfg.qdrant.host}:{cfg.qdrant.port}/dashboard"))
    logger.info("\nLần sau khởi động app, dense retrieval sẽ tự động dùng collection này.")
    logger.info("Chạy lại script này với --recreate-collection nếu muốn đổi model embedding.")


if __name__ == "__main__":
    main()
