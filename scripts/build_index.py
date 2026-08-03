"""Dựng sẵn FAISS index (Tier 2, bi-encoder pretrained) cho corpus mẫu (Mục 6/M4).

Đây là bước tối ưu (cache embeddings xuống đĩa) — KHÔNG bắt buộc để chạy pipeline,
vì `RetrieverModule` (Tier 3, BM25) luôn hoạt động mà không cần chạy script này trước
(Degraded Mode, Mục 2.1). Nếu `sentence-transformers`/`faiss-cpu` chưa cài được hoặc
không tải được model, script log rõ và thoát — không phải lỗi chặn tiến độ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.config import get_models_settings, resolve_path  # noqa: E402
from autotender.rag.chunker import chunk_corpus_dir  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402
from autotender.utils.logging import get_logger  # noqa: E402

ensure_utf8_console()
logger = get_logger(__name__)

CORPUS_DIR = resolve_path("data/samples/corpus")
OUT_INDEX = resolve_path("data/processed/faiss.index")
OUT_META = resolve_path("data/processed/faiss_meta.jsonl")


def main() -> None:
    chunks = chunk_corpus_dir(CORPUS_DIR)
    logger.info("Đã chunk %d đoạn từ corpus tại %s", len(chunks), CORPUS_DIR)

    cfg = get_models_settings().retriever
    model_name = cfg.get("bi_encoder_base", "bkai-foundation-models/vietnamese-bi-encoder")

    try:
        from sentence_transformers import SentenceTransformer

        from autotender.rag.index import FaissChunkIndex
    except ImportError as e:
        logger.warning("Chưa cài `sentence-transformers`/`faiss-cpu` (%s). Bỏ qua — Tier 3 (BM25) vẫn dùng được.", e)
        return

    try:
        encoder = SentenceTransformer(model_name)
        embeddings = encoder.encode([c.text for c in chunks], show_progress_bar=True)
    except Exception as e:  # noqa: BLE001 — thường do lỗi mạng khi tải model từ HF Hub
        logger.warning("Không tải/chạy được bi-encoder '%s' (%s). Bỏ qua — Tier 3 (BM25) vẫn dùng được.", model_name, e)
        return

    index = FaissChunkIndex(dim=embeddings.shape[1])
    index.add(embeddings)
    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    index.save(OUT_INDEX)

    with open(OUT_META, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"chunk_id": c.chunk_id, "text": c.text, "source_doc": c.source_doc}, ensure_ascii=False) + "\n")

    logger.info("Đã ghi FAISS index (%d vector, dim=%d) vào %s", len(chunks), embeddings.shape[1], OUT_INDEX)


if __name__ == "__main__":
    main()
