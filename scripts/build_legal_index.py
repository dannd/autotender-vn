"""Dựng index RAG cho kho tri thức luật thật: chunk theo Điều/Khoản, embed bằng 2 model
để so sánh (Mục "Phân tích DL"), build FAISS (dense) cho mỗi model. BM25 (sparse) không
cần lưu — dựng lại nhanh từ chunk text lúc truy vấn (`rag/bm25.py`).

Kết quả:
  data/index/chunks.jsonl              — metadata chunk dùng chung cho mọi model (thứ tự
                                          cố định = thứ tự dòng, ánh xạ với vị trí trong FAISS)
  data/index/{model_key}/index.faiss   — FAISS IndexFlatIP cho từng model trong EMBEDDING_MODELS
  data/index/{model_key}/dim.txt       — số chiều embedding (cần khi load lại FaissChunkIndex)

Ví dụ: python scripts/build_legal_index.py
       python scripts/build_legal_index.py --model vi_bi_encoder
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.config import resolve_path  # noqa: E402
from autotender.rag.chunker import RawChunk, chunk_legal_corpus_dir  # noqa: E402
from autotender.rag.embedding_models import EMBEDDING_MODELS  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402
from autotender.utils.logging import get_logger  # noqa: E402

ensure_utf8_console()
logger = get_logger(__name__)

LEGAL_CORPUS_DIR = resolve_path("data/samples/legal_corpus")
INDEX_DIR = resolve_path("data/index")


def save_chunk_metadata(chunks: list[RawChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "source_doc": c.source_doc,
                        "law_id": c.law_id,
                        "dieu_so": c.dieu_so,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_index_for_model(model_key: str, model_name: str, chunks: list[RawChunk]) -> None:
    from sentence_transformers import SentenceTransformer

    from autotender.rag.index import FaissChunkIndex

    logger.info("Đang tải model '%s' (%s)...", model_key, model_name)
    t0 = time.time()
    encoder = SentenceTransformer(model_name)
    logger.info("Tải xong sau %.1fs.", time.time() - t0)

    texts = [c.text for c in chunks]
    logger.info("Đang embed %d chunk...", len(texts))
    t0 = time.time()
    embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=32)
    logger.info("Embed xong sau %.1fs, dim=%d.", time.time() - t0, embeddings.shape[1])

    index = FaissChunkIndex(dim=embeddings.shape[1])
    index.add(embeddings)

    model_dir = INDEX_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    index.save(model_dir / "index.faiss")
    (model_dir / "dim.txt").write_text(str(embeddings.shape[1]), encoding="utf-8")
    (model_dir / "model_name.txt").write_text(model_name, encoding="utf-8")
    logger.info("Đã lưu FAISS index tại %s", model_dir / "index.faiss")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(EMBEDDING_MODELS), help="Chỉ build 1 model (mặc định: cả 2)")
    args = parser.parse_args()

    chunks = chunk_legal_corpus_dir(LEGAL_CORPUS_DIR)
    if not chunks:
        logger.error("Không có chunk nào — chạy scripts/fetch_legal_corpus.py --all trước.")
        return
    logger.info("Đã chunk %d đoạn từ %s.", len(chunks), LEGAL_CORPUS_DIR)

    save_chunk_metadata(chunks, INDEX_DIR / "chunks.jsonl")

    model_keys = [args.model] if args.model else list(EMBEDDING_MODELS)
    for model_key in model_keys:
        build_index_for_model(model_key, EMBEDDING_MODELS[model_key], chunks)

    logger.info("Hoàn tất. %d chunk, %d model.", len(chunks), len(model_keys))


if __name__ == "__main__":
    main()
