"""Phân tích DL (Giai đoạn 3, đề cương RAG+LLM): trích xuất embedding từ 2 model đã đăng ký
(`rag/embedding_models.py`), so sánh không gian biểu diễn bằng t-SNE/UMAP + độ tách biệt
intra/inter-Điều. KHÔNG huấn luyện lại — chỉ dùng model pretrained có sẵn đúng tinh thần
đề cương ("DL thể hiện qua phân tích, không phải qua tự train").

Ví dụ: python scripts/analyze_embeddings.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.config import resolve_path  # noqa: E402
from autotender.eval.embedding_compare import intra_inter_article_similarity, reduce_dimensions  # noqa: E402
from autotender.rag.chunker import chunk_legal_corpus_dir  # noqa: E402
from autotender.rag.embedding_models import EMBEDDING_MODELS, encode_texts  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402
from autotender.utils.logging import get_logger  # noqa: E402

ensure_utf8_console()
logger = get_logger(__name__)

LEGAL_CORPUS_DIR = resolve_path("data/samples/legal_corpus")
FIGURES_DIR = resolve_path("reports/figures")
OUT_PATH = resolve_path("reports/embedding_comparison.json")

_LAW_COLORS = {"luat_22_2023_qh15": "#2563eb", "nd_214_2025_ndcp": "#dc2626"}
_LAW_LABELS = {"luat_22_2023_qh15": "Luật Đấu thầu 22/2023/QH15", "nd_214_2025_ndcp": "Nghị định 214/2025/NĐ-CP"}


def plot_projection(coords, law_ids: list[str], title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for law_id in sorted(set(law_ids)):
        mask = [lid == law_id for lid in law_ids]
        pts = coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.6, c=_LAW_COLORS.get(law_id, "gray"), label=_LAW_LABELS.get(law_id, law_id))
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Đã lưu %s", out_path)


def analyze_model(model_key: str, model_name: str, chunks) -> dict:
    from sentence_transformers import SentenceTransformer

    logger.info("Đang tải model '%s' (%s)...", model_key, model_name)
    encoder = SentenceTransformer(model_name)

    texts = [c.text for c in chunks]
    logger.info("Đang embed %d chunk...", len(texts))
    t0 = time.time()
    # `encode_texts` — cùng đường mã hoá với `scripts/build_legal_index.py` (sliding-window
    # mean-pooling cho chunk dài hơn max_seq_length), để số liệu so sánh ở đây phản ánh
    # ĐÚNG embedding thật sự đang nằm trong FAISS index, không phải bản bị cắt.
    embeddings = encode_texts(encoder, texts, batch_size=32, show_progress_bar=True)
    logger.info("Embed xong sau %.1fs, dim=%d.", time.time() - t0, embeddings.shape[1])

    article_ids = [(c.law_id, c.dieu_so) for c in chunks]
    separation = intra_inter_article_similarity(embeddings, article_ids)
    logger.info(
        "  intra=%.4f inter=%.4f separation=%.4f", separation["intra_mean"], separation["inter_mean"], separation["separation"]
    )

    law_ids = [c.law_id for c in chunks]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for method in ["tsne", "umap"]:
        logger.info("  Đang chạy %s...", method)
        coords = reduce_dimensions(embeddings, method=method)
        plot_projection(
            coords, law_ids, f"{model_key} ({embeddings.shape[1]}d) — {method.upper()}",
            FIGURES_DIR / f"embedding_{model_key}_{method}.png",
        )

    return {"dim": int(embeddings.shape[1]), "n_chunks": len(chunks), **separation}


def main() -> None:
    chunks = chunk_legal_corpus_dir(LEGAL_CORPUS_DIR)
    logger.info("Đã chunk %d đoạn từ %s.", len(chunks), LEGAL_CORPUS_DIR)
    if not chunks:
        logger.error("Không có chunk nào — chạy scripts/fetch_legal_corpus.py --all trước.")
        return

    results = {}
    for model_key, model_name in EMBEDDING_MODELS.items():
        results[model_key] = analyze_model(model_key, model_name, chunks)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Đã ghi kết quả so sánh vào %s", OUT_PATH)

    logger.info("=" * 60)
    for model_key, r in results.items():
        logger.info("%s (%dd): separation=%.4f (intra=%.4f, inter=%.4f)", model_key, r["dim"], r["separation"], r["intra_mean"], r["inter_mean"])


if __name__ == "__main__":
    main()
