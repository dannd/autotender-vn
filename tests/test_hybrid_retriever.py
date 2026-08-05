import json

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from autotender.rag.hybrid_retriever import HybridLegalRetriever  # noqa: E402
from autotender.rag.index import FaissChunkIndex  # noqa: E402

_MODEL_KEY = "vi_bi_encoder"

_CHUNKS = [
    {"chunk_id": "c1", "text": "quy định về hồ sơ mời thầu phần mềm", "source_doc": "Điều 1", "law_id": "x", "dieu_so": 1},
    {"chunk_id": "c2", "text": "điều kiện năng lực kinh nghiệm nhà thầu", "source_doc": "Điều 2", "law_id": "x", "dieu_so": 2},
    {"chunk_id": "c3", "text": "thời gian thực hiện hợp đồng xây lắp", "source_doc": "Điều 3", "law_id": "x", "dieu_so": 3},
]

# Vector cố định (dim=4), c1 gần trục X nhất -> query [1,0,0,0] phải trả c1 đầu tiên khi dense.
_VECS = {
    "c1": [1.0, 0.0, 0.0, 0.0],
    "c2": [0.0, 1.0, 0.0, 0.0],
    "c3": [0.0, 0.0, 1.0, 0.0],
}


class _FakeEncoder:
    def encode(self, texts, show_progress_bar=False):
        if texts == ["hồ sơ mời thầu phần mềm"]:
            return np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype="float32")
        return np.asarray([_VECS.get(t, [0, 0, 0, 1]) for t in texts], dtype="float32")


@pytest.fixture
def index_dir(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in _CHUNKS:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    model_dir = tmp_path / _MODEL_KEY
    model_dir.mkdir()
    index = FaissChunkIndex(dim=4)
    index.add([_VECS[c["chunk_id"]] for c in _CHUNKS])
    index.save(model_dir / "index.faiss")
    (model_dir / "dim.txt").write_text("4", encoding="utf-8")
    (model_dir / "model_name.txt").write_text("fake", encoding="utf-8")
    return tmp_path


def test_retrieve_dense_ranks_closest_vector_first(index_dir):
    retriever = HybridLegalRetriever(model_key=_MODEL_KEY, index_dir=index_dir)
    retriever._encoder = _FakeEncoder()

    results = retriever.retrieve_dense("hồ sơ mời thầu phần mềm", top_k=3)

    assert results[0][0] == 0  # index của c1 trong _CHUNKS


def test_retrieve_sparse_finds_keyword_match(index_dir):
    retriever = HybridLegalRetriever(model_key=_MODEL_KEY, index_dir=index_dir)

    results = retriever.retrieve_sparse("năng lực kinh nghiệm", top_k=3)

    assert results[0][0] == 1  # index của c2 (chứa đúng 2 từ khoá)


def test_retrieve_fuses_dense_and_sparse_via_rrf(index_dir):
    retriever = HybridLegalRetriever(model_key=_MODEL_KEY, index_dir=index_dir)
    retriever._encoder = _FakeEncoder()

    results = retriever.retrieve("hồ sơ mời thầu phần mềm", top_k=3, candidate_k=3)

    assert len(results) == 3
    assert results[0].chunk_id == "c1"
    assert all(r.law_id == "x" for r in results)


def test_retrieve_reranked_uses_cross_encoder_order(monkeypatch, index_dir):
    retriever = HybridLegalRetriever(model_key=_MODEL_KEY, index_dir=index_dir)
    retriever._encoder = _FakeEncoder()

    # Cross-encoder giả lập: đảo ngược thứ hạng RRF (đưa ứng viên cuối lên đầu) để xác
    # nhận kết quả thật sự đi qua bước rerank, không chỉ trả nguyên fusion RRF.
    def _fake_rerank(model_name, query, candidates, top_k):
        n = len(candidates)
        ranked = [(n - 1 - i, float(i)) for i in range(n)]
        return ranked[:top_k]

    monkeypatch.setattr("autotender.rag.hybrid_retriever.rerank_with_cross_encoder", _fake_rerank)

    results = retriever.retrieve_reranked("hồ sơ mời thầu phần mềm", top_k=3, candidate_k=3)

    assert len(results) == 3
    assert all(r.law_id == "x" for r in results)


def test_retrieve_reranked_returns_empty_when_no_candidates(index_dir):
    retriever = HybridLegalRetriever(model_key=_MODEL_KEY, index_dir=index_dir)
    retriever._fuse_rrf = lambda query, candidate_k: []  # type: ignore[method-assign]

    assert retriever.retrieve_reranked("bất kỳ", top_k=3) == []


def test_missing_faiss_index_falls_back_to_bm25_only_without_error(tmp_path):
    """Chưa chạy scripts/build_legal_index.py (thư mục index rỗng) — retrieve() vẫn phải
    thành công bằng cách tự chunk từ data/samples/legal_corpus/*.jsonl (đã commit) và chỉ
    dùng BM25 (không dense), giữ đúng nguyên tắc "Tier 3 luôn chạy được" cho generator/QA."""
    retriever = HybridLegalRetriever(model_key=_MODEL_KEY, index_dir=tmp_path)

    assert retriever.num_chunks > 0  # tự chunk từ corpus luật thật đã commit
    assert retriever._faiss_index is None
    assert retriever.has_dense_index is False

    results = retriever.retrieve("hồ sơ mời thầu gồm những nội dung gì", top_k=3)
    assert len(results) > 0


def test_has_dense_index_true_when_faiss_built(index_dir):
    retriever = HybridLegalRetriever(model_key=_MODEL_KEY, index_dir=index_dir)
    assert retriever.has_dense_index is True


def test_retrieve_dense_raises_clear_error_when_faiss_missing(tmp_path):
    retriever = HybridLegalRetriever(model_key=_MODEL_KEY, index_dir=tmp_path)

    with pytest.raises(RuntimeError, match="build_legal_index"):
        retriever.retrieve_dense("bất kỳ")
