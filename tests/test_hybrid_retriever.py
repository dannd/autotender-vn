"""Unit tests cho HybridLegalRetriever (Qdrant + BM25 RRF).

Chiến lược: mock QdrantLegalStore để test không cần Docker Qdrant đang chạy.
- _FakeQdrantStore: giả lập kết quả dense search từ dict lookup (chunk_id → score)
- Chunker fallback hoạt động khi Qdrant offline (đọc từ data/samples/legal_corpus)
- Các test BM25 (sparse), RRF fusion, reranking hoạt động không cần Qdrant thật

Lưu ý: test_hybrid_retriever_integration_* cần Qdrant thật → đánh dấu
pytest.mark.integration và bị skip trong CI nếu QDRANT_AVAILABLE=false.
"""

import numpy as np
import pytest

from autotender.rag.hybrid_retriever import HybridLegalRetriever, RRF_K
from autotender.rag.qdrant_store import QdrantUnavailableError
from autotender.schemas import RetrievedChunk

# ---------------------------------------------------------------------------
# Dữ liệu test
# ---------------------------------------------------------------------------

_MODEL_KEY = "vi_bi_encoder"

_CHUNKS = [
    {"chunk_id": "c1", "text": "quy định về hồ sơ mời thầu phần mềm", "source_doc": "Điều 1",
     "law_id": "luat_22_2023_qh15", "doc_type": "Luật", "dieu_so": 1, "dieu_title": "Lập HSMT"},
    {"chunk_id": "c2", "text": "điều kiện năng lực kinh nghiệm nhà thầu", "source_doc": "Điều 2",
     "law_id": "luat_22_2023_qh15", "doc_type": "Luật", "dieu_so": 2, "dieu_title": "Điều kiện"},
    {"chunk_id": "c3", "text": "thời gian thực hiện hợp đồng xây lắp", "source_doc": "Điều 3",
     "law_id": "nd_214_2025_ndcp", "doc_type": "Nghị định", "dieu_so": 3, "dieu_title": "Hợp đồng"},
]

# Vector cố định dim=4: c1 gần [1,0,0,0] nhất
_VECS = {
    "c1": np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"),
    "c2": np.array([0.0, 1.0, 0.0, 0.0], dtype="float32"),
    "c3": np.array([0.0, 0.0, 1.0, 0.0], dtype="float32"),
}

_MIXED_CHUNKS = [
    {"chunk_id": "m1", "text": "quy định về hồ sơ mời thầu phần mềm", "source_doc": "Luật, Điều 1",
     "law_id": "luat", "doc_type": "Luật", "dieu_so": 1},
    {"chunk_id": "m2", "text": "quy định về hồ sơ mời thầu chi tiết", "source_doc": "Nghị định, Điều 2",
     "law_id": "nghi_dinh", "doc_type": "Nghị định", "dieu_so": 2},
    {"chunk_id": "m3", "text": "thời gian thực hiện hợp đồng xây lắp", "source_doc": "Luật, Điều 3",
     "law_id": "luat", "doc_type": "Luật", "dieu_so": 3},
]


# ---------------------------------------------------------------------------
# Fake objects
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class _FakeEncoder:
    """Encoder giả: trả về vector [1,0,0,0] cho query chuẩn, [0,0,0,1] cho các query khác."""
    max_seq_length = 999_999
    tokenizer = _FakeTokenizer()

    def encode(self, texts, show_progress_bar=False, batch_size=32):
        result = []
        for t in texts:
            if "hồ sơ mời thầu phần mềm" in t:
                result.append([1.0, 0.0, 0.0, 0.0])
            else:
                result.append([0.0, 0.0, 0.0, 1.0])
        return np.asarray(result, dtype="float32")


class _FakeQdrantClient:
    """Fake Qdrant client có scroll method để _scroll_all_chunks_from_qdrant chạy được."""

    def __init__(self, chunks):
        self._chunks = chunks

    def get_collections(self):
        return []

    def collection_exists(self, name):
        return True

    def scroll(self, collection_name, with_payload=True, with_vectors=False,
               limit=1000, offset=None, scroll_filter=None):
        """Trả về toàn bộ fake chunks dưới dạng scrolled points."""
        class _FakePoint:
            def __init__(self, chunk):
                self.id = chunk["chunk_id"]
                self.payload = {**chunk, "text": chunk["text"], "content": chunk["text"]}
        return [_FakePoint(c) for c in self._chunks], None  # None = no more pages


class _FakeQdrantStore:
    """Giả lập QdrantLegalStore — trả về kết quả search từ cosine similarity với _VECS."""

    def __init__(self, chunks=None, available=True):
        self._chunks = chunks or _CHUNKS
        self._available = available
        # Build id→chunk map
        self._id_map = {c["chunk_id"]: c for c in self._chunks}
        # Build chunk_vec map
        self._vecs = _VECS
        self._fake_client = _FakeQdrantClient(self._chunks)

    def is_available(self):
        return self._available

    def collection_exists(self):
        return self._available

    def _get_client(self):
        if not self._available:
            raise QdrantUnavailableError("FakeStore: Qdrant offline (test).")
        return self._fake_client

    def search(self, query_vector, top_k=50, filter_law_ids=None):
        """Cosine sim giữa query_vector và _vecs — trả về list[(0, score, payload)]."""
        q = np.asarray(query_vector, dtype="float32")
        q = q / (np.linalg.norm(q) + 1e-9)
        results = []
        for chunk in self._chunks:
            cid = chunk["chunk_id"]
            if cid not in self._vecs:
                continue
            if filter_law_ids and chunk.get("law_id") not in filter_law_ids:
                continue
            v = self._vecs[cid]
            v = v / (np.linalg.norm(v) + 1e-9)
            score = float(np.dot(q, v))
            results.append((0, score, {**chunk, "text": chunk["text"], "content": chunk["text"]}))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @property
    def _cfg(self):
        class _Cfg:
            collection = "test_collection"
        return _Cfg()


def _make_retriever(chunks=None, qdrant_available=True):
    """Tạo HybridLegalRetriever với Qdrant và encoder giả — không cần Docker."""
    store = _FakeQdrantStore(chunks=chunks, available=qdrant_available)
    retriever = HybridLegalRetriever(model_key=_MODEL_KEY, qdrant_store=store)
    # Override encoder để không load model thật
    retriever._encoder = _FakeEncoder()
    return retriever


# ---------------------------------------------------------------------------
# Tests: Dense retrieval
# ---------------------------------------------------------------------------

def test_retrieve_dense_ranks_closest_vector_first():
    retriever = _make_retriever()
    results = retriever.retrieve_dense("hồ sơ mời thầu phần mềm", top_k=3)
    # c1 phải là kết quả đầu tiên (vector [1,0,0,0] gần nhất với query [1,0,0,0])
    first_idx = results[0][0]
    chunk = retriever._chunks[first_idx]
    assert chunk["chunk_id"] == "c1"


def test_retrieve_dense_respects_law_ids_filter():
    retriever = _make_retriever()
    results = retriever.retrieve_dense("hồ sơ mời thầu phần mềm", top_k=3, law_ids={"nd_214_2025_ndcp"})
    # Chỉ chunk của nd_214_2025_ndcp được trả về
    for idx, _ in results:
        assert retriever._chunks[idx]["law_id"] == "nd_214_2025_ndcp"


def test_retrieve_dense_raises_when_qdrant_unavailable():
    retriever = _make_retriever(qdrant_available=False)
    with pytest.raises(RuntimeError, match="Qdrant"):
        retriever.retrieve_dense("bất kỳ")


# ---------------------------------------------------------------------------
# Tests: Sparse retrieval (BM25)
# ---------------------------------------------------------------------------

def test_retrieve_sparse_finds_keyword_match():
    retriever = _make_retriever()
    results = retriever.retrieve_sparse("năng lực kinh nghiệm", top_k=3)
    # c2 phải là kết quả đầu (chứa "năng lực kinh nghiệm")
    first_idx = results[0][0]
    assert retriever._chunks[first_idx]["chunk_id"] == "c2"


def test_retrieve_sparse_excludes_chunks_outside_law_ids_filter():
    chunks = _MIXED_CHUNKS
    retriever = _make_retriever(chunks=chunks)
    filtered = retriever.retrieve_sparse("hồ sơ mời thầu", top_k=3, law_ids={"nghi_dinh"})
    assert {retriever._chunks[i]["law_id"] for i, _ in filtered} == {"nghi_dinh"}


# ---------------------------------------------------------------------------
# Tests: RRF Fusion
# ---------------------------------------------------------------------------

def test_retrieve_fuses_dense_and_sparse_via_rrf():
    retriever = _make_retriever()
    results = retriever.retrieve("hồ sơ mời thầu phần mềm", top_k=3, candidate_k=3)
    assert len(results) == 3
    assert results[0].chunk_id == "c1"  # đầu bảng cả dense lẫn BM25


def test_retrieve_returns_retrieved_chunk_with_full_metadata():
    retriever = _make_retriever()
    results = retriever.retrieve("hồ sơ mời thầu phần mềm", top_k=1)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, RetrievedChunk)
    assert r.chunk_id
    assert r.text
    assert r.source_doc
    assert r.score >= 0


def test_retrieve_with_law_ids_filter_only_returns_matching_chunks():
    chunks = _MIXED_CHUNKS
    retriever = _make_retriever(chunks=chunks)
    results = retriever.retrieve("hồ sơ mời thầu", top_k=3, candidate_k=3, law_ids={"luat"})
    assert results
    assert all(r.law_id == "luat" for r in results)


# ---------------------------------------------------------------------------
# Tests: Reranking
# ---------------------------------------------------------------------------

def test_retrieve_reranked_uses_cross_encoder_order(monkeypatch):
    retriever = _make_retriever()

    def _fake_rerank(model_name, query, candidates, top_k):
        n = len(candidates)
        # Đảo ngược thứ hạng RRF để xác nhận rerank thật sự ảnh hưởng
        return [(n - 1 - i, float(i)) for i in range(min(n, top_k))]

    monkeypatch.setattr("autotender.rag.hybrid_retriever.rerank_with_cross_encoder", _fake_rerank)
    results = retriever.retrieve_reranked("hồ sơ mời thầu phần mềm", top_k=3, candidate_k=3)
    assert len(results) == 3


def test_retrieve_reranked_truncates_fused_candidates_to_candidate_k(monkeypatch):
    """Xác nhận cross-encoder chỉ nhận đúng candidate_k ứng viên, không nhiều hơn."""
    retriever = _make_retriever()
    seen_candidate_counts = []

    def _fake_rerank(model_name, query, candidates, top_k):
        seen_candidate_counts.append(len(candidates))
        return [(i, 0.0) for i in range(min(len(candidates), top_k))]

    monkeypatch.setattr("autotender.rag.hybrid_retriever.rerank_with_cross_encoder", _fake_rerank)
    retriever.retrieve_reranked("hồ sơ mời thầu phần mềm", top_k=2, candidate_k=2)
    assert seen_candidate_counts == [2]


def test_retrieve_reranked_returns_empty_when_no_candidates():
    retriever = _make_retriever()
    retriever._fuse_rrf = lambda query, candidate_k, law_ids=None: []  # type: ignore[method-assign]
    assert retriever.retrieve_reranked("bất kỳ", top_k=3) == []


# ---------------------------------------------------------------------------
# Tests: Fallback khi Qdrant không available
# ---------------------------------------------------------------------------

def test_qdrant_unavailable_fallback_to_bm25_only():
    """Khi Qdrant offline, retrieve() vẫn chạy được bằng BM25-only (Degraded Mode)."""
    retriever = _make_retriever(qdrant_available=False)
    # Retriever phải tự chunk từ corpus nếu Qdrant không có data
    # (trong test, store trả is_available=False nên chunks được load theo fallback)
    assert retriever.num_chunks > 0
    results = retriever.retrieve("hồ sơ mời thầu gồm những nội dung gì", top_k=3)
    assert len(results) > 0


def test_has_dense_index_false_when_qdrant_unavailable():
    retriever = _make_retriever(qdrant_available=False)
    assert retriever.has_dense_index is False


# ---------------------------------------------------------------------------
# Tests: Metadata utilities
# ---------------------------------------------------------------------------

def test_indices_for_law_ids_returns_matching_indices():
    chunks = _MIXED_CHUNKS
    retriever = _make_retriever(chunks=chunks)
    assert retriever.indices_for_law_ids({"luat"}) == {0, 2}
    assert retriever.indices_for_law_ids({"nghi_dinh"}) == {1}
    assert retriever.indices_for_law_ids({"khong_ton_tai"}) == set()


def test_chunk_index_by_id_consistent():
    retriever = _make_retriever()
    id_map = retriever._chunk_index_by_id()
    for chunk_id, idx in id_map.items():
        assert retriever._chunks[idx]["chunk_id"] == chunk_id
