import json

import pytest

faiss = pytest.importorskip("faiss")

from autotender.generation.claude_client import ClaudeUnavailableError  # noqa: E402
from autotender.models import legal_qa as legal_qa_module  # noqa: E402
from autotender.models.legal_qa import LegalQAModule  # noqa: E402
from autotender.rag.hybrid_retriever import HybridLegalRetriever  # noqa: E402
from autotender.rag.index import FaissChunkIndex  # noqa: E402

_MODEL_KEY = "vi_bi_encoder"
_CHUNKS = [
    {"chunk_id": "c1", "text": "hồ sơ mời thầu gồm chỉ dẫn nhà thầu và bảng dữ liệu", "source_doc": "Điều 44", "law_id": "x", "dieu_so": 44},
]


@pytest.fixture
def retriever(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in _CHUNKS:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    model_dir = tmp_path / _MODEL_KEY
    model_dir.mkdir()
    index = FaissChunkIndex(dim=4)
    index.add([[1.0, 0.0, 0.0, 0.0]])
    index.save(model_dir / "index.faiss")
    (model_dir / "dim.txt").write_text("4", encoding="utf-8")

    r = HybridLegalRetriever(model_key=_MODEL_KEY, index_dir=tmp_path)

    class _FakeEncoder:
        def encode(self, texts, show_progress_bar=False):
            import numpy as np

            return np.asarray([[1.0, 0.0, 0.0, 0.0] for _ in texts], dtype="float32")

    r._encoder = _FakeEncoder()
    return r


def test_ask_uses_claude_when_available(monkeypatch, retriever):
    module = LegalQAModule(retriever=retriever)
    module._cfg = dict(module._cfg, use_rerank=False)  # tránh gọi cross-encoder thật (cần mạng) trong unit test
    monkeypatch.setattr(
        legal_qa_module, "call_claude", lambda **kwargs: "Câu trả lời từ Claude (Điều 44)."
    )

    answer = module.ask("Hồ sơ mời thầu gồm những gì?")

    assert module.active_tier == 1
    assert answer.model_used == "claude-sonnet-5"
    assert "Claude" in answer.answer
    assert len(answer.citations) == 1
    assert answer.citations[0].dieu_so == 44


def test_ask_falls_back_to_template_when_claude_unavailable(monkeypatch, retriever):
    module = LegalQAModule(retriever=retriever)
    module._cfg = dict(module._cfg, use_rerank=False)  # tránh gọi cross-encoder thật (cần mạng) trong unit test

    def _raise(**kwargs):
        raise ClaudeUnavailableError("no api key")

    monkeypatch.setattr(legal_qa_module, "call_claude", _raise)

    answer = module.ask("Hồ sơ mời thầu gồm những gì?")

    assert module.active_tier == 3
    assert answer.model_used == "template"
    assert len(answer.citations) == 1
    assert "Điều 44" in answer.answer
