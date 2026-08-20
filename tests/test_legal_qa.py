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
        # Đủ lớn để `encode_texts` luôn đi qua nhánh 1-cửa-sổ với câu hỏi ngắn trong test.
        max_seq_length = 999_999

        class tokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return text.split()

        def encode(self, texts, show_progress_bar=False, batch_size=32):
            import numpy as np

            return np.asarray([[1.0, 0.0, 0.0, 0.0] for _ in texts], dtype="float32")

    r._encoder = _FakeEncoder()
    return r


def test_ask_uses_claude_when_available(monkeypatch, retriever):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    module = LegalQAModule(retriever=retriever)
    module._cfg = dict(module._cfg, use_rerank=False)  # tránh gọi cross-encoder thật (cần mạng) trong unit test
    monkeypatch.setattr(
        legal_qa_module, "call_claude", lambda **kwargs: "Câu trả lời từ Claude (Điều 44)."
    )
    # Không gọi Claude Haiku thật (query rewrite) trong unit test — giữ nguyên câu hỏi gốc.
    monkeypatch.setattr(legal_qa_module, "rewrite_query", lambda question, model=None: question)

    answer = module.ask("Hồ sơ mời thầu gồm những gì?")

    assert module.active_tier == 1
    assert "claude" in answer.model_used.lower()
    assert "Claude" in answer.answer
    assert len(answer.citations) == 1
    assert answer.citations[0].dieu_so == 44


def test_ask_uses_rewritten_query_for_retrieval_not_for_displayed_question(monkeypatch, retriever):
    """Câu hỏi hiển thị/trả lời (`QAAnswer.question`) phải giữ nguyên bản gốc của người dùng —
    chỉ câu dùng để TRUY HỒI mới bị viết lại (HyDE-lite, xem `generation/query_rewrite.py`)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    module = LegalQAModule(retriever=retriever)
    # `use_query_rewrite` mặc định False (đo thật cho thấy làm giảm chất lượng truy hồi, xem
    # configs/models.yaml) — bật tường minh ở đây vì test này kiểm tra riêng NHÁNH bật.
    module._cfg = dict(module._cfg, use_rerank=False, use_query_rewrite=True)
    monkeypatch.setattr(legal_qa_module, "call_claude", lambda **kwargs: "Câu trả lời từ Claude.")

    seen_queries: list[str] = []
    original_retrieve = retriever.retrieve

    def _spy_retrieve(query, **kwargs):
        seen_queries.append(query)
        return original_retrieve(query, **kwargs)

    monkeypatch.setattr(retriever, "retrieve", _spy_retrieve)
    monkeypatch.setattr(
        legal_qa_module, "rewrite_query", lambda question, model=None: "hồ sơ mời thầu (thuật ngữ chuẩn hoá)"
    )

    answer = module.ask("thầu qua mạng thế nào")

    assert answer.question == "thầu qua mạng thế nào"
    assert seen_queries == ["hồ sơ mời thầu (thuật ngữ chuẩn hoá)"]


def test_ask_skips_rewrite_when_disabled_in_config(monkeypatch, retriever):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    module = LegalQAModule(retriever=retriever)
    module._cfg = dict(module._cfg, use_rerank=False, use_query_rewrite=False)
    monkeypatch.setattr(legal_qa_module, "call_claude", lambda **kwargs: "Câu trả lời từ Claude.")

    def _fail_if_called(question, model=None):
        raise AssertionError("rewrite_query không nên được gọi khi use_query_rewrite=False")

    monkeypatch.setattr(legal_qa_module, "rewrite_query", _fail_if_called)

    module.ask("Hồ sơ mời thầu gồm những gì?")


def test_ask_uses_classifier_law_ids_filter_when_enabled(monkeypatch, retriever):
    """`use_law_id_filter=True` phải truyền law_ids từ bộ phân loại xuống `retriever.retrieve`
    (metadata filtering, xem `generation/law_classifier.py`)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    module = LegalQAModule(retriever=retriever)
    module._cfg = dict(module._cfg, use_rerank=False, use_law_id_filter=True)
    monkeypatch.setattr(legal_qa_module, "call_claude", lambda **kwargs: "Câu trả lời từ Claude.")

    seen_law_ids: list[set[str] | None] = []
    original_retrieve = retriever.retrieve

    def _spy_retrieve(query, **kwargs):
        seen_law_ids.append(kwargs.get("law_ids"))
        return original_retrieve(query, **kwargs)

    monkeypatch.setattr(retriever, "retrieve", _spy_retrieve)
    monkeypatch.setattr(legal_qa_module, "classify_relevant_law_ids", lambda question, model=None: {"x"})

    module.ask("Nghị định 45/2026 quy định gì về nghiệm thu phần mềm?")

    assert seen_law_ids == [{"x"}]


def test_ask_skips_law_id_filter_when_disabled_in_config(monkeypatch, retriever):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    module = LegalQAModule(retriever=retriever)
    module._cfg = dict(module._cfg, use_rerank=False, use_law_id_filter=False)
    monkeypatch.setattr(legal_qa_module, "call_claude", lambda **kwargs: "Câu trả lời từ Claude.")

    def _fail_if_called(question, model=None):
        raise AssertionError("classify_relevant_law_ids không nên được gọi khi use_law_id_filter=False")

    monkeypatch.setattr(legal_qa_module, "classify_relevant_law_ids", _fail_if_called)

    module.ask("Hồ sơ mời thầu gồm những gì?")


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
