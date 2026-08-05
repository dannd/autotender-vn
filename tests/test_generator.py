import json

import pytest

from autotender.models import generator as generator_module
from autotender.models.generator import GeneratorModule, verify_numeric_consistency
from autotender.schemas import ExtractedField


def _fields() -> list[ExtractedField]:
    return [
        ExtractedField(name="PACKAGE_NAME", value="Mua sắm thiết bị CNTT", confidence=0.9, source="regex"),
        ExtractedField(name="INVESTOR", value="Sở Y tế tỉnh X", confidence=0.9, source="regex"),
        ExtractedField(name="VALUE", value="5.200.000.000", confidence=0.9, source="regex"),
        ExtractedField(name="FUNDING", value="Ngân sách nhà nước năm 2026", confidence=0.9, source="regex"),
        ExtractedField(name="DURATION", value="90 ngày", confidence=0.9, source="regex"),
    ]


def test_generator_tier3_fills_slots_without_any_checkpoint():
    module = GeneratorModule()
    result = module.generate_section("chuong_III.muc_4", _fields())

    assert module.active_tier == 3
    assert "Mua sắm thiết bị CNTT" in result.text
    assert "5.200.000.000" in result.text
    assert len(result.citations) > 0


def test_generator_rejects_out_of_scope_section():
    module = GeneratorModule()
    try:
        module.generate_section("chuong_I.muc_1", _fields())
        assert False, "phải raise ValueError cho section ngoài phạm vi"
    except ValueError:
        pass


def test_verify_numeric_consistency_flags_unknown_number():
    fields = _fields()
    text = "Giá gói thầu là 5.200.000.000 đồng nhưng doanh thu yêu cầu tối thiểu 99.999.999.999 đồng."
    flags = verify_numeric_consistency(text, fields)
    assert len(flags) == 1
    assert flags[0].rule_code == "R4"
    assert "99.999.999.999" in flags[0].sentence


def test_verify_numeric_consistency_no_flag_when_numbers_match():
    fields = _fields()
    text = "Giá gói thầu là 5.200.000.000 đồng, thời gian thực hiện 90 ngày."
    flags = verify_numeric_consistency(text, fields)
    assert flags == []


def test_generate_section_uses_claude_when_available(monkeypatch, tmp_path):
    faiss = pytest.importorskip("faiss")
    from autotender.rag.hybrid_retriever import HybridLegalRetriever
    from autotender.rag.index import FaissChunkIndex

    chunk = {
        "chunk_id": "c1", "text": "Yêu cầu về nhãn hiệu, xuất xứ hàng hóa phải nêu tương đương.",
        "source_doc": "Điều 44, Khoản 3", "law_id": "x", "dieu_so": 44,
    }
    with open(tmp_path / "chunks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    model_dir = tmp_path / "vi_bi_encoder"
    model_dir.mkdir()
    index = FaissChunkIndex(dim=4)
    index.add([[1.0, 0.0, 0.0, 0.0]])
    index.save(model_dir / "index.faiss")
    (model_dir / "dim.txt").write_text("4", encoding="utf-8")

    retriever = HybridLegalRetriever(model_key="vi_bi_encoder", index_dir=tmp_path)

    class _FakeEncoder:
        def encode(self, texts, show_progress_bar=False):
            import numpy as np

            return np.asarray([[1.0, 0.0, 0.0, 0.0] for _ in texts], dtype="float32")

    retriever._encoder = _FakeEncoder()
    monkeypatch.setattr(
        "autotender.rag.hybrid_retriever.rerank_with_cross_encoder",
        lambda model_name, query, candidates, top_k: [(0, 1.0)],
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(generator_module, "call_claude", lambda **kwargs: "Nội dung do Claude soạn (Điều 44).")

    module = GeneratorModule(retriever=retriever)
    result = module.generate_section("chuong_III.muc_4", _fields())

    assert module.active_tier == 1
    assert "Claude" in result.text
    assert len(result.citations) == 1
    assert result.citations[0].dieu_so == 44
