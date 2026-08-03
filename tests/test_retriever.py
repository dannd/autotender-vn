from autotender.models.retriever import RetrieverModule


def test_retriever_tier3_bm25_returns_relevant_chunks():
    module = RetrieverModule()
    assert module.num_chunks > 0

    results = module.retrieve("nhãn hiệu xuất xứ hàng hóa", top_k=5)

    assert module.active_tier == 3
    assert 0 < len(results) <= 5
    assert all(r.score >= 0 for r in results)
    assert any("nhãn hiệu" in r.text.lower() for r in results)


def test_retriever_tier3_respects_top_k():
    module = RetrieverModule()
    results = module.retrieve("tiến độ thực hiện hợp đồng", top_k=2)
    assert len(results) <= 2
