from autotender.rag import rerank as rerank_module


class _FakeCrossEncoder:
    instances_created = 0

    def __init__(self, model_name):
        _FakeCrossEncoder.instances_created += 1
        self.model_name = model_name

    def predict(self, pairs):
        return [1.0 / (i + 1) for i in range(len(pairs))]


def test_rerank_with_cross_encoder_reuses_cached_model(monkeypatch):
    """`CrossEncoder(model_name)` tải trọng số + gọi HF Hub — tốn ~6-7s đo thực tế. Gọi
    `rerank_with_cross_encoder` nhiều lần với CÙNG model_name (đúng cách dùng thật của
    `HybridLegalRetriever.retrieve_reranked`, luôn dùng `self._cross_encoder_name` cố
    định) không được tải lại model mỗi lần."""
    rerank_module._MODEL_CACHE.clear()
    _FakeCrossEncoder.instances_created = 0

    def _fake_get_cross_encoder(model_name):
        if model_name not in rerank_module._MODEL_CACHE:
            rerank_module._MODEL_CACHE[model_name] = _FakeCrossEncoder(model_name)
        return rerank_module._MODEL_CACHE[model_name]

    monkeypatch.setattr(rerank_module, "_get_cross_encoder", _fake_get_cross_encoder)

    rerank_module.rerank_with_cross_encoder("fake-model", "câu hỏi 1", ["a", "b"], top_k=2)
    rerank_module.rerank_with_cross_encoder("fake-model", "câu hỏi 2", ["c", "d", "e"], top_k=2)
    rerank_module.rerank_with_cross_encoder("fake-model", "câu hỏi 3", ["f"], top_k=1)

    assert _FakeCrossEncoder.instances_created == 1


def test_rerank_with_cross_encoder_returns_sorted_top_k(monkeypatch):
    rerank_module._MODEL_CACHE.clear()
    monkeypatch.setattr(rerank_module, "_get_cross_encoder", lambda name: _FakeCrossEncoder(name))

    result = rerank_module.rerank_with_cross_encoder("fake-model", "q", ["a", "b", "c"], top_k=2)

    assert len(result) == 2
    assert result[0][1] >= result[1][1]  # giảm dần
